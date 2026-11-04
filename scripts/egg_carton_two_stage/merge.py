"""Merge the relabeled egg-carton dataset with the open-carton dataset.

Runs LeRobot's own ``merge_datasets`` with its default settings (what ``lerobot-edit-dataset
--operation.type merge`` calls), which deduplicates tasks by string and remaps ``task_index`` / episode /
frame indices. Before merging it:

* stops if fps, robot_type, camera keys or any feature (dtype, shape, names, video info) differ;
* requires the relabeled dataset's tasks to be exactly [OPEN_PROMPT, EGG_PROMPT];
* compares the open dataset's tasks with OPEN_PROMPT; on mismatch it rewrites them to OPEN_PROMPT in a
  temporary staging copy (videos hard-linked, the source stays untouched) and reports the rewrite;
* reports the range of dim 19 (``right_gripper.pos``) in the open dataset's state and action.

Usage:
    python scripts/egg_carton_two_stage/merge.py \
        --relabeled <root> --relabeled-repo-id <id> --open <root> --open-repo-id <id> \
        --dst <root> --dst-repo-id <id>
"""

import argparse
import os
import pathlib
import shutil
import sys

from egg_prompts import EGG_PROMPT
from egg_prompts import OPEN_PROMPT
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

GRIPPER_DIM = 19


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--relabeled", type=pathlib.Path, required=True)
    parser.add_argument("--relabeled-repo-id", required=True)
    parser.add_argument("--open", type=pathlib.Path, required=True)
    parser.add_argument("--open-repo-id", required=True)
    parser.add_argument("--dst", type=pathlib.Path, required=True, help="Output root; must not exist.")
    parser.add_argument("--dst-repo-id", required=True)
    return parser.parse_args()


def compare_metadata(a, b) -> list[str]:
    """Differences that make two LeRobot datasets unmergeable."""
    problems = [
        f"{attr}: {getattr(a, attr)!r} vs {getattr(b, attr)!r}"
        for attr in ("fps", "robot_type", "video_keys")
        if getattr(a, attr) != getattr(b, attr)
    ]
    for key in sorted(set(a.features) | set(b.features)):
        fa, fb = a.features.get(key), b.features.get(key)
        if fa is None or fb is None:
            problems.append(f"feature {key!r} only in {'relabeled' if fb is None else 'open'}")
        elif fa != fb:
            fields = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
            problems += [f"feature {key!r} {f}: {fa.get(f)!r} vs {fb.get(f)!r}" for f in fields]
    return problems


def gripper_range(root: pathlib.Path) -> dict[str, tuple[float, float]]:
    """Min/max of dim GRIPPER_DIM of observation.state and action over all frames."""
    table = pa.concat_tables(
        [pq.read_table(p, columns=["observation.state", "action"]) for p in sorted((root / "data").glob("*/*.parquet"))]
    )
    ranges = {}
    for key in ("observation.state", "action"):
        values = np.stack(table[key].to_numpy(zero_copy_only=False))[:, GRIPPER_DIM]
        ranges[key] = (float(values.min()), float(values.max()))
    return ranges


def stage_with_open_prompt(src: pathlib.Path, repo_id: str, staging: pathlib.Path) -> None:
    """Copy data/meta, hard-link videos, then rewrite every task to OPEN_PROMPT in the copy."""
    from lerobot.datasets.dataset_tools import modify_tasks
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    shutil.copytree(src / "meta", staging / "meta")
    shutil.copytree(src / "data", staging / "data")
    shutil.copytree(src / "videos", staging / "videos", copy_function=os.link)
    modify_tasks(LeRobotDataset(repo_id, root=staging), new_task=OPEN_PROMPT)


def main() -> int:
    args = parse_args()
    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    if args.dst.exists():
        print(f"{args.dst} already exists; refusing to overwrite", file=sys.stderr)
        return 1
    relabeled = LeRobotDatasetMetadata(args.relabeled_repo_id, root=args.relabeled)
    open_meta = LeRobotDatasetMetadata(args.open_repo_id, root=args.open)

    problems = compare_metadata(relabeled, open_meta)
    if list(relabeled.tasks.index) != [OPEN_PROMPT, EGG_PROMPT]:
        problems.append(f"relabeled tasks are {list(relabeled.tasks.index)}, expected [OPEN_PROMPT, EGG_PROMPT]")
    if problems:
        print("datasets cannot be merged; nothing written:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print(f"open dataset {GRIPPER_DIM=} (right_gripper.pos) range, not modified:")
    for key, (lo, hi) in gripper_range(args.open).items():
        print(f"  {key}: min={lo:.6f} max={hi:.6f}")

    open_tasks = list(open_meta.tasks.index)
    open_root, open_repo_id = args.open, args.open_repo_id
    staging = args.dst.parent / f".{args.dst.name}.open_staging"
    if open_tasks != [OPEN_PROMPT]:
        print(f"open dataset tasks {open_tasks} != [{OPEN_PROMPT!r}]; rewriting them to OPEN_PROMPT for the merge")
        if staging.exists():
            print(f"{staging} already exists; remove it first", file=sys.stderr)
            return 1
        stage_with_open_prompt(args.open, args.open_repo_id, staging)
        open_root, open_repo_id = staging, f"{args.open_repo_id}_open_prompt"
    else:
        print("open dataset task matches OPEN_PROMPT")

    try:
        merge_datasets(
            [
                LeRobotDataset(args.relabeled_repo_id, root=args.relabeled),
                LeRobotDataset(open_repo_id, root=open_root),
            ],
            output_repo_id=args.dst_repo_id,
            output_dir=args.dst,
        )
    finally:
        if open_root == staging:
            shutil.rmtree(staging)

    merged = LeRobotDatasetMetadata(args.dst_repo_id, root=args.dst)
    task_index = pa.concat_tables(
        [
            pq.read_table(p, columns=["episode_index", "task_index"])
            for p in sorted((args.dst / "data").glob("*/*.parquet"))
        ]
    ).to_pandas()
    from_open = task_index["episode_index"] >= relabeled.total_episodes
    checks = {
        "total_frames": merged.total_frames == relabeled.total_frames + open_meta.total_frames == len(task_index),
        "total_episodes": merged.total_episodes == relabeled.total_episodes + open_meta.total_episodes,
        "tasks": list(merged.tasks.index) == [OPEN_PROMPT, EGG_PROMPT],
        "open frames all task 0": bool((task_index.loc[from_open, "task_index"] == 0).all()),
        "relabeled task counts kept": task_index.loc[~from_open, "task_index"].value_counts().sort_index().tolist()
        == [int(x) for x in np.bincount(_relabeled_task_index(args.relabeled))],
    }
    print(f"merged {args.dst}: {merged.total_episodes} episodes, {merged.total_frames} frames, tasks:")
    for task, row in merged.tasks.iterrows():
        print(f"  task_index={int(row['task_index'])}: {task!r}")
    for name, ok in checks.items():
        print(f"  check {name}: {'ok' if ok else 'FAILED'}")
    return 0 if all(checks.values()) else 1


def _relabeled_task_index(root: pathlib.Path) -> np.ndarray:
    return pa.concat_tables(
        [pq.read_table(p, columns=["task_index"]) for p in sorted((root / "data").glob("*/*.parquet"))]
    )["task_index"].to_numpy()


if __name__ == "__main__":
    sys.exit(main())
