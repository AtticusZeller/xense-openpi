"""Check that relabel.py changed only the task bookkeeping of its source dataset.

Compares the output with the source frame by frame: every data column except ``task_index``, every
episodes column except ``tasks`` and ``stats/task_index/*``, info.json except ``total_tasks`` and
stats.json except ``task_index`` must be identical; videos must be hard links; ``task_index`` must follow
the cut rule recorded in ``relabel_episode_map.csv``.

Usage:
    python scripts/egg_carton_two_stage/verify_relabel.py --src <relabel source> --dst <relabeled root>
"""

import argparse
import csv
import json
import pathlib
import sys

from egg_prompts import EGG_PROMPT
from egg_prompts import OPEN_PROMPT
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from relabel import EPISODE_MAP_FILE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=pathlib.Path, required=True)
    parser.add_argument("--dst", type=pathlib.Path, required=True)
    return parser.parse_args()


def read_cuts(dst: pathlib.Path) -> dict[int, int]:
    """Dataset episode index -> cut frame, from relabel.py's map file."""
    with (dst / EPISODE_MAP_FILE).open() as f:
        return {int(r["episode_index"]): int(r["cut_frame"]) for r in csv.DictReader(f)}


def concat_parquet(root: pathlib.Path, pattern: str) -> pa.Table:
    return pa.concat_tables([pq.read_table(p) for p in sorted(root.glob(pattern))])


def check_data(src: pa.Table, dst: pa.Table, cuts: dict[int, int]) -> list[str]:
    if src.num_rows != dst.num_rows or src.column_names != dst.column_names:
        return [f"data: {dst.num_rows} rows / {dst.column_names} vs {src.num_rows} / {src.column_names}"]
    errors = [f"data column {n!r} differs" for n in src.column_names if n != "task_index" and not src[n].equals(dst[n])]
    episode = dst["episode_index"].to_numpy()
    expected = np.where(dst["frame_index"].to_numpy() < np.array([cuts[e] for e in episode]), 0, 1)
    bad = np.flatnonzero(dst["task_index"].to_numpy() != expected)
    if len(bad):
        errors.append(f"task_index: {len(bad)} wrong rows, first at row {bad[0]}")
    return errors


def check_episodes(src: pa.Table, dst: pa.Table) -> list[str]:
    errors = []
    for s, d in zip(src.to_pylist(), dst.to_pylist(), strict=True):
        changed = [k for k in s if k != "tasks" and not k.startswith("stats/task_index/") and s[k] != d[k]]
        if changed:
            errors.append(f"episode {s['episode_index']}: columns changed {changed}")
        if d["tasks"] != [OPEN_PROMPT, EGG_PROMPT]:
            errors.append(f"episode {d['episode_index']}: tasks={d['tasks']}")
    return errors


def check_meta(src: pathlib.Path, dst: pathlib.Path) -> list[str]:
    errors = []
    tasks = pd.read_parquet(dst / "meta" / "tasks.parquet")
    if list(zip(tasks.index, tasks["task_index"], strict=True)) != [(OPEN_PROMPT, 0), (EGG_PROMPT, 1)]:
        errors.append(f"tasks.parquet: {tasks}")
    info_src = json.loads((src / "meta" / "info.json").read_text())
    info_dst = json.loads((dst / "meta" / "info.json").read_text())
    if {**info_src, "total_tasks": 2} != info_dst:
        errors.append("info.json differs outside total_tasks")
    stats_src = json.loads((src / "meta" / "stats.json").read_text())
    stats_dst = json.loads((dst / "meta" / "stats.json").read_text())
    if {k: v for k, v in stats_src.items() if k != "task_index"} != {
        k: v for k, v in stats_dst.items() if k != "task_index"
    }:
        errors.append("stats.json differs outside task_index")
    return errors


def check_videos(src: pathlib.Path, dst: pathlib.Path) -> list[str]:
    src_files = sorted(p.relative_to(src) for p in (src / "videos").rglob("*") if p.is_file())
    dst_files = sorted(p.relative_to(dst) for p in (dst / "videos").rglob("*") if p.is_file())
    if src_files != dst_files:
        return ["video file lists differ"]
    return [f"video {p} is not a hard link" for p in src_files if (src / p).stat().st_ino != (dst / p).stat().st_ino]


def main() -> int:
    args = parse_args()
    src, dst = args.src, args.dst
    data_dst = concat_parquet(dst, "data/*/*.parquet")
    errors = [
        *check_data(concat_parquet(src, "data/*/*.parquet"), data_dst, read_cuts(dst)),
        *check_episodes(
            concat_parquet(src, "meta/episodes/*/*.parquet"), concat_parquet(dst, "meta/episodes/*/*.parquet")
        ),
        *check_meta(src, dst),
        *check_videos(src, dst),
    ]
    stats = json.loads((dst / "meta" / "stats.json").read_text())["task_index"]
    task = data_dst["task_index"].to_numpy()
    print(f"output: {data_dst.num_rows} frames, task_index counts {np.bincount(task).tolist()}")
    print(f"stats.json task_index mean={stats['mean']} count={stats['count']} (data mean={task.mean():.6f})")
    if not np.isclose(stats["mean"][0], task.mean()) or stats["count"][0] != len(task):
        errors.append("stats.json task_index does not match the data")
    for e in errors:
        print(f"ERROR {e}")
    print("OK" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
