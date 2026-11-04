"""Split every episode of the egg-carton dataset into two prompts.

Frames with ``frame_index < round(cut_s * fps)`` get ``OPEN_PROMPT`` (task_index 0); the rest get
``EGG_PROMPT`` (task_index 1). LeRobot has no per-frame task tool (``modify_tasks`` sets one task per
episode), so this script changes only the task bookkeeping: ``task_index`` in ``data/``, the episodes'
``tasks`` and ``stats/task_index/*``, ``meta/tasks.parquet``, ``total_tasks`` and the ``task_index`` entry
of ``meta/stats.json``. Every other file is copied unchanged; videos are hard-linked.

Bad episodes are deleted beforehand with ``drop_episodes.py``. ``cuts.csv`` keeps the episode numbering
of the dataset before that deletion; pass the deleted episodes with ``--deleted-episodes`` so the cuts are
mapped onto the renumbered episodes.

The cuts are validated strictly: any problem is listed and the script exits without writing.

Usage:
    python scripts/egg_carton_two_stage/relabel.py --src <dataset after deletion> --dst <new root> \
        --cuts scripts/egg_carton_two_stage/cuts.csv --deleted-episodes 108
"""

import argparse
import csv
import json
import os
import pathlib
import shutil
import sys

from egg_prompts import EGG_PROMPT
from egg_prompts import OPEN_PROMPT
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TASKS = [OPEN_PROMPT, EGG_PROMPT]  # Position == task_index; LeRobot looks tasks up with iloc.
EPISODE_MAP_FILE = "relabel_episode_map.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=pathlib.Path, required=True, help="LeRobot v3.0 dataset root (after deletion).")
    parser.add_argument("--dst", type=pathlib.Path, required=True, help="Output root; must not exist.")
    parser.add_argument("--cuts", type=pathlib.Path, required=True, help="CSV with columns episode_index,cut_s.")
    parser.add_argument(
        "--deleted-episodes",
        type=int,
        nargs="*",
        default=[],
        help="Episodes (in the cuts.csv numbering) already deleted from --src.",
    )
    return parser.parse_args()


def load_episode_lengths(root: pathlib.Path) -> dict[int, int]:
    files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "length"]) for f in files])
    return dict(zip(df["episode_index"].tolist(), df["length"].tolist(), strict=True))


def original_to_current(num_current: int, deleted: list[int]) -> dict[int, int]:
    """Episode index in the cuts.csv numbering -> index in --src, for the kept episodes."""
    kept = [ep for ep in range(num_current + len(set(deleted))) if ep not in deleted]
    return {old: new for new, old in enumerate(kept)}


def validate_cuts(
    rows: list[dict[str, str]], deleted: list[int], lengths: dict[int, int], fps: int
) -> tuple[dict[int, int], list[str]]:
    """Check the cuts against episode lengths keyed by cuts.csv numbering; return (cut frames, problems)."""
    problems: list[str] = []
    cut_frames: dict[int, int] = {}
    listed: set[int] = set()
    total = len(lengths) + len(set(deleted))

    problems += [
        f"--deleted-episodes lists episode {ep} more than once"
        for ep in sorted({e for e in deleted if deleted.count(e) > 1})
    ]
    problems += [
        f"--deleted-episodes episode {ep} is outside 0..{total - 1}" for ep in sorted(set(deleted) - set(range(total)))
    ]

    for line_no, row in enumerate(rows, start=2):  # Line 1 is the header.
        try:
            ep = int(row["episode_index"])
            cut_s = float(row["cut_s"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"line {line_no}: cannot parse {row}")
            continue
        if ep in deleted:
            problems.append(f"line {line_no}: episode {ep} is also in --deleted-episodes")
            continue
        if ep not in lengths:
            problems.append(f"line {line_no}: episode {ep} is outside 0..{total - 1}")
            continue
        if ep in listed:
            problems.append(f"line {line_no}: episode {ep} appears more than once")
            continue
        listed.add(ep)
        cut = round(cut_s * fps)
        if not 0 < cut < lengths[ep]:
            problems.append(
                f"line {line_no}: episode {ep} cut_s={cut_s} -> cut_frame={cut} is outside (0, length={lengths[ep]})"
            )
            continue
        cut_frames[ep] = cut

    problems += [f"episode {ep} has no cut and is not in --deleted-episodes" for ep in sorted(set(lengths) - listed)]
    return cut_frames, problems


def relabel_data_table(table: pa.Table, cut_frames: dict[int, int]) -> pa.Table:
    """Set task_index from the cut; cut_frames is keyed by the table's episode_index."""
    cut = np.array([cut_frames[e] for e in table["episode_index"].to_numpy()], dtype=np.int64)
    task_index = np.where(table["frame_index"].to_numpy() < cut, 0, 1).astype(np.int64)
    position = table.schema.get_field_index("task_index")
    field = table.schema.field(position)
    return table.set_column(position, field, pa.array(task_index, field.type))


def task_index_stats(table: pa.Table) -> dict[int, dict[str, np.ndarray]]:
    """Per-episode LeRobot stats of task_index, keyed by episode index."""
    from lerobot.datasets.compute_stats import get_feature_stats

    df = table.select(["episode_index", "task_index"]).to_pandas()
    return {
        int(ep): get_feature_stats(group["task_index"].to_numpy(), axis=0, keepdims=True)
        for ep, group in df.groupby("episode_index")
    }


def relabel_episodes_table(table: pa.Table, stats: dict[int, dict[str, np.ndarray]]) -> pa.Table:
    rows = table.to_pylist()
    for row in rows:
        row["tasks"] = list(TASKS)
        for name, value in stats[row["episode_index"]].items():
            row[f"stats/task_index/{name}"] = np.asarray(value).tolist()
    return pa.Table.from_pylist(rows, schema=table.schema)


def aggregate_task_index_stats(stats: dict[int, dict[str, np.ndarray]]) -> dict[str, list]:
    """Dataset-level task_index stats, aggregated the way LeRobot writes meta/stats.json."""
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.utils import serialize_dict

    aggregated = aggregate_stats([{"task_index": s} for _, s in sorted(stats.items())])["task_index"]
    return serialize_dict(aggregated)


def main() -> int:
    args = parse_args()
    src, dst = args.src, args.dst
    info = json.loads((src / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise RuntimeError(f"expected a v3.0 dataset, got {info.get('codebase_version')}")
    current_lengths = load_episode_lengths(src)
    if sorted(current_lengths) != list(range(info["total_episodes"])):
        raise RuntimeError("episode indices in meta/episodes are not 0..total_episodes-1")
    to_current = original_to_current(info["total_episodes"], args.deleted_episodes)
    lengths = {old: current_lengths[new] for old, new in to_current.items()}

    with args.cuts.open(newline="") as f:
        rows = list(csv.DictReader(f))
    cut_frames, problems = validate_cuts(rows, args.deleted_episodes, lengths, info["fps"])
    if problems:
        print(f"{len(problems)} problem(s) in {args.cuts} / --deleted-episodes; nothing written:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    if dst.exists():
        print(f"{dst} already exists; refusing to overwrite", file=sys.stderr)
        return 1
    current_cuts = {to_current[old]: cut for old, cut in cut_frames.items()}

    stats: dict[int, dict[str, np.ndarray]] = {}
    for path in sorted((src / "data").glob("*/*.parquet")):
        table = relabel_data_table(pq.read_table(path), current_cuts)
        out = dst / path.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out)
        file_stats = task_index_stats(table)
        if split := set(file_stats) & set(stats):
            raise RuntimeError(f"episodes {sorted(split)} span several data files; per-episode stats would be wrong")
        stats.update(file_stats)

    for path in sorted((src / "meta" / "episodes").glob("*/*.parquet")):
        out = dst / path.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(relabel_episodes_table(pq.read_table(path), stats), out)

    from lerobot.datasets.utils import write_info
    from lerobot.datasets.utils import write_json
    from lerobot.datasets.utils import write_tasks

    write_tasks(pd.DataFrame({"task_index": range(len(TASKS))}, index=pd.Index(TASKS, name="task")), dst)
    dataset_stats = json.loads((src / "meta" / "stats.json").read_text())
    dataset_stats["task_index"] = aggregate_task_index_stats(stats)
    write_json(dataset_stats, dst / "meta" / "stats.json")
    info["total_tasks"] = len(TASKS)
    write_info(info, dst)
    shutil.copytree(src / "videos", dst / "videos", copy_function=os.link)

    with (dst / EPISODE_MAP_FILE).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cuts_episode_index", "episode_index", "length", "cut_frame"])
        for old, new in to_current.items():
            writer.writerow([old, new, lengths[old], cut_frames[old]])

    print(f"wrote {dst}: {info['total_episodes']} episodes, task_index stats for {len(stats)} episodes")
    print(f"episode numbering map (cuts.csv -> dataset) in {EPISODE_MAP_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
