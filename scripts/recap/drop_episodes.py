"""Delete episodes from a LeRobot v3.0 dataset without re-encoding any video.

LeRobot's ``delete_episodes`` re-encodes every mp4 that holds a deleted episode with a hard-coded
``libsvtav1``. On an H.264 dataset that leaves AV1 and H.264 files side by side while info.json still says
h264, and LeRobot's own merge then fails to stream-concatenate them. This script removes the episodes
from the metadata only:

* ``data/``: rows of deleted episodes are dropped; ``episode_index`` and ``index`` are renumbered
  contiguously; every other column is unchanged.
* ``meta/episodes``: rows of deleted episodes are dropped; ``episode_index``, ``dataset_from_index`` /
  ``dataset_to_index`` and the per-episode ``episode_index`` / ``index`` stats are updated; video file
  references and timestamps are unchanged.
* ``meta/stats.json`` is re-aggregated from the kept episodes with LeRobot's ``aggregate_stats``;
  ``info.json`` counters are updated; ``tasks.parquet`` is copied.
* Video files are hard-linked unchanged. A deleted episode's frames stay inside its mp4 but no episode
  references them any more.

Usage:
    python scripts/egg_carton_two_stage/drop_episodes.py --src <root> --dst <new root> --episodes 108
"""

import argparse
import json
import os
import pathlib
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_RENUMBERED = ("episode_index", "index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=pathlib.Path, required=True, help="Source LeRobot v3.0 dataset root.")
    parser.add_argument("--dst", type=pathlib.Path, required=True, help="Output root; must not exist.")
    parser.add_argument("--episodes", type=int, nargs="+", required=True, help="Episode indices to delete.")
    return parser.parse_args()


def build_episode_map(total_episodes: int, deleted: set[int]) -> dict[int, int]:
    """Old episode index -> new contiguous index for the kept episodes."""
    kept = [ep for ep in range(total_episodes) if ep not in deleted]
    return {old: new for new, old in enumerate(kept)}


def check_source_layout(src: pathlib.Path, data_files: list[pathlib.Path], total_frames: int) -> None:
    """Renumbering assumes the data files hold the global index 0..N-1 in file order."""
    expected = 0
    for path in data_files:
        index = pq.read_table(path, columns=["index"])["index"].to_numpy()
        if len(index) and not (index[0] == expected and np.all(np.diff(index) == 1)):
            raise RuntimeError(f"{path.relative_to(src)}: global index is not contiguous from {expected}")
        expected += len(index)
    if expected != total_frames:
        raise RuntimeError(f"data files hold {expected} frames, info.json says {total_frames}")


def drop_data_table(table: pa.Table, episode_map: dict[int, int], index_start: int) -> pa.Table:
    """Drop unmapped episodes and renumber episode_index / index."""
    table = table.filter(pa.array(np.isin(table["episode_index"].to_numpy(), list(episode_map))))
    columns = {
        "episode_index": np.array([episode_map[e] for e in table["episode_index"].to_numpy()], dtype=np.int64),
        "index": np.arange(index_start, index_start + len(table), dtype=np.int64),
    }
    for name, values in columns.items():
        position = table.schema.get_field_index(name)
        field = table.schema.field(position)
        table = table.set_column(position, field, pa.array(values, field.type))
    return table


def renumbered_stats(table: pa.Table) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    """Per-episode LeRobot stats of the renumbered columns, keyed by new episode index."""
    from lerobot.datasets.compute_stats import get_feature_stats

    df = table.select(list(_RENUMBERED)).to_pandas()
    return {
        int(ep): {key: get_feature_stats(group[key].to_numpy(), axis=0, keepdims=True) for key in _RENUMBERED}
        for ep, group in df.groupby("episode_index")
    }


def drop_episodes_table(
    table: pa.Table,
    episode_map: dict[int, int],
    index_ranges: dict[int, tuple[int, int]],
    stats: dict[int, dict[str, dict[str, np.ndarray]]],
) -> pa.Table:
    rows = [row for row in table.to_pylist() if row["episode_index"] in episode_map]
    for row in rows:
        new = episode_map[row["episode_index"]]
        row["episode_index"] = new
        row["dataset_from_index"], row["dataset_to_index"] = index_ranges[new]
        for key, key_stats in stats[new].items():
            for name, value in key_stats.items():
                row[f"stats/{key}/{name}"] = np.asarray(value).tolist()
    return pa.Table.from_pylist(rows, schema=table.schema)


def aggregate_dataset_stats(episodes_tables: list[pa.Table], features: dict) -> dict:
    """meta/stats.json from per-episode stats, aggregated as LeRobot's dataset tools do."""
    from lerobot.datasets.compute_stats import aggregate_stats

    stats_list = []
    for table in episodes_tables:
        for row in table.to_pylist():
            episode_stats: dict[str, dict[str, np.ndarray]] = {}
            for column, value in row.items():
                if column.startswith("stats/"):
                    key, name = column.removeprefix("stats/").rsplit("/", 1)
                    episode_stats.setdefault(key, {})[name] = np.asarray(value)
            stats_list.append(episode_stats)
    return {k: v for k, v in aggregate_stats(stats_list).items() if k in features}


def main() -> int:
    args = parse_args()
    src, dst = args.src, args.dst
    info = json.loads((src / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise RuntimeError(f"expected a v3.0 dataset, got {info.get('codebase_version')}")
    deleted = set(args.episodes)
    if len(deleted) != len(args.episodes) or not deleted <= set(range(info["total_episodes"])):
        print(
            f"--episodes must be distinct and within 0..{info['total_episodes'] - 1}: {args.episodes}", file=sys.stderr
        )
        return 1
    if len(deleted) == info["total_episodes"]:
        print("refusing to delete every episode", file=sys.stderr)
        return 1
    if dst.exists():
        print(f"{dst} already exists; refusing to overwrite", file=sys.stderr)
        return 1

    data_files = sorted((src / "data").glob("*/*.parquet"))
    check_source_layout(src, data_files, info["total_frames"])
    episode_map = build_episode_map(info["total_episodes"], deleted)

    index_ranges: dict[int, tuple[int, int]] = {}
    stats: dict[int, dict] = {}
    next_index = 0
    for path in data_files:
        table = drop_data_table(pq.read_table(path), episode_map, next_index)
        out = dst / path.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out)
        file_stats = renumbered_stats(table)
        if split := set(file_stats) & set(stats):
            raise RuntimeError(f"episodes {sorted(split)} span several data files; per-episode stats would be wrong")
        stats.update(file_stats)
        for ep, group in table.select(list(_RENUMBERED)).to_pandas().groupby("episode_index"):
            index_ranges[int(ep)] = (int(group["index"].min()), int(group["index"].max()) + 1)
        next_index += len(table)

    episodes_tables = []
    for path in sorted((src / "meta" / "episodes").glob("*/*.parquet")):
        table = drop_episodes_table(pq.read_table(path), episode_map, index_ranges, stats)
        out = dst / path.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out)
        episodes_tables.append(table)

    from lerobot.datasets.utils import write_info
    from lerobot.datasets.utils import write_stats

    shutil.copy2(src / "meta" / "tasks.parquet", dst / "meta" / "tasks.parquet")
    write_stats(aggregate_dataset_stats(episodes_tables, info["features"]), dst)
    info.update(
        total_episodes=len(episode_map),
        total_frames=next_index,
        splits={"train": f"0:{len(episode_map)}"},
    )
    write_info(info, dst)
    shutil.copytree(src / "videos", dst / "videos", copy_function=os.link)

    print(f"wrote {dst}: deleted episodes {sorted(deleted)}, {len(episode_map)} episodes, {next_index} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
