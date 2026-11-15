"""Check a drop_episodes.py output against its source row by row.

Kept rows must be identical except for the renumbered ``episode_index`` / ``index``; episodes metadata
must keep every video reference and timestamp; videos must be hard links; and the stats aggregation
must reproduce the source's own meta/stats.json before the output's stats.json is trusted.

Usage:
    python scripts/egg_carton_two_stage/verify_drop.py --src <source root> --dst <output root> --episodes 108
"""

import argparse
import json
import pathlib
import sys

from drop_episodes import aggregate_dataset_stats
from drop_episodes import build_episode_map
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_RENUMBERED = {"episode_index", "index"}
_STAT_NAMES = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
_EPISODE_UPDATED = {"episode_index", "dataset_from_index", "dataset_to_index"} | {
    f"stats/{key}/{name}" for key in _RENUMBERED for name in _STAT_NAMES
}
_INFO_COUNTERS = {"total_episodes", "total_frames", "splits"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=pathlib.Path, required=True)
    parser.add_argument("--dst", type=pathlib.Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    return parser.parse_args()


def concat_parquet(root: pathlib.Path, pattern: str) -> pa.Table:
    return pa.concat_tables([pq.read_table(p) for p in sorted(root.glob(pattern))])


def check_data(src: pa.Table, dst: pa.Table, episode_map: dict[int, int]) -> list[str]:
    kept = src.filter(pa.array(np.isin(src["episode_index"].to_numpy(), list(episode_map))))
    if kept.num_rows != dst.num_rows:
        return [f"data: {dst.num_rows} rows, expected {kept.num_rows}"]
    errors = [
        f"data column {n!r} differs" for n in kept.column_names if n not in _RENUMBERED and not kept[n].equals(dst[n])
    ]
    expected = {
        "episode_index": np.array([episode_map[e] for e in kept["episode_index"].to_numpy()]),
        "index": np.arange(dst.num_rows),
    }
    for name, values in expected.items():
        bad = np.flatnonzero(dst[name].to_numpy() != values)
        if len(bad):
            errors.append(f"data column {name!r}: {len(bad)} wrong rows, first at output row {bad[0]}")
    return errors


def check_episodes(src: pa.Table, dst: pa.Table, data: pa.Table, episode_map: dict[int, int]) -> list[str]:
    src_rows = {r["episode_index"]: r for r in src.to_pylist()}
    dst_rows = {r["episode_index"]: r for r in dst.to_pylist()}
    if sorted(dst_rows) != sorted(episode_map.values()):
        return [f"episodes: indices {sorted(dst_rows)[:5]}..., expected 0..{len(episode_map) - 1}"]
    index = data.select(["episode_index", "index"]).to_pandas().groupby("episode_index")["index"]
    lo, hi = index.min(), index.max() + 1
    errors = []
    for old, new in episode_map.items():
        s, d = src_rows[old], dst_rows[new]
        changed = [k for k in s if k not in _EPISODE_UPDATED and s[k] != d[k]]
        if changed:
            errors.append(f"episode {old}->{new}: columns changed {changed}")
        if (d["dataset_from_index"], d["dataset_to_index"]) != (lo[new], hi[new]):
            errors.append(f"episode {new}: index range does not match data")
    return errors


def check_videos(src: pathlib.Path, dst: pathlib.Path) -> list[str]:
    src_files = sorted(p.relative_to(src) for p in (src / "videos").rglob("*") if p.is_file())
    dst_files = sorted(p.relative_to(dst) for p in (dst / "videos").rglob("*") if p.is_file())
    if src_files != dst_files:
        return ["video file lists differ"]
    return [f"video {p} is not a hard link" for p in src_files if (src / p).stat().st_ino != (dst / p).stat().st_ino]


def stats_close(a: dict, b: dict) -> list[str]:
    errors = []
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            errors.append(f"stats key {key!r} missing on one side")
            continue
        errors += [
            f"stats {key}/{name} differs"
            for name in a[key]
            if not np.allclose(np.asarray(a[key][name], float), np.asarray(b[key][name], float), rtol=1e-6, atol=1e-9)
        ]
    return errors


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    args = parse_args()
    src, dst = args.src, args.dst
    info_src, info_dst = load_json(src / "meta" / "info.json"), load_json(dst / "meta" / "info.json")
    episode_map = build_episode_map(info_src["total_episodes"], set(args.episodes))

    data_src, data_dst = concat_parquet(src, "data/*/*.parquet"), concat_parquet(dst, "data/*/*.parquet")
    eps_src = concat_parquet(src, "meta/episodes/*/*.parquet")
    eps_dst = concat_parquet(dst, "meta/episodes/*/*.parquet")
    deleted_frames = int(np.isin(data_src["episode_index"].to_numpy(), args.episodes).sum())

    errors = [
        *check_data(data_src, data_dst, episode_map),
        *check_episodes(eps_src, eps_dst, data_dst, episode_map),
        *check_videos(src, dst),
    ]
    if not info_dst["total_frames"] == info_src["total_frames"] - deleted_frames == data_dst.num_rows:
        errors.append(f"total_frames {info_dst['total_frames']} != {info_src['total_frames']} - {deleted_frames}")
    if info_dst["total_episodes"] != len(episode_map):
        errors.append(f"total_episodes {info_dst['total_episodes']} != {len(episode_map)}")
    if {k: v for k, v in info_src.items() if k not in _INFO_COUNTERS} != {
        k: v for k, v in info_dst.items() if k not in _INFO_COUNTERS
    }:
        errors.append("info.json differs outside the counters")
    if (src / "meta" / "tasks.parquet").read_bytes() != (dst / "meta" / "tasks.parquet").read_bytes():
        errors.append("tasks.parquet differs")
    # The aggregation must reproduce the source's own stats.json before its output can be trusted.
    errors += [
        f"source {e}"
        for e in stats_close(
            aggregate_dataset_stats([eps_src], info_src["features"]), load_json(src / "meta" / "stats.json")
        )
    ]
    errors += [
        f"output {e}"
        for e in stats_close(
            aggregate_dataset_stats([eps_dst], info_dst["features"]), load_json(dst / "meta" / "stats.json")
        )
    ]

    print(f"deleted episodes {sorted(args.episodes)} ({deleted_frames} frames)")
    print(f"output: {info_dst['total_episodes']} episodes, {data_dst.num_rows} frames")
    for e in errors:
        print(f"ERROR {e}")
    print("OK" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
