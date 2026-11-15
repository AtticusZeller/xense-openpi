"""Checks a LeRobot v3.0 dataset before it is used to train the RECAP value model.

Reads metadata and parquet files only; nothing is modified. Reports

* problems that make the dataset unusable as it is: episode metadata, `info.json` and `data/` disagreeing
  (for example a leftover metadata file), frames stored out of order, or camera videos missing; the script
  exits with status 1 when it finds any;
* suspect episodes for a person to review: far longer than the rest of the dataset, or with the robot
  standing still for most of the episode, at its start or at its end (a recording that was not stopped).

With `--sheet-dir` it also decodes a row of frames per suspect episode. Nothing is dropped; use
scripts/recap/drop_episodes.py for that.

Usage:
    python scripts/recap/check_dataset.py --dataset-root <lerobot dataset dir> [--sheet-dir <dir>]
"""

import dataclasses
import json
import logging
import pathlib
import sys

import numpy as np
import pandas as pd
import tyro

from openpi.recap import dataset_check as _dataset_check


@dataclasses.dataclass(frozen=True)
class Args:
    # Root directory of the LeRobot v3.0 dataset (contains `meta/`, `data/` and `videos/`).
    dataset_root: pathlib.Path
    # Cameras whose video files must exist for every episode.
    cameras: tuple[str, ...] = (
        "observation.images.head",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    )
    # Largest per-frame state change, as a share of each dimension's range, that counts as standing still.
    still_threshold: float = 5e-4
    # An episode is suspect when it stands still for longer than this at its start or at its end.
    edge_still_seconds: float = 5.0
    # An episode is suspect when more than this share of its frames stands still.
    still_fraction: float = 0.5
    # An episode is suspect when it is longer than the third quartile plus this many interquartile ranges.
    length_iqr_factor: float = 3.0
    # Where to write one row of frames per suspect episode. Needs the videos; skipped when unset.
    sheet_dir: pathlib.Path | None = None
    # Camera shown on the sheets.
    sheet_camera: str = "observation.images.head"
    # Where to write the report as JSON. Printed only when unset.
    output: pathlib.Path | None = None


def load_episodes(dataset_root: pathlib.Path) -> pd.DataFrame:
    """Loads the episode metadata of every file under `meta/episodes`, in file order."""
    files = sorted((dataset_root / "meta" / "episodes").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata under {dataset_root / 'meta' / 'episodes'}")
    tables = []
    for file in files:
        table = pd.read_parquet(file)
        tables.append(table[[c for c in table.columns if not c.startswith("stats/")]].assign(meta_file=file.name))
    return pd.concat(tables, ignore_index=True)


def load_frames(dataset_root: pathlib.Path) -> pd.DataFrame:
    """Loads the indices and `observation.state` of every frame, in storage order."""
    files = sorted((dataset_root / "data").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data files under {dataset_root / 'data'}")
    columns = ["index", "episode_index", "frame_index", "observation.state"]
    return pd.concat([pd.read_parquet(file, columns=columns) for file in files], ignore_index=True)


def metadata_problems(info: dict, episodes: pd.DataFrame, frame_episodes: np.ndarray) -> list[str]:
    """Lists the disagreements between `info.json`, the episode metadata and the stored frames."""
    problems = []
    indices, lengths = episodes["episode_index"].to_numpy(), episodes["length"].to_numpy()
    if len(episodes) != info["total_episodes"]:
        problems.append(f"meta/episodes has {len(episodes)} rows, info.json says {info['total_episodes']} episodes")
    duplicated = episodes[episodes["episode_index"].duplicated(keep=False)]
    if len(duplicated):
        problems.append(
            f"episode indices listed more than once: {sorted(set(duplicated['episode_index'].tolist()))} "
            f"(files {sorted(set(duplicated['meta_file']))})"
        )
    elif not np.array_equal(np.sort(indices), np.arange(len(indices))):
        problems.append("episode indices are not consecutive from 0")
    if lengths.sum() != info["total_frames"]:
        problems.append(f"episode lengths sum to {lengths.sum()}, info.json says {info['total_frames']} frames")
    if len(frame_episodes) != info["total_frames"]:
        problems.append(f"data/ has {len(frame_episodes)} rows, info.json says {info['total_frames']} frames")
    if not len(duplicated):
        order = np.argsort(indices)
        if not np.array_equal(
            episodes["dataset_from_index"].to_numpy()[order], np.cumsum(lengths[order]) - lengths[order]
        ):
            problems.append("episodes are not stored back to back in episode order")
        counts = pd.concat(
            [
                pd.Series(frame_episodes).value_counts().rename("stored"),
                pd.Series(lengths, index=indices).rename("listed"),
            ],
            axis=1,
        )
        differing = sorted(counts.index[counts["stored"] != counts["listed"]].tolist())
        if differing:
            problems.append(f"data/ and meta/episodes disagree on the frames of episodes {differing[:20]}")
    return problems


def missing_videos(
    dataset_root: pathlib.Path, info: dict, episodes: pd.DataFrame, cameras: tuple[str, ...]
) -> list[str]:
    """Lists the camera video files that the episode metadata references but that are not on disk."""
    problems = []
    for camera in cameras:
        chunk, file = f"videos/{camera}/chunk_index", f"videos/{camera}/file_index"
        if chunk not in episodes:
            problems.append(f"{camera}: not a video stream of this dataset")
            continue
        missing = [
            path
            # Rows of a metadata file written without video columns hold NaN here.
            for c, f in episodes[[chunk, file]].dropna().astype(int).drop_duplicates().itertuples(index=False)
            if not (
                dataset_root / (path := info["video_path"].format(video_key=camera, chunk_index=c, file_index=f))
            ).is_file()
        ]
        if missing:
            problems.append(f"{camera}: {len(missing)} video files missing, first {missing[0]}")
    return problems


def suspect_episodes(episodes: pd.DataFrame, frames: pd.DataFrame, fps: float, args: Args) -> pd.DataFrame:
    """Measures every episode and returns the suspect ones with the reasons, longest standstill first."""
    states = np.stack(frames["observation.state"].to_numpy()).astype(np.float64)
    spans = _dataset_check.motion_spans(states)
    rows = []
    for episode_index, positions in frames.groupby("episode_index", sort=True).indices.items():
        still = _dataset_check.still_frames(states[positions], spans, args.still_threshold)
        head, tail = _dataset_check.edge_still_lengths(still)
        rows.append((episode_index, len(still), still.mean(), head / fps, tail / fps))
    table = pd.DataFrame(rows, columns=["episode_index", "length", "still_fraction", "head_still_s", "tail_still_s"])
    table["length_ratio"] = table["length"] / table["length"].median()

    checks = {
        "long": _dataset_check.length_outliers(table["length"].to_numpy(), args.length_iqr_factor),
        "mostly still": (table["still_fraction"] > args.still_fraction).to_numpy(),
        "still at the start": (table["head_still_s"] > args.edge_still_seconds).to_numpy(),
        "still at the end": (table["tail_still_s"] > args.edge_still_seconds).to_numpy(),
    }
    table["reasons"] = [", ".join(name for name, hit in checks.items() if hit[i]) for i in range(len(table))]
    logging.info(
        "%d episodes, %.1f s median; unflagged episodes: longest %.2fx the median, still at most %.1f s at the "
        "start and %.1f s at the end, still share at most %.2f",
        len(table), table["length"].median() / fps,
        *(table[table["reasons"] == ""][c].max() for c in ("length_ratio", "head_still_s", "tail_still_s", "still_fraction")),
    )  # fmt: skip
    suspects = table[table["reasons"] != ""]
    return suspects.sort_values(["tail_still_s", "length"], ascending=False, ignore_index=True)


def write_sheets(args: Args, repo_id: str, suspects: pd.DataFrame, fps: float) -> None:
    """Writes one PNG per suspect episode with six evenly spaced frames of `sheet_camera`."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    from openpi.training import data_loader as _data_loader

    assert args.sheet_dir is not None
    dataset = _data_loader.SelectiveVideoLeRobotDataset(repo_id, root=args.dataset_root, tolerance_s=1e-2)
    dataset.decode_video_keys = frozenset({args.sheet_camera})
    starts = dataset.meta.episodes["dataset_from_index"]
    args.sheet_dir.mkdir(parents=True, exist_ok=True)
    for row in suspects.itertuples():
        positions = np.linspace(0, row.length - 1, 6).round().astype(int)
        figure, axes = plt.subplots(1, 6, figsize=(18, 2.9), constrained_layout=True)
        for axis, position in zip(axes, positions, strict=True):
            image = dataset[int(starts[row.episode_index]) + int(position)][args.sheet_camera]
            axis.imshow(image.permute(1, 2, 0).numpy())
            axis.set_title(f"{position / fps:.1f} s", fontsize=9)
            axis.axis("off")
        figure.suptitle(
            f"{repo_id} episode {row.episode_index}: {row.reasons} | {row.length / fps:.1f} s "
            f"({row.length_ratio:.2f}x median), still {row.still_fraction:.0%}, "
            f"start {row.head_still_s:.1f} s, end {row.tail_still_s:.1f} s",
            fontsize=10,
        )
        figure.savefig(args.sheet_dir / f"episode_{row.episode_index:04d}.png", dpi=90)
        plt.close(figure)


def main(args: Args) -> None:
    info = json.loads((args.dataset_root / "meta" / "info.json").read_text())
    episodes, frames = load_episodes(args.dataset_root), load_frames(args.dataset_root)
    problems = metadata_problems(info, episodes, frames["episode_index"].to_numpy())
    problems += _dataset_check.row_order_problems(
        frames["index"].to_numpy(), frames["episode_index"].to_numpy(), frames["frame_index"].to_numpy()
    )
    problems += missing_videos(args.dataset_root, info, episodes, args.cameras)
    for problem in problems:
        logging.error("PROBLEM %s", problem)
    suspects = suspect_episodes(episodes, frames, info["fps"], args)
    for row in suspects.itertuples():
        logging.warning(
            "SUSPECT episode %d: %s | %.1f s (%.2fx median), still %.0f%%, start %.1f s, end %.1f s",
            row.episode_index, row.reasons, row.length / info["fps"], row.length_ratio, 100 * row.still_fraction,
            row.head_still_s, row.tail_still_s,
        )  # fmt: skip
    logging.info("%s: %d problems, %d suspect episodes", args.dataset_root.name, len(problems), len(suspects))
    if args.output:
        report = {"dataset": args.dataset_root.name, "problems": problems, "suspects": suspects.to_dict("records")}
        args.output.write_text(json.dumps(report, indent=2))
    if args.sheet_dir and len(suspects) and not problems:
        write_sheets(args, f"local/{args.dataset_root.name}", suspects, info["fps"])
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    main(tyro.cli(Args))
