"""Writes an approved labeling attempt into a copy of the dataset, as the column `advantage_positive`.

Policy training with advantage conditioning reads this column, so which labels a run trains on is decided by
the dataset its config names. The copy is made with LeRobot's own `add_features`: the source dataset is not
modified, and the copy holds a full copy of the videos. The attempt's JSON summary is copied to
`meta/advantage_labels.json` of the copy, recording how its labels were made.

The labels must hold exactly the frames of the dataset; otherwise nothing is written.

Usage:
    python scripts/recap/write_advantage_labels.py --src <lerobot dataset dir> --labels <labels.parquet> \
        --dst <new dir>
"""

import dataclasses
import logging
import pathlib

import tyro

from openpi.recap import dataset_edit as _dataset_edit


@dataclasses.dataclass(frozen=True)
class Args:
    # Root directory of the LeRobot v3.0 dataset the labels were made for.
    src: pathlib.Path
    # Labels parquet written by scripts/recap/label_advantage.py; its JSON summary must lie beside it.
    labels: pathlib.Path
    # Root directory of the labeled copy; must not exist.
    dst: pathlib.Path


def main(args: Args) -> None:
    summary = _dataset_edit.write_advantage_labels(args.src, args.labels, args.dst)
    logging.info(
        "Wrote %s: %d frames, %.1f%% positive (threshold frames: %s, horizon %d, fraction %.2f).",
        args.dst, summary["frames"], 100 * summary["positive_share"], summary["threshold_frames"],
        summary["advantage_horizon"], summary["positive_fraction"],
    )  # fmt: skip


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
