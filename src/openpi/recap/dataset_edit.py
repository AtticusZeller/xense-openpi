"""Copies of a LeRobot v3.0 dataset with RECAP columns added.

Every function writes a new dataset with LeRobot's own `modify_features` and leaves the source untouched.
`modify_features` copies the video files in full, so each copy takes the disk space of the videos it keeps.
"""

from collections.abc import Sequence
import json
import pathlib
import shutil

from lerobot.datasets import dataset_tools
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

from openpi.recap import advantage_labels as _advantage_labels
from openpi.recap import collection_columns as _collection_columns


def add_columns(
    src_root: pathlib.Path,
    dst_root: pathlib.Path,
    columns: dict[str, tuple[np.ndarray, dict]],
    remove_features: Sequence[str] = (),
) -> None:
    """Writes a copy of the dataset at `src_root` to `dst_root` with per-frame columns added.

    Args:
        src_root: Root directory of the source dataset.
        dst_root: Root directory of the copy; must not exist.
        columns: Column name -> (values in storage order, shape (total_frames,); LeRobot feature definition).
        remove_features: Features of the source the copy leaves out; the videos of a removed stream are
            not copied.
    """
    if dst_root.exists():
        raise FileExistsError(f"{dst_root} already exists; refusing to overwrite")
    dataset = LeRobotDataset(f"local/{src_root.name}", root=src_root)
    for name, (values, _) in columns.items():
        if values.shape != (dataset.meta.total_frames,):
            raise ValueError(f"`{name}` has shape {values.shape}, the dataset has {dataset.meta.total_frames} frames")
    dataset_tools.modify_features(
        dataset,
        add_features=columns,
        remove_features=list(remove_features) or None,
        output_dir=dst_root,
        repo_id=f"local/{dst_root.name}",
    )


def mark_initial_demonstrations(
    src_root: pathlib.Path, dst_root: pathlib.Path, remove_features: Sequence[str] = ()
) -> None:
    """Writes a copy of a dataset of initial demonstrations with the collection columns added.

    Every frame gets `is_intervention = False`, `episode_success = True` and `round = 0`. LeRobot merges
    two datasets only when their features are identical, so the copy must end up with exactly the features
    of a rollout dataset: `remove_features` names what the demonstrations hold beyond them (the tactile
    video streams, which rollout collection does not record).
    """
    total_frames = json.loads((src_root / "meta" / "info.json").read_text())["total_frames"]
    values = _collection_columns.initial_demonstration_columns(total_frames)
    features = _collection_columns.COLLECTION_FEATURES
    add_columns(src_root, dst_root, {name: (values[name], features[name]) for name in features}, remove_features)


def write_advantage_labels(src_root: pathlib.Path, labels_path: pathlib.Path, dst_root: pathlib.Path) -> dict:
    """Writes a copy of a dataset with a labeling attempt as the column `advantage_positive`.

    The attempt's summary is copied to `meta/advantage_labels.json` of the copy.

    Args:
        src_root: Root directory of the dataset the labels were made for.
        labels_path: Labels parquet of the attempt; its JSON summary lies beside it.
        dst_root: Root directory of the copy; must not exist.

    Returns:
        The summary of the attempt.

    Raises:
        ValueError: When the labels do not hold exactly the frames of the dataset. Nothing is written then.
    """
    labels, summary = _advantage_labels.read_labels(labels_path)
    positive = _advantage_labels.aligned_positive(labels, _collection_columns.load_frames(src_root))
    add_columns(src_root, dst_root, {_advantage_labels.COLUMN: (positive, _advantage_labels.FEATURE)})
    shutil.copy2(_advantage_labels.summary_path(labels_path), dst_root / "meta" / "advantage_labels.json")
    return summary
