"""Advantage labels of a whole dataset as a table, from the labeling attempt to the dataset column.

One labeling attempt is a parquet file with one row per frame plus a JSON file of the same name with the
parameters and counts. Several attempts can exist side by side; the one a person approves is written into
the dataset as the column `advantage_positive`, which policy training reads.
"""

import json
import pathlib

import numpy as np
import pandas as pd

from openpi.recap import advantage as _advantage

# Dataset column holding the approved labels.
COLUMN = "advantage_positive"
FEATURE = {"dtype": "bool", "shape": (1,), "names": None}

_KEYS = ["episode_index", "frame_index"]


def build_labels(
    frames: pd.DataFrame,
    returns: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    horizon: int,
    positive_fraction: float,
    threshold_frames: _advantage.ThresholdFrames = "rollout",
) -> tuple[pd.DataFrame, dict]:
    """Labels every frame of a dataset.

    Args:
        frames: One row per frame of the dataset with `episode_index`, `frame_index`, `is_intervention` and
            `round`.
        returns: One row per frame with `episode_index`, `frame_index` and `return`.
        predictions: Value predictions with `episode_index`, `frame_index` and `value`; must cover every frame
            of the dataset exactly once. A `return` column, when present, must agree with `returns`.
        horizon: Steps the advantage estimate looks ahead.
        positive_fraction: Share of the threshold's frames labeled positive.
        threshold_frames: Which frames the threshold is taken over, see `advantage.label_frames`.

    Returns:
        The labels, sorted by `(episode_index, frame_index)`, with the columns `episode_index`, `frame_index`,
        `return`, `value`, `advantage`, `positive` and `source`; and a summary of the attempt.
    """
    if predictions.duplicated(_KEYS).any():
        raise ValueError("the predictions hold some frames more than once")
    stray = predictions.merge(frames[_KEYS], on=_KEYS, how="left", indicator=True)["_merge"] != "both"
    if stray.any():
        raise ValueError(
            f"{int(stray.sum())} predicted frames are not in the dataset: the predictions were made on another "
            "copy of it"
        )
    table = frames[[*_KEYS, "is_intervention", "round"]].merge(returns[[*_KEYS, "return"]], on=_KEYS, how="left")
    prediction_columns = [*_KEYS, "value", *(["return"] if "return" in predictions else [])]
    table = table.merge(predictions[prediction_columns], on=_KEYS, how="left", suffixes=("", "_predicted"))
    table = table.sort_values(_KEYS, ignore_index=True)
    for column, origin in (("return", "the return file"), ("value", "the predictions")):
        missing = table[table[column].isna()]
        if len(missing):
            first = missing.iloc[0]
            raise ValueError(
                f"{origin} cover {len(table) - len(missing)} of the dataset's {len(table)} frames; the first "
                f"missing one is episode {int(first['episode_index'])} frame {int(first['frame_index'])}"
            )
    if "return_predicted" in table and not np.allclose(table["return"], table["return_predicted"], atol=1e-6):
        raise ValueError("the predictions were made against other returns than the return file holds")

    advantages = []
    for episode_index, episode in table.groupby("episode_index", sort=False):
        if not np.array_equal(episode["frame_index"], np.arange(len(episode))):
            raise ValueError(f"the frames of episode {episode_index} are not numbered 0..{len(episode) - 1}")
        advantages.append(
            _advantage.n_step_advantages(episode["return"].to_numpy(), episode["value"].to_numpy(), horizon)
        )
    table["advantage"] = np.concatenate(advantages)
    table["positive"], table["source"], threshold = _advantage.label_frames(
        table["advantage"].to_numpy(),
        (table["round"] == 0).to_numpy(),
        table["is_intervention"].to_numpy(),
        positive_fraction,
        threshold_frames,
    )
    labels = table[[*_KEYS, "return", "value", "advantage", "positive", "source"]]
    summary = {
        "advantage_horizon": horizon,
        "positive_fraction": positive_fraction,
        "threshold_frames": threshold_frames,
        # JSON has no NaN; the threshold is undefined when no frame is labeled by its advantage.
        "threshold": None if np.isnan(threshold) else threshold,
        "frames": len(labels),
        "positive_share": float(labels["positive"].mean()),
        "sources": {
            source: {"frames": len(group), "positive_share": float(group["positive"].mean())}
            for source, group in labels.groupby("source")
        },
    }
    return labels, summary


def summary_path(labels_path: pathlib.Path) -> pathlib.Path:
    return labels_path.with_suffix(".json")


def write_labels(path: pathlib.Path, labels: pd.DataFrame, summary: dict) -> None:
    """Writes one labeling attempt: the labels to `path` and the summary to the JSON file beside it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(path, index=False)
    summary_path(path).write_text(json.dumps(summary, indent=2))


def read_labels(path: pathlib.Path) -> tuple[pd.DataFrame, dict]:
    """Reads a labeling attempt written by `write_labels`."""
    return pd.read_parquet(path), json.loads(summary_path(path).read_text())


def aligned_positive(labels: pd.DataFrame, frames: pd.DataFrame) -> np.ndarray:
    """Returns the `positive` label of every frame of a dataset, in the dataset's storage order.

    Args:
        labels: A labeling attempt with `episode_index`, `frame_index` and `positive`.
        frames: `episode_index` and `frame_index` of every frame of the dataset, in storage order.

    Raises:
        ValueError: When the labels and the dataset do not hold exactly the same frames.
    """
    if labels.duplicated(_KEYS).any():
        raise ValueError("the labels hold some frames more than once")
    merged = frames[_KEYS].merge(labels[[*_KEYS, "positive"]], on=_KEYS, how="left")
    missing = merged[merged["positive"].isna()]
    if len(missing) or len(labels) != len(frames):
        detail = (
            f"the first unlabeled frame is episode {int(missing.iloc[0]['episode_index'])} frame "
            f"{int(missing.iloc[0]['frame_index'])}"
            if len(missing)
            else "the labels hold frames that are not in the dataset"
        )
        raise ValueError(f"the labels hold {len(labels)} frames, the dataset {len(frames)}; {detail}")
    return merged["positive"].to_numpy(dtype=np.bool_)
