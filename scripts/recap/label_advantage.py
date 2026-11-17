"""Labels every frame of a dataset with a binary advantage, as one labeling attempt in a sidecar file.

The dataset is not modified. Frames of the initial demonstrations (`round == 0`) and frames the operator
executed (`is_intervention`) are positive; every other frame is positive when its N-step advantage exceeds a
threshold (`openpi.recap.advantage.label_frames`). A dataset without the collection columns is rejected:
rollouts get them from collection mode, initial demonstrations from scripts/recap/mark_initial_dataset.py.

The value predictions must cover every frame of the dataset. scripts/recap/train_value.py writes the
held-out episodes to `heldout_predictions.parquet` and, with `--final-train-episodes`, the training episodes
to `train_predictions.parquet`; pass both.

Writes `<output>` with one row per frame (`episode_index`, `frame_index`, `return`, `value`, `advantage`,
`positive`, `source`) and a JSON file of the same name with the parameters, the threshold and the counts per
source. scripts/recap/plot_value.py draws an attempt for review and scripts/recap/write_advantage_labels.py
writes an approved one into the dataset.

Usage:
    python scripts/recap/label_advantage.py --dataset-root <lerobot dataset dir> --returns <returns.parquet> \
        --predictions <run>/heldout_predictions.parquet <run>/train_predictions.parquet --output <labels.parquet>
"""

import dataclasses
import json
import logging
import pathlib

import pandas as pd
import tyro

from openpi.recap import advantage as _advantage
from openpi.recap import advantage_labels as _advantage_labels
from openpi.recap import collection_columns as _collection_columns


@dataclasses.dataclass(frozen=True)
class Args:
    # Root directory of the LeRobot v3.0 dataset to label.
    dataset_root: pathlib.Path
    # Returns of the dataset, written by scripts/recap/compute_returns.py.
    returns: pathlib.Path
    # Value predictions written by scripts/recap/train_value.py; together they must cover every frame.
    predictions: tuple[pathlib.Path, ...]
    # Path of the labels parquet to write; the summary goes to the JSON file of the same name.
    output: pathlib.Path
    # Repo id the predictions call this dataset. Needed when a prediction file holds several datasets.
    repo_id: str | None = None
    # Steps the advantage estimate looks ahead.
    advantage_horizon: int = 50
    # Share of the threshold's frames labeled positive.
    positive_fraction: float = 0.4
    # Frames the threshold is the quantile of: "rollout" takes only the frames labeled by their advantage
    # (rollout frames the policy executed), "all" takes every frame of the dataset.
    threshold_frames: _advantage.ThresholdFrames = "rollout"


def load_predictions(paths: tuple[pathlib.Path, ...], repo_id: str | None) -> pd.DataFrame:
    """Loads the prediction files and keeps the rows of the dataset called `repo_id`."""
    predictions = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    if "dataset" not in predictions:
        return predictions
    datasets = sorted(predictions["dataset"].unique())
    if repo_id is None:
        if len(datasets) > 1:
            raise ValueError(f"The predictions hold the datasets {datasets}; choose one with --repo-id.")
        return predictions
    if repo_id not in datasets:
        raise ValueError(f"The predictions hold the datasets {datasets}, not {repo_id}.")
    return predictions[predictions["dataset"] == repo_id]


def main(args: Args) -> None:
    labels, summary = _advantage_labels.build_labels(
        _collection_columns.load_frames(args.dataset_root),
        pd.read_parquet(args.returns),
        load_predictions(args.predictions, args.repo_id),
        horizon=args.advantage_horizon,
        positive_fraction=args.positive_fraction,
        threshold_frames=args.threshold_frames,
    )
    summary |= {
        "dataset_root": str(args.dataset_root),
        "returns": str(args.returns),
        "predictions": [str(path) for path in args.predictions],
    }
    _advantage_labels.write_labels(args.output, labels, summary)
    logging.info("Wrote %s:\n%s", args.output, json.dumps(summary, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    main(tyro.cli(Args))
