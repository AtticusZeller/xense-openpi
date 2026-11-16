"""Writes normalized per-frame returns of a LeRobot v3.0 dataset to a sidecar parquet file.

The dataset itself is not modified. The output has one row per frame with the columns
`episode_index`, `frame_index` and `return`. Success and failure come from the dataset's per-frame
`episode_success` column; a dataset without it (initial demonstrations) counts every episode as a success.

Usage:
    python scripts/recap/compute_returns.py --dataset-root <lerobot dataset dir> --output <returns.parquet>
"""

import dataclasses
import json
import logging
import pathlib

import numpy as np
import pandas as pd
import tyro

from openpi.recap import returns as _returns


@dataclasses.dataclass(frozen=True)
class Args:
    # Root directory of the LeRobot v3.0 dataset (contains `meta/` and `data/`).
    dataset_root: pathlib.Path
    # Path of the sidecar parquet file to write.
    output: pathlib.Path


def load_success(dataset_root: pathlib.Path) -> pd.Series | None:
    """Loads the success of every episode from the `episode_success` column, indexed by episode index.

    Returns None when the dataset has no such column.
    """
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    if "episode_success" not in info["features"]:
        return None
    files = sorted((dataset_root / "data").glob("*/*.parquet"))
    columns = ["episode_index", "episode_success"]
    frames = pd.concat([pd.read_parquet(file, columns=columns) for file in files], ignore_index=True)
    grouped = frames.groupby("episode_index")["episode_success"]
    changing = grouped.nunique()
    if (changing > 1).any():
        raise ValueError(f"`episode_success` changes within episodes {changing.index[changing > 1].tolist()[:20]}")
    return grouped.first().astype(bool)


def load_episodes(dataset_root: pathlib.Path) -> pd.DataFrame:
    """Loads `episode_index`, `length`, `task` and `success` of every episode, sorted by episode index."""
    files = sorted((dataset_root / "meta" / "episodes").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata under {dataset_root / 'meta' / 'episodes'}")
    columns = ["episode_index", "length", "tasks"]
    episodes = pd.concat([pd.read_parquet(file, columns=columns) for file in files], ignore_index=True)
    episodes = episodes.sort_values("episode_index", ignore_index=True)
    success = load_success(dataset_root)
    if success is None:
        logging.info("No `episode_success` column: treating all %d episodes as successes.", len(episodes))
        episodes["success"] = True
    else:
        episodes["success"] = success.reindex(episodes["episode_index"]).to_numpy()
        if bool(episodes["success"].isna().any()):
            raise ValueError("`episode_success` is missing for some episodes of meta/episodes")
        episodes["success"] = episodes["success"].astype(bool)
        logging.info("%d of %d episodes are failures.", (~episodes["success"]).sum(), len(episodes))
    # An episode with several task strings is normalized together with episodes sharing the same set.
    episodes["task"] = episodes["tasks"].map(lambda tasks: " | ".join(sorted(tasks)))
    return episodes[["episode_index", "length", "task", "success"]]


def build_table(episodes: pd.DataFrame) -> pd.DataFrame:
    """Builds the per-frame return table from the episode table."""
    returns = _returns.dataset_returns(
        episodes["length"].to_numpy(), episodes["success"].to_numpy(), episodes["task"].to_numpy()
    )
    lengths = episodes["length"].to_numpy()
    return pd.DataFrame(
        {
            "episode_index": np.repeat(episodes["episode_index"].to_numpy(), lengths).astype(np.int64),
            "frame_index": np.concatenate([np.arange(length, dtype=np.int64) for length in lengths]),
            "return": np.concatenate(returns),
        }
    )


def main(args: Args) -> None:
    episodes = load_episodes(args.dataset_root)
    table = build_table(episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output, index=False)
    l_max = episodes.groupby("task")["length"].max().to_dict()
    logging.info(
        "Wrote %d rows for %d episodes to %s. l_max per task: %s. min return: %.4f",
        len(table),
        len(episodes),
        args.output,
        l_max,
        table["return"].min(),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
