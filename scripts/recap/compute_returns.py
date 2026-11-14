"""Writes normalized per-frame returns of a LeRobot v3.0 dataset to a sidecar parquet file.

The dataset itself is not modified. The output has one row per frame with the columns
`episode_index`, `frame_index` and `return`.

Usage:
    python scripts/recap/compute_returns.py --dataset-root <lerobot dataset dir> --output <returns.parquet>
"""

import dataclasses
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
    # Boolean column of the episode metadata marking success. If unset, every episode counts as a success.
    success_column: str | None = None


def load_episodes(dataset_root: pathlib.Path, success_column: str | None) -> pd.DataFrame:
    """Loads `episode_index`, `length`, `task` and `success` of every episode, sorted by episode index."""
    files = sorted((dataset_root / "meta" / "episodes").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata under {dataset_root / 'meta' / 'episodes'}")
    columns = ["episode_index", "length", "tasks"] + ([success_column] if success_column else [])
    episodes = pd.concat([pd.read_parquet(file, columns=columns) for file in files], ignore_index=True)
    episodes = episodes.sort_values("episode_index", ignore_index=True)
    if success_column is None:
        logging.info("No success column given: treating all %d episodes as successes.", len(episodes))
        episodes["success"] = True
    else:
        episodes["success"] = episodes[success_column].astype(bool)
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
    episodes = load_episodes(args.dataset_root, args.success_column)
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
