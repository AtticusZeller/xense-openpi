"""The per-frame facts a RECAP dataset carries on top of states, actions and images.

* `is_intervention`: the action of this frame came from the operator, not from the policy.
* `episode_success`: whether the episode of this frame ended in success; constant within an episode.
* `round`: the improvement round that collected the frame; 0 for the initial demonstrations.

The robot records them during rollout collection (examples/bi_flexiv_rizon4_rt/recorder.py) and
scripts/recap/mark_initial_dataset.py adds them to the initial demonstrations, so both kinds of data can
be merged into one dataset.
"""

import json
import pathlib

import numpy as np
import pandas as pd

# LeRobot only merges datasets whose features are identical, so these must stay equal to
# COLLECTION_FEATURES in examples/bi_flexiv_rizon4_rt/recorder.py (the robot host does not install openpi).
COLLECTION_FEATURES: dict[str, dict] = {
    "is_intervention": {"dtype": "bool", "shape": (1,), "names": None},
    "episode_success": {"dtype": "bool", "shape": (1,), "names": None},
    "round": {"dtype": "int64", "shape": (1,), "names": None},
}
COLUMNS = tuple(COLLECTION_FEATURES)


def initial_demonstration_columns(num_frames: int) -> dict[str, np.ndarray]:
    """Returns the columns of a dataset of initial demonstrations: all successful, no intervention, round 0."""
    return {
        "is_intervention": np.zeros(num_frames, dtype=np.bool_),
        "episode_success": np.ones(num_frames, dtype=np.bool_),
        "round": np.zeros(num_frames, dtype=np.int64),
    }


def load_frames(dataset_root: pathlib.Path) -> pd.DataFrame:
    """Loads `episode_index`, `frame_index` and the three columns of every frame, in storage order.

    Raises:
        ValueError: When the dataset lacks any of the columns. Guessing them would label a rollout
            recorded without them as initial demonstrations, that is, every frame positive.
    """
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    missing = [column for column in COLUMNS if column not in info["features"]]
    if missing:
        raise ValueError(
            f"{dataset_root} lacks the collection columns {missing}. Rollouts get them from collection mode "
            "(--args.record-round), initial demonstrations from scripts/recap/mark_initial_dataset.py."
        )
    files = sorted((dataset_root / "data").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data files under {dataset_root / 'data'}")
    columns = ["episode_index", "frame_index", *COLUMNS]
    return pd.concat([pd.read_parquet(file, columns=columns) for file in files], ignore_index=True)


def episode_table(frames: pd.DataFrame) -> pd.DataFrame:
    """Reduces the per-frame columns to one row per episode.

    Args:
        frames: One row per frame with `episode_index` and the three columns of `COLUMNS`.

    Returns:
        One row per episode, sorted by `episode_index`, with `length`, `round`, `success`,
        `intervention_frames` and, for the two columns that must not change within an episode,
        `round_values` and `success_values` (the number of distinct values).
    """
    grouped = frames.groupby("episode_index", sort=True)
    return pd.DataFrame(
        {
            "length": grouped.size(),
            "round": grouped["round"].first(),
            "success": grouped["episode_success"].first(),
            "intervention_frames": grouped["is_intervention"].sum(),
            "round_values": grouped["round"].nunique(),
            "success_values": grouped["episode_success"].nunique(),
        }
    ).reset_index()


def collection_problems(episodes: pd.DataFrame) -> list[str]:
    """Lists the ways the three columns contradict their definition.

    Args:
        episodes: The table returned by `episode_table`.
    """
    problems = []
    for column, count in (("round", "round_values"), ("episode_success", "success_values")):
        changing = episodes.loc[episodes[count] > 1, "episode_index"].tolist()
        if changing:
            problems.append(f"`{column}` changes within episodes {changing[:20]}")
    initial = episodes[episodes["round"] == 0]
    intervened = initial.loc[initial["intervention_frames"] > 0, "episode_index"].tolist()
    if intervened:
        problems.append(f"initial demonstrations (round 0) with intervention frames: episodes {intervened[:20]}")
    return problems


def round_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    """Counts, per round, the episodes, the failures, the episodes with intervention and the intervention share.

    Args:
        episodes: The table returned by `episode_table`.

    Returns:
        One row per round with `episodes`, `failures`, `episodes_with_intervention`, `frames` and
        `intervention_share` (intervention frames over all frames of the round).
    """
    rows = []
    for round_index, group in episodes.groupby("round", sort=True):
        frames = int(group["length"].sum())
        rows.append(
            {
                "round": int(round_index),
                "episodes": len(group),
                "failures": int((~group["success"].astype(bool)).sum()),
                "episodes_with_intervention": int((group["intervention_frames"] > 0).sum()),
                "frames": frames,
                "intervention_share": float(group["intervention_frames"].sum() / frames),
            }
        )
    return pd.DataFrame(rows)
