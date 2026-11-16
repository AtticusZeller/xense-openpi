import numpy as np
import pandas as pd

from openpi.recap import collection_columns as _collection_columns


def _frames(*episodes: tuple[int, int, bool, set[int]]) -> pd.DataFrame:
    """Frames of episodes given as (length, round, success, intervention steps)."""
    rows = []
    for episode_index, (length, round_index, success, intervention) in enumerate(episodes):
        rows += [
            {
                "episode_index": episode_index,
                "is_intervention": step in intervention,
                "episode_success": success,
                "round": round_index,
            }
            for step in range(length)
        ]
    return pd.DataFrame(rows)


_DATASET = _frames(
    (5, 0, True, set()), (4, 0, True, set()), (6, 1, True, {2, 3}), (3, 1, False, set()), (7, 2, False, {0})
)


def test_initial_demonstration_columns_are_successful_round_zero_without_intervention():
    columns = _collection_columns.initial_demonstration_columns(4)
    assert set(columns) == set(_collection_columns.COLUMNS)
    assert columns["is_intervention"].tolist() == [False] * 4
    assert columns["episode_success"].tolist() == [True] * 4
    assert columns["round"].tolist() == [0] * 4
    for name, values in columns.items():
        assert values.dtype == np.dtype(_collection_columns.COLLECTION_FEATURES[name]["dtype"])


def test_consistent_columns_have_no_problems():
    assert _collection_columns.collection_problems(_collection_columns.episode_table(_DATASET)) == []


def test_round_summary_counts_episodes_failures_and_interventions():
    summary = _collection_columns.round_summary(_collection_columns.episode_table(_DATASET))
    assert summary.to_dict("records") == [
        {"round": 0, "episodes": 2, "failures": 0, "episodes_with_intervention": 0, "frames": 9, "intervention_share": 0.0},
        {"round": 1, "episodes": 2, "failures": 1, "episodes_with_intervention": 1, "frames": 9, "intervention_share": 2 / 9},
        {"round": 2, "episodes": 1, "failures": 1, "episodes_with_intervention": 1, "frames": 7, "intervention_share": 1 / 7},
    ]  # fmt: skip


def test_episode_success_changing_within_an_episode_is_a_problem():
    frames = _DATASET.copy()
    frames.loc[frames.index[frames["episode_index"] == 2][-1], "episode_success"] = False
    problems = _collection_columns.collection_problems(_collection_columns.episode_table(frames))
    assert problems == ["`episode_success` changes within episodes [2]"]


def test_round_changing_within_an_episode_is_a_problem():
    frames = _DATASET.copy()
    frames.loc[frames.index[frames["episode_index"] == 3][0], "round"] = 2
    problems = _collection_columns.collection_problems(_collection_columns.episode_table(frames))
    assert problems == ["`round` changes within episodes [3]"]


def test_intervention_in_an_initial_demonstration_is_a_problem():
    frames = _DATASET.copy()
    frames.loc[frames.index[frames["episode_index"] == 1][2], "is_intervention"] = True
    problems = _collection_columns.collection_problems(_collection_columns.episode_table(frames))
    assert problems == ["initial demonstrations (round 0) with intervention frames: episodes [1]"]
