"""Success and failure read from the `episode_success` column, on hand-written dataset files."""

import json
import pathlib

import compute_returns
import numpy as np
import pandas as pd
import pytest


def _write_dataset(root: pathlib.Path, lengths: list[int], success: list[list[bool]] | None) -> None:
    """Writes the three files compute_returns.py reads; `success` holds the per-frame column of each episode."""
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    features = {"episode_index": {}} | ({"episode_success": {}} if success else {})
    (root / "meta" / "info.json").write_text(json.dumps({"features": features}))
    # Episode metadata in reverse order: the script must sort it.
    episodes = pd.DataFrame(
        {"episode_index": range(len(lengths)), "length": lengths, "tasks": [["stack"]] * len(lengths)}
    )
    episodes.iloc[::-1].to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)
    frames = pd.DataFrame({"episode_index": np.repeat(np.arange(len(lengths)), lengths)})
    if success:
        frames["episode_success"] = np.concatenate(success)
    frames.to_parquet(root / "data" / "chunk-000" / "file-000.parquet", index=False)


def test_dataset_without_the_column_counts_every_episode_as_a_success(tmp_path):
    _write_dataset(tmp_path, [3, 5], None)
    episodes = compute_returns.load_episodes(tmp_path)
    assert episodes["success"].tolist() == [True, True]
    table = compute_returns.build_table(episodes)
    # Successes end at 0; the longest episode starts at -(l_max - 1) / (2 * l_max).
    assert table["return"].iloc[[2, 7]].tolist() == [0.0, 0.0]
    assert table["return"].iloc[3] == pytest.approx(-4 / 10)


def test_failed_episodes_get_the_failure_penalty(tmp_path):
    _write_dataset(tmp_path, [3, 5, 4], [[True] * 3, [False] * 5, [True] * 4])
    episodes = compute_returns.load_episodes(tmp_path)
    assert episodes["episode_index"].tolist() == [0, 1, 2]
    assert episodes["success"].tolist() == [True, False, True]
    table = compute_returns.build_table(episodes)
    by_episode = [table.loc[table["episode_index"] == index, "return"].to_numpy() for index in range(3)]
    # l_max = c_fail = 5: a failure ends at -5/10 and starts at -(4 + 5)/10; successes still end at 0.
    assert by_episode[1][-1] == pytest.approx(-0.5)
    assert by_episode[1][0] == pytest.approx(-0.9)
    assert by_episode[0][-1] == by_episode[2][-1] == 0.0


def test_episode_success_changing_within_an_episode_is_rejected(tmp_path):
    _write_dataset(tmp_path, [3, 2], [[True, True, False], [True, True]])
    with pytest.raises(ValueError, match="changes within episodes \\[0\\]"):
        compute_returns.load_episodes(tmp_path)
