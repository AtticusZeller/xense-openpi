import numpy as np
import pandas as pd
import pytest

from openpi.recap import advantage as _advantage
from openpi.recap import advantage_labels as _advantage_labels
from openpi.recap import returns as _returns

_HORIZON = 5
# (length, round, success, intervention steps): two initial demonstrations and two rollouts.
_EPISODES = [(30, 0, True, set()), (40, 0, True, set()), (50, 1, True, {10, 11, 12}), (45, 1, False, set())]


def _dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Frames, returns and noisy value predictions of `_EPISODES`."""
    rng = np.random.default_rng(0)
    frames, returns, predictions = [], [], []
    for episode_index, (length, round_index, success, intervention) in enumerate(_EPISODES):
        keys = {"episode_index": episode_index, "frame_index": np.arange(length)}
        episode_returns = _returns.episode_returns(length, success=success, l_max=50)
        frames.append(
            pd.DataFrame({**keys, "is_intervention": np.isin(np.arange(length), list(intervention)), "round": round_index})
        )  # fmt: skip
        returns.append(pd.DataFrame({**keys, "return": episode_returns}))
        values = episode_returns + rng.normal(scale=0.02, size=length).astype(np.float32)
        predictions.append(pd.DataFrame({**keys, "return": episode_returns, "value": values}))
    return tuple(pd.concat(tables, ignore_index=True) for tables in (frames, returns, predictions))


def _build(frames, returns, predictions, **kwargs):
    kwargs = {"horizon": _HORIZON, "positive_fraction": 0.4, **kwargs}
    return _advantage_labels.build_labels(frames, returns, predictions, **kwargs)


def test_labels_follow_the_rule_and_the_summary_counts_them():
    frames, returns, predictions = _dataset()
    labels, summary = _build(frames, returns, predictions)

    assert labels[["episode_index", "frame_index"]].equals(frames[["episode_index", "frame_index"]])
    assert (labels.loc[labels["episode_index"] < 2, "source"] == "initial").all()
    rollout = labels[labels["episode_index"] == 2]
    assert rollout.loc[rollout["source"] == "intervention", "frame_index"].tolist() == [10, 11, 12]
    assert labels.loc[labels["source"] != "value", "positive"].all()
    by_value = labels[labels["source"] == "value"]
    np.testing.assert_array_equal(by_value["positive"], by_value["advantage"] > summary["threshold"])

    assert summary["frames"] == len(frames) == sum(source["frames"] for source in summary["sources"].values())
    assert summary["sources"]["initial"] == {"frames": 70, "positive_share": 1.0}
    assert summary["sources"]["intervention"] == {"frames": 3, "positive_share": 1.0}
    assert summary["sources"]["value"]["frames"] == 92
    assert summary["sources"]["value"]["positive_share"] == pytest.approx(0.4, abs=0.01)
    assert summary["threshold_frames"] == "rollout"


def test_advantages_are_computed_within_each_episode():
    frames, returns, predictions = _dataset()
    labels, _ = _build(frames, returns, predictions)
    for episode_index in range(len(_EPISODES)):
        episode = labels[labels["episode_index"] == episode_index]
        expected = _advantage.n_step_advantages(episode["return"].to_numpy(), episode["value"].to_numpy(), _HORIZON)
        np.testing.assert_array_equal(episode["advantage"], expected)


def test_predictions_may_arrive_in_several_shuffled_parts():
    frames, returns, predictions = _dataset()
    expected, _ = _build(frames, returns, predictions)
    shuffled = pd.concat(
        [predictions[predictions["episode_index"] % 2 == 1], predictions[predictions["episode_index"] % 2 == 0]]
    )
    labels, _ = _build(frames, returns, shuffled.sample(frac=1.0, random_state=0))
    pd.testing.assert_frame_equal(labels, expected)


def test_all_threshold_is_taken_over_every_frame():
    frames, returns, predictions = _dataset()
    labels, summary = _build(frames, returns, predictions, threshold_frames="all")
    assert summary["threshold"] == _advantage.positive_threshold(labels["advantage"].to_numpy(), 0.4)


def test_a_dataset_of_initial_demonstrations_has_no_threshold():
    frames, returns, predictions = _dataset()
    labels, summary = _build(frames.assign(round=0, is_intervention=False), returns, predictions)
    assert labels["positive"].all()
    assert summary["threshold"] is None


def test_a_missing_prediction_is_rejected():
    frames, returns, predictions = _dataset()
    with pytest.raises(ValueError, match=r"cover 164 of the dataset's 165 frames.*episode 2 frame 7"):
        _build(frames, returns, predictions.drop(index=30 + 40 + 7))


def test_predictions_of_another_dataset_copy_are_rejected():
    frames, returns, predictions = _dataset()
    extra = predictions.iloc[[0]].assign(episode_index=9)
    with pytest.raises(ValueError, match="1 predicted frames are not in the dataset"):
        _build(frames, returns, pd.concat([predictions, extra]))
    with pytest.raises(ValueError, match="more than once"):
        _build(frames, returns, pd.concat([predictions, predictions.iloc[[0]]]))


def test_predictions_made_against_other_returns_are_rejected():
    frames, returns, predictions = _dataset()
    with pytest.raises(ValueError, match="other returns"):
        _build(frames, returns.assign(**{"return": returns["return"] - 0.1}), predictions)


def test_labels_survive_a_round_trip_through_their_files(tmp_path):
    labels, summary = _build(*_dataset())
    _advantage_labels.write_labels(tmp_path / "attempt" / "labels.parquet", labels, summary)
    read, read_summary = _advantage_labels.read_labels(tmp_path / "attempt" / "labels.parquet")
    pd.testing.assert_frame_equal(read, labels)
    assert read_summary == summary


def test_aligned_positive_follows_the_storage_order_of_the_dataset():
    frames, returns, predictions = _dataset()
    labels, _ = _build(frames, returns, predictions)
    storage = frames.iloc[::-1].reset_index(drop=True)
    np.testing.assert_array_equal(
        _advantage_labels.aligned_positive(labels, storage), labels["positive"].to_numpy()[::-1]
    )


def test_aligned_positive_rejects_labels_that_do_not_match_the_dataset():
    frames, returns, predictions = _dataset()
    labels, _ = _build(frames, returns, predictions)
    with pytest.raises(ValueError, match=r"hold 164 frames, the dataset 165.*episode 0 frame 3"):
        _advantage_labels.aligned_positive(labels.drop(index=3), frames)
    with pytest.raises(ValueError, match="not in the dataset"):
        _advantage_labels.aligned_positive(labels, frames.iloc[:-1])
