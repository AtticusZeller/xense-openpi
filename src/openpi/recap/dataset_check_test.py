import numpy as np

from openpi.recap import dataset_check as _dataset_check


def _episode(still_head: int, moving: int, still_tail: int) -> np.ndarray:
    """A 2-D state that rests, moves one unit per frame along the first dimension, then rests again."""
    position = np.concatenate([np.zeros(still_head), np.arange(1, moving + 1), np.full(still_tail, moving)])
    return np.stack([position, np.zeros_like(position)], axis=1)


def test_motion_spans_ignore_constant_dimensions():
    spans = _dataset_check.motion_spans(_episode(0, 100, 0))
    assert spans[0] == np.quantile(np.arange(1, 101), 0.99) - np.quantile(np.arange(1, 101), 0.01)
    assert np.isinf(spans[1])


def test_still_frames_mark_the_resting_parts():
    states = _episode(5, 20, 8)
    still = _dataset_check.still_frames(states, np.array([20.0, np.inf]), 0.01)

    assert still[:5].all()
    assert not still[5:25].any()
    assert still[25:].all()


def test_still_frames_ignore_changes_below_the_threshold():
    states = _episode(0, 20, 0) * 1e-4  # Moves, but far less than the threshold share of the span.
    assert _dataset_check.still_frames(states, np.array([1.0, np.inf]), 0.01).all()
    assert not _dataset_check.still_frames(np.zeros((1, 2)), np.array([1.0, 1.0]), 0.01).any()


def test_edge_still_lengths():
    still = _dataset_check.still_frames(_episode(5, 20, 8), np.array([20.0, np.inf]), 0.01)
    assert _dataset_check.edge_still_lengths(still) == (5, 8)
    assert _dataset_check.edge_still_lengths(np.array([False, True, False])) == (0, 0)
    assert _dataset_check.edge_still_lengths(np.ones(7, bool)) == (7, 7)


def test_length_outliers():
    lengths = np.array([100, 105, 110, 115, 120, 125, 130, 400])
    np.testing.assert_array_equal(_dataset_check.length_outliers(lengths, 3.0), [False] * 7 + [True])
    assert not _dataset_check.length_outliers(lengths[:7], 3.0).any()
