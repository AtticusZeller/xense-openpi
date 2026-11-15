import numpy as np

from openpi.recap import dataset_check as _dataset_check


def _episode(still_head: int, moving: int, still_tail: int) -> np.ndarray:
    """A 2-D state that rests, moves one unit per frame along the first dimension, then rests again."""
    position = np.concatenate([np.zeros(still_head), np.arange(1, moving + 1), np.full(still_tail, moving)])
    return np.stack([position, np.zeros_like(position)], axis=1)


def _rows(*lengths: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index, episode index and frame index of episodes of the given lengths stored in order."""
    episode_index = np.repeat(np.arange(len(lengths)), lengths)
    frame_index = np.concatenate([np.arange(length) for length in lengths])
    return np.arange(len(episode_index)), episode_index, frame_index


def test_row_order_problems_accept_episodes_stored_in_order():
    assert _dataset_check.row_order_problems(*_rows(3, 1, 4)) == []


def test_row_order_problems_report_swapped_data_files():
    index, episode_index, frame_index = _rows(3, 4)
    swapped = np.r_[3:7, 0:3]  # The second episode's file is read first.
    problems = _dataset_check.row_order_problems(index[swapped], episode_index[swapped], frame_index[swapped])
    assert len(problems) == 2


def test_row_order_problems_report_frames_out_of_order_within_an_episode():
    index, episode_index, frame_index = _rows(3, 4)
    frame_index[3:] = frame_index[3:][::-1]
    assert _dataset_check.row_order_problems(index, episode_index, frame_index) == [
        "the frames of an episode are not stored together in frame order"
    ]


def test_row_order_problems_report_a_shifted_global_index():
    index, episode_index, frame_index = _rows(3, 4)
    assert _dataset_check.row_order_problems(index + 1, episode_index, frame_index) == [
        "stored frames are not in the order of the global index 0..N-1"
    ]


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


def test_still_frames_see_motion_in_any_dimension_and_direction():
    # Only the second dimension moves, and it moves backwards.
    position = -np.arange(10, dtype=np.float64)
    states = np.stack([np.zeros_like(position), position, np.zeros_like(position)], axis=1)
    assert not _dataset_check.still_frames(states, np.array([np.inf, 10.0, np.inf]), 0.01).any()


def test_still_frames_take_the_largest_change_over_dimensions():
    # One dimension moves by 5% of its span per frame, the other nine do not move: the mean change is 0.5%.
    states = np.zeros((10, 10))
    states[:, 3] = np.arange(10) * 0.05
    assert not _dataset_check.still_frames(states, np.ones(10), 0.01).any()
