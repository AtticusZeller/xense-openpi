import numpy as np
import pytest

from openpi.recap import returns as _returns


def test_success_episode_returns():
    np.testing.assert_allclose(
        _returns.episode_returns(6, success=True, l_max=6), np.array([-5, -4, -3, -2, -1, 0]) / 12, rtol=1e-6
    )
    np.testing.assert_allclose(
        _returns.episode_returns(4, success=True, l_max=6), np.array([-3, -2, -1, 0]) / 12, rtol=1e-6
    )


def test_failed_episode_returns():
    np.testing.assert_allclose(
        _returns.episode_returns(6, success=False, l_max=6), np.array([-11, -10, -9, -8, -7, -6]) / 12, rtol=1e-6
    )
    np.testing.assert_allclose(
        _returns.episode_returns(4, success=False, l_max=6), np.array([-9, -8, -7, -6]) / 12, rtol=1e-6
    )


def test_returns_stay_in_range_and_are_float32():
    for success in (True, False):
        for length in range(1, 7):
            returns = _returns.episode_returns(length, success=success, l_max=6)
            assert returns.dtype == np.float32
            assert returns.min() >= -1.0
            assert returns.max() <= 0.0


def test_explicit_c_fail_is_clipped():
    returns = _returns.episode_returns(3, success=False, l_max=3, c_fail=0)
    np.testing.assert_allclose(returns, np.array([-2, -1, 0]) / 3, rtol=1e-6)


def test_rejects_episode_longer_than_l_max():
    with pytest.raises(ValueError, match="length must be in"):
        _returns.episode_returns(7, success=True, l_max=6)


def test_dataset_returns_normalizes_each_task_by_its_longest_episode():
    lengths = np.array([4, 6, 2, 3])
    successes = np.array([True, False, True, True])
    tasks = np.array([0, 0, 1, 1])

    returns = _returns.dataset_returns(lengths, successes, tasks)

    np.testing.assert_allclose(returns[0], np.array([-3, -2, -1, 0]) / 12, rtol=1e-6)
    np.testing.assert_allclose(returns[1], np.array([-11, -10, -9, -8, -7, -6]) / 12, rtol=1e-6)
    # Task 1 has l_max = 3.
    np.testing.assert_allclose(returns[2], np.array([-1, 0]) / 6, rtol=1e-6)
    np.testing.assert_allclose(returns[3], np.array([-2, -1, 0]) / 6, rtol=1e-6)
