import numpy as np
import pytest

from openpi.recap import advantage as _advantage
from openpi.recap import returns as _returns


@pytest.mark.parametrize("success", [True, False])
def test_exact_values_give_zero_advantage(success):
    returns = _returns.episode_returns(40, success=success, l_max=50)
    np.testing.assert_allclose(_advantage.n_step_advantages(returns, returns, 5), 0.0, atol=1e-7)


def test_stalled_value_is_negative_and_the_catch_up_positive():
    returns = _returns.episode_returns(40, success=True, l_max=40)
    step = float(returns[1] - returns[0])
    values = returns.copy()
    values[10:20] = returns[10]  # The value stops rising for 10 frames, then jumps back onto the return.
    advantages = _advantage.n_step_advantages(returns, values, 5)

    np.testing.assert_allclose(advantages[10:15], -5 * step, rtol=1e-5)
    assert (advantages[16:20] > 0).all()
    np.testing.assert_allclose(advantages[20:], 0.0, atol=1e-7)


def test_frames_near_the_end_do_not_bootstrap():
    returns = _returns.episode_returns(40, success=True, l_max=40)
    values = returns + 0.1  # A constant offset cancels wherever the value is bootstrapped.
    advantages = _advantage.n_step_advantages(returns, values, 5)

    np.testing.assert_allclose(advantages[:35], 0.0, atol=1e-6)
    np.testing.assert_allclose(advantages[35:], -0.1, atol=1e-6)


def test_episode_shorter_than_the_horizon():
    returns = _returns.episode_returns(3, success=False, l_max=10)
    values = np.zeros(3, np.float32)
    np.testing.assert_allclose(_advantage.n_step_advantages(returns, values, 5), returns, atol=1e-7)


def test_advantages_reject_bad_input():
    with pytest.raises(ValueError, match="equally long"):
        _advantage.n_step_advantages(np.zeros(3), np.zeros(4), 2)
    with pytest.raises(ValueError, match="horizon"):
        _advantage.n_step_advantages(np.zeros(3), np.zeros(3), 0)


def test_threshold_leaves_the_given_fraction_positive():
    advantages = np.random.default_rng(0).normal(size=1000)
    threshold = _advantage.positive_threshold(advantages, 0.4)
    assert (advantages > threshold).mean() == pytest.approx(0.4, abs=1e-3)
    with pytest.raises(ValueError, match="positive_fraction"):
        _advantage.positive_threshold(advantages, 1.0)


def test_label_runs():
    starts, ends = _advantage.label_runs(np.array([True, True, False, True, True, True]))
    np.testing.assert_array_equal(starts, [0, 2, 3])
    np.testing.assert_array_equal(ends, [2, 3, 6])

    starts, ends = _advantage.label_runs(np.array([False]))
    np.testing.assert_array_equal(starts, [0])
    np.testing.assert_array_equal(ends, [1])
