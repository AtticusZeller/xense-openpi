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


def _mixed_frames():
    """1000 frames: 400 initial, then 600 rollout of which every sixth is an intervention."""
    advantages = np.random.default_rng(1).normal(size=1000)
    # Initial and intervention frames sit far below every threshold, so only the rule can make them positive.
    initial = np.arange(1000) < 400
    intervention = ~initial & (np.arange(1000) % 6 == 0)
    advantages[initial | intervention] -= 10.0
    return advantages, initial, intervention


@pytest.mark.parametrize("threshold_frames", ["rollout", "all"])
def test_initial_and_intervention_frames_are_positive_whatever_their_advantage(threshold_frames):
    advantages, initial, intervention = _mixed_frames()
    positive, source, _ = _advantage.label_frames(advantages, initial, intervention, 0.4, threshold_frames)

    assert positive[initial].all()
    assert positive[intervention].all()
    assert (source[initial] == "initial").all()
    assert (source[intervention] == "intervention").all()
    assert (source[~initial & ~intervention] == "value").all()


def test_rollout_threshold_leaves_the_given_fraction_of_the_policy_frames_positive():
    advantages, initial, intervention = _mixed_frames()
    positive, source, threshold = _advantage.label_frames(advantages, initial, intervention, 0.4, "rollout")

    by_value = source == "value"
    assert positive[by_value].mean() == pytest.approx(0.4, abs=2e-3)
    assert threshold == _advantage.positive_threshold(advantages[by_value], 0.4)
    np.testing.assert_array_equal(positive[by_value], advantages[by_value] > threshold)


def test_all_threshold_is_the_quantile_of_every_frame():
    advantages, initial, intervention = _mixed_frames()
    positive, source, threshold = _advantage.label_frames(advantages, initial, intervention, 0.4, "all")

    assert threshold == _advantage.positive_threshold(advantages, 0.4)
    by_value = source == "value"
    np.testing.assert_array_equal(positive[by_value], advantages[by_value] > threshold)
    # The forced frames hold the lowest advantages here, so the 40% above the threshold are all policy frames.
    assert positive[by_value].sum() == 400


def test_dataset_of_initial_demonstrations_is_all_positive_without_a_threshold():
    positive, source, threshold = _advantage.label_frames(
        np.linspace(-1, 1, 10), np.ones(10, bool), np.zeros(10, bool), 0.4, "rollout"
    )
    assert positive.all()
    assert (source == "initial").all()
    assert np.isnan(threshold)


def test_label_frames_reject_bad_input():
    with pytest.raises(ValueError, match="equally long"):
        _advantage.label_frames(np.zeros(3), np.zeros(4, bool), np.zeros(3, bool), 0.4)
    with pytest.raises(ValueError, match="threshold_frames"):
        _advantage.label_frames(np.zeros(3), np.zeros(3, bool), np.zeros(3, bool), 0.4, "episode")


def test_label_runs_split_more_than_two_labels():
    starts, ends = _advantage.label_runs(np.array([2, 2, 0, 1, 1]))
    np.testing.assert_array_equal(starts, [0, 2, 3])
    np.testing.assert_array_equal(ends, [2, 3, 5])
