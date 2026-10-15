import functools

import jax

from openpi.models import pi0_config
from openpi.policies import policy as _policy


def test_infer_batch_for_training_runs_under_caller_jit():
    """The EXPO-FT learner builds a Policy and samples candidates inside its own `jax.jit`."""
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=6,
        max_token_len=8,
        pi05=True,
    )
    model = config.create(jax.random.key(0))
    inputs = config.fake_obs(2).to_dict()

    @functools.partial(jax.jit, static_argnames=("num_samples",))
    def sample(inputs, rng, num_samples):
        policy = _policy.Policy(model, rng=rng, sample_kwargs={"num_samples": num_samples, "num_steps": 2})
        return policy.infer(inputs, is_batch=True, for_training=True)

    outputs = sample(inputs, jax.random.key(1), num_samples=3)
    assert outputs["actions"].shape == (2 * 3, config.action_horizon, config.action_dim)
    assert isinstance(outputs["actions"], jax.Array)
