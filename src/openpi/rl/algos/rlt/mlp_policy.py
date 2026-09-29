"""RLT phase two: MLP actor and twin-Q critic over the frozen RL token.

Port of TacXense's ``tacxense/rlt/mlp_policy.py`` (itself adapted from RLinf's
``rlt_mlp_policy.py``). Observations are dicts with

- ``z_rl``: ``(B, Z)`` RL-token feature from the frozen phase-one encoder,
- ``proprio``: ``(B, S)`` normalized robot state (``ActionSpace.normalize_state``),
- ``ref_chunk``: ``(B, R, A)`` normalized VLA reference chunk; the actor reads its first C steps.

The actor is a Gaussian with fixed std in normalized action space: the last
linear layer is the mean, training samples ``mean + std * eps``, inference uses
the mean, and both are clipped to ``[-1, 1]``. The critic is an ensemble of
independent Q heads over ``[z_rl, proprio, action_chunk]``. There is no target
actor: the TD bootstrap scores the online actor's next action with a target copy
of the critic, whose parameters the learner keeps as a separate state.
"""

import itertools
import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.rl.algos.rlt import config as _config

_RELU_GAIN = math.sqrt(2.0)


class _MLP(nnx.Module):
    """Linear layers with ReLU (and optional LayerNorm before it) between them."""

    def __init__(
        self,
        dims: tuple[int, ...],
        *,
        layer_norm: bool,
        hidden_init: nnx.Initializer,
        output_init: nnx.Initializer,
        rngs: nnx.Rngs,
    ):
        last = len(dims) - 2
        self.linears = [
            nnx.Linear(d_in, d_out, kernel_init=output_init if i == last else hidden_init, rngs=rngs)
            for i, (d_in, d_out) in enumerate(itertools.pairwise(dims))
        ]
        self.norms = [nnx.LayerNorm(d, epsilon=1e-5, rngs=rngs) for d in dims[1:-1]] if layer_norm else None

    def __call__(self, x: jax.Array) -> jax.Array:
        for i, linear in enumerate(self.linears[:-1]):
            x = linear(x)
            if self.norms is not None:
                x = self.norms[i](x)
            x = jax.nn.relu(x)
        return self.linears[-1](x)


class Actor(nnx.Module):
    def __init__(self, config: _config.RLConfig, *, z_dim: int, state_dim: int, action_dim: int, rngs: nnx.Rngs):
        self.num_action_chunks = config.num_action_chunks
        self.action_dim = action_dim
        self.fixed_std = config.fixed_std
        flat = config.num_action_chunks * action_dim
        self.net = _MLP(
            (z_dim + state_dim + flat, *config.actor_hidden_dims, flat),
            layer_norm=config.actor_layer_norm,
            hidden_init=nnx.initializers.orthogonal(_RELU_GAIN),
            # Tiny output head: initial actions sit near the normalized origin.
            output_init=nnx.initializers.orthogonal(0.01 * _RELU_GAIN),
            rngs=rngs,
        )

    def __call__(
        self,
        obs: dict[str, jax.Array],
        *,
        noise_rng: jax.Array | None = None,
        dropout_rng: jax.Array | None = None,
        reference_dropout_prob: float = 0.0,
    ) -> jax.Array:
        """Normalized action chunk ``(B, C, A)``; the mean when ``noise_rng`` is None."""
        batch = obs["z_rl"].shape[0]
        ref = obs["ref_chunk"][:, : self.num_action_chunks].reshape(batch, -1)
        if dropout_rng is not None and reference_dropout_prob > 0:
            keep = jax.random.uniform(dropout_rng, (batch, 1)) >= reference_dropout_prob
            ref = ref * keep
        mean = self.net(jnp.concatenate([obs["z_rl"], obs["proprio"], ref], axis=-1))
        action = mean if noise_rng is None else mean + self.fixed_std * jax.random.normal(noise_rng, mean.shape)
        return jnp.clip(action, -1.0, 1.0).reshape(batch, self.num_action_chunks, self.action_dim)


class Critic(nnx.Module):
    def __init__(self, config: _config.RLConfig, *, z_dim: int, state_dim: int, action_dim: int, rngs: nnx.Rngs):
        dims = (z_dim + state_dim + config.num_action_chunks * action_dim, *config.critic_hidden_dims, 1)
        self.heads = [
            _MLP(
                dims,
                layer_norm=config.critic_layer_norm,
                hidden_init=nnx.initializers.variance_scaling(_RELU_GAIN**2, "fan_avg", "uniform"),  # xavier_uniform
                output_init=nnx.initializers.normal(0.02),
                rngs=rngs,
            )
            for _ in range(config.num_q_heads)
        ]

    def __call__(self, obs: dict[str, jax.Array], actions: jax.Array) -> jax.Array:
        """Q values ``(B, num_q_heads)`` of a full action chunk ``(B, C, A)``."""
        x = jnp.concatenate([obs["z_rl"], obs["proprio"], actions.reshape(actions.shape[0], -1)], axis=-1)
        return jnp.concatenate([head(x) for head in self.heads], axis=-1)
