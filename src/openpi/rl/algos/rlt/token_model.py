"""RLT phase one: RL-token encoder/decoder over frozen VLA prefix hidden states.

A JAX port of RLinf's ``rlinf/models/embodiment/modules/rlt_token_transformer.py``
(same blocks, initialization and objective). The RL token is a learned
continuous embedding - never a vocab token, never fed back into the VLA. The
encoder compresses the prefix into it (``z_rl``); the decoder, a training-only
auxiliary, reconstructs the prefix autoregressively from it with teacher
forcing, under a masked MSE.

Masking relies on the pi0 prefix layout being fixed: each camera owns a
256-token block and the prompt is right-padded to ``max_token_len``, so a
token's slot - and with it its absolute positional encoding - does not depend
on what else is in the batch. Padded slots are excluded as attention keys and
from the loss; since they are never queried for the RL token and the decoder
is causal, trailing padding can be trimmed without changing any result.
"""

import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.rl.algos.rlt import config as _config
from openpi.shared import array_typing as at


def sinusoidal_table(seq_len: int, dim: int) -> np.ndarray:
    """Transformer sin/cos table, interleaved (even dims sin, odd dims cos)."""
    position = np.arange(seq_len, dtype=np.float32)[:, None]
    div_term = np.exp(np.arange(0, dim, 2, dtype=np.float32) * -(math.log(10000.0) / dim))
    table = np.zeros((seq_len, dim), dtype=np.float32)
    table[:, 0::2] = np.sin(position * div_term)
    table[:, 1::2] = np.cos(position * div_term[: table[:, 1::2].shape[1]])
    return table


def _uniform(limit: float) -> nnx.Initializer:
    return lambda key, shape, dtype=jnp.float32: jax.random.uniform(key, shape, dtype, -limit, limit)


def _linear(in_dim: int, out_dim: int, *, dtype, rngs: nnx.Rngs, zero_bias: bool = False) -> nnx.Linear:
    """``nn.Linear`` with torch's default init: weight and bias ~ U(+-1/sqrt(fan_in))."""
    limit = 1.0 / math.sqrt(in_dim)
    return nnx.Linear(
        in_dim,
        out_dim,
        kernel_init=_uniform(limit),
        bias_init=nnx.initializers.zeros_init() if zero_bias else _uniform(limit),
        dtype=dtype,
        rngs=rngs,
    )


class _Attention(nnx.Module):
    """``torch.nn.MultiheadAttention`` (batch-first, biased, no dropout)."""

    def __init__(self, dim: int, num_heads: int, *, dtype, rngs: nnx.Rngs):
        self.num_heads = num_heads
        # torch initializes the fused (3*dim, dim) in_proj with xavier_uniform and zero bias.
        self.qkv = nnx.Linear(
            dim,
            3 * dim,
            kernel_init=_uniform(math.sqrt(6.0 / (dim + 3 * dim))),
            bias_init=nnx.initializers.zeros_init(),
            dtype=dtype,
            rngs=rngs,
        )
        self.out = _linear(dim, dim, dtype=dtype, rngs=rngs, zero_bias=True)

    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        """``mask`` is True where attention is allowed, broadcastable to (b, heads, q, k)."""
        b, s, d = x.shape
        q, k, v = jnp.split(self.qkv(x).reshape(b, s, 3 * self.num_heads, d // self.num_heads), 3, axis=2)
        return self.out(jax.nn.dot_product_attention(q, k, v, mask=mask).reshape(b, s, d))


class _Block(nnx.Module):
    """Pre-LN block: ``x + attn(ln(x))`` then ``x + mlp(ln(x))``, with RLinf's GeGLU MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, *, dtype, rngs: nnx.Rngs):
        mlp_dim = int(dim * mlp_ratio)
        self.attn_norm = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        self.attn = _Attention(dim, num_heads, dtype=dtype, rngs=rngs)
        self.mlp_norm = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        self.mlp_in = _linear(dim, mlp_dim, dtype=dtype, rngs=rngs)
        self.geglu = _linear(mlp_dim, 2 * mlp_dim, dtype=dtype, rngs=rngs)
        self.mlp_out = _linear(mlp_dim, dim, dtype=dtype, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        x = x + self.attn(self.attn_norm(x), mask)
        h, gate = jnp.split(self.geglu(self.mlp_in(self.mlp_norm(x))), 2, axis=-1)
        return x + self.mlp_out(h * jax.nn.gelu(gate, approximate=False))


class _Stack(nnx.Module):
    def __init__(self, config: _config.RLTModelConfig, *, dtype, rngs: nnx.Rngs):
        self.layers = [
            _Block(config.embed_dim, config.num_heads, config.mlp_ratio, dtype=dtype, rngs=rngs)
            for _ in range(config.num_layers)
        ]
        self.remat = config.remat

    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = nnx.remat(lambda layer, x, mask: layer(x, mask))(layer, x, mask) if self.remat else layer(x, mask)
        return x


def _maybe_proj(in_dim: int, out_dim: int, *, dtype, rngs: nnx.Rngs) -> nnx.Linear | None:
    return _linear(in_dim, out_dim, dtype=dtype, rngs=rngs) if in_dim != out_dim else None


class RLTTokenTransformer(nnx.Module):
    """Single-RL-token encoder with an autoregressive reconstruction decoder."""

    def __init__(self, config: _config.RLTModelConfig, *, input_dim: int, rngs: nnx.Rngs):
        dtype = jnp.dtype(config.dtype)
        dim = config.embed_dim
        self.prefix_seq_len = config.prefix_seq_len

        # Encoder.
        self.input_proj = _maybe_proj(input_dim, dim, dtype=dtype, rngs=rngs)
        self.rl_token = nnx.Param(jnp.asarray(sinusoidal_table(1, dim)))
        self.rl_token_pos = nnx.Param(jnp.asarray(sinusoidal_table(1, dim)))
        self.prefix_pos = nnx.Param(jnp.asarray(sinusoidal_table(config.prefix_seq_len, dim)))
        self.encoder = _Stack(config, dtype=dtype, rngs=rngs)

        # Decoder.
        self.teacher_proj = _maybe_proj(input_dim, dim, dtype=dtype, rngs=rngs)
        self.decoder_pos = nnx.Param(jnp.asarray(sinusoidal_table(config.prefix_seq_len, dim)))
        self.decoder = _Stack(config, dtype=dtype, rngs=rngs)
        self.output_proj = _linear(dim, input_dim, dtype=dtype, rngs=rngs)

    def _check_len(self, seq_len: int) -> None:
        if seq_len > self.prefix_seq_len:
            raise ValueError(f"prefix length {seq_len} exceeds prefix_seq_len {self.prefix_seq_len}.")

    def encode(self, prefix: at.Float[at.Array, "b s d"], mask: at.Bool[at.Array, "b s"]) -> at.Float[at.Array, "b e"]:
        """Compress the prefix into the RL token; returns ``z_rl``."""
        b, s, _ = prefix.shape
        self._check_len(s)
        x = prefix.astype(jnp.float32)
        if self.input_proj is not None:
            x = self.input_proj(x).astype(jnp.float32)
        x = x + self.prefix_pos[:s]
        rl_token = jnp.broadcast_to(self.rl_token[...] + self.rl_token_pos[...], (b, 1, x.shape[-1]))
        x = jnp.concatenate([x, rl_token], axis=1)
        keys = jnp.concatenate([mask, jnp.ones((b, 1), dtype=bool)], axis=1)
        return self.encoder(x, keys[:, None, None, :])[:, -1]

    def decode(
        self, z_rl: at.Float[at.Array, "b e"], prefix: at.Float[at.Array, "b s d"], mask: at.Bool[at.Array, "b s"]
    ) -> at.Float[at.Array, "b s d"]:
        """Teacher-forced reconstruction: output i predicts prefix token i from [z_rl, prefix[:i]]."""
        s = prefix.shape[1]
        self._check_len(s)
        teacher = jnp.where(mask[:, :-1, None], prefix[:, :-1], 0).astype(jnp.float32)
        if self.teacher_proj is not None:
            teacher = self.teacher_proj(teacher).astype(jnp.float32)
        x = jnp.concatenate([z_rl[:, None].astype(jnp.float32), teacher], axis=1) + self.decoder_pos[:s]
        keys = jnp.concatenate([jnp.ones_like(mask[:, :1]), mask[:, :-1]], axis=1)
        causal = jnp.tril(jnp.ones((s, s), dtype=bool))
        return self.output_proj(self.decoder(x, causal[None, None] & keys[:, None, None, :]))

    def __call__(
        self, prefix: at.Float[at.Array, "b s d"], mask: at.Bool[at.Array, "b s"]
    ) -> tuple[at.Float[at.Array, ""], at.Float[at.Array, "b e"]]:
        """Masked reconstruction MSE (over valid tokens and features) and ``z_rl``."""
        prefix = jax.lax.stop_gradient(prefix)
        z_rl = self.encode(prefix, mask)
        recon = self.decode(z_rl, prefix, mask).astype(jnp.float32)
        weights = mask[..., None].astype(jnp.float32)
        sq_error = jnp.square(recon - prefix.astype(jnp.float32)) * weights
        mse = sq_error.sum() / jnp.maximum(weights.sum() * prefix.shape[-1], 1.0)
        return mse, z_rl
