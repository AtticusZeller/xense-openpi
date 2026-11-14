"""Distributional value model for RECAP (pi*0.6).

The model reads camera images and the task text and predicts a categorical distribution over
`num_bins` return bins spread uniformly over [v_min, v_max]. It is trained with the cross-entropy
against a two-hot encoding of the normalized return; the value is the expectation of the distribution.

The backbones are injected so the model can be unit-tested with small stand-ins. The real ones are
SigLIP So400m (`openpi.models.siglip`) and Gemma3-270M (`openpi.recap.gemma3`).
"""

import flax.linen as nn
import jax
import jax.numpy as jnp

NUM_BINS = 201
V_MIN = -1.0
V_MAX = 0.0


def two_hot(values: jax.Array, *, num_bins: int = NUM_BINS, v_min: float = V_MIN, v_max: float = V_MAX) -> jax.Array:
    """Encodes scalars as distributions over the two neighboring bins, preserving the expectation.

    Args:
        values: Scalars of shape (...,). Values outside [v_min, v_max] are clipped.
        num_bins: Number of bins.
        v_min: Center of the first bin.
        v_max: Center of the last bin.

    Returns:
        Probabilities of shape (..., num_bins).
    """
    position = (jnp.clip(values, v_min, v_max) - v_min) / (v_max - v_min) * (num_bins - 1)
    lower = jnp.clip(jnp.floor(position), 0, num_bins - 2).astype(jnp.int32)
    upper_weight = position - lower
    return (
        jax.nn.one_hot(lower, num_bins) * (1.0 - upper_weight)[..., None]
        + jax.nn.one_hot(lower + 1, num_bins) * upper_weight[..., None]
    )


def expected_value(logits: jax.Array, *, v_min: float = V_MIN, v_max: float = V_MAX) -> jax.Array:
    """Returns the expectation of the categorical distribution given by logits of shape (..., num_bins)."""
    centers = jnp.linspace(v_min, v_max, logits.shape[-1])
    return jnp.sum(jax.nn.softmax(logits.astype(jnp.float32), axis=-1) * centers, axis=-1)


def value_loss(logits: jax.Array, targets: jax.Array, *, v_min: float = V_MIN, v_max: float = V_MAX) -> jax.Array:
    """Cross-entropy between the predicted distribution and the two-hot target, per sample."""
    target_probs = two_hot(targets, num_bins=logits.shape[-1], v_min=v_min, v_max=v_max)
    return -jnp.sum(target_probs * jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1), axis=-1)


class ValueModel(nn.Module):
    """Images and task text to return-bin logits.

    The sequence fed to the language backbone is [image tokens of every camera, text tokens, CLS]. The
    backbone is causal, so the trailing learned CLS token sees everything; its output is read out.

    Attributes:
        vision: Module mapping images (B, H, W, 3) in [-1, 1] to `(tokens (B, P, Dv), aux)`.
        language: Module with `embed(tokens) -> (B, L, width)` and
            `encode(embeddings (B, S, width), mask (B, S)) -> (B, S, width)`.
        width: Embedding width of the language backbone.
        num_bins: Number of return bins.
    """

    vision: nn.Module
    language: nn.Module
    width: int
    num_bins: int = NUM_BINS

    @nn.compact
    def __call__(self, images: jax.Array, tokens: jax.Array, token_mask: jax.Array) -> jax.Array:
        """Computes return-bin logits.

        Args:
            images: Camera images of shape (B, N, H, W, 3) in [-1, 1].
            tokens: Task text tokens of shape (B, L).
            token_mask: True for real text tokens, shape (B, L).

        Returns:
            Logits of shape (B, num_bins).
        """
        batch, num_cameras = images.shape[:2]
        image_tokens, _ = self.vision(images.reshape(batch * num_cameras, *images.shape[2:]), train=False)
        image_tokens = nn.Dense(self.width, name="image_proj")(image_tokens.astype(jnp.float32))
        image_tokens = image_tokens.reshape(batch, -1, self.width)

        text = self.language.embed(tokens)
        cls = self.param("cls", nn.initializers.normal(0.02), (1, 1, self.width))
        cls = jnp.broadcast_to(cls, (batch, 1, self.width))

        embeddings = jnp.concatenate([image_tokens.astype(text.dtype), text, cls.astype(text.dtype)], axis=1)
        mask = jnp.concatenate(
            [
                jnp.ones((batch, image_tokens.shape[1]), dtype=bool),
                token_mask.astype(bool),
                jnp.ones((batch, 1), dtype=bool),
            ],
            axis=1,
        )
        hidden = self.language.encode(embeddings, mask)
        return nn.Dense(self.num_bins, name="value_head")(hidden[:, -1].astype(jnp.float32))
