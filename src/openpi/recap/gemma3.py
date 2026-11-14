"""Gemma3-270M as the language backbone of the RECAP value model.

Wraps the `gemma` (google-deepmind/gemma) Flax implementation so it accepts precomputed embeddings,
which lets image tokens from our own SigLIP be placed in front of the text. `gemma` is an optional
dependency: import this module only where it is installed.
"""

from typing import Any

from gemma import gm
from gemma.gm.nn import _transformer
from gemma.gm.utils import _dtype_params
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

WIDTH = 640
PARAMS_PATH = "gs://gemma-data/checkpoints/gemma3-270m-pt"
TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"


class Gemma3Backbone(gm.nn.Gemma3_270M):
    """Gemma3-270M exposing token embedding and causal encoding of an embedding sequence.

    Parameter names match `gm.nn.Gemma3_270M`, so the released checkpoint loads unchanged.
    """

    def embed(self, tokens: jax.Array) -> jax.Array:
        """Embeds text tokens of shape (B, L) to (B, L, WIDTH)."""
        with _dtype_params.initialize_param_with_dtype(self.dtype):
            return self.embedder.encode(tokens)

    def encode(self, embeddings: jax.Array, mask: jax.Array) -> jax.Array:
        """Runs the causal transformer over embeddings (B, S, WIDTH); `mask` (B, S) marks real tokens."""
        length = mask.shape[1]
        causal = jnp.tril(jnp.ones((length, length), dtype=bool))
        inputs = _transformer._Inputs(
            embeddings=embeddings,
            # Padding tokens do not advance the position, as in gemma's own position builder.
            positions=jnp.maximum(jnp.cumsum(mask, axis=-1) - 1, 0),
            attention_mask=causal[None] & mask[:, None, :],
            inputs_mask=mask,
        )
        with _dtype_params.initialize_param_with_dtype(self.dtype):
            hidden, _ = self._apply_attention(inputs, None)
        return hidden


def load_pretrained_params(path: str) -> dict[str, Any]:
    """Loads the released Gemma3-270M checkpoint as the nested params of `Gemma3Backbone`.

    `gm.ckpts.load_params` needs a newer orbax than this repository pins, so the checkpoint is read with
    orbax directly. It stores a flat dict keyed like "transformer/layer_0/attn/q_einsum".

    Args:
        path: Local directory of the checkpoint (a copy of `PARAMS_PATH`).

    Returns:
        Nested params, e.g. `params["layer_0"]["attn"]["q_einsum"]["w"]`.
    """
    flat = ocp.PyTreeCheckpointer().restore(path)
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split("/")
        if parts[0] != "transformer":
            raise ValueError(f"Unexpected checkpoint key: {key}")
        node = nested
        for part in parts[1:-1]:
            node = node.setdefault(part, {})
        # The MLP weights are bare arrays in the model (`nn.share_scope`) but `{"w": array}` in the checkpoint.
        is_shared_scope_weight = parts[-2] == "mlp" and isinstance(value, dict) and value.keys() == {"w"}
        node[parts[-1]] = value["w"] if is_shared_scope_weight else value
    return nested
