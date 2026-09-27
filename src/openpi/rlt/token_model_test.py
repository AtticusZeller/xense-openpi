import importlib.util
import os
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.rlt import config as _config

_CONFIG = _config.RLTModelConfig(embed_dim=64, prefix_seq_len=40, num_heads=4, dtype="float32")
_RLINF_MODULE = "rlinf/models/embodiment/modules/rlt_token_transformer.py"


def _model(input_dim: int = 64, config: _config.RLTModelConfig = _CONFIG):
    return config.create(input_dim, nnx.Rngs(0))


def _inputs(batch: int = 3, seq: int = 24, dim: int = 64):
    prefix = jax.random.normal(jax.random.key(1), (batch, seq, dim))
    mask = np.ones((batch, seq), dtype=bool)
    mask[1, 4:8] = False  # a missing "camera" block mid-sequence
    mask[1, 18:] = False  # prompt padding
    mask[2, 12:] = False
    return prefix, jnp.asarray(mask)


def _valid(x, mask):
    return np.asarray(x)[np.asarray(mask)]


def test_padding_content_is_inert():
    model = _model()
    prefix, mask = _inputs()
    noisy = jnp.where(mask[..., None], prefix, 1e3)
    loss_a, z_a = model(prefix, mask)
    loss_b, z_b = model(noisy, mask)
    np.testing.assert_allclose(z_a, z_b, atol=1e-5)
    np.testing.assert_allclose(loss_a, loss_b, rtol=1e-5)


def test_sample_is_independent_of_batch():
    model = _model()
    prefix, mask = _inputs()
    z_batch = model.encode(prefix, mask)
    recon_batch = model.decode(z_batch, prefix, mask)
    for i in range(prefix.shape[0]):
        z_one = model.encode(prefix[i : i + 1], mask[i : i + 1])
        np.testing.assert_allclose(z_one[0], z_batch[i], atol=1e-5)
        recon_one = model.decode(z_one, prefix[i : i + 1], mask[i : i + 1])
        np.testing.assert_allclose(_valid(recon_one[0], mask[i]), _valid(recon_batch[i], mask[i]), atol=1e-5)


def test_trailing_padding_can_be_trimmed():
    model = _model()
    prefix, mask = _inputs()
    trimmed = int(np.asarray(mask).any(axis=0).nonzero()[0].max()) + 1
    padded = jnp.concatenate([prefix, jnp.zeros_like(prefix[:, :8])], axis=1)
    padded_mask = jnp.concatenate([mask, jnp.zeros_like(mask[:, :8])], axis=1)
    loss_a, z_a = model(prefix[:, :trimmed], mask[:, :trimmed])
    loss_b, z_b = model(padded, padded_mask)
    np.testing.assert_allclose(z_a, z_b, atol=1e-5)
    np.testing.assert_allclose(loss_a, loss_b, rtol=1e-5)


def test_remat_matches():
    prefix, mask = _inputs()
    plain = _model()
    remat = _model(config=_config.RLTModelConfig(**{**_CONFIG.__dict__, "remat": True}))
    grad_plain = nnx.grad(lambda m: m(prefix, mask)[0])(plain)
    grad_remat = nnx.grad(lambda m: m(prefix, mask)[0])(remat)
    for a, b in zip(jax.tree.leaves(grad_plain), jax.tree.leaves(grad_remat), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-6)


def _load_rlinf():
    root = pathlib.Path(os.environ.get("RLINF_ROOT", pathlib.Path(__file__).resolve().parents[4] / "RLinf"))
    path = root / _RLINF_MODULE
    if not path.is_file():
        pytest.skip(f"RLinf reference not found at {path} (set RLINF_ROOT)")
    spec = importlib.util.spec_from_file_location("rlinf_rlt_token_transformer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_block(block, layer) -> None:
    def t(x):
        return jnp.asarray(x.detach().numpy())

    block.attn_norm.scale.value, block.attn_norm.bias.value = t(layer.self_norm.weight), t(layer.self_norm.bias)
    block.attn.qkv.kernel.value = t(layer.self_attn.in_proj_weight).T
    block.attn.qkv.bias.value = t(layer.self_attn.in_proj_bias)
    block.attn.out.kernel.value = t(layer.self_attn.out_proj.weight).T
    block.attn.out.bias.value = t(layer.self_attn.out_proj.bias)
    block.mlp_norm.scale.value, block.mlp_norm.bias.value = t(layer.mlp_norm.weight), t(layer.mlp_norm.bias)
    for ours, theirs in ((block.mlp_in, layer.mlp[0]), (block.geglu, layer.mlp[2].proj), (block.mlp_out, layer.mlp[3])):
        ours.kernel.value, ours.bias.value = t(theirs.weight).T, t(theirs.bias)


def _copy_linear(ours, theirs) -> None:
    ours.kernel.value = jnp.asarray(theirs.weight.detach().numpy()).T
    ours.bias.value = jnp.asarray(theirs.bias.detach().numpy())


@pytest.mark.parametrize("input_dim", [64, 48])
def test_matches_rlinf(input_dim):
    torch = pytest.importorskip("torch")
    rlinf = _load_rlinf()
    torch.manual_seed(0)
    reference = rlinf.RLTTokenTransformer(
        input_dim=input_dim,
        embed_dim=_CONFIG.embed_dim,
        prefix_seq_len=_CONFIG.prefix_seq_len,
        num_layers=_CONFIG.num_layers,
        num_heads=_CONFIG.num_heads,
        mlp_ratio=_CONFIG.mlp_ratio,
    ).eval()

    model = _model(input_dim)
    enc, dec = reference.encoder, reference.decoder
    model.rl_token.value = jnp.asarray(enc.rl_token_embed.detach().numpy())
    model.rl_token_pos.value = jnp.asarray(enc.rl_token_pos_enc.detach().numpy())
    model.prefix_pos.value = jnp.asarray(enc.prefix_pos_enc.detach().numpy())
    model.decoder_pos.value = jnp.asarray(dec.decoder_pos_enc.detach().numpy())
    if input_dim != _CONFIG.embed_dim:
        _copy_linear(model.input_proj, enc.input_proj)
        _copy_linear(model.teacher_proj, dec.teacher_input_proj)
    _copy_linear(model.output_proj, dec.output_proj)
    for block, layer in zip(model.encoder.layers, enc.layers, strict=True):
        _copy_block(block, layer)
    for block, layer in zip(model.decoder.layers, dec.layers, strict=True):
        _copy_block(block, layer)

    prefix, mask = _inputs(dim=input_dim)
    with torch.no_grad():
        ref_loss, ref_info = reference(torch.from_numpy(np.array(prefix)), torch.from_numpy(np.array(mask)))
    loss, z_rl = model(prefix, mask)
    np.testing.assert_allclose(z_rl, ref_info["z_rl"].numpy(), atol=1e-4)
    np.testing.assert_allclose(loss, ref_loss.item(), rtol=1e-4)
