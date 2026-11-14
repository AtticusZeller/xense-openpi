import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.recap import checkpoints as _checkpoints


def _params() -> dict:
    return {"cls": jnp.arange(4, dtype=jnp.float32), "head": {"kernel": jnp.ones((2, 3)), "bias": jnp.zeros(3)}}


def test_saved_params_restore_to_the_same_values(tmp_path):
    params = _params()

    _checkpoints.save_params(tmp_path / "params", params)
    # A freshly initialized model has the same structure but other values.
    template = jax.tree.map(jnp.zeros_like, params)
    restored = _checkpoints.match_params(template, _checkpoints.restore_params(tmp_path / "params"), "init")

    assert jax.tree.structure(restored) == jax.tree.structure(params)
    for expected, actual in zip(jax.tree.leaves(params), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype


def test_save_params_replaces_an_existing_checkpoint(tmp_path):
    _checkpoints.save_params(tmp_path / "params", _params())
    newer = {**_params(), "cls": jnp.full(4, 7.0)}

    _checkpoints.save_params(tmp_path / "params", newer)

    np.testing.assert_array_equal(_checkpoints.restore_params(tmp_path / "params")["cls"], np.full(4, 7.0))


def test_match_params_rejects_other_names_and_shapes():
    params = _params()

    with pytest.raises(ValueError, match="missing \\['head/bias'\\]"):
        _checkpoints.match_params(params, {"cls": params["cls"], "head": {"kernel": params["head"]["kernel"]}}, "x")
    with pytest.raises(ValueError, match="expected shape \\(4,\\)"):
        _checkpoints.match_params(params, {**params, "cls": jnp.zeros(5)}, "x")
