"""Saving, restoring and shape-checking the params of the RECAP value model."""

import pathlib

import flax.traverse_util
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp


def match_params(expected: dict, loaded: dict, name: str) -> dict:
    """Returns `loaded` cast to the dtypes of `expected` after checking that names and shapes match.

    Args:
        expected: Params tree that defines the required names, shapes and dtypes.
        loaded: Params tree to take the values from.
        name: Label of the tree in error messages.

    Raises:
        ValueError: If a name is missing or unexpected, or a shape differs.
    """
    expected_flat = flax.traverse_util.flatten_dict(expected, sep="/")
    loaded_flat = flax.traverse_util.flatten_dict(loaded, sep="/")
    if expected_flat.keys() != loaded_flat.keys():
        raise ValueError(
            f"{name}: missing {sorted(expected_flat.keys() - loaded_flat.keys())[:5]}, "
            f"unexpected {sorted(loaded_flat.keys() - expected_flat.keys())[:5]}"
        )
    for key, value in expected_flat.items():
        if value.shape != loaded_flat[key].shape:
            raise ValueError(f"{name}/{key}: expected shape {value.shape}, got {loaded_flat[key].shape}")
    converted = {key: jnp.asarray(value, dtype=expected_flat[key].dtype) for key, value in loaded_flat.items()}
    return flax.traverse_util.unflatten_dict(converted, sep="/")


def save_params(directory: pathlib.Path, params: dict) -> None:
    """Writes `params` to `directory`, replacing what is there."""
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(directory.resolve(), {"params": jax.device_get(params)}, force=True)


def restore_params(directory: pathlib.Path) -> dict:
    """Reads the params written by `save_params` as numpy arrays."""
    with ocp.PyTreeCheckpointer() as checkpointer:
        return checkpointer.restore(directory.resolve())["params"]
