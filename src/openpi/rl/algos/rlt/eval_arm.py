"""RLT as an evaluation arm (``openpi.rl.eval``).

``rlt`` runs the trained actor's mean chunk inside open windows - the only place it
ever drove during training - and the frozen VLA's reference outside them. ``vla`` runs
the VLA reference everywhere, from the same feature pass, so the two arms differ only
in who drives the windows. Both execute C steps per chunk, the training cadence.
"""

from __future__ import annotations

import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi.rl.algos.rlt import config as _rlt_config
from openpi.rl.algos.rlt import features as _features
from openpi.rl.algos.rlt import mlp_policy
from openpi.rl.algos.rlt import serving

ARMS = ("vla", "rlt")


class RLTArm:
    def __init__(self, extractor: _features.FeatureExtractor, actor: mlp_policy.Actor | None, *, horizon: int):
        self.name = "vla" if actor is None else "rlt"
        self.space = extractor.space
        self._extractor = extractor
        self._horizon = horizon
        self._act = None
        if actor is not None:
            space = extractor.space

            @jax.jit
            def act(obs):
                return space.decode(actor(obs), obs["state"])[0]

            self._act = act

    def act(self, obs: dict[str, Any], *, window_open: bool) -> tuple[np.ndarray, str]:
        features = self._extractor.extract(obs)
        if self._act is not None and window_open:
            batched = {key: jnp.asarray(features[key])[None] for key in ("z_rl", "state", "proprio", "ref_chunk")}
            return np.asarray(self._act(batched)), "actor"
        return features["ref_exec"][: self._horizon], "vla"


def create_arm(config: _rlt_config.RLTConfig, arm: str, actor_checkpoint: pathlib.Path | str | None = None) -> RLTArm:
    """The ``vla`` or ``rlt`` arm of ``config``'s run; ``rlt`` defaults to the run's latest RL round."""
    if arm not in ARMS:
        raise ValueError(f"Unknown RLT arm {arm!r}; expected one of {ARMS}.")
    rl_checkpoint = serving.resolve_rl_checkpoint(config, actor_checkpoint) if arm == "rlt" else None
    extractor, binding = serving.load_extractor(config)
    actor = None if rl_checkpoint is None else serving.load_trained_actor(config, rl_checkpoint, extractor, binding)
    return RLTArm(extractor, actor, horizon=config.rl.num_action_chunks)
