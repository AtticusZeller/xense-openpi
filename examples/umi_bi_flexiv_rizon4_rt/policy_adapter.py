"""Client-side policy adapter between BiFlexiv execution and UMI training spaces."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from typing_extensions import override
from xense_client import base_policy as _base_policy

from examples.umi_bi_flexiv_rizon4_rt.frame_transform import PerArmUmiBiFlexivTransform


class BiFlexivUmiPolicyAdapter(_base_policy.BasePolicy):
    """Keep brokers in BiFlexiv space while the server stays in UMI space.

    This conversion boundary also handles RTC's ``prev_chunk_left_over``. That
    array is made of absolute BiFlexiv actions held by the client-side queue and
    must be converted to UMI before the server applies its training transforms.
    """

    def __init__(
        self,
        inner: _base_policy.BasePolicy,
        *,
        transform: PerArmUmiBiFlexivTransform | None = None,
        prompt: str | None = None,
    ) -> None:
        self._inner = inner
        self._transform = transform or PerArmUmiBiFlexivTransform.identity()
        self._prompt = prompt

    @override
    def infer(self, obs: dict, **kwargs) -> dict:
        images = obs.get("images")
        if not isinstance(images, Mapping):
            raise ValueError("Observation must contain an 'images' mapping")

        missing = {"left_wrist", "right_wrist"} - set(images)
        if missing:
            raise ValueError(f"Missing required wrist cameras: {tuple(sorted(missing))}")

        server_obs: dict[str, Any] = {
            "state": self._transform.flexiv_state_to_umi(np.asarray(obs["state"])),
            "images": {
                "left_wrist": np.asarray(images["left_wrist"]),
                "right_wrist": np.asarray(images["right_wrist"]),
            },
        }
        prompt = obs.get("prompt", self._prompt)
        if prompt is not None:
            server_obs["prompt"] = prompt

        server_kwargs = dict(kwargs)
        previous = server_kwargs.get("prev_chunk_left_over")
        if previous is not None:
            server_kwargs["prev_chunk_left_over"] = self._transform.flexiv_actions_to_umi(np.asarray(previous))

        result = self._inner.infer(server_obs, **server_kwargs)
        if "actions" not in result:
            raise ValueError(f"Policy response is missing 'actions'; got keys {tuple(result)}")

        # Only action chunks and scalar/dict timing metadata should reach the
        # action broker. In particular, drop model-space `actions_original` and
        # the array-valued server `state`, which are not executable actions.
        converted = {
            "actions": self._transform.umi_actions_to_flexiv(np.asarray(result["actions"])),
        }
        for key in ("server_timing", "policy_timing"):
            if key in result:
                converted[key] = result[key]
        return converted

    @override
    def reset(self) -> None:
        self._inner.reset()

    @override
    def warmup(self, obs: dict) -> None:
        # Brokers in this repository warm up through infer(). Keeping this as a
        # no-op avoids a second, subtly different conversion path.
        return None
