"""Client-side policy adapter between BiFlexiv execution and policy coordinate space."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from typing_extensions import override
from xense_client import base_policy as _base_policy

from examples.umi_bi_flexiv_rizon4_rt.frame_transform import PerArmFirstFrameTransform


class BiFlexivUmiPolicyAdapter(_base_policy.BasePolicy):
    """Keep brokers in native BiFlexiv space while the server stays in first-frame-relative space.

    Both sides use the native BiFlexiv layout
    [left_tcp(0-8), right_tcp(9-17), left_gripper(18), right_gripper(19)] — the
    training data was converted to it by scripts/convert_umi_first_frame_relative.py —
    so the only conversion here is the per-arm coordinate frame: the episode's
    first observation defines each arm's reference frame, captured lazily on the
    first infer() after reset().

    This conversion boundary also handles RTC's ``prev_chunk_left_over``. That
    array is made of absolute BiFlexiv actions held by the client-side queue and
    must be converted to first-frame-relative space before the server applies
    its training transforms.
    """

    def __init__(
        self,
        inner: _base_policy.BasePolicy,
        *,
        transform: PerArmFirstFrameTransform | None = None,
        prompt: str | None = None,
    ) -> None:
        self._inner = inner
        self._transform = transform or PerArmFirstFrameTransform()
        self._prompt = prompt

    @override
    def infer(self, obs: dict, **kwargs) -> dict:
        # Note: obs may carry "images_raw" (original-resolution HWC images for
        # recording); only the resized CHW "images" are forwarded to the server.
        images = obs.get("images")
        if not isinstance(images, Mapping):
            raise ValueError("Observation must contain an 'images' mapping")

        missing = {"left_wrist", "right_wrist"} - set(images)
        if missing:
            raise ValueError(f"Missing required wrist cameras: {tuple(sorted(missing))}")

        server_obs: dict[str, Any] = {
            "state": self._transform.flexiv_to_policy(np.asarray(obs["state"])),
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
            server_kwargs["prev_chunk_left_over"] = self._transform.flexiv_to_policy(np.asarray(previous))

        result = self._inner.infer(server_obs, **server_kwargs)
        if "actions" not in result:
            raise ValueError(f"Policy response is missing 'actions'; got keys {tuple(result)}")

        # Only action chunks and scalar/dict timing metadata should reach the
        # action broker. In particular, drop model-space `actions_original` and
        # the array-valued server `state`, which are not executable actions.
        # result["actions"] are ABSOLUTE first-frame-relative poses — the
        # server's output pipeline (AbsoluteActions) has already un-deltaed the
        # model output against the state we sent — so the rigid transform below
        # is applied to absolute poses, as required.
        converted = {
            "actions": self._transform.policy_to_flexiv(np.asarray(result["actions"])),
        }
        for key in ("server_timing", "policy_timing"):
            if key in result:
                converted[key] = result[key]
        return converted

    @override
    def reset(self) -> None:
        # New episode: drop the captured first-frame reference so the next
        # infer() re-captures it from the new initial observation.
        self._transform.reset()
        self._inner.reset()

    @override
    def warmup(self, obs: dict) -> None:
        # Brokers in this repository warm up through infer(). Keeping this as a
        # no-op avoids a second, subtly different conversion path.
        return None
