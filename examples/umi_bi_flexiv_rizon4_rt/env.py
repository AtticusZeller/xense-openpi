"""OpenPI environment for UMI inference on a BiFlexiv Rizon4 RT robot.

State/action format (native BiFlexiv 20D, identical to bi_flexiv_rizon4_rt):
    [left_tcp.x/y/z/r1-r6 (0-8), right_tcp.x/y/z/r1-r6 (9-17),
     left_gripper.pos (18), right_gripper.pos (19)]

Unlike examples/bi_flexiv_rizon4_rt, this layer does NOT talk to the policy
in the model's own space: conversion to the policy's first-frame-relative
coordinate space (per arm, layout unchanged) happens in
policy_adapter.BiFlexivUmiPolicyAdapter, so everything below
the broker stays in the native BiFlexiv space the robot driver understands.
"""

import einops
from lerobot.utils.robot_utils import get_logger
import numpy as np
from typing_extensions import override
from xense_client import image_tools
from xense_client.runtime import environment as _environment

from examples.umi_bi_flexiv_rizon4_rt.real_env import UmiBiFlexivRizon4RTRealEnv

logger = get_logger("UmiBiFlexivRizon4RTEnv")

# Policy-facing camera names. UMI training data has no third-person view, so
# only the two wrist cameras are connected and sent; the model's base_0_rgb
# slot is filled with a black image server-side (see umi_policy.UmiInputs).
_WRIST_CAMERAS = ("left_wrist", "right_wrist")

# Action dimension labels for debug logging (20D Cartesian, native BiFlexiv order)
BI_FLEXIV_ACTION_NAMES = (
    "left_tcp.x",
    "left_tcp.y",
    "left_tcp.z",
    "left_tcp.r1",
    "left_tcp.r2",
    "left_tcp.r3",
    "left_tcp.r4",
    "left_tcp.r5",
    "left_tcp.r6",
    "right_tcp.x",
    "right_tcp.y",
    "right_tcp.z",
    "right_tcp.r1",
    "right_tcp.r2",
    "right_tcp.r3",
    "right_tcp.r4",
    "right_tcp.r5",
    "right_tcp.r6",
    "left_gripper.pos",
    "right_gripper.pos",
)


class UmiBiFlexivRizon4RTEnvironment(_environment.Environment):
    """OpenPI environment exposing native BiFlexiv obs/actions for UMI inference.

    Same obs/action decoupling as examples/bi_flexiv_rizon4_rt: get_observation()
    reads the cameras + robot state fresh, and apply_action() only sends a target
    pose — no observation read. The outer runtime loop owns obs scheduling.

    All policy-space conversion (per-arm first-frame-relative frame transform)
    happens outside this class, in BiFlexivUmiPolicyAdapter. This layer stays
    strictly in the native BiFlexiv 20D space.

    Camera name mapping (real → policy):
        left_wrist  -> left_wrist
        right_wrist -> right_wrist
        (no head camera; UMI checkpoints mask out the base_0_rgb slot)
    """

    def __init__(
        self,
        *,
        bi_mount_type: str = "forward",
        use_force: bool = False,
        go_to_start: bool = True,
        stiffness_ratio: float = 0.2,
        inner_control_hz: int = 1000,
        interpolate_cmds: bool = True,
        log_level: str = "INFO",
        render_height: int = 224,
        render_width: int = 224,
        setup_robot: bool = True,
    ) -> None:
        self._env = UmiBiFlexivRizon4RTRealEnv(
            bi_mount_type=bi_mount_type,
            use_force=use_force,
            go_to_start=go_to_start,
            stiffness_ratio=stiffness_ratio,
            inner_control_hz=inner_control_hz,
            interpolate_cmds=interpolate_cmds,
            log_level=log_level,
            setup_robot=setup_robot,
        )
        self._render_height = render_height
        self._render_width = render_width

    @override
    def reset(self) -> None:
        self._env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        # Reads cameras + robot state fresh. Returns the obs the policy should
        # see for THIS step's action — matching the bi_flexiv_rizon4_rt env and
        # the lerobot recorder convention used to train this stack.
        raw_obs = self._env.get_observation()
        images = {}
        raw_images = {}
        for camera in _WRIST_CAMERAS:
            # Unlike bi_flexiv_rizon4_rt (which silently skips missing cameras),
            # a missing wrist camera is fatal here: the UMI checkpoint was
            # trained on exactly these two views, so running without one would
            # feed the policy a silently wrong observation.
            if camera not in raw_obs["images"]:
                raise RuntimeError(f"Required camera {camera!r} is missing; got {tuple(raw_obs['images'])}")
            image = raw_obs["images"][camera]
            resized = image_tools.resize_with_pad(
                np.expand_dims(image, axis=0), self._render_height, self._render_width
            )[0]
            # (H, W, C) -> (C, H, W) for OpenPI policy input
            images[camera] = einops.rearrange(resized, "h w c -> c h w")
            # Raw images (original resolution HWC) passed through for recording.
            raw_images[camera] = image

        # Fail fast on a malformed state vector: the frame transform downstream
        # would otherwise turn a bad read into a large, hard-to-debug arm motion.
        state = np.asarray(raw_obs["qpos"], dtype=np.float32)
        if state.shape != (20,) or not np.all(np.isfinite(state)):
            raise RuntimeError(f"Invalid BiFlexiv state: shape={state.shape}, finite={np.all(np.isfinite(state))}")
        return {"state": state, "images": images, "images_raw": raw_images}

    @override
    def apply_action(self, action: dict) -> None:
        # Validate before sending: the action leaving this layer is executed by
        # the RT driver verbatim (it is already in native BiFlexiv space).
        actions = np.asarray(action["actions"], dtype=np.float32)
        if actions.shape != (20,):
            raise ValueError(f"Expected one BiFlexiv 20D action, got {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("Refusing to execute an action containing NaN or Inf")
        logger.debug(f"BiFlexiv action: {dict(zip(BI_FLEXIV_ACTION_NAMES, actions, strict=True))}")
        # Pure send — no observation read. The outer loop owns obs scheduling.
        self._env.send_action(actions)

    def disconnect(self) -> None:
        self._env.disconnect()
