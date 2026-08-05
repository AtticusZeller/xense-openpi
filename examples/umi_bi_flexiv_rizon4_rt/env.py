"""OpenPI environment for UMI inference on a BiFlexiv Rizon4 RT robot."""

import einops
from lerobot.utils.robot_utils import get_logger
import numpy as np
from typing_extensions import override
from xense_client import image_tools
from xense_client.runtime import environment as _environment

from examples.umi_bi_flexiv_rizon4_rt.real_env import UmiBiFlexivRizon4RTRealEnv

logger = get_logger("UmiBiFlexivRizon4RTEnv")
_WRIST_CAMERAS = ("left_wrist", "right_wrist")
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
    """Expose BiFlexiv observations/actions; policy conversion happens outside."""

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
        raw_obs = self._env.get_observation()
        images = {}
        raw_images = {}
        for camera in _WRIST_CAMERAS:
            if camera not in raw_obs["images"]:
                raise RuntimeError(f"Required camera {camera!r} is missing; got {tuple(raw_obs['images'])}")
            image = raw_obs["images"][camera]
            resized = image_tools.resize_with_pad(
                np.expand_dims(image, axis=0), self._render_height, self._render_width
            )[0]
            images[camera] = einops.rearrange(resized, "h w c -> c h w")
            raw_images[camera] = image

        state = np.asarray(raw_obs["qpos"], dtype=np.float32)
        if state.shape != (20,) or not np.all(np.isfinite(state)):
            raise RuntimeError(f"Invalid BiFlexiv state: shape={state.shape}, finite={np.all(np.isfinite(state))}")
        return {"state": state, "images": images, "images_raw": raw_images}

    @override
    def apply_action(self, action: dict) -> None:
        actions = np.asarray(action["actions"], dtype=np.float32)
        if actions.shape != (20,):
            raise ValueError(f"Expected one BiFlexiv 20D action, got {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("Refusing to execute an action containing NaN or Inf")
        logger.debug(f"BiFlexiv action: {dict(zip(BI_FLEXIV_ACTION_NAMES, actions, strict=True))}")
        self._env.send_action(actions)

    def disconnect(self) -> None:
        self._env.disconnect()
