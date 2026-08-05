"""BiFlexiv Rizon4 RT real environment with the head camera disabled."""

import collections
import time

from lerobot.robots.bi_flexiv_rizon4_rt.config_bi_flexiv_rizon4_rt import BiFlexivRizon4RTConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.robot_utils import get_logger
import numpy as np

logger = get_logger("UmiBiFlexivRizon4RTRealEnv")
_WRIST_CAMERAS = ("left_wrist", "right_wrist")
_MOUNT_ALIASES = {
    # The current lerobot-xense presets use station-qualified names. Preserve
    # the short CLI requested by the deployment scripts.
    "forward": "forward-06",
}


class UmiBiFlexivRizon4RTRealEnv:
    """Expose the native BiFlexiv 20D order and connect only wrist cameras.

    This class intentionally does not inherit the older example environment:
    that module imports a legacy emergency-stop helper absent from the current
    ``lerobot-xense`` environment. The robot driver's own ``disconnect()``
    performs the safe RT-thread stop, home motion, and resource cleanup.
    """

    def __init__(
        self,
        bi_mount_type: str = "forward",
        use_force: bool = False,
        go_to_start: bool = True,
        stiffness_ratio: float = 0.2,
        inner_control_hz: int = 1000,
        interpolate_cmds: bool = True,
        log_level: str = "INFO",
        setup_robot: bool = True,
    ) -> None:
        if use_force:
            raise ValueError("UMI inference uses the 20D pose/gripper space; use_force must be False")

        resolved_mount_type = _MOUNT_ALIASES.get(bi_mount_type, bi_mount_type)
        self.config = BiFlexivRizon4RTConfig(
            bi_mount_type=resolved_mount_type,
            use_force=False,
            go_to_start=go_to_start,
            stiffness_ratio=stiffness_ratio,
            inner_control_hz=inner_control_hz,
            interpolate_cmds=interpolate_cmds,
            enable_tactile_sensors=False,
            log_level=log_level,
        )
        # BiFlexivRizon4RTConfig.__post_init__ always injects a head RealSense.
        # Remove it before the robot object constructs or connects its cameras.
        self.config.cameras.pop("head", None)
        if set(self.config.cameras) != set(_WRIST_CAMERAS):
            raise RuntimeError(
                "Expected exactly the two wrist cameras after disabling head/tactile cameras, "
                f"got {tuple(self.config.cameras)}"
            )

        self.robot = make_robot_from_config(self.config)
        if setup_robot:
            self.setup_robot()

    def setup_robot(self) -> None:
        logger.info(f"Connecting BiFlexiv Rizon4 RT with wrist cameras only: {_WRIST_CAMERAS}")
        self.robot.connect(calibrate=False, go_to_start=self.config.go_to_start)
        logger.info("BiFlexiv Rizon4 RT connected; head camera is disabled")

    @staticmethod
    def get_qpos(obs: dict) -> np.ndarray:
        """Build native BiFlexiv order: left TCP, right TCP, left/right gripper."""
        left_tcp = [obs["left_tcp.x"], obs["left_tcp.y"], obs["left_tcp.z"]]
        left_tcp += [obs[f"left_tcp.r{i}"] for i in range(1, 7)]
        right_tcp = [obs["right_tcp.x"], obs["right_tcp.y"], obs["right_tcp.z"]]
        right_tcp += [obs[f"right_tcp.r{i}"] for i in range(1, 7)]
        return np.asarray(
            left_tcp + right_tcp + [obs["left_gripper.pos"], obs["right_gripper.pos"]],
            dtype=np.float32,
        )

    @staticmethod
    def get_images(obs: dict) -> dict:
        missing = set(_WRIST_CAMERAS) - set(obs)
        if missing:
            raise RuntimeError(f"Missing wrist camera observations: {tuple(sorted(missing))}")
        return {camera: obs[camera] for camera in _WRIST_CAMERAS}

    def get_observation(self) -> dict:
        raw_obs = self.robot.get_observation()
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(raw_obs)
        obs["images"] = self.get_images(raw_obs)
        return obs

    def reset(self) -> None:
        logger.info("Resetting BiFlexiv Rizon4 RT to its configured start pose")
        self.robot.reset_to_initial_position()

        # reset_to_initial_position() uses a non-blocking RT trajectory. Wait
        # for it to start and then finish before policy execution begins.
        start = time.monotonic()
        while not self.robot.rt_moving:
            if time.monotonic() - start > 1.0:
                logger.warning("RT reset trajectory did not report a start within 1 second")
                return
            time.sleep(0.001)
        while self.robot.rt_moving:
            if time.monotonic() - start > 15.0:
                raise TimeoutError("BiFlexiv RT reset trajectory exceeded 15 seconds")
            time.sleep(0.05)

    @staticmethod
    def _build_action_dict(action: np.ndarray) -> dict[str, float]:
        values = np.asarray(action)
        if values.shape != (20,):
            raise ValueError(f"Expected a BiFlexiv action with shape (20,), got {values.shape}")

        action_dict = {
            "left_tcp.x": float(values[0]),
            "left_tcp.y": float(values[1]),
            "left_tcp.z": float(values[2]),
            "right_tcp.x": float(values[9]),
            "right_tcp.y": float(values[10]),
            "right_tcp.z": float(values[11]),
            "left_gripper.pos": float(np.clip(values[18], 0.0, 1.0)),
            "right_gripper.pos": float(np.clip(values[19], 0.0, 1.0)),
        }
        for index in range(6):
            action_dict[f"left_tcp.r{index + 1}"] = float(values[3 + index])
            action_dict[f"right_tcp.r{index + 1}"] = float(values[12 + index])
        return action_dict

    def send_action(self, action: np.ndarray) -> None:
        self.robot.send_action(self._build_action_dict(action))

    def disconnect(self) -> None:
        if self.robot.is_connected:
            logger.info("Disconnecting BiFlexiv Rizon4 RT")
            self.robot.disconnect()
