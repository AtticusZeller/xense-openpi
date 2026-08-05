#!/usr/bin/env python
"""Run a UMI-trained OpenPI checkpoint on a BiFlexiv Rizon4 RT robot.

Example::

    python -m examples.umi_bi_flexiv_rizon4_rt.main \
        --args.host 192.168.142.220 \
        --args.port 8000 \
        --args.bi-mount-type forward \
        --args.inner-control-hz 1000 \
        --args.interpolate-cmds \
        --args.runtime-hz 30 \
        --args.rtc-enabled \
        --args.dry-run
"""

from dataclasses import dataclass
from pathlib import Path
import signal
import sys

from lerobot.utils.robot_utils import get_logger
import numpy as np
from typing_extensions import override
import tyro
from xense_client import action_chunk_broker
from xense_client import rtc_action_chunk_broker
from xense_client import websocket_client_policy as _websocket_client_policy
from xense_client.runtime import environment as _environment
from xense_client.runtime import runtime as _runtime
from xense_client.runtime.agents import policy_agent as _policy_agent

from examples.umi_bi_flexiv_rizon4_rt.env import BI_FLEXIV_ACTION_NAMES
from examples.umi_bi_flexiv_rizon4_rt.env import UmiBiFlexivRizon4RTEnvironment
from examples.umi_bi_flexiv_rizon4_rt.frame_transform import PerArmUmiBiFlexivTransform
from examples.umi_bi_flexiv_rizon4_rt.policy_adapter import BiFlexivUmiPolicyAdapter

logger = get_logger("UmiBiFlexivRizon4RTMain")


class DryRunEnvironmentWrapper(_environment.Environment):
    """Print BiFlexiv-space actions without sending them to the robot."""

    def __init__(self, wrapped: UmiBiFlexivRizon4RTEnvironment) -> None:
        self._wrapped_env = wrapped
        self._step = 0

    @override
    def reset(self) -> None:
        self._step = 0
        self._wrapped_env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return self._wrapped_env.is_episode_complete()

    @override
    def get_observation(self) -> dict:
        return self._wrapped_env.get_observation()

    @override
    def apply_action(self, action: dict) -> None:
        self._step += 1
        values = np.asarray(action["actions"])
        logger.info(f"DRY RUN step {self._step}: action not executed")
        for index, (name, value) in enumerate(zip(BI_FLEXIV_ACTION_NAMES, values, strict=True)):
            logger.info(f"  [{index:02d}] {name:<19s} {value:+0.6f}")
        rtc_metrics = action.get("rtc_metrics")
        if rtc_metrics is not None:
            logger.info(f"RTC metrics: {rtc_metrics}")

    def disconnect(self) -> None:
        self._wrapped_env.disconnect()


@dataclass
class Args:
    host: str = "localhost"
    port: int = 8000
    prompt: str | None = None

    bi_mount_type: str = "forward"
    use_force: bool = False
    go_to_start: bool = True
    stiffness_ratio: float = 0.2
    inner_control_hz: int = 1000
    interpolate_cmds: bool = True
    log_level: str = "INFO"

    render_height: int = 224
    render_width: int = 224
    runtime_hz: float = 30.0
    num_episodes: int = 1
    max_episode_steps: int = 1_000_000

    dry_run: bool = False
    frame_calibration: Path | None = None
    action_horizon: int = 50

    rtc_enabled: bool = False
    action_queue_size_to_get_new_actions: int = 30
    execution_horizon: int = 50
    blend_steps: int = 0
    default_delay: int = 4


def main(args: Args) -> None:
    if args.use_force:
        raise SystemExit("UMI checkpoints use a 20D pose/gripper state; --args.use-force is unsupported")
    if not args.dry_run and args.frame_calibration is None:
        raise SystemExit(
            "Real execution requires separate left/right UMI↔Flexiv calibration matrices. "
            "Pass --args.frame-calibration <json>; use --args.dry-run while uncalibrated."
        )
    if args.runtime_hz != 30.0:
        logger.warning(f"UMI data was recorded at 30 Hz; requested runtime_hz={args.runtime_hz}")

    websocket_policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logger.info(f"Server metadata: {websocket_policy.get_server_metadata()}")
    transform = (
        PerArmUmiBiFlexivTransform.from_json(args.frame_calibration)
        if args.frame_calibration is not None
        else PerArmUmiBiFlexivTransform.identity()
    )
    if args.frame_calibration is not None:
        logger.info(f"Loaded per-arm frame calibration from {args.frame_calibration}")
    else:
        logger.warning("No frame calibration supplied; dry-run uses identity transforms for both arms")
    converted_policy = BiFlexivUmiPolicyAdapter(websocket_policy, transform=transform, prompt=args.prompt)

    base_environment = UmiBiFlexivRizon4RTEnvironment(
        bi_mount_type=args.bi_mount_type,
        use_force=args.use_force,
        go_to_start=args.go_to_start,
        stiffness_ratio=args.stiffness_ratio,
        inner_control_hz=args.inner_control_hz,
        interpolate_cmds=args.interpolate_cmds,
        log_level=args.log_level,
        render_height=args.render_height,
        render_width=args.render_width,
        setup_robot=True,
    )
    environment: _environment.Environment
    if args.dry_run:
        logger.info("DRY RUN enabled: the robot is connected/read, but policy actions are not sent")
        environment = DryRunEnvironmentWrapper(base_environment)
    else:
        environment = base_environment

    if args.rtc_enabled:
        broker = rtc_action_chunk_broker.RTCActionChunkBroker(
            policy=converted_policy,
            frequency_hz=args.runtime_hz,
            action_queue_size_to_get_new_actions=args.action_queue_size_to_get_new_actions,
            rtc_enabled=True,
            execution_horizon=args.execution_horizon,
            blend_steps=args.blend_steps,
            default_delay=args.default_delay,
            dry_run=args.dry_run,
        )
    else:
        broker = action_chunk_broker.ActionChunkBroker(
            policy=converted_policy,
            action_horizon=args.action_horizon,
        )

    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(policy=broker),
        subscribers=[],
        max_hz=args.runtime_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    def disconnect() -> None:
        try:
            environment.disconnect()
        except Exception as error:
            logger.warning(f"Error while disconnecting: {error}")

    def signal_handler(sig, frame) -> None:
        del sig, frame
        logger.info("Ctrl+C received; stopping runtime")
        if hasattr(runtime, "request_stop"):
            runtime.request_stop()
        else:
            disconnect()
            sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    try:
        runtime.run()
    finally:
        disconnect()


if __name__ == "__main__":
    tyro.cli(main)
