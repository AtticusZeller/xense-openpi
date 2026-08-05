#!/usr/bin/env python
"""Run a UMI-trained OpenPI checkpoint on a BiFlexiv Rizon4 RT robot.

Pipeline overview (everything client-side stays in native BiFlexiv space):
    env (BiFlexiv 20D obs) -> broker (BiFlexiv-space action queue)
        -> BiFlexivUmiPolicyAdapter (layout regroup + per-arm frame transform)
        -> WebsocketClientPolicy -> UMI policy server

This is a reduced variant of examples/bi_flexiv_rizon4_rt/main.py: no
recording, no Pico4 intervention, no decoupled runtime — just the synchronous
Runtime with optional RTC action chunking.

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

    python -m examples.umi_bi_flexiv_rizon4_rt.main \
        --args.host 192.168.142.220 \
        --args.port 8000 \
        --args.bi-mount-type forward \
        --args.inner-control-hz 1000 \
        --args.interpolate-cmds \
        --args.runtime-hz 30 \
        --args.rtc-enabled \
        --args.frame-calibration frame_calibration_identity.json
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
    """Intercepts policy actions and prints them without executing on robot.

    The actions printed here are already converted back to native BiFlexiv
    space by the policy adapter — i.e. exactly what the robot would execute.
    """

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
    """Arguments for UMI inference on BiFlexiv Rizon4 RT."""

    # Policy server
    host: str = "localhost"
    port: int = 8000
    # Task prompt for the policy. None = fall back to the checkpoint's
    # default_prompt (see the training config's LeRobotUmiDataConfig).
    prompt: str | None = None

    # Robot configuration
    bi_mount_type: str = "forward"  # short name; "forward" resolves to the "forward-06" preset
    use_force: bool = False
    go_to_start: bool = True
    stiffness_ratio: float = 0.2
    inner_control_hz: int = 1000
    interpolate_cmds: bool = True
    log_level: str = "INFO"

    # Image rendering
    render_height: int = 224
    render_width: int = 224

    # Runtime settings
    runtime_hz: float = 30.0
    num_episodes: int = 1
    max_episode_steps: int = 1_000_000

    # Dry run mode
    dry_run: bool = False

    # Per-arm UMI-world -> Flexiv-arm frame calibration (JSON with two 4x4
    # matrices). Required for real execution; dry-run may use identity.
    frame_calibration: Path | None = None

    # Non-RTC action chunking
    action_horizon: int = 50

    # RTC config
    rtc_enabled: bool = False
    action_queue_size_to_get_new_actions: int = 30
    execution_horizon: int = 50
    blend_steps: int = 0
    default_delay: int = 4


def main(args: Args) -> None:
    # UMI checkpoints consume the 20D pose/gripper space only — force/wrench
    # readings would change the state layout, so refuse them up front.
    if args.use_force:
        raise SystemExit("UMI checkpoints use a 20D pose/gripper state; --args.use-force is unsupported")
    # BiFlexiv reports each TCP in its own arm frame while UMI data lives in a
    # shared Pico4 world frame. Executing with identity transforms by accident
    # would send poses in the wrong frame, so real execution requires an
    # explicit calibration file (identity included — an explicit choice).
    if not args.dry_run and args.frame_calibration is None:
        raise SystemExit(
            "Real execution requires separate left/right UMI↔Flexiv calibration matrices. "
            "Pass --args.frame-calibration <json>; use --args.dry-run while uncalibrated."
        )
    if args.runtime_hz != 30.0:
        # The checkpoint was trained on 30 Hz data; RTC's delay estimation also
        # reads frequency_hz, so drifting from 30 Hz hurts both the policy
        # distribution and the RTC merge math.
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
    # Conversion boundary: brokers/queues below this adapter stay in native
    # BiFlexiv space; only websocket requests/responses are in UMI space.
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
        # The broker's queue holds BiFlexiv-space actions (the adapter converts
        # server responses back). prev_chunk_left_over takes the opposite trip:
        # broker (BiFlexiv) -> adapter -> server (UMI), where the server
        # re-bases it through the training input pipeline before freezing the
        # prefix. frequency_hz must match the real consumption rate — the
        # synchronous Runtime pops at runtime_hz.
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

    # SIGINT handling: ask the runtime to wind down; the `finally` then runs
    # disconnect(), which homes the arms via the driver's own shutdown — same
    # end-state as bi_flexiv_rizon4_rt. Unlike bi_flexiv_rizon4_rt there is no
    # os._exit escape on a second Ctrl+C: the handler is idempotent, so repeated
    # presses just re-request the stop (forcing exit mid-homing would leave the
    # arms in an unknown pose).
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
