#!/usr/bin/env python
"""Main script for BiFlexiv Rizon4 RT dual-arm robot inference with OpenPI.

Example usage:
    # Basic inference
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --host 192.168.2.100 --port 8000

    # With RTC enabled
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --host 192.168.2.100 --port 8000 --rtc_enabled

    # Side-mount configuration
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --host 192.168.2.100 --port 8000 --bi_mount_type side

    # Dry run (robot connected but actions not sent)
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --host 192.168.2.100 --port 8000 --dry_run

    # Inference + simultaneous recording in LeRobot format
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --host 192.168.2.100 --port 8000 \\
        --record \\
        --record_repo_id Xense/my_new_dataset \\
        --task "pack 6 cosmetic bottles into the carton"

    # Inference with Pico4 human intervention (both grips held → teleop takes over)
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --host 192.168.2.100 --port 8000 --pico4_intervention
"""

from dataclasses import dataclass
import pathlib
import signal
import sys

from lerobot.teleoperators.bi_pico4 import BiPico4
from lerobot.teleoperators.bi_pico4.config_bi_pico4 import BiPico4Config
from lerobot.utils.robot_utils import get_logger
import numpy as np
from PIL import Image, ImageDraw
from typing_extensions import override
import tyro
from xense_client import action_chunk_broker
from xense_client import rtc_action_chunk_broker
from xense_client import websocket_client_policy as _websocket_client_policy
from xense_client.runtime import environment as _environment
from xense_client.runtime import runtime as _runtime
from xense_client.runtime.agents import policy_agent as _policy_agent

import examples.bi_flexiv_rizon4_rt.env as _env
import examples.bi_flexiv_rizon4_rt.intervention as _intervention
import examples.bi_flexiv_rizon4_rt.recorder as _recorder

logger = get_logger("BiFlexivRizon4RTMain")

_TACTILE_CAMERA_KEYS = (
    "left_tactile_top",
    "left_tactile_bottom",
    "right_tactile_top",
    "right_tactile_bottom",
)

# Action dimension labels for dry-run logging
_ACTION_LABELS = [
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
]


def _image_to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected CHW or HWC RGB image, got shape={image.shape}")

    if np.issubdtype(image.dtype, np.integer):
        return np.clip(image, 0, 255).astype(np.uint8)

    image = image.astype(np.float32)
    image_min = float(np.nanmin(image))
    image_max = float(np.nanmax(image))
    if image_min >= -1.05 and image_max <= 1.05:
        image = (np.clip(image, -1.0, 1.0) + 1.0) * 127.5
    elif image_min >= -0.05 and image_max <= 1.05:
        image = np.clip(image, 0.0, 1.0) * 255.0
    else:
        image = np.clip(image, 0.0, 255.0)
    return image.astype(np.uint8)


def _save_tactile_strip(images: dict, output_path: pathlib.Path) -> None:
    tiles = [_image_to_uint8_hwc(np.asarray(images[key])) for key in _TACTILE_CAMERA_KEYS]
    strip = np.concatenate(tiles, axis=1)

    label_height = 24
    canvas = np.zeros((strip.shape[0] + label_height, strip.shape[1], 3), dtype=np.uint8)
    canvas[label_height:] = strip

    pil_image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_image)
    x = 0
    for key, tile in zip(_TACTILE_CAMERA_KEYS, tiles, strict=True):
        draw.text((x + 4, 5), key, fill=(255, 255, 255))
        x += tile.shape[1]
    pil_image.save(output_path)


class DryRunEnvironmentWrapper(_environment.Environment):
    """Intercepts policy actions and prints them without executing on robot."""

    def __init__(self, wrapped_env: _environment.Environment):
        self._wrapped_env = wrapped_env
        self._step_count = 0
        self._episode_count = 0
        self._last_rtc_inference_seq = -1

    @override
    def reset(self) -> None:
        self._episode_count += 1
        self._step_count = 0
        self._last_rtc_inference_seq = -1
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Episode {self._episode_count} - reset (dry run)")
        logger.info(f"{'=' * 80}\n")
        self._wrapped_env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return self._wrapped_env.is_episode_complete()

    @override
    def get_observation(self) -> dict:
        return self._wrapped_env.get_observation()

    @override
    def apply_action(self, action: dict) -> None:
        self._step_count += 1
        actions = action.get("actions")
        if actions is not None:
            logger.info(f"\n{'─' * 80}")
            logger.info(f"Step {self._step_count} - policy action (20D Cartesian):")
            logger.info(f"{'─' * 80}")
            for i, (label, value) in enumerate(zip(_ACTION_LABELS, actions)):
                logger.info(f"  [{i:2d}] {label:18s}: {value:+.6f}")
            logger.info(f"{'─' * 80}")
            rtc = action.get("rtc_metrics")
            if rtc is not None:
                seq = int(rtc.get("inference_seq", 0))
                if seq != self._last_rtc_inference_seq:
                    self._last_rtc_inference_seq = seq
                    next_d = rtc["delay_for_next_infer_steps"]
                    logger.info(
                        f"RTC [dry run] new chunk #{seq}: "
                        f"infer+server RTT={rtc['infer_round_trip_ms']:.1f} ms "
                        f"(model + WebSocket round-trip); "
                        f"delay est={rtc['estimated_delay_steps']} steps, "
                        f"real={rtc['real_delay_steps']} steps, "
                        f"time-based={rtc['inference_delay_steps']} steps; "
                        f"delay_for_next_infer={next_d}; "
                        f"merge={rtc['merge_ms']:.2f} ms; "
                        f"queue_after={rtc['queue_size_after_merge']}; "
                        f"infer RTT p95={rtc['latency_p95_ms']:.1f} ms"
                    )
            logger.info("DRY RUN: action NOT sent to robot")
            logger.info(f"{'─' * 80}\n")

    def disconnect(self) -> None:
        self._wrapped_env.disconnect()


class TactileFlowAuditEnvironmentWrapper(_environment.Environment):
    """Audits tactile images at the exact observation boundary sent to policy."""

    def __init__(
        self,
        wrapped_env: _environment.Environment,
        *,
        audit_steps: int,
        require_tactile_images: bool,
        audit_dir: str | None,
        expected_height: int,
        expected_width: int,
    ) -> None:
        self._wrapped_env = wrapped_env
        self._audit_steps = audit_steps
        self._require_tactile_images = require_tactile_images
        self._audit_dir = pathlib.Path(audit_dir) if audit_dir else None
        self._expected_shape = (3, expected_height, expected_width)
        self._episode_audit_count = 0
        self._total_audit_count = 0

        if self._audit_dir is not None:
            self._audit_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Tactile flow audit images will be saved to {self._audit_dir}")

    @override
    def reset(self) -> None:
        self._wrapped_env.reset()
        self._episode_audit_count = 0

    @override
    def is_episode_complete(self) -> bool:
        return self._wrapped_env.is_episode_complete()

    @override
    def get_observation(self) -> dict:
        obs = self._wrapped_env.get_observation()
        self._audit_observation(obs)
        return obs

    @override
    def apply_action(self, action: dict) -> None:
        self._wrapped_env.apply_action(action)

    def disconnect(self) -> None:
        self._wrapped_env.disconnect()

    def _audit_observation(self, obs: dict) -> None:
        if self._audit_steps <= 0 or self._episode_audit_count >= self._audit_steps:
            return

        self._episode_audit_count += 1
        self._total_audit_count += 1

        images = obs.get("images", {})
        missing = [key for key in _TACTILE_CAMERA_KEYS if key not in images]
        if missing:
            message = (
                "Tactile flow audit failed before WebSocket send: "
                f"missing={missing}, available={tuple(images.keys())}"
            )
            if self._require_tactile_images:
                raise RuntimeError(message)
            logger.warning(message)
            return

        failures = []
        for key in _TACTILE_CAMERA_KEYS:
            image = np.asarray(images[key])
            if image.shape != self._expected_shape:
                failures.append(f"{key}: shape={image.shape}, expected={self._expected_shape}")
                continue

            if not np.isfinite(image).all():
                failures.append(f"{key}: contains NaN or Inf")
                continue

            image_min = float(image.min())
            image_max = float(image.max())
            image_mean = float(image.mean())
            image_std = float(image.std())
            if image_std <= 1e-6:
                failures.append(f"{key}: appears constant, std={image_std:.6g}")

            logger.info(
                "[TACTILE CLIENT] %s shape=%s dtype=%s min=%.3f max=%.3f mean=%.3f std=%.3f",
                key,
                image.shape,
                image.dtype,
                image_min,
                image_max,
                image_mean,
                image_std,
            )

        if failures:
            message = "Tactile flow audit found invalid images: " + "; ".join(failures)
            if self._require_tactile_images:
                raise RuntimeError(message)
            logger.warning(message)

        if self._audit_dir is not None:
            output_path = self._audit_dir / f"tactile_flow_{self._total_audit_count:04d}.png"
            _save_tactile_strip(images, output_path)
            logger.info(f"Saved tactile flow audit image: {output_path}")


@dataclass
class Args:
    """Arguments for BiFlexiv Rizon4 RT inference."""

    # Policy server
    host: str = "localhost"
    port: int = 8000

    # Robot configuration
    bi_mount_type: str = "side"  # "forward" or "side"
    use_force: bool = False
    go_to_start: bool = True
    stiffness_ratio: float = 0.2
    inner_control_hz: int = 1000
    interpolate_cmds: bool = True
    enable_tactile_sensors: bool = True
    log_level: str = "DEBUG"

    # Tactile flow audit at the client boundary, before sending observations to the policy server.
    tactile_flow_audit_steps: int = 3
    require_tactile_flow: bool = False
    tactile_flow_audit_dir: str | None = None

    # Tactile camera mapping (lerobot camera name -> policy-side name).
    # The four values below correspond to BiFlexivTactileInputs.EXPECTED_CAMERAS;
    # only edit them if your robot exposes the tactile cameras under different keys.
    left_tactile_top_cam: str = "left_tactile_top"
    left_tactile_bottom_cam: str = "left_tactile_bottom"
    right_tactile_top_cam: str = "right_tactile_top"
    right_tactile_bottom_cam: str = "right_tactile_bottom"

    # Image rendering
    render_height: int = 224
    render_width: int = 224

    # Runtime settings
    runtime_hz: float = 30.0
    num_episodes: int = 1
    max_episode_steps: int = 1000000

    # Dry run mode
    dry_run: bool = False

    # Non-RTC action chunking
    action_horizon: int = 50

    # RTC config
    rtc_enabled: bool = False
    action_queue_size_to_get_new_actions: int = 30
    execution_horizon: int = 50
    blend_steps: int = 0
    default_delay: int = 4

    # Recording (LeRobot format, raw 640x480 images + absolute actions)
    record: bool = False
    record_repo_id: str = "Xense/recorded_dataset"
    record_root: str | None = None  # local save path, defaults to ~/.cache/huggingface/lerobot
    task: str = "pack 6 cosmetic bottles into the carton"

    # Pico4 human-in-the-loop intervention (hold both grips to take over)
    pico4_intervention: bool = False
    pico4_pos_sensitivity: float = 1.0
    pico4_ori_sensitivity: float = 1.0


def main(args: Args) -> None:
    if args.pico4_intervention and args.rtc_enabled:
        # RTCActionChunkBroker owns an execution queue + blending; its reset
        # semantics differ from ActionChunkBroker. A correct RTC handoff needs
        # a separate design pass — refuse the combo rather than silently doing
        # the wrong thing.
        raise SystemExit(
            "--pico4_intervention is not supported with --rtc_enabled in this release. "
            "Run without --rtc_enabled, or disable intervention."
        )

    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logger.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    base_environment = _env.BiFlexivRizon4RTEnvironment(
        bi_mount_type=args.bi_mount_type,
        use_force=args.use_force,
        go_to_start=args.go_to_start,
        stiffness_ratio=args.stiffness_ratio,
        inner_control_hz=args.inner_control_hz,
        interpolate_cmds=args.interpolate_cmds,
        enable_tactile_sensors=args.enable_tactile_sensors,
        log_level=args.log_level,
        render_height=args.render_height,
        render_width=args.render_width,
        setup_robot=True,
        tactile_camera_mapping={
            "left_tactile_top": args.left_tactile_top_cam,
            "left_tactile_bottom": args.left_tactile_bottom_cam,
            "right_tactile_top": args.right_tactile_top_cam,
            "right_tactile_bottom": args.right_tactile_bottom_cam,
        },
    )

    if args.dry_run:
        logger.info("DRY RUN mode: actions will be printed, not executed")
        environment = DryRunEnvironmentWrapper(base_environment)
    else:
        environment = base_environment

    if args.tactile_flow_audit_steps > 0:
        environment = TactileFlowAuditEnvironmentWrapper(
            environment,
            audit_steps=args.tactile_flow_audit_steps,
            require_tactile_images=args.require_tactile_flow,
            audit_dir=args.tactile_flow_audit_dir,
            expected_height=args.render_height,
            expected_width=args.render_width,
        )

    subscribers = []
    if args.record:
        if args.dry_run:
            logger.warning(
                "Recording is enabled in dry-run mode — state/action data will be from policy output only (no real robot motion)"
            )
        recorder = _recorder.make_recorder_subscriber(
            repo_id=args.record_repo_id,
            task=args.task,
            fps=int(args.runtime_hz),
            root=args.record_root,
        )
        subscribers.append(recorder)
        logger.info(f"Recording enabled: repo_id={args.record_repo_id}, task='{args.task}'")

    if args.rtc_enabled:
        policy = rtc_action_chunk_broker.RTCActionChunkBroker(
            policy=ws_client_policy,
            frequency_hz=args.runtime_hz,
            action_queue_size_to_get_new_actions=args.action_queue_size_to_get_new_actions,
            rtc_enabled=args.rtc_enabled,
            execution_horizon=args.execution_horizon,
            blend_steps=args.blend_steps,
            default_delay=args.default_delay,
            dry_run=args.dry_run,
            # Training used DeltaActions(mask=[True]*18 + [False]*2): the first
            # 18 action dims (bi-arm TCP XYZ + 6D rot) are deltas relative to
            # obs.state; the last 2 (grippers) are absolute. Telling the broker
            # this lets it re-base prev_chunk_left_over from the previous
            # inference's state into the current obs.state before sending, so
            # the model sees a prefix consistent with its training distribution
            # and the postfix joins smoothly at merge.
            delta_state_dim=18,
        )
    else:
        policy = action_chunk_broker.ActionChunkBroker(
            policy=ws_client_policy,
            action_horizon=args.action_horizon,
        )

    intervention_controller: _intervention.Pico4InterventionController | None = None
    if args.pico4_intervention:
        teleop = BiPico4(
            BiPico4Config(
                pos_sensitivity=args.pico4_pos_sensitivity,
                ori_sensitivity=args.pico4_ori_sensitivity,
            )
        )
        intervention_controller = _intervention.Pico4InterventionController(teleop, base_environment)
        intervention_controller.start()
        environment = _intervention.InterventionEnvironmentWrapper(environment, intervention_controller)
        agent = _intervention.InterventionPolicyAgent(
            inner_agent=_policy_agent.PolicyAgent(policy=policy),
            controller=intervention_controller,
            broker=policy,
        )
    else:
        agent = _policy_agent.PolicyAgent(policy=policy)

    runtime = _runtime.Runtime(
        environment=environment,
        agent=agent,
        subscribers=subscribers,
        max_hz=args.runtime_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    def safe_disconnect() -> None:
        try:
            if intervention_controller is not None:
                intervention_controller.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting Pico4: {e}")
        try:
            actual_env = environment
            if isinstance(actual_env, _intervention.InterventionEnvironmentWrapper):
                actual_env = actual_env._wrapped_env
            if isinstance(actual_env, DryRunEnvironmentWrapper):
                actual_env = actual_env._wrapped_env
            actual_env.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting: {e}")

    def signal_handler(sig, frame):
        logger.info("Ctrl+C detected, disconnecting...")
        safe_disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        runtime.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt")
    except Exception as e:
        logger.error(f"Runtime error: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        safe_disconnect()


if __name__ == "__main__":
    tyro.cli(main)
