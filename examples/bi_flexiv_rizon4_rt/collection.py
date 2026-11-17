"""RECAP rollout collection on the bench: button-driven episodes with step-level takeover.

Entered from ``main.py`` with ``--args.record-round <k>``. The policy runs in the
synchronous runtime (no RTC) and the operator drives each episode from the Pico4:

- ``A``: once the arms have homed, start the episode; during an episode, end it unsaved.
- ``B``: end the episode as a success.
- ``Y``: end the episode as a failure.
- ``X``: end the episode unsaved.
- Grip held, then moved: take over (the grip alone only arms the takeover).

The step on which ``B`` or ``Y`` is read is the episode's last recorded frame; the arms
home afterwards, unrecorded. The return of a frame counts the steps left until the
episode ends, so recording the homing motion would shift it. An episode that ends
without a label (``X``, ``A``, the step limit, Ctrl+C) is not saved.
"""

from __future__ import annotations

from collections.abc import Callable
import math
import os
import threading
import time
from typing import Any, override

from lerobot.utils.robot_utils import get_logger
import numpy as np
from xense_client.runtime import environment as _environment

import examples.bi_flexiv_rizon4_rt.intervention as _intervention

logger = get_logger("RecapCollection")

# Controller motion that turns a held grip into a takeover (the values RLT collection
# used). Taking over on the grip alone would record the frames before the operator
# moves as human actions, which the advantage labels later force to positive.
TAKEOVER_MOTION = _intervention.TakeoverMotion(position_m=0.005, rotation_rad=math.radians(3.0))

_GATE_POLL_S = 0.05
# Face buttons that label the episode; the other two end it unsaved.
_LABELS = {"B": True, "Y": False}


class CollectionController(_intervention.Pico4InterventionController):
    """Takeover controller whose face buttons drive the episode instead of reaching the policy."""

    @override
    def consume_button_events(self) -> list[str]:
        # InterventionPolicyAgent forwards whatever this returns to the policy server as
        # ``__button_events__``. During collection the presses belong to
        # EpisodeControlEnvironment, which reads them through take_button_events.
        return []

    def take_button_events(self) -> list[str]:
        return super().consume_button_events()


class EpisodeControlEnvironment(_intervention.InterventionEnvironmentWrapper):
    """Starts each episode on ``A`` and ends it on a face button, reporting the label.

    ``on_label`` receives True (``B``, success) or False (``Y``, failure) on the step that
    ends the episode. An episode that ends any other way reports nothing.
    """

    def __init__(
        self,
        wrapped_env: _environment.Environment,
        controller: CollectionController,
        on_label: Callable[[bool], None],
    ) -> None:
        super().__init__(wrapped_env, controller)
        self._collection_controller = controller
        self._on_label = on_label
        self._awaiting_start = False
        self._ended = False
        self._stop_requested = False
        self._last_gripper: np.ndarray | None = None

    def request_stop(self) -> None:
        """Give up waiting for ``A`` (Ctrl+C). A running episode is ended by the runtime itself."""
        self._stop_requested = True

    @override
    def reset(self) -> None:
        super().reset()
        # The wait for A happens in the next get_observation, not here: the runtime also
        # resets once after the last episode, and that homing must not wait for a press.
        self._awaiting_start = True
        self._ended = False
        self._last_gripper = None

    @override
    def is_episode_complete(self) -> bool:
        return self._ended or super().is_episode_complete()

    @override
    def get_observation(self) -> dict:
        if self._awaiting_start:
            self._wait_for_start()
            self._awaiting_start = False
        # The last commanded grippers, so a hand that is not driving keeps its commanded
        # opening rather than the measured one (a gripper holding an object measures the
        # object's width, and commanding that would loosen the grasp).
        self._collection_controller.poll_and_decide(gripper_command=self._last_gripper)
        for button in self._collection_controller.take_button_events():
            self._end_episode(button)
        return self._wrapped_env.get_observation()

    @override
    def apply_action(self, action: dict) -> None:
        super().apply_action(action)
        self._last_gripper = np.asarray(action["actions"][-2:], dtype=np.float32).copy()

    def _wait_for_start(self) -> None:
        # Presses made while the arms homed are stale: a stray A must not start the
        # episode before the operator has reset the scene.
        stale = self._collection_controller.take_button_events()
        if stale:
            logger.info(f"Ignoring Pico4 presses made before the arms homed: {', '.join(stale)}")
        logger.info("Arms homed. Reset the scene, then press Pico4 A to start the episode.")
        while True:
            if self._stop_requested:
                raise KeyboardInterrupt
            self._collection_controller.poll_buttons()
            if "A" in self._collection_controller.take_button_events():
                return
            time.sleep(_GATE_POLL_S)

    def _end_episode(self, button: str) -> None:
        if self._ended:
            return
        self._ended = True
        if button in _LABELS:
            success = _LABELS[button]
            self._on_label(success)
            logger.info(f"Pico4 {button}: episode ends as a {'success' if success else 'failure'}.")
        else:
            logger.info(f"Pico4 {button}: episode ends unsaved.")


def stop_on_sigint(
    runtime: Any,
    episode_control: EpisodeControlEnvironment | None = None,
    recorder: Any = None,
    force_exit: Callable[[int], Any] = os._exit,
) -> Callable[[int, Any], None]:
    """Returns the SIGINT handler of a run.

    The first Ctrl+C asks the runtime to wind down (DecoupledRuntime joins its action and
    observation threads, ~0.5 s), so that main's ``finally`` closes the dataset and homes the
    arms via MoveJ before releasing the SDK. A second Ctrl+C while that is in progress exits
    at once, accepting that the arms may not return home.

    Args:
        runtime: The runtime to stop.
        episode_control: Collection mode's episode control, which waits for Pico4 A outside
            the runtime's reach.
        recorder: Collection mode's recorder. The forced exit closes it first: LeRobot writes
            the parquet footers only on ``finalize``, and without them no episode saved in
            this session can be read.
        force_exit: Ends the process without running ``finally`` blocks.
    """
    shutdown_in_progress = threading.Event()

    def handler(sig: int, frame: Any) -> None:
        if shutdown_in_progress.is_set():
            logger.warn("Second Ctrl+C — forcing exit. Arms may not return home cleanly.")
            if recorder is not None:
                try:
                    recorder.finalize()
                except Exception as e:
                    logger.warn(f"Could not close the dataset before the forced exit: {e}")
            force_exit(1)
        shutdown_in_progress.set()
        logger.info("Ctrl+C — stopping runtime gracefully (press Ctrl+C again to force exit)")
        runtime.request_stop()
        if episode_control is not None:
            # The runtime cannot interrupt the wait for Pico4 A; the episode control raises out of it.
            episode_control.request_stop()

    return handler
