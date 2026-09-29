"""One operator round over the robot session protocol, and the recording window as the server sees it.

For every executed segment the robot reports whether the recording window was
open while it ran, plus any label or discard the operator pressed (see
``protocol``). ``RoundSession`` turns that stream into window events that every
consumer interprets the same way:

- ``opened``: the window was closed and this segment ran inside it;
- ``closed_unlabeled``: the window was open and this segment ran outside it;
- ``label``: success or failure closing the open window, reported by the segment
  that completed the labeled unit (a label with no open window is ignored);
- ``discard``: the operator dropped the round's labeled data, which also closes
  the window.

``recording`` is the window state the robot latched for the next chunk; a policy
that may only run inside windows decides from it. What a window becomes - replay
rows in RLT training, a trial in evaluation - is up to the consumer.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from openpi.rl.env import protocol as _protocol


@dataclasses.dataclass(frozen=True)
class Segment:
    obs: dict[str, Any]  # observation after the segment
    executed: np.ndarray  # (n, A) absolute actions that ran, policy or human
    human: np.ndarray  # (n,) bool, True where a human drove
    recording: bool  # the window was open while the segment ran
    opened: bool
    closed_unlabeled: bool
    label: str | None
    discard: bool
    round_end: bool


@dataclasses.dataclass(frozen=True)
class ChunkResult:
    segments: list[Segment]
    captures: dict[int, dict[str, Any]]  # window step -> observation taken there


class RoundSession:
    """Resets the robot for a round and executes chunks until the operator ends it."""

    def __init__(
        self,
        env: _protocol.RemoteEnv,
        *,
        action_dim: int,
        takeover_position_m: float,
        takeover_rotation_deg: float,
        capture_stride: int,
    ):
        self._env = env
        self._action_dim = action_dim
        self._reset_request = {
            "op": "reset",
            "takeover_position_m": takeover_position_m,
            "takeover_rotation_deg": takeover_rotation_deg,
            "capture_stride": capture_stride,
        }
        self.recording = False
        self.window_open = False
        self.round_over = True

    def reset(self) -> dict[str, Any]:
        """Home the robot, wait for the operator to start the round, and return the first observation."""
        reply = self._env.request(self._reset_request)
        self.recording = bool(reply["recording"])
        self.window_open = False
        self.round_over = False
        return reply["obs"]

    def execute(self, actions: np.ndarray, *, source: str) -> ChunkResult:
        reply = self._env.request({"op": "chunk", "actions": actions, "source": source})
        segments = [self._segment(raw) for raw in reply["segments"]]
        self.recording = bool(reply["recording_next"])
        return ChunkResult(segments, {int(c["step"]): c["obs"] for c in reply["captures"]})

    def _segment(self, raw: dict[str, Any]) -> Segment:
        recording = bool(raw["recording"])
        opened = recording and not self.window_open
        closed_unlabeled = self.window_open and not recording
        self.window_open = recording
        discard = bool(raw["discard"])
        label = None
        if discard:
            self.window_open = False
        elif raw["label"] is not None and self.window_open:
            label = raw["label"]
            self.window_open = False
        self.round_over = self.round_over or bool(raw["round_end"])
        human = np.asarray(raw["human"], bool)
        return Segment(
            obs=raw["obs"],
            executed=np.asarray(raw["executed"], np.float32).reshape(len(human), self._action_dim),
            human=human,
            recording=recording,
            opened=opened,
            closed_unlabeled=closed_unlabeled,
            label=label,
            discard=discard,
            round_end=bool(raw["round_end"]),
        )
