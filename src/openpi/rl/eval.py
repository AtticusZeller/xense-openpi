"""Labeled evaluation of a frozen policy on the robot, over the same session as online training.

One run evaluates one arm: a frozen policy that returns a chunk for an observation,
given whether the recording window is open. The operator drives rounds exactly as in
online collection (``env.session``): a labeled window is one trial; a window closed
without a label is dropped; a discard marks the round's trials so far, and the open
one, as discarded; a lost connection drops the round. Nothing trains and nothing
enters a replay.

A trial is an autonomous success when it is labeled success and no human step ran
inside its window. The success rate is autonomous successes over the labeled trials
that were not discarded. Steps are exact; ``wall_s`` runs from the request of the
chunk the window opened in to the reply carrying the label, so it also covers the
part of that first chunk before the window opened.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import time
from typing import Any, Protocol

import numpy as np

from openpi.rl import run_logger as _run_logger
from openpi.rl.env import protocol as _protocol
from openpi.rl.env import session as _session

AXES = ("round", "trial")


class Arm(Protocol):
    name: str

    def act(self, obs: dict[str, Any], *, window_open: bool) -> tuple[np.ndarray, str]:
        """A chunk of absolute robot actions and its source; the robot runs "actor" only inside open windows."""
        ...


@dataclasses.dataclass
class Trial:
    arm: str
    round: int
    label: str | None = None
    steps: int = 0
    human_steps: int = 0
    first_human_step: int | None = None  # window step of the first human step
    chunks: int = 0
    wall_s: float = 0.0
    discarded: bool = False

    @property
    def autonomous_success(self) -> bool:
        return self.label == "success" and self.human_steps == 0 and not self.discarded


def summarize(trials: list[Trial]) -> dict[str, float]:
    counted = [t for t in trials if not t.discarded]
    autonomous = sum(t.autonomous_success for t in counted)
    successes = sum(t.label == "success" for t in counted)
    return {
        "trials": len(counted),
        "successes": successes,
        "failures": len(counted) - successes,
        "autonomous_successes": autonomous,
        "intervened": sum(t.human_steps > 0 for t in counted),
        "discarded": len(trials) - len(counted),
        "autonomous_success_rate": autonomous / len(counted) if counted else float("nan"),
    }


class Evaluation:
    """Runs rounds until ``trials`` labeled trials count, appending each round's trials to ``trials.jsonl``."""

    def __init__(
        self,
        env: _protocol.RemoteEnv,
        arm: Arm,
        *,
        action_dim: int,
        takeover_position_m: float,
        takeover_rotation_deg: float,
        capture_stride: int,
        trials: int,
        out_dir: pathlib.Path,
        logger: _run_logger.RunLogger | None = None,
    ):
        self.env = env
        self.arm = arm
        self.target = trials
        self.out_dir = out_dir
        self.logger = logger or _run_logger.RunLogger(None, axes=AXES)
        self._session = {
            "action_dim": action_dim,
            "takeover_position_m": takeover_position_m,
            "takeover_rotation_deg": takeover_rotation_deg,
            "capture_stride": capture_stride,
        }
        self.trials: list[Trial] = []
        self.rounds = 0
        out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def counted(self) -> int:
        return sum(not t.discarded for t in self.trials)

    def run(self) -> dict[str, float]:
        while self.counted < self.target:
            try:
                self.run_round()
            except _protocol.EnvConnectionLostError as exc:
                logging.warning("Robot connection lost (%s); this round's trials are dropped.", exc)
        return summarize(self.trials)

    def run_round(self) -> list[Trial]:
        """Evaluate one round; its trials are recorded only once it ends."""
        session = _session.RoundSession(self.env, **self._session)
        obs = session.reset()
        number = self.rounds + 1
        kept: list[Trial] = []
        trial: Trial | None = None
        opened_at = 0.0
        while not session.round_over:
            actions, source = self.arm.act(obs, window_open=session.recording)
            requested = time.monotonic()
            result = session.execute(actions, source=source)
            if trial is not None:
                trial.chunks += 1
            for segment in result.segments:
                if segment.opened:
                    trial = Trial(arm=self.arm.name, round=number, chunks=1)
                    opened_at = requested
                elif segment.closed_unlabeled:
                    logging.info("Window closed without a label; no trial.")
                    trial = None
                if trial is not None and segment.recording:
                    if trial.first_human_step is None and segment.human.any():
                        trial.first_human_step = trial.steps + int(np.argmax(segment.human))
                    trial.steps += len(segment.human)
                    trial.human_steps += int(segment.human.sum())
                if segment.discard:
                    if trial is not None:
                        kept.append(trial)
                    for discarded in kept:
                        discarded.discarded = True
                    trial = None
                    self.env.status(f"Discard: this round's {len(kept)} trials will not count.")
                elif segment.label is not None:
                    assert trial is not None, "the session only reports a label that closes an open window"
                    trial.label = segment.label
                    trial.wall_s = time.monotonic() - opened_at
                    kept.append(trial)
                    trial = None
                    counted = self.counted + sum(not t.discarded for t in kept)
                    self.env.status(f"{segment.label.capitalize()} labeled: {counted}/{self.target} trials.")
            obs = result.segments[-1].obs
        self.rounds = number
        self._record(kept)
        return kept

    def _record(self, kept: list[Trial]) -> None:
        with (self.out_dir / "trials.jsonl").open("a") as f:
            for trial in kept:
                self.trials.append(trial)
                f.write(json.dumps(dataclasses.asdict(trial)) + "\n")
                self.logger.log(
                    "trial",
                    len(self.trials),
                    {
                        "round": trial.round,
                        "success": trial.label == "success",
                        "autonomous_success": trial.autonomous_success,
                        "steps": trial.steps,
                        "human_steps": trial.human_steps,
                        "first_human_step": trial.first_human_step,
                        "chunks": trial.chunks,
                        "wall_s": trial.wall_s,
                        "discarded": trial.discarded,
                    },
                )
        summary = summarize(self.trials)
        self.logger.log("round", self.rounds, {**summary, "round_trials": len(kept)})
        self.env.status(
            f"Round {self.rounds}: {len(kept)} trials; {summary['trials']}/{self.target} counted, "
            f"{summary['autonomous_successes']} autonomous successes."
        )
