import json
import math

import numpy as np

from openpi.rl import eval as _eval
from openpi.rl.env import protocol as _protocol
from openpi.rl.env import session_test
from openpi.rl.env.session_test import _chunk
from openpi.rl.env.session_test import _segment

_RESET = {"obs": {"n": 0}, "recording": False}


class Robot(session_test.ScriptedEnv):
    def status(self, text):
        pass


class FakeArm:
    name = "fake"

    def __init__(self):
        self.window_open = []

    def act(self, obs, *, window_open):
        self.window_open.append(window_open)
        return np.zeros((2, session_test.A), np.float32), "actor" if window_open else "vla"


def _evaluate(script, tmp_path, trials):
    robot, arm = Robot(script), FakeArm()
    evaluation = _eval.Evaluation(
        robot,
        arm,
        action_dim=session_test.A,
        takeover_position_m=0.005,
        takeover_rotation_deg=3.0,
        capture_stride=2,
        trials=trials,
        out_dir=tmp_path,
    )
    summary = evaluation.run()
    assert not robot.replies, "the evaluation stopped before the script ran out"
    records = [json.loads(line) for line in (tmp_path / "trials.jsonl").read_text().splitlines()]
    return summary, records, arm, robot


def test_labeled_windows_become_trials_and_the_run_stops_at_the_target(tmp_path):
    script = [
        # Round 1: a window requested mid-chunk opens with the next chunk and is labeled success untouched.
        _RESET,
        _chunk(_segment(), recording_next=True),
        _chunk(_segment(rec=True), recording_next=True),
        _chunk(_segment(rec=True, label="success")),
        _chunk(_segment(end=True)),
        # Round 2: a takeover inside the window (one continuation segment), labeled failure as the round ends.
        _RESET,
        _chunk(_segment(), recording_next=True),
        _chunk(_segment(rec=True, human=(1,)), _segment(3, rec=True, human=(0, 1, 2)), recording_next=True),
        _chunk(_segment(rec=True, label="failure", end=True)),
    ]
    summary, records, arm, robot = _evaluate(script, tmp_path, trials=2)
    assert [(r["round"], r["label"], r["steps"], r["human_steps"], r["chunks"]) for r in records] == [
        (1, "success", 4, 0, 2),
        (2, "failure", 7, 4, 2),
    ]
    assert records[0]["first_human_step"] is None
    assert records[1]["first_human_step"] == 1
    assert all(r["arm"] == "fake" and not r["discarded"] and r["wall_s"] >= 0 for r in records)
    assert summary == {
        "trials": 2,
        "successes": 1,
        "failures": 1,
        "autonomous_successes": 1,
        "intervened": 1,
        "discarded": 0,
        "autonomous_success_rate": 0.5,
    }
    # The arm is told the window the robot latched for each chunk, so it only drives open windows.
    assert arm.window_open == [False, True, True, False, False, True, True]
    assert [r["source"] for r in robot.requests if r["op"] == "chunk"] == [
        "vla",
        "actor",
        "actor",
        "vla",
        "vla",
        "actor",
        "actor",
    ]


def test_a_discard_marks_the_round_trials_and_they_do_not_count(tmp_path):
    script = [
        # Round 1: one labeled trial, then a second window discarded one step in: both are marked.
        _RESET,
        _chunk(_segment(), recording_next=True),
        _chunk(_segment(rec=True, label="success")),
        _chunk(_segment(), recording_next=True),
        _chunk(_segment(1, rec=True, discard=True)),
        _chunk(_segment(end=True)),
        # Round 2: one labeled trial, the only one that counts.
        _RESET,
        _chunk(_segment(), recording_next=True),
        _chunk(_segment(rec=True, label="failure", end=True)),
    ]
    summary, records, _, _ = _evaluate(script, tmp_path, trials=1)
    assert [(r["round"], r["label"], r["discarded"]) for r in records] == [
        (1, "success", True),
        (1, None, True),
        (2, "failure", False),
    ]
    assert summary["trials"] == 1
    assert summary["discarded"] == 2
    assert summary["autonomous_successes"] == 0
    assert summary["autonomous_success_rate"] == 0.0


def test_a_lost_connection_drops_the_round(tmp_path):
    script = [
        _RESET,
        _chunk(_segment(), recording_next=True),
        _chunk(_segment(rec=True, label="success")),
        _protocol.EnvConnectionLostError("robot host gone"),
        _RESET,
        _chunk(_segment(), recording_next=True),
        _chunk(_segment(rec=True, label="success", end=True)),
    ]
    summary, records, _, _ = _evaluate(script, tmp_path, trials=1)
    # The dropped round recorded nothing and did not take a round number.
    assert [(r["round"], r["label"]) for r in records] == [(1, "success")]
    assert summary["autonomous_success_rate"] == 1.0


def test_a_window_closed_without_a_label_is_no_trial(tmp_path):
    script = [
        _RESET,
        _chunk(_segment(), recording_next=True),
        _chunk(_segment(rec=True), _segment(), recording_next=True),
        _chunk(_segment(rec=True, label="success", end=True)),
    ]
    summary, records, _, _ = _evaluate(script, tmp_path, trials=1)
    assert [(r["label"], r["steps"]) for r in records] == [("success", 2)]
    assert summary["trials"] == 1


def test_summary_of_no_counted_trials_has_no_rate():
    discarded = _eval.Trial(arm="a", round=1, label="success", discarded=True)
    summary = _eval.summarize([discarded])
    assert summary["trials"] == 0
    assert summary["discarded"] == 1
    assert math.isnan(summary["autonomous_success_rate"])
