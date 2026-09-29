import numpy as np
import pytest

from openpi.rl.env import protocol as _protocol
from openpi.rl.env import session as _session

A = 3


class ScriptedEnv:
    """Replies to requests from a fixed script; an exception in the script is raised instead."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def request(self, message):
        self.requests.append(message)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _segment(n=2, *, rec=False, human=(), label=None, discard=False, end=False):
    return {
        "obs": {"n": n},
        "executed": np.zeros((n, A), np.float32),
        "human": np.isin(np.arange(n), human),
        "recording": rec,
        "label": label,
        "round_end": end,
        "discard": discard,
    }


def _chunk(*segments, captures=(), recording_next=False):
    return {
        "segments": list(segments),
        "captures": [{"step": step, "obs": {"step": step}} for step in captures],
        "recording_next": recording_next,
    }


def _started(*replies):
    env = ScriptedEnv([{"obs": {"n": 0}, "recording": False}, *replies])
    session = _session.RoundSession(
        env, action_dim=A, takeover_position_m=0.005, takeover_rotation_deg=3.0, capture_stride=2
    )
    assert session.reset() == {"n": 0}
    return session, env


def _run(session):
    return session.execute(np.zeros((2, A), np.float32), source="vla")


def test_reset_forwards_the_takeover_thresholds_and_capture_stride():
    session, env = _started()
    assert env.requests[0] == {
        "op": "reset",
        "takeover_position_m": 0.005,
        "takeover_rotation_deg": 3.0,
        "capture_stride": 2,
    }
    assert not session.round_over
    assert not session.recording


def test_window_requested_mid_chunk_opens_at_the_next_segment():
    session, env = _started(_chunk(_segment(), recording_next=True), _chunk(_segment(rec=True), captures=[2]))
    first = _run(session)
    assert not first.segments[0].opened
    assert session.recording  # latched for the next chunk, which the caller may route to a window-only policy
    assert not session.window_open
    second = _run(session)
    assert second.segments[0].opened
    assert session.window_open
    assert second.captures == {2: {"step": 2}}
    assert env.requests[-1]["source"] == "vla"


def test_takeover_continuation_segments_stay_in_one_window():
    session, _ = _started(
        _chunk(_segment(rec=True, human=(1,)), _segment(4, rec=True, human=(0, 1, 2, 3)), _segment(1, rec=True))
    )
    segments = _run(session).segments
    assert [s.opened for s in segments] == [True, False, False]
    assert [s.closed_unlabeled for s in segments] == [False, False, False]
    assert [s.executed.shape for s in segments] == [(2, A), (4, A), (1, A)]
    assert [int(s.human.sum()) for s in segments] == [1, 4, 0]


def test_a_pending_label_is_reported_before_the_round_ends():
    session, _ = _started(
        _chunk(_segment(rec=True), recording_next=True), _chunk(_segment(rec=True, label="failure", end=True))
    )
    assert _run(session).segments[0].label is None
    assert not session.round_over
    segment = _run(session).segments[0]
    assert segment.label == "failure"
    assert segment.round_end
    assert session.round_over
    assert not session.window_open


def test_discard_closes_the_window_and_a_new_one_can_open():
    session, _ = _started(
        _chunk(_segment(rec=True), _segment(0, rec=True, discard=True)),
        _chunk(_segment(rec=True, label="success")),
    )
    segments = _run(session).segments
    assert segments[0].opened
    assert segments[1].discard
    assert segments[1].executed.shape == (0, A)
    assert not session.window_open
    segment = _run(session).segments[0]
    assert segment.opened
    assert segment.label == "success"


def test_a_window_closing_without_a_label_is_reported_and_a_stray_label_ignored():
    session, _ = _started(_chunk(_segment(rec=True), _segment(label="success")))
    segments = _run(session).segments
    assert segments[0].opened
    assert segments[1].closed_unlabeled
    assert segments[1].label is None
    assert not session.window_open


def test_a_lost_connection_propagates():
    session, _ = _started(_protocol.EnvConnectionLostError("gone"))
    with pytest.raises(_protocol.EnvConnectionLostError):
        _run(session)
