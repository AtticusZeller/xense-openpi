"""Prompt switching against the real chunk brokers with a fake policy; no robot or server needed."""

import io
import os
import pty
import termios
import threading
import time

import numpy as np
import pytest
from xense_client.action_chunk_broker import ActionChunkBroker
from xense_client.base_policy import BasePolicy
from xense_client.rtc_action_chunk_broker import RTCActionChunkBroker

from examples.bi_elite_cs66_rt.prompt_switch import EGG_PROMPT
from examples.bi_elite_cs66_rt.prompt_switch import OPEN_PROMPT
from examples.bi_elite_cs66_rt.prompt_switch import KeyboardPromptSwitcher
from examples.bi_elite_cs66_rt.prompt_switch import PromptState
from examples.bi_elite_cs66_rt.prompt_switch import PromptSwitchingPolicy

HORIZON = 10
ACTION_DIM = 20


class RecordingPolicy(BasePolicy):
    """Returns a chunk filled with its call number and records the prompt of each call."""

    def __init__(self, latency_s: float = 0.0) -> None:
        self.prompts: list[str] = []
        self._latency_s = latency_s
        self._lock = threading.Lock()

    def infer(self, obs: dict, **kwargs) -> dict:
        time.sleep(self._latency_s)
        with self._lock:
            self.prompts.append(obs["prompt"])
            seq = len(self.prompts) - 1
        return {"actions": np.full((HORIZON, ACTION_DIM), seq, dtype=np.float32)}

    def prompt_of(self, action: np.ndarray) -> str:
        with self._lock:
            return self.prompts[int(action[0])]


def test_chunk_broker_drops_old_chunk_on_switch():
    inner = RecordingPolicy()
    state = PromptState()
    policy = PromptSwitchingPolicy(ActionChunkBroker(inner, action_horizon=HORIZON), state)

    first = [policy.infer({})["actions"] for _ in range(3)]
    assert inner.prompts == [OPEN_PROMPT]
    assert all(inner.prompt_of(a) == OPEN_PROMPT for a in first)

    assert state.set(EGG_PROMPT)
    after = policy.infer({})["actions"]
    # The 4th step would have continued chunk 0; it must come from a fresh EGG inference instead.
    assert inner.prompts == [OPEN_PROMPT, EGG_PROMPT]
    assert inner.prompt_of(after) == EGG_PROMPT


def test_same_key_does_not_reset():
    inner = RecordingPolicy()
    state = PromptState()
    policy = PromptSwitchingPolicy(ActionChunkBroker(inner, action_horizon=HORIZON), state)
    policy.infer({})
    assert not state.set(OPEN_PROMPT)
    policy.infer({})
    assert inner.prompts == [OPEN_PROMPT]


def test_rtc_broker_executes_only_new_prompt_actions_after_switch():
    hz = 50.0
    inner = RecordingPolicy(latency_s=0.01)
    state = PromptState()
    broker = RTCActionChunkBroker(inner, frequency_hz=hz, action_queue_size_to_get_new_actions=5, default_delay=2)
    policy = PromptSwitchingPolicy(broker, state)

    def run_steps(n: int) -> list[np.ndarray]:
        # Paced like the runtime loop so the background inference keeps the queue filled.
        actions = []
        for _ in range(n):
            actions.append(policy.infer({})["actions"])
            time.sleep(1.0 / hz)
        return actions

    try:
        policy.warmup({})
        before = run_steps(30)
        assert all(inner.prompt_of(a) == OPEN_PROMPT for a in before)

        assert state.set(EGG_PROMPT)
        after = run_steps(30)
        assert all(inner.prompt_of(a) == EGG_PROMPT for a in after)
        assert EGG_PROMPT in inner.prompts
    finally:
        broker.stop()


def test_warmup_and_reset_use_current_prompt():
    inner = RecordingPolicy()
    state = PromptState()
    policy = PromptSwitchingPolicy(ActionChunkBroker(inner, action_horizon=HORIZON), state)
    state.set(EGG_PROMPT)
    policy.reset()  # Episode start: the new version is applied without a second reset.
    policy.infer({})
    assert inner.prompts == [EGG_PROMPT]


def _wait_for(predicate, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_keyboard_switches_on_single_key_and_restores_terminal():
    master, slave = pty.openpty()
    with os.fdopen(slave, "r") as stream:
        attrs_before = termios.tcgetattr(slave)
        state = PromptState()
        switcher = KeyboardPromptSwitcher(state, stream=stream)
        switcher.start()
        try:
            assert not termios.tcgetattr(slave)[3] & termios.ICANON  # cbreak: no Enter needed.
            os.write(master, b"2")
            assert _wait_for(lambda: state.snapshot() == (EGG_PROMPT, 1))
            os.write(master, b"x2")  # Unknown key ignored, same key does not bump the version.
            os.write(master, b"1")
            assert _wait_for(lambda: state.snapshot() == (OPEN_PROMPT, 2))
        finally:
            switcher.stop()
        assert termios.tcgetattr(slave) == attrs_before
    os.close(master)


def test_keyboard_requires_tty():
    with pytest.raises(RuntimeError, match="not a TTY"):
        KeyboardPromptSwitcher(PromptState(), stream=io.StringIO())


def test_main_rejects_action_hz_with_prompt_switch(monkeypatch):
    pytest.importorskip("lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt", reason="Elite driver not installed")
    import examples.bi_elite_cs66_rt.main as entrypoint

    def fail_if_connecting(**kwargs):
        raise AssertionError("must fail before connecting to the policy server")

    monkeypatch.setattr(entrypoint._websocket_client_policy, "WebsocketClientPolicy", fail_if_connecting)
    args = entrypoint.Args(robot_recipe="diagonal-08-taccap", action_hz=50.0)
    with pytest.raises(SystemExit, match=r"does not support --args\.action-hz"):
        entrypoint.main(args)
