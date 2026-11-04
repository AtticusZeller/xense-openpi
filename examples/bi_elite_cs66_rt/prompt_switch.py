"""Keyboard-switched task prompt for the two-stage egg-carton policy.

The policy was trained with one prompt per stage. A background thread reads single key presses from
the controlling terminal (works over SSH, no GUI keyboard library): ``1`` selects OPEN_PROMPT, ``2``
selects EGG_PROMPT. The control thread notices the change on its next step, drops the cached action
chunk through the broker's ``reset()`` and infers with the new prompt.
"""

import os
import select
import sys
import termios
import threading
import tty
from typing import TextIO, override

from xense_client import base_policy as _base_policy
from xense_client.logger import get_logger

logger = get_logger("PromptSwitch")

OPEN_PROMPT = "Open the egg carton."
EGG_PROMPT = "Take out the eggs one by one and place them in the egg tray."
KEY_PROMPTS = {"1": OPEN_PROMPT, "2": EGG_PROMPT}


class PromptState:
    """Current prompt plus a version that increments on every change; shared between threads."""

    def __init__(self, prompt: str = OPEN_PROMPT) -> None:
        self._lock = threading.Lock()
        self._prompt = prompt
        self._version = 0

    def snapshot(self) -> tuple[str, int]:
        with self._lock:
            return self._prompt, self._version

    def set(self, prompt: str) -> bool:
        """Select ``prompt``; returns False when it is already current."""
        with self._lock:
            if prompt == self._prompt:
                return False
            self._prompt = prompt
            self._version += 1
            return True


class PromptSwitchingPolicy(_base_policy.BasePolicy):
    """Adds the current prompt to every observation and resets the inner broker after a switch."""

    def __init__(self, policy: _base_policy.BasePolicy, state: PromptState) -> None:
        self._policy = policy
        self._state = state
        self._applied_version = state.snapshot()[1]

    @override
    def infer(self, obs: dict) -> dict:
        prompt, version = self._state.snapshot()
        if version != self._applied_version:
            # Chunks already queued were planned for the old prompt; drop them before inferring.
            self._policy.reset()
            self._applied_version = version
            logger.info(f"Prompt switch applied, cached actions cleared: {prompt!r}")
        return self._policy.infer({**obs, "prompt": prompt})

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._applied_version = self._state.snapshot()[1]

    @override
    def warmup(self, obs: dict) -> None:
        prompt, self._applied_version = self._state.snapshot()
        self._policy.warmup({**obs, "prompt": prompt})


class KeyboardPromptSwitcher:
    """Reads single keys from a terminal in cbreak mode and updates a PromptState.

    cbreak keeps ISIG, so Ctrl+C still reaches the SIGINT handler. ``stop()`` restores the terminal.
    """

    def __init__(self, state: PromptState, stream: TextIO = sys.stdin) -> None:
        if not stream.isatty():
            raise RuntimeError("prompt switching reads keys from the terminal, but stdin is not a TTY")
        self._state = state
        self._fd = stream.fileno()
        self._saved_attrs = termios.tcgetattr(self._fd)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="prompt-keys", daemon=True)

    def start(self) -> None:
        tty.setcbreak(self._fd)
        self._thread.start()
        prompt, _ = self._state.snapshot()
        logger.info(f"Prompt keys: 1 = {OPEN_PROMPT!r}, 2 = {EGG_PROMPT!r}. Current prompt: {prompt!r}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_attrs)

    def _loop(self) -> None:
        while not self._stop.is_set():
            readable, _, _ = select.select([self._fd], [], [], 0.1)
            if not readable:
                continue
            data = os.read(self._fd, 1)
            if not data:
                return
            prompt = KEY_PROMPTS.get(data.decode(errors="ignore"))
            if prompt is None:
                continue
            if self._state.set(prompt):
                logger.info(f"Prompt -> {prompt!r}")
            else:
                logger.info(f"Prompt unchanged: {prompt!r}")
