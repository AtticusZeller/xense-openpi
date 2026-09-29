"""Run logging for online RL: W&B rows on independent axes, mirrored to ``events.jsonl``.

Each axis - RLT training logs per round, per gradient update and per executed chunk,
evaluation per round and per trial - has its own step counter, so rows of different
cadence never share one. With a dump directory, every row is also appended to
``events.jsonl``. Logging failures are warned about once and never interrupt the
robot loop.
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from typing import Any


class RunLogger:
    def __init__(self, run: Any | None, dump_dir: pathlib.Path | None = None, *, axes: tuple[str, ...]):
        self._run = run
        self._dump_dir = dump_dir
        self._warned = False
        if run is not None:
            for axis in axes:
                run.define_metric(f"{axis}/step")
                run.define_metric(f"{axis}/*", step_metric=f"{axis}/step")

    def log(self, axis: str, step: int, values: dict[str, Any]) -> None:
        try:
            row = {key: float(value) for key, value in values.items() if value is not None}
            if self._run is not None:
                self._run.log({**{f"{axis}/{k}": v for k, v in row.items()}, f"{axis}/step": step})
            if self._dump_dir is not None:
                with (self._dump_dir / "events.jsonl").open("a") as f:
                    f.write(json.dumps({"axis": axis, "step": step, "time": time.time(), **row}) + "\n")
        except Exception:
            if not self._warned:
                logging.warning("Metric logging failed; further failures are silent.", exc_info=True)
                self._warned = True
