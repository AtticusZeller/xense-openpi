"""Test helper: import TacXense's torch RLT implementation as a numerical reference."""

import os
import pathlib
import sys

import pytest


def import_tacxense():
    """Return the ``tacxense.rlt`` package, or skip if the TacXense checkout is absent (set TACXENSE_ROOT)."""
    root = pathlib.Path(os.environ.get("TACXENSE_ROOT", pathlib.Path(__file__).resolve().parents[6] / "TacXense"))
    if not (root / "src" / "tacxense" / "rlt").is_dir():
        pytest.skip(f"TacXense reference not found at {root} (set TACXENSE_ROOT)")
    pytest.importorskip("torch")
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    import tacxense.rlt

    return tacxense.rlt
