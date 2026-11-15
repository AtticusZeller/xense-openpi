"""Episode-level checks run on a dataset before it is used to train the RECAP value model.

A demonstration that is recorded while the robot does not move teaches the value model that time passes
without progress, and one overlong episode stretches the return normalization of its whole task. These
functions find such episodes from the robot state alone; deciding what to drop stays with a person.
"""

import numpy as np


def motion_spans(states: np.ndarray) -> np.ndarray:
    """Returns the range each state dimension covers over a dataset, from its 1st to its 99th percentile.

    Dimensions that never change get an infinite span, so they can never count as motion.
    """
    spans = np.quantile(states, 0.99, axis=0) - np.quantile(states, 0.01, axis=0)
    return np.where(spans > 1e-6, spans, np.inf)


def still_frames(states: np.ndarray, spans: np.ndarray, threshold: float) -> np.ndarray:
    """Marks the frames of one episode during which the robot does not move.

    A frame is still when no state dimension changed by more than `threshold` of its span since the
    previous frame. The first frame has no predecessor and takes the mark of the second.

    Args:
        states: Robot state of every frame, shape (length, dims).
        spans: Range of every dimension, shape (dims,), from `motion_spans`.
        threshold: Largest per-frame change, as a fraction of the span, that still counts as no motion.

    Returns:
        Boolean array of shape (length,).
    """
    if len(states) < 2:
        return np.zeros(len(states), bool)
    still = (np.abs(np.diff(states, axis=0)) / spans).max(axis=1) < threshold
    return np.concatenate([still[:1], still])


def edge_still_lengths(still: np.ndarray) -> tuple[int, int]:
    """Returns how many frames are still at the start and at the end of an episode."""
    moving = np.flatnonzero(~np.asarray(still, bool))
    if len(moving) == 0:
        return len(still), len(still)
    return int(moving[0]), int(len(still) - 1 - moving[-1])


def length_outliers(lengths: np.ndarray, iqr_factor: float) -> np.ndarray:
    """Marks episodes longer than the third quartile plus `iqr_factor` interquartile ranges."""
    lengths = np.asarray(lengths, np.float64)
    q1, q3 = np.quantile(lengths, [0.25, 0.75])
    return lengths > q3 + iqr_factor * (q3 - q1)
