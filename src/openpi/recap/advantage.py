"""N-step advantage estimates and binary advantage labels for RECAP (pi*0.6)."""

import numpy as np


def n_step_advantages(returns: np.ndarray, values: np.ndarray, horizon: int) -> np.ndarray:
    """Estimates the advantage of every frame of one episode.

    `A_t = (sum of the next `horizon` rewards) + V(t + horizon) - V(t)`. Rewards are not stored; their sum
    over a stretch is the drop of the return across it, which also carries the penalty of a failed episode.
    Once `t + horizon` passes the last frame the episode is over and there is no value left to bootstrap
    from, so the remaining rewards are the return itself: `A_t = R_t - V(t)`.

    Args:
        returns: Normalized return of every frame, shape (length,).
        values: Predicted value of every frame, shape (length,).
        horizon: Number of steps to look ahead.

    Returns:
        Float32 array of shape (length,).
    """
    returns, values = np.asarray(returns, np.float64), np.asarray(values, np.float64)
    if returns.ndim != 1 or returns.shape != values.shape:
        raise ValueError(f"returns and values must be 1-D and equally long, got {returns.shape} and {values.shape}")
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    advantages = returns - values
    bootstrapped = len(returns) - horizon
    if bootstrapped > 0:
        advantages[:bootstrapped] = (
            returns[:bootstrapped] - returns[horizon:] + values[horizon:] - values[:bootstrapped]
        )
    return advantages.astype(np.float32)


def positive_threshold(advantages: np.ndarray, positive_fraction: float) -> float:
    """Returns the threshold above which `positive_fraction` of `advantages` lie."""
    if not 0.0 < positive_fraction < 1.0:
        raise ValueError(f"positive_fraction must be in (0, 1), got {positive_fraction}")
    return float(np.quantile(np.asarray(advantages, np.float64), 1.0 - positive_fraction))


def label_runs(positive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Splits a boolean label sequence into runs of equal labels.

    Returns:
        `(starts, ends)`: run `i` covers the frames `starts[i]` to `ends[i]` (exclusive) and carries the
        label `positive[starts[i]]`.
    """
    positive = np.asarray(positive, bool)
    if positive.ndim != 1 or len(positive) == 0:
        raise ValueError(f"positive must be a non-empty 1-D array, got shape {positive.shape}")
    changes = np.flatnonzero(positive[1:] != positive[:-1]) + 1
    return np.concatenate([[0], changes]), np.concatenate([changes, [len(positive)]])
