"""Normalized per-frame returns for the RECAP (pi*0.6) value model.

Every non-terminal step costs 1 and a failed episode pays an extra `c_fail` at its end. Returns are
divided by `l_max + c_fail`, with `c_fail = l_max` (the frame count of the longest episode of the
task), so successful episodes span [-0.5, 0] and failed ones stay inside [-1, 0].
"""

import numpy as np


def episode_returns(length: int, *, success: bool, l_max: int, c_fail: int | None = None) -> np.ndarray:
    """Computes the normalized return of every frame of one episode.

    Args:
        length: Number of frames of the episode.
        success: Whether the episode ended in success.
        l_max: Frame count of the longest episode of the same task.
        c_fail: Penalty added at the end of a failed episode. Defaults to `l_max`.

    Returns:
        Float32 array of shape (length,) with values in [-1, 0]; the last frame of a success is 0.
    """
    if length < 1 or length > l_max:
        raise ValueError(f"length must be in [1, l_max={l_max}], got {length}")
    c_fail = l_max if c_fail is None else c_fail
    steps_to_go = np.arange(length - 1, -1, -1, dtype=np.float64)
    penalty = 0.0 if success else float(c_fail)
    returns = -(steps_to_go + penalty) / (l_max + c_fail)
    return np.clip(returns, -1.0, 0.0).astype(np.float32)


def dataset_returns(lengths: np.ndarray, successes: np.ndarray, tasks: np.ndarray) -> list[np.ndarray]:
    """Computes normalized returns for all episodes, normalizing each task by its own longest episode.

    Args:
        lengths: Frame count of each episode, shape (num_episodes,).
        successes: Whether each episode succeeded, shape (num_episodes,).
        tasks: Task identifier of each episode, shape (num_episodes,).

    Returns:
        One return array per episode, in the input order.
    """
    lengths, successes, tasks = np.asarray(lengths), np.asarray(successes), np.asarray(tasks)
    l_max_by_task = {task: int(lengths[tasks == task].max()) for task in np.unique(tasks)}
    return [
        episode_returns(int(length), success=bool(success), l_max=l_max_by_task[task])
        for length, success, task in zip(lengths, successes, tasks, strict=True)
    ]
