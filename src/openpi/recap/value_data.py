"""Data for training the RECAP value model: frames of a LeRobot dataset paired with normalized returns."""

from collections.abc import Mapping, Sequence
import pathlib

import numpy as np
import pandas as pd
import torch

# Dataset columns of the bi_flexiv policy cameras, in model order.
BI_FLEXIV_CAMERA_KEYS = (
    "observation.images.head",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
IMAGE_SIZE = 224


def split_episodes(num_episodes: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Splits episode indices into sorted train and held-out sets; at least one episode is held out."""
    order = np.random.default_rng(seed).permutation(num_episodes)
    num_val = max(1, round(num_episodes * val_fraction))
    return np.sort(order[num_val:]), np.sort(order[:num_val])


def frame_indices(episode_starts: np.ndarray, episode_lengths: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    """Returns the global frame indices of the given episodes, in episode then frame order."""
    return np.concatenate(
        [np.arange(episode_starts[e], episode_starts[e] + episode_lengths[e], dtype=np.int64) for e in episodes]
    )


def align_returns(table: pd.DataFrame, episode_starts: np.ndarray, episode_lengths: np.ndarray) -> np.ndarray:
    """Orders the sidecar return table by global frame index.

    Args:
        table: Columns `episode_index`, `frame_index`, `return`.
        episode_starts: Global index of the first frame of each episode.
        episode_lengths: Frame count of each episode.

    Returns:
        Float32 array with one return per dataset frame.

    Raises:
        ValueError: If the table does not cover every frame exactly once.
    """
    num_frames = int(episode_lengths.sum())
    episode = table["episode_index"].to_numpy()
    frame = table["frame_index"].to_numpy()
    if len(table) != num_frames or (frame >= episode_lengths[episode]).any():
        raise ValueError(f"Return table has {len(table)} rows but the dataset has {num_frames} frames.")
    index = episode_starts[episode] + frame
    if len(np.unique(index)) != num_frames:
        raise ValueError("Return table has duplicated (episode_index, frame_index) rows.")
    returns = np.empty(num_frames, dtype=np.float32)
    returns[index] = table["return"].to_numpy()
    return returns


def tokenize_tasks(encode, tasks: Mapping[int, str], max_len: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Tokenizes every task once.

    Args:
        encode: Maps text to token ids, including the BOS token.
        tasks: Task text by task index.
        max_len: Padded length; longer texts raise instead of being truncated.

    Returns:
        `(tokens, mask)` of shape (max_len,) by task index.
    """
    result = {}
    for task_index, text in tasks.items():
        ids = list(encode(text))
        if len(ids) > max_len:
            raise ValueError(f"Task {task_index} has {len(ids)} tokens, more than max_len={max_len}.")
        tokens = np.zeros(max_len, dtype=np.int32)
        tokens[: len(ids)] = ids
        mask = np.arange(max_len) < len(ids)
        result[int(task_index)] = (tokens, mask)
    return result


def resize_with_pad(image: torch.Tensor, size: int = IMAGE_SIZE) -> np.ndarray:
    """Resizes a float CHW image in [0, 1] to uint8 HWC (size, size, 3), padding with black to keep the aspect."""
    _, height, width = image.shape
    scale = size / max(height, width)
    new_height, new_width = max(1, round(height * scale)), max(1, round(width * scale))
    resized = torch.nn.functional.interpolate(
        image[None], size=(new_height, new_width), mode="bilinear", align_corners=False, antialias=True
    )[0]
    canvas = torch.zeros(3, size, size, dtype=resized.dtype)
    top, left = (size - new_height) // 2, (size - new_width) // 2
    canvas[:, top : top + new_height, left : left + new_width] = resized
    return (canvas.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()


class ValueFrames(torch.utils.data.Dataset):
    """Frames of a LeRobot dataset as value-model samples.

    Each sample holds `images` uint8 (N, 224, 224, 3), `tokens` int32 (L,), `token_mask` bool (L,),
    `target` float32 and the global frame `index`.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        indices: np.ndarray,
        returns: np.ndarray,
        task_tokens: Mapping[int, tuple[np.ndarray, np.ndarray]],
        camera_keys: Sequence[str] = BI_FLEXIV_CAMERA_KEYS,
    ):
        self._dataset = dataset
        self._indices = np.asarray(indices, dtype=np.int64)
        self._returns = returns
        self._task_tokens = task_tokens
        self._camera_keys = tuple(camera_keys)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        index = int(self._indices[i])
        item = self._dataset[index]
        tokens, mask = self._task_tokens[int(item["task_index"])]
        return {
            "images": np.stack([resize_with_pad(item[key]) for key in self._camera_keys]),
            "tokens": tokens,
            "token_mask": mask,
            "target": np.float32(self._returns[index]),
            "index": np.int64(index),
        }


def load_episode_table(dataset_root: pathlib.Path) -> pd.DataFrame:
    """Loads `episode_index`, `length` and `dataset_from_index` of every episode, sorted by episode index."""
    files = sorted((dataset_root / "meta" / "episodes").glob("*/*.parquet"))
    columns = ["episode_index", "length", "dataset_from_index"]
    episodes = pd.concat([pd.read_parquet(file, columns=columns) for file in files], ignore_index=True)
    episodes = episodes.sort_values("episode_index", ignore_index=True)
    # Episode indices double as row positions everywhere in this module.
    if not np.array_equal(episodes["episode_index"].to_numpy(), np.arange(len(episodes))):
        raise ValueError("Episode indices must be consecutive and start at 0.")
    return episodes
