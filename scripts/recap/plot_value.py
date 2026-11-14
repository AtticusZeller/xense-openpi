"""Plots predicted value against the return for held-out episodes, with evenly spaced frames on top.

Reads the predictions written by scripts/recap/train_value.py and writes one PNG per episode.

Usage:
    python scripts/recap/plot_value.py --predictions <run>/heldout_predictions.parquet \
        --repo-id Xense/cylinder-stack-0909 --output-dir <dir> [--episodes 3 17]
"""

import dataclasses
import pathlib

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro

from openpi.training import data_loader as _data_loader


@dataclasses.dataclass(frozen=True)
class Args:
    # Parquet with the columns episode_index, frame_index, return, value.
    predictions: pathlib.Path
    # LeRobot repo id of the dataset the predictions were computed on.
    repo_id: str
    output_dir: pathlib.Path
    # Root of the dataset. Defaults to the LeRobot cache.
    dataset_root: pathlib.Path | None = None
    # Episodes to plot. Defaults to every episode in the predictions.
    episodes: tuple[int, ...] = ()
    # Camera shown above the curves.
    camera_key: str = "observation.images.head"
    num_frames: int = 8


def frame_positions(length: int, num_frames: int) -> np.ndarray:
    """Returns `num_frames` evenly spaced frame indices covering the first and last frame."""
    return np.linspace(0, length - 1, min(num_frames, length)).round().astype(int)


def plot_episode(episode: pd.DataFrame, frames: list[np.ndarray], positions: np.ndarray, title: str) -> plt.Figure:
    """Draws the frames in a top row and the value and return curves below.

    Args:
        episode: Rows of one episode sorted by frame_index, with the columns frame_index, return, value.
        frames: HWC uint8 images shown above the curves.
        positions: Frame index of each image.
        title: Figure title.
    """
    figure = plt.figure(figsize=(2.2 * len(frames), 5.2), constrained_layout=True)
    grid = figure.add_gridspec(2, len(frames), height_ratios=[1, 2.2])
    for column, (image, position) in enumerate(zip(frames, positions, strict=True)):
        axis = figure.add_subplot(grid[0, column])
        axis.imshow(image)
        axis.set_title(f"t={position}", fontsize=8)
        axis.axis("off")
    axis = figure.add_subplot(grid[1, :])
    axis.plot(episode["frame_index"], episode["return"], color="tab:gray", linestyle="--", label="return")
    axis.plot(episode["frame_index"], episode["value"], color="tab:blue", label="predicted value")
    for position in positions:
        axis.axvline(position, color="tab:gray", alpha=0.2, linewidth=0.8)
    mae = float(np.abs(episode["value"] - episode["return"]).mean())
    axis.set_title(f"{title}  (MAE {mae:.4f})", fontsize=10)
    axis.set_xlabel("frame")
    axis.set_ylabel("normalized return")
    axis.legend(loc="lower right")
    return figure


def main(args: Args) -> None:
    predictions = pd.read_parquet(args.predictions)
    episodes = args.episodes or tuple(sorted(predictions["episode_index"].unique().tolist()))
    dataset = _data_loader.SelectiveVideoLeRobotDataset(args.repo_id, root=args.dataset_root, tolerance_s=1e-2)
    dataset.decode_video_keys = frozenset({args.camera_key})
    starts = {
        int(e): int(s)
        for e, s in zip(
            dataset.meta.episodes["episode_index"], dataset.meta.episodes["dataset_from_index"], strict=True
        )
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for episode_index in episodes:
        episode = predictions[predictions["episode_index"] == episode_index].sort_values("frame_index")
        if episode.empty:
            raise ValueError(f"Episode {episode_index} is not in {args.predictions}")
        positions = frame_positions(len(episode), args.num_frames)
        frames = [
            (dataset[starts[episode_index] + int(p)][args.camera_key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            for p in positions
        ]
        figure = plot_episode(episode, frames, positions, f"episode {episode_index}")
        figure.savefig(args.output_dir / f"value_episode_{episode_index:04d}.png", dpi=130)
        plt.close(figure)


if __name__ == "__main__":
    main(tyro.cli(Args))
