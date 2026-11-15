"""Plots predicted value against the return for whole episodes, with the advantage labels the value implies.

Reads the predictions written by scripts/recap/train_value.py and writes one PNG per episode: evenly spaced
frames on top, the value and return curves over a background colored by advantage label, and the advantage
below. The labels follow the RECAP rule: an N-step advantage estimate, positive above a per-task threshold.

With `--videos` it also writes one MP4 per episode: the camera image inside a border in the color of its
label, over a timeline of the episode's labels with a cursor.

Usage:
    python scripts/recap/plot_value.py --predictions <run>/heldout_predictions.parquet \
        --repo-id Xense/cylinder-stack-0909 --output-dir <dir> [--episodes 3 17] [--videos]
"""

import dataclasses
import logging
import pathlib

import av
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro

from openpi.recap import advantage as _advantage
from openpi.recap import label_video as _label_video
from openpi.training import data_loader as _data_loader

_LABEL_COLORS = {True: "tab:green", False: "tab:red"}
_VIDEO_BORDER = 8
_VIDEO_STRIP_HEIGHT = 24


@dataclasses.dataclass(frozen=True)
class Args:
    # Parquet with the columns episode_index, frame_index, return, value, holding whole episodes.
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
    # Steps the advantage estimate looks ahead.
    advantage_horizon: int = 50
    # Share of the frames labeled positive. The threshold is taken over every frame of this dataset in the
    # predictions, which is per task as long as the dataset holds one task.
    positive_fraction: float = 0.4
    # Also write one MP4 per episode with every frame bordered in the color of its label.
    videos: bool = False


def frame_positions(length: int, num_frames: int) -> np.ndarray:
    """Returns `num_frames` evenly spaced frame indices covering the first and last frame."""
    return np.linspace(0, length - 1, min(num_frames, length)).round().astype(int)


def label_advantages(predictions: pd.DataFrame, horizon: int, positive_fraction: float) -> tuple[pd.DataFrame, float]:
    """Adds the columns `advantage` and `positive` to the predictions of one dataset.

    Returns:
        The predictions sorted by `(episode_index, frame_index)` and the advantage threshold.
    """
    predictions = predictions.sort_values(["episode_index", "frame_index"], ignore_index=True)
    advantages = []
    for episode_index, episode in predictions.groupby("episode_index", sort=False):
        if not np.array_equal(episode["frame_index"], np.arange(len(episode))):
            raise ValueError(f"Episode {episode_index} is not predicted in full; advantages need every frame.")
        advantages.append(
            _advantage.n_step_advantages(episode["return"].to_numpy(), episode["value"].to_numpy(), horizon)
        )
    predictions["advantage"] = np.concatenate(advantages)
    threshold = _advantage.positive_threshold(predictions["advantage"].to_numpy(), positive_fraction)
    predictions["positive"] = predictions["advantage"] > threshold
    return predictions, threshold


def label_statistics(predictions: pd.DataFrame, fps: float) -> dict[str, float]:
    """Summarizes how the labels are spread: advantage spread, label flips per second, positive run length."""
    flips, positive_runs = 0, []
    for _, episode in predictions.groupby("episode_index", sort=False):
        positive = episode["positive"].to_numpy()
        starts, ends = _advantage.label_runs(positive)
        flips += len(starts) - 1
        positive_runs.extend((ends - starts)[positive[starts]])
    return {
        "advantage_std": float(predictions["advantage"].std()),
        "flips_per_second": flips / (len(predictions) / fps),
        "median_positive_run_seconds": float(np.median(positive_runs)) / fps if positive_runs else 0.0,
    }


def plot_episode(
    episode: pd.DataFrame, frames: list[np.ndarray], positions: np.ndarray, threshold: float, title: str
) -> plt.Figure:
    """Draws the frames in a top row, the value and return curves below and the advantage at the bottom.

    Args:
        episode: Rows of one episode sorted by frame_index, with the columns frame_index, return, value,
            advantage, positive.
        frames: HWC uint8 images shown above the curves.
        positions: Frame index of each image.
        threshold: Advantage threshold separating positive from negative.
        title: Figure title.
    """
    positive = episode["positive"].to_numpy()
    figure = plt.figure(figsize=(2.2 * len(frames), 6.6), constrained_layout=True)
    grid = figure.add_gridspec(3, len(frames), height_ratios=[1, 2.2, 1])
    for column, (image, position) in enumerate(zip(frames, positions, strict=True)):
        axis = figure.add_subplot(grid[0, column])
        axis.imshow(image)
        axis.set_title(f"t={position}", fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            # The border tells the label of the moment the frame was taken.
            spine.set_edgecolor(_LABEL_COLORS[bool(positive[position])])
            spine.set_linewidth(3)

    value_axis = figure.add_subplot(grid[1, :])
    advantage_axis = figure.add_subplot(grid[2, :], sharex=value_axis)
    for start, end in zip(*_advantage.label_runs(positive), strict=True):
        for axis in (value_axis, advantage_axis):
            axis.axvspan(start, end, color=_LABEL_COLORS[bool(positive[start])], alpha=0.15, linewidth=0)
    for position in positions:
        value_axis.axvline(position, color="tab:gray", alpha=0.3, linewidth=0.8)

    value_axis.plot(episode["frame_index"], episode["return"], color="tab:gray", linestyle="--", label="return")
    value_axis.plot(episode["frame_index"], episode["value"], color="tab:blue", label="predicted value")
    mae = float(np.abs(episode["value"] - episode["return"]).mean())
    value_axis.set_title(
        f"{title}  (MAE {mae:.4f}, {positive.mean():.0%} positive; green = positive, red = negative)", fontsize=10
    )
    value_axis.set_ylabel("normalized return")
    value_axis.legend(loc="lower right")
    value_axis.tick_params(labelbottom=False)

    advantage_axis.plot(episode["frame_index"], episode["advantage"], color="black", linewidth=0.8)
    advantage_axis.axhline(threshold, color="tab:purple", linestyle=":", label=f"threshold {threshold:.4f}")
    advantage_axis.set_xlabel("frame")
    advantage_axis.set_ylabel("advantage")
    advantage_axis.legend(loc="upper left", fontsize=8)
    return figure


def load_frame(dataset: _data_loader.SelectiveVideoLeRobotDataset, index: int, camera_key: str) -> np.ndarray:
    """Decodes one camera image of the dataset as HWC uint8."""
    return (dataset[index][camera_key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def write_video(
    path: pathlib.Path,
    episode: pd.DataFrame,
    dataset: _data_loader.SelectiveVideoLeRobotDataset,
    start: int,
    camera_key: str,
) -> None:
    """Writes one episode as H.264 video with the label of every frame drawn on it.

    Args:
        path: Output MP4.
        episode: Rows of one episode sorted by frame_index, with the columns episode_index, advantage, positive.
        dataset: Dataset the frames are decoded from.
        start: Dataset index of the episode's first frame.
        camera_key: Camera to show.
    """
    positive, advantages = episode["positive"].to_numpy(), episode["advantage"].to_numpy()
    episode_index = int(episode["episode_index"].iloc[0])
    with av.open(str(path), "w") as container:
        stream, strip = None, None
        for position in range(len(episode)):
            image = load_frame(dataset, start + position, camera_key)
            if strip is None:
                strip = _label_video.label_strip(positive, image.shape[1] + 2 * _VIDEO_BORDER, _VIDEO_STRIP_HEIGHT)
            frame = _label_video.annotate_frame(
                image, strip, position, len(episode), positive=bool(positive[position]), border=_VIDEO_BORDER,
                caption=f"episode {episode_index}  t={position}  A={advantages[position]:+.4f}",
            )  # fmt: skip
            if stream is None:
                # yuv420p needs even sizes; the border and the strip keep the parity of the camera image.
                stream = container.add_stream("libx264", rate=dataset.meta.fps, options={"crf": "26"})
                stream.height, stream.width, stream.pix_fmt = frame.shape[0], frame.shape[1], "yuv420p"
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        if stream is not None:
            container.mux(stream.encode())


def main(args: Args) -> None:
    predictions = pd.read_parquet(args.predictions)
    if "dataset" in predictions:
        # Runs on several datasets write all of them into one file.
        predictions = predictions[predictions["dataset"] == args.repo_id]
    predictions, threshold = label_advantages(predictions, args.advantage_horizon, args.positive_fraction)
    episodes = args.episodes or tuple(sorted(predictions["episode_index"].unique().tolist()))
    dataset = _data_loader.SelectiveVideoLeRobotDataset(args.repo_id, root=args.dataset_root, tolerance_s=1e-2)
    dataset.decode_video_keys = frozenset({args.camera_key})
    starts = {
        int(e): int(s)
        for e, s in zip(
            dataset.meta.episodes["episode_index"], dataset.meta.episodes["dataset_from_index"], strict=True
        )
    }
    lengths = dict(zip(dataset.meta.episodes["episode_index"], dataset.meta.episodes["length"], strict=True))
    for episode_index, length in predictions.groupby("episode_index").size().items():
        if length != lengths[episode_index]:
            # A copy of the dataset with episodes dropped numbers them differently; frames would then come
            # from another episode than the curve.
            raise ValueError(
                f"Episode {episode_index} has {length} predicted frames but {lengths[episode_index]} frames in "
                f"{dataset.root}: the predictions are cut short or were made on another copy of the dataset."
            )
    statistics = label_statistics(predictions, dataset.meta.fps)
    logging.info(
        "%s: %d episodes, threshold=%.5f advantage_std=%.5f flips_per_second=%.2f median_positive_run=%.2fs",
        args.repo_id, predictions["episode_index"].nunique(), threshold, statistics["advantage_std"],
        statistics["flips_per_second"], statistics["median_positive_run_seconds"],
    )  # fmt: skip
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for episode_index in episodes:
        episode = predictions[predictions["episode_index"] == episode_index]
        if episode.empty:
            raise ValueError(f"Episode {episode_index} is not in {args.predictions}")
        positions = frame_positions(len(episode), args.num_frames)
        frames = [load_frame(dataset, starts[episode_index] + int(p), args.camera_key) for p in positions]
        figure = plot_episode(episode, frames, positions, threshold, f"episode {episode_index}")
        figure.savefig(args.output_dir / f"value_episode_{episode_index:04d}.png", dpi=130)
        plt.close(figure)
        if args.videos:
            video = args.output_dir / f"value_episode_{episode_index:04d}.mp4"
            write_video(video, episode, dataset, starts[episode_index], args.camera_key)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    main(tyro.cli(Args))
