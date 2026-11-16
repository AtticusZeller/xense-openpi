"""LeRobot-format episode recorder for BiFlexiv Rizon4 RT inference.

Records observations and actions during policy execution in the same format
as the original training dataset (raw state, raw 640x480 HWC images, absolute actions).

Usage:
    from examples.bi_flexiv_rizon4_rt.recorder import make_recorder_subscriber

    subscriber = make_recorder_subscriber(
        repo_id="Xense/my_new_dataset",
        task="pack 6 cosmetic bottles into the carton",
        fps=30,
    )
    runtime = Runtime(..., subscribers=[subscriber])
"""

import pathlib
from typing import override

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import get_logger
import numpy as np
from xense_client.runtime import subscriber as _subscriber

logger = get_logger("BiFlexivRecorder")

# Feature names matching Xense/pack_6_cosmetic_bottles_into_carton and all bi_flexiv datasets.
_STATE_NAMES = [
    "left_tcp.x",
    "left_tcp.y",
    "left_tcp.z",
    "left_tcp.r1",
    "left_tcp.r2",
    "left_tcp.r3",
    "left_tcp.r4",
    "left_tcp.r5",
    "left_tcp.r6",
    "right_tcp.x",
    "right_tcp.y",
    "right_tcp.z",
    "right_tcp.r1",
    "right_tcp.r2",
    "right_tcp.r3",
    "right_tcp.r4",
    "right_tcp.r5",
    "right_tcp.r6",
    "left_gripper.pos",
    "right_gripper.pos",
]

_ACTION_NAMES = _STATE_NAMES  # action space identical to state space

# Policy camera names (order matches BiFlexivInputs.EXPECTED_CAMERAS)
_POLICY_CAMERAS = ("head", "left_wrist", "right_wrist")

# Per-frame facts a RECAP rollout records on top of the plain features. LeRobot only
# merges datasets whose features are identical, so these must stay equal to
# COLLECTION_FEATURES in src/openpi/recap/collection_columns.py, which adds the same
# columns to the initial demonstrations (the robot host does not install openpi).
COLLECTION_FEATURES: dict = {
    "is_intervention": {"dtype": "bool", "shape": (1,), "names": None},
    "episode_success": {"dtype": "bool", "shape": (1,), "names": None},
    "round": {"dtype": "int64", "shape": (1,), "names": None},
}


def make_bi_flexiv_dataset_features(
    image_height: int = 480,
    image_width: int = 640,
    use_videos: bool = True,
) -> dict:
    """Build the LeRobot features dict for a bi_flexiv_rizon4_rt dataset."""
    dtype = "video" if use_videos else "image"
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(_STATE_NAMES),),
            "names": _STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(_ACTION_NAMES),),
            "names": _ACTION_NAMES,
        },
    }
    for cam in _POLICY_CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": dtype,
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


class LeRobotRecorderSubscriber(_subscriber.Subscriber):
    """Records inference episodes to disk in LeRobot dataset format.

    Taps into the Runtime subscriber interface:
    - on_episode_start: resets per-episode state
    - on_step: appends (observation, action) frame to the dataset
    - on_episode_end: flushes episode to disk

    The observation dict is expected to contain:
        "state": np.ndarray (20,) — raw robot state, no normalization
        "images_raw": dict[str, np.ndarray (H, W, C)] — original resolution images

    The action dict is expected to contain:
        "actions": np.ndarray (20,) — absolute action after output transforms
    """

    def __init__(self, dataset: LeRobotDataset, task: str):
        self._dataset = dataset
        self._task = task
        self._step_count = 0

    @override
    def on_episode_start(self) -> None:
        self._step_count = 0
        logger.info("Episode recording started")

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        self._dataset.add_frame(self._build_frame(observation, action))
        self._step_count += 1

    def _build_frame(self, observation: dict, action: dict) -> dict:
        images_raw = observation.get("images_raw", {})

        frame: dict = {
            "observation.state": np.asarray(observation["state"], dtype=np.float32),
            "action": np.asarray(action["actions"], dtype=np.float32),
            "task": self._task,
        }
        for cam in _POLICY_CAMERAS:
            if cam in images_raw:
                frame[f"observation.images.{cam}"] = np.asarray(images_raw[cam], dtype=np.uint8)
            else:
                logger.warn(f"Camera '{cam}' missing from images_raw, skipping frame image")
        return frame

    @override
    def on_episode_end(self) -> None:
        if self._step_count == 0:
            logger.warn("Episode ended with 0 steps — not saving")
            return
        logger.info(f"Saving episode ({self._step_count} steps)...")
        self._dataset.save_episode()
        logger.info(
            f"Episode saved. Total episodes: {self._dataset.meta.total_episodes}, "
            f"total frames: {self._dataset.meta.total_frames}"
        )


class CollectionRecorderSubscriber(LeRobotRecorderSubscriber):
    """Records RECAP rollout episodes: the plain frames plus ``COLLECTION_FEATURES``.

    ``is_intervention`` comes from each step's action dict and ``round`` is constant.
    ``episode_success`` is only known once the operator labels the episode, so it is
    written over the whole episode right before saving. An episode that ends without
    a label is dropped and does not take an episode index.
    """

    def __init__(self, dataset: LeRobotDataset, task: str, round_index: int):
        super().__init__(dataset, task)
        self._round_index = round_index
        self._success: bool | None = None

    def label_episode(self, success: bool) -> None:
        """Label the running episode; it is saved when the episode ends."""
        self._success = success

    def finalize(self) -> None:
        """Close the dataset writers. Without this the parquet files have no footer."""
        self._dataset.finalize()

    @override
    def on_episode_start(self) -> None:
        super().on_episode_start()
        self._success = None

    @override
    def _build_frame(self, observation: dict, action: dict) -> dict:
        frame = super()._build_frame(observation, action)
        frame["is_intervention"] = np.array([bool(action["is_intervention"])])
        frame["round"] = np.array([self._round_index], dtype=np.int64)
        # Placeholder: LeRobot wants every feature on every frame, on_episode_end fills it.
        frame["episode_success"] = np.array([False])
        return frame

    @override
    def on_episode_end(self) -> None:
        if self._step_count > 0 and self._success is None:
            self._dataset.clear_episode_buffer()
            logger.info(f"Episode ended without a success/failure label - {self._step_count} steps discarded")
            return
        if self._step_count > 0:
            buffer = _episode_buffer(self._dataset)
            buffer["episode_success"] = [np.array([self._success])] * buffer["size"]
        super().on_episode_end()


def _episode_buffer(dataset: LeRobotDataset) -> dict:
    """The frames added since the last save.

    lerobot-xense (the robot host) keeps the buffer on the dataset; upstream lerobot 0.5
    (the training environment, where the tests run) moved it to ``dataset.writer``.
    """
    holder = dataset if hasattr(dataset, "episode_buffer") else dataset.writer
    return holder.episode_buffer  # pyright: ignore[reportOptionalMemberAccess]


def _create_dataset(
    repo_id: str,
    fps: int,
    root: str | pathlib.Path | None,
    features: dict,
    use_videos: bool,
    image_writer_threads: int,
    vcodec: str | None = None,
) -> LeRobotDataset:
    local_dir = pathlib.Path(root) if root else None

    logger.info(f"Creating dataset repo_id={repo_id}" + (f" root={local_dir}" if local_dir else ""))
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=local_dir,
        robot_type="bi_flexiv_rizon4_rt",
        use_videos=use_videos,
        image_writer_threads=image_writer_threads,
        # None keeps LeRobot's own default codec.
        **({} if vcodec is None else {"vcodec": vcodec}),
    )


def make_collection_recorder_subscriber(
    repo_id: str,
    task: str,
    round_index: int,
    fps: int = 30,
    root: str | pathlib.Path | None = None,
    image_height: int = 480,
    image_width: int = 640,
    use_videos: bool = True,
    image_writer_threads: int = 4,
    vcodec: str = "h264",
) -> CollectionRecorderSubscriber:
    """Create a CollectionRecorderSubscriber for a new RECAP rollout dataset.

    Args:
        repo_id: HuggingFace dataset repo id (e.g. "Xense/my_dataset_v1").
        task: Language description of the task being recorded.
        round_index: Improvement round these rollouts belong to (>= 1; 0 is the initial demonstrations).
        fps: Dataset frame rate. Should match the inference runtime_hz.
        root: Local root directory. Defaults to ~/.cache/huggingface/lerobot/<repo_id>.
        image_height: Raw image height in pixels (default 480).
        image_width: Raw image width in pixels (default 640).
        use_videos: Encode images as video (True) or individual frames (False).
        image_writer_threads: Async image writer thread count.
        vcodec: Video codec. LeRobot only merges datasets whose videos share one codec, so this must be
            the codec of the initial demonstrations the rollouts are merged with (h264 for the bi_flexiv
            demonstrations).

    Returns:
        A configured CollectionRecorderSubscriber ready to attach to Runtime.
    """
    features = make_bi_flexiv_dataset_features(image_height, image_width, use_videos) | COLLECTION_FEATURES
    dataset = _create_dataset(repo_id, fps, root, features, use_videos, image_writer_threads, vcodec)
    return CollectionRecorderSubscriber(dataset=dataset, task=task, round_index=round_index)


def make_recorder_subscriber(
    repo_id: str,
    task: str,
    fps: int = 30,
    root: str | pathlib.Path | None = None,
    image_height: int = 480,
    image_width: int = 640,
    use_videos: bool = True,
    image_writer_threads: int = 4,
) -> LeRobotRecorderSubscriber:
    """Create a LeRobotRecorderSubscriber for a new dataset.

    Args:
        repo_id: HuggingFace dataset repo id (e.g. "Xense/my_dataset").
        task: Language description of the task being recorded.
        fps: Dataset frame rate. Should match the inference runtime_hz.
        root: Local root directory. Defaults to ~/.cache/huggingface/lerobot/<repo_id>.
        image_height: Raw image height in pixels (default 480).
        image_width: Raw image width in pixels (default 640).
        use_videos: Encode images as video (True) or individual frames (False).
        image_writer_threads: Async image writer thread count.

    Returns:
        A configured LeRobotRecorderSubscriber ready to attach to Runtime.
    """
    features = make_bi_flexiv_dataset_features(image_height, image_width, use_videos)
    dataset = _create_dataset(repo_id, fps, root, features, use_videos, image_writer_threads)
    return LeRobotRecorderSubscriber(dataset=dataset, task=task)
