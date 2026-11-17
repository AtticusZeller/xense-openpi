"""Dataset copies with RECAP columns, on small real LeRobot datasets written to a temporary directory."""

import json
import pathlib

from lerobot.datasets import dataset_tools
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pytest

from openpi.recap import advantage_labels as _advantage_labels
from openpi.recap import collection_columns as _collection_columns
from openpi.recap import dataset_edit as _dataset_edit

_CAMERA = "observation.images.head"
# A stream only the demonstrations hold, as the tactile cameras of the bi_flexiv demonstrations.
_TACTILE = "observation.images.left_tactile_left"
_TACTILE_FEATURE = {"dtype": "video", "shape": (40, 72, 3), "names": ["height", "width", "channels"]}
_BASE_FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    _CAMERA: {"dtype": "video", "shape": (48, 64, 3), "names": ["height", "width", "channels"]},
}


def _write_dataset(
    root: pathlib.Path, lengths: list[int], *, offset: float, collection: dict | None = None, tactile: bool = False
) -> None:
    """Writes episodes whose state, action and image brightness all encode `offset` and the frame."""
    features = _BASE_FEATURES | (_collection_columns.COLLECTION_FEATURES if collection else {})
    if tactile:
        features = features | {_TACTILE: _TACTILE_FEATURE}
    dataset = LeRobotDataset.create(repo_id=f"local/{root.name}", fps=30, features=features, root=root)
    for episode, length in enumerate(lengths):
        for step in range(length):
            value = offset + 10 * episode + step
            frame = {
                "observation.state": np.array([value, -value], np.float32),
                "action": np.array([value + 0.5, value - 0.5], np.float32),
                _CAMERA: np.full((48, 64, 3), int(value) % 256, np.uint8),
                "task": "stack the cylinders",
            }
            if tactile:
                frame[_TACTILE] = np.zeros((40, 72, 3), np.uint8)
            if collection:
                frame["is_intervention"] = np.array([step in collection["intervention"][episode]])
                frame["episode_success"] = np.array([collection["success"][episode]])
                frame["round"] = np.array([1], np.int64)
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()


def _columns(root: pathlib.Path) -> dict[str, np.ndarray]:
    table = LeRobotDataset(f"local/{root.name}", root=root).hf_dataset.with_format("numpy")
    return {name: np.asarray(values) for name, values in table[:].items()}


def _image(root: pathlib.Path, index: int) -> np.ndarray:
    return LeRobotDataset(f"local/{root.name}", root=root, video_backend="pyav")[index][_CAMERA].numpy()


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    base = tmp_path_factory.mktemp("recap_datasets")
    demos, marked, rollouts, merged = (base / name for name in ("demos", "demos_marked", "rollouts", "merged"))
    _write_dataset(demos, [5, 7], offset=0.0, tactile=True)
    _write_dataset(
        rollouts, [6, 4], offset=100.0, collection={"intervention": [{2, 3}, set()], "success": [True, False]}
    )
    _dataset_edit.mark_initial_demonstrations(demos, marked, remove_features=[_TACTILE])
    dataset_tools.merge_datasets(
        [LeRobotDataset(f"local/{root.name}", root=root) for root in (marked, rollouts)],
        output_repo_id="local/merged",
        output_dir=merged,
    )
    return {"demos": demos, "marked": marked, "rollouts": rollouts, "merged": merged}


def _features(root: pathlib.Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())["features"]


def test_marked_demonstrations_have_exactly_the_features_of_the_rollouts(roots):
    assert _TACTILE in _features(roots["demos"])
    assert _features(roots["marked"]) == _features(roots["rollouts"])
    assert not (roots["marked"] / "videos" / _TACTILE).exists()
    assert (roots["marked"] / "videos" / _CAMERA).exists()


def test_demonstrations_with_a_stream_the_rollouts_lack_are_refused_by_the_merge(roots, tmp_path):
    _dataset_edit.mark_initial_demonstrations(roots["demos"], tmp_path / "marked_with_tactile")
    datasets = [
        LeRobotDataset(f"local/{root.name}", root=root)
        for root in (tmp_path / "marked_with_tactile", roots["rollouts"])
    ]
    with pytest.raises(ValueError, match="features"):
        dataset_tools.merge_datasets(datasets, output_repo_id="local/refused", output_dir=tmp_path / "refused")


def test_marked_demonstrations_gain_the_three_columns_and_keep_everything_else(roots):
    source, marked = _columns(roots["demos"]), _columns(roots["marked"])
    assert set(marked) - set(source) == set(_collection_columns.COLUMNS)
    for name, values in source.items():
        np.testing.assert_array_equal(marked[name], values)
    assert not marked["is_intervention"].any()
    assert marked["episode_success"].all()
    assert (marked["round"] == 0).all()
    assert marked["is_intervention"].dtype == np.bool_
    assert marked["round"].dtype == np.int64


def test_marking_leaves_the_source_untouched_and_refuses_an_existing_destination(roots):
    assert set(_collection_columns.COLUMNS).isdisjoint(_columns(roots["demos"]))
    with pytest.raises(FileExistsError):
        _dataset_edit.mark_initial_demonstrations(roots["demos"], roots["marked"])


def test_demonstrations_can_be_labeled_only_once_they_are_marked(roots):
    with pytest.raises(ValueError, match="lacks the collection columns"):
        _collection_columns.load_frames(roots["demos"])
    marked = _collection_columns.load_frames(roots["marked"])
    assert len(marked) == 12
    assert (marked["round"] == 0).all()


def test_merged_dataset_keeps_every_frame_of_both_sources(roots):
    demos, rollouts, merged = (_columns(roots[name]) for name in ("marked", "rollouts", "merged"))
    split = len(demos["index"])
    assert len(merged["index"]) == split + len(rollouts["index"]) == 22
    for name in ("observation.state", "action", "frame_index", *_collection_columns.COLUMNS):
        np.testing.assert_array_equal(merged[name][:split], demos[name])
        np.testing.assert_array_equal(merged[name][split:], rollouts[name])
    # Episodes of the rollout dataset follow the demonstrations.
    np.testing.assert_array_equal(merged["episode_index"][split:], rollouts["episode_index"] + 2)
    # The demonstrations are round 0 without intervention; the rollout columns arrive unchanged.
    assert (merged["round"][:split] == 0).all()
    assert not merged["is_intervention"][:split].any()
    assert merged["episode_success"][:split].all()
    assert np.flatnonzero(merged["is_intervention"]).tolist() == [split + 2, split + 3]
    assert merged["episode_success"][split:].tolist() == [True] * 6 + [False] * 4


def test_merged_dataset_decodes_the_images_of_both_sources(roots):
    split = len(_columns(roots["marked"])["index"])
    for index in (0, 4, 5, 11):
        np.testing.assert_array_equal(_image(roots["merged"], index), _image(roots["demos"], index))
    for index in (0, 5, 6, 9):
        np.testing.assert_array_equal(_image(roots["merged"], split + index), _image(roots["rollouts"], index))
    # Brightness encodes the frame, so a frame read from the wrong episode or dataset would differ.
    assert abs(float(_image(roots["merged"], split + 9).mean()) * 255 - (100 + 10 + 3)) < 3


def _labels(root: pathlib.Path, directory: pathlib.Path) -> pathlib.Path:
    """Writes a labeling attempt for the dataset at `root` whose labels alternate from frame to frame."""
    frames = _collection_columns.load_frames(root)
    labels = frames[["episode_index", "frame_index"]].assign(positive=(np.arange(len(frames)) % 3 == 0))
    # Shuffled: the write-back must match frames by their keys, not by row position.
    labels = labels.sample(frac=1.0, random_state=0, ignore_index=True)
    path = directory / "labels.parquet"
    _advantage_labels.write_labels(path, labels, {"frames": len(labels), "threshold_frames": "rollout"})
    return path


def test_written_labels_match_the_attempt_frame_by_frame_and_nothing_else_changes(roots, tmp_path):
    labeled = tmp_path / "labeled"
    summary = _dataset_edit.write_advantage_labels(roots["merged"], _labels(roots["merged"], tmp_path), labeled)

    source, copy = _columns(roots["merged"]), _columns(labeled)
    assert set(copy) - set(source) == {_advantage_labels.COLUMN}
    for name, values in source.items():
        np.testing.assert_array_equal(copy[name], values)
    assert copy[_advantage_labels.COLUMN].dtype == np.bool_
    np.testing.assert_array_equal(copy[_advantage_labels.COLUMN], np.arange(22) % 3 == 0)
    assert summary == json.loads((labeled / "meta" / "advantage_labels.json").read_text())
    np.testing.assert_array_equal(_image(labeled, 15), _image(roots["merged"], 15))


def test_labels_missing_a_frame_are_rejected_and_no_copy_is_written(roots, tmp_path):
    path = _labels(roots["merged"], tmp_path)
    labels, summary = _advantage_labels.read_labels(path)
    _advantage_labels.write_labels(path, labels.iloc[1:], summary)
    with pytest.raises(ValueError, match="hold 21 frames, the dataset 22"):
        _dataset_edit.write_advantage_labels(roots["merged"], path, tmp_path / "labeled")
    assert not (tmp_path / "labeled").exists()
