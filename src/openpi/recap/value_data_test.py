import numpy as np
import pandas as pd
import pytest
import torch

from openpi.recap import value_data as _value_data


def test_split_episodes_is_disjoint_deterministic_and_holds_out_the_fraction():
    train, val = _value_data.split_episodes(151, 0.1, seed=0)

    assert len(val) == 15
    assert len(train) == 136
    assert not set(train) & set(val)
    np.testing.assert_array_equal(_value_data.split_episodes(151, 0.1, seed=0)[1], val)
    assert len(_value_data.split_episodes(3, 0.1, seed=0)[1]) == 1


def test_frame_indices_follow_episode_ranges():
    starts, lengths = np.array([0, 3, 5]), np.array([3, 2, 4])

    np.testing.assert_array_equal(_value_data.frame_indices(starts, lengths, np.array([0, 2])), [0, 1, 2, 5, 6, 7, 8])


def test_align_returns_orders_rows_by_global_index():
    starts, lengths = np.array([0, 2]), np.array([2, 3])
    table = pd.DataFrame(
        {
            "episode_index": [1, 1, 0, 1, 0],
            "frame_index": [2, 0, 1, 1, 0],
            "return": [-0.1, -0.3, -0.4, -0.2, -0.5],
        }
    )

    np.testing.assert_allclose(_value_data.align_returns(table, starts, lengths), [-0.5, -0.4, -0.3, -0.2, -0.1])


def test_align_returns_rejects_missing_or_duplicated_rows():
    starts, lengths = np.array([0, 2]), np.array([2, 3])
    missing = pd.DataFrame({"episode_index": [0, 0], "frame_index": [0, 1], "return": [0.0, 0.0]})
    duplicated = pd.DataFrame({"episode_index": [0, 0, 1, 1, 1], "frame_index": [0, 0, 0, 1, 2], "return": [0.0] * 5})

    with pytest.raises(ValueError, match="rows but the dataset has"):
        _value_data.align_returns(missing, starts, lengths)
    with pytest.raises(ValueError, match="duplicated"):
        _value_data.align_returns(duplicated, starts, lengths)


def test_tokenize_tasks_pads_and_rejects_long_text():
    encode = lambda text: [2, *range(10, 10 + len(text.split()))]  # noqa: E731

    tokens, mask = _value_data.tokenize_tasks(encode, {0: "stack the cylinders"}, max_len=6)[0]

    np.testing.assert_array_equal(tokens, [2, 10, 11, 12, 0, 0])
    np.testing.assert_array_equal(mask, [True, True, True, True, False, False])
    with pytest.raises(ValueError, match="more than max_len"):
        _value_data.tokenize_tasks(encode, {0: "a b c d e f g"}, max_len=6)


def test_resize_with_pad_keeps_aspect_and_pads_with_black():
    image = torch.ones(3, 480, 640)

    out = _value_data.resize_with_pad(image)

    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8
    # 640x480 scales to 224x168, leaving 28 black rows above and below.
    assert (out[:28] == 0).all()
    assert (out[-28:] == 0).all()
    assert (out[28:196] == 255).all()


def test_value_frames_pairs_frames_with_returns():
    class _Frames(torch.utils.data.Dataset):
        def __getitem__(self, index):
            image = torch.full((3, 8, 8), index / 10)
            return {"cam_a": image, "cam_b": image, "task_index": torch.tensor(1)}

        def __len__(self):
            return 10

    returns = np.linspace(-0.9, 0.0, 10, dtype=np.float32)
    task_tokens = {1: (np.array([2, 5, 0], dtype=np.int32), np.array([True, True, False]))}
    frames = _value_data.ValueFrames(_Frames(), np.array([7, 3]), returns, task_tokens, camera_keys=("cam_a", "cam_b"))

    sample = frames[0]

    assert len(frames) == 2
    assert sample["images"].shape == (2, 224, 224, 3)
    assert sample["index"] == 7
    assert sample["target"] == returns[7]
    np.testing.assert_array_equal(sample["tokens"], [2, 5, 0])
    assert int(sample["images"][0, 0, 0, 0]) == round(0.7 * 255)
