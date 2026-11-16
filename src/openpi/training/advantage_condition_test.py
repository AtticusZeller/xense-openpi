"""Tests of the RECAP advantage text condition across the tokenizer, the training chain and the serving chain."""

import dataclasses
import logging

from lerobot.datasets import lerobot_dataset
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.recap import advantage_labels as _advantage_labels
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.transforms as _transforms

_MAX_LEN = 64
_TASK = "stack the cylinders"
_STATE = np.array([-1.0, 0.0, 0.5], dtype=np.float32)
# The Pi05 prompt before the advantage condition existed; np.digitize maps _STATE to bins 0, 128, 192.
_PI05_PROMPT = "Task: stack the cylinders, State: 0 128 192;\nAction: "


def _decode(tokenizer: _tokenizer.PaligemmaTokenizer, tokens: np.ndarray, mask: np.ndarray) -> str:
    return tokenizer._tokenizer.decode(tokens[mask].tolist())


def _reference_tokens(tokenizer: _tokenizer.PaligemmaTokenizer, text: str) -> list[int]:
    return tokenizer._tokenizer.encode(text, add_bos=True)


class _Samples(_data_loader.Dataset):
    """Returns the same raw sample for every index."""

    def __init__(self, size: int):
        self._size = size

    def __getitem__(self, index) -> dict:
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        return {"prompt": _TASK, "state": _STATE.copy(), "image": {"base_0_rgb": image}}

    def __len__(self) -> int:
        return self._size


def _model_config(*, advantage_conditioned: bool) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True, action_dim=3, max_token_len=_MAX_LEN, advantage_conditioned=advantage_conditioned
    )


def _training_chain(model_config: pi0_config.Pi0Config, size: int) -> _data_loader.Dataset:
    data_config = _config.DataConfig(repo_id="fake", model_transforms=_config.ModelTransformFactory()(model_config))
    return _data_loader.transform_dataset(
        _Samples(size),
        data_config,
        advantage_dropout=_data_loader._advantage_dropout(model_config),
    )


@pytest.mark.parametrize(
    ("advantage", "expected"),
    [
        ("positive", "Task: stack the cylinders, State: 0 128 192;\nAdvantage: positive;\nAction: "),
        ("negative", "Task: stack the cylinders, State: 0 128 192;\nAdvantage: negative;\nAction: "),
        (None, _PI05_PROMPT),
    ],
)
def test_tokenizer_places_advantage_before_action(advantage, expected):
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=_MAX_LEN)

    tokens, mask = tokenizer.tokenize(_TASK, _STATE, advantage)

    assert tokens[mask].tolist() == _reference_tokens(tokenizer, expected)
    assert _decode(tokenizer, tokens, mask) == expected


def test_tokenizer_rejects_unsupported_advantage():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=_MAX_LEN)

    with pytest.raises(ValueError, match="advantage must be one of"):
        tokenizer.tokenize(_TASK, _STATE, "neutral")
    with pytest.raises(ValueError, match="only supported by the Pi05"):
        tokenizer.tokenize(_TASK, None, "positive")


def test_config_requires_discrete_state_input():
    with pytest.raises(ValueError, match="requires discrete_state_input"):
        pi0_config.Pi0Config(pi05=False, advantage_conditioned=True)


def test_switch_off_keeps_the_pi05_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=_MAX_LEN)
    reference = _reference_tokens(tokenizer, _PI05_PROMPT)
    np.random.seed(0)
    dataset = _training_chain(_model_config(advantage_conditioned=False), 200)

    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample["tokenized_prompt"][sample["tokenized_prompt_mask"]].tolist() == reference
        assert "advantage" not in sample


def test_training_chain_drops_the_condition_at_the_configured_rate():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=_MAX_LEN)
    conditioned = _reference_tokens(
        tokenizer, "Task: stack the cylinders, State: 0 128 192;\nAdvantage: positive;\nAction: "
    )
    dropped = _reference_tokens(tokenizer, _PI05_PROMPT)
    model_config = _model_config(advantage_conditioned=True)
    assert model_config.advantage_dropout == 0.3
    np.random.seed(0)
    dataset = _training_chain(model_config, 10_000)

    num_dropped = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        tokens = sample["tokenized_prompt"][sample["tokenized_prompt_mask"]].tolist()
        assert tokens in (conditioned, dropped)
        num_dropped += tokens == dropped

    assert abs(num_dropped / len(dataset) - 0.3) < 0.02


def test_serving_chain_never_drops_the_condition():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=_MAX_LEN)
    conditioned = _reference_tokens(
        tokenizer, "Task: stack the cylinders, State: 0 128 192;\nAdvantage: positive;\nAction: "
    )
    # Serving applies `model_transforms` without the training-only dropout (see policy_config.py).
    serving = _transforms.compose(_config.ModelTransformFactory()(_model_config(advantage_conditioned=True)).inputs)
    samples = _Samples(1)
    np.random.seed(0)

    for _ in range(1000):
        sample = serving(samples[0])
        assert sample["tokenized_prompt"][sample["tokenized_prompt_mask"]].tolist() == conditioned


_CAMERA = "observation.images.head"
# Labels of the frames of the small dataset below; frame 2 is the only positive one of the second episode.
_LABELS = [True, False, False, True, False, False, True, False]


def _write_dataset(root, *, labeled: bool) -> None:
    """Writes two 4-frame episodes; with `labeled`, the dataset carries `_LABELS` as `advantage_positive`."""
    features = {
        "observation.state": {"dtype": "float32", "shape": (3,), "names": ["a", "b", "c"]},
        "action": {"dtype": "float32", "shape": (3,), "names": ["a", "b", "c"]},
        _CAMERA: {"dtype": "video", "shape": (48, 64, 3), "names": ["height", "width", "channels"]},
    }
    if labeled:
        features[_advantage_labels.COLUMN] = _advantage_labels.FEATURE
    dataset = lerobot_dataset.LeRobotDataset.create(repo_id=f"local/{root.name}", fps=30, features=features, root=root)
    for index, positive in enumerate(_LABELS):
        frame = {
            "observation.state": _STATE.copy(),
            "action": np.zeros(3, np.float32),
            _CAMERA: np.zeros((48, 64, 3), np.uint8),
            "task": _TASK,
        }
        if labeled:
            frame[_advantage_labels.COLUMN] = np.array([positive])
        dataset.add_frame(frame)
        if index % 4 == 3:
            dataset.save_episode()
    dataset.finalize()


def _dataset_chain(root, model_config: pi0_config.Pi0Config) -> _data_loader.Dataset:
    """The dataset as `create_torch_data_loader` builds it, on a real LeRobot dataset."""

    def to_model_image(data: dict) -> dict:
        image = (np.asarray(data["image"]["base_0_rgb"]).transpose(1, 2, 0) * 255).astype(np.uint8)
        return {**data, "image": {"base_0_rgb": image}}

    data_config = _config.DataConfig(
        repo_id=f"local/{root.name}",
        dataset_root=str(root),
        video_backend="pyav",
        prompt_from_task=True,
        action_sequence_keys=("action",),
        repack_transforms=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {"image": {"base_0_rgb": _CAMERA}, "state": "observation.state", "prompt": "prompt"}
                )
            ]
        ),
        data_transforms=_transforms.Group(inputs=[to_model_image]),
        model_transforms=_config.ModelTransformFactory()(model_config),
    )
    return _data_loader.transform_dataset(
        _data_loader.create_torch_dataset(data_config, model_config.action_horizon, model_config),
        data_config,
        skip_norm_stats=True,
        advantage_dropout=_data_loader._advantage_dropout(model_config),
        advantage_column=_data_loader._advantage_column(data_config, model_config),
    )


def _prompts(dataset: _data_loader.Dataset, tokenizer: _tokenizer.PaligemmaTokenizer) -> list[str]:
    samples = [dataset[index] for index in range(len(dataset))]
    return [_decode(tokenizer, sample["tokenized_prompt"], sample["tokenized_prompt_mask"]) for sample in samples]


def test_training_takes_the_advantage_label_of_every_frame_from_the_dataset_column(tmp_path, caplog):
    _write_dataset(tmp_path / "labeled", labeled=True)
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=_MAX_LEN)
    model_config = dataclasses.replace(_model_config(advantage_conditioned=True), advantage_dropout=0.0)
    with caplog.at_level(logging.INFO):
        dataset = _dataset_chain(tmp_path / "labeled", model_config)

    expected = [
        _PI05_PROMPT.replace("Action: ", f"Advantage: {'positive' if p else 'negative'};\nAction: ") for p in _LABELS
    ]
    assert _prompts(dataset, tokenizer) == expected
    assert "37.5% of 8 frames positive" in caplog.text


def test_dropped_samples_carry_neither_label(tmp_path):
    _write_dataset(tmp_path / "labeled", labeled=True)
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=_MAX_LEN)
    dataset = _dataset_chain(tmp_path / "labeled", _model_config(advantage_conditioned=True))
    np.random.seed(0)

    prompts = [prompt for _ in range(25) for prompt in _prompts(dataset, tokenizer)]
    labels = _LABELS * 25
    dropped = [prompt == _PI05_PROMPT for prompt in prompts]
    assert 0.2 < np.mean(dropped) < 0.4
    for prompt, positive, was_dropped in zip(prompts, labels, dropped, strict=True):
        if not was_dropped:
            assert f"Advantage: {'positive' if positive else 'negative'};" in prompt
    # Both labels are dropped, not only one of them.
    assert {positive for positive, was_dropped in zip(labels, dropped, strict=True) if was_dropped} == {True, False}


def test_dataset_without_the_column_trains_every_frame_as_positive(tmp_path, caplog):
    _write_dataset(tmp_path / "plain", labeled=False)
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=_MAX_LEN)
    model_config = dataclasses.replace(_model_config(advantage_conditioned=True), advantage_dropout=0.0)
    with caplog.at_level(logging.INFO):
        dataset = _dataset_chain(tmp_path / "plain", model_config)

    positive = _reference_tokens(tokenizer, _PI05_PROMPT.replace("Action: ", "Advantage: positive;\nAction: "))
    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample["tokenized_prompt"][sample["tokenized_prompt_mask"]].tolist() == positive
    assert "every frame is trained as positive" in caplog.text


def test_unconditioned_model_ignores_the_column(tmp_path):
    _write_dataset(tmp_path / "labeled", labeled=True)
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=_MAX_LEN)
    dataset = _dataset_chain(tmp_path / "labeled", _model_config(advantage_conditioned=False))
    assert _prompts(dataset, tokenizer) == [_PI05_PROMPT] * 8
