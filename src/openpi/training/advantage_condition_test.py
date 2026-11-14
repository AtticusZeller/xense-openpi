"""Tests of the RECAP advantage text condition across the tokenizer, the training chain and the serving chain."""

import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
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
