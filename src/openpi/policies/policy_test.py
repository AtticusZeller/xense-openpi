import numpy as np
import pytest
from xense_client import action_chunk_broker

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def test_policy_input_keeps_state_with_training_dropout(monkeypatch, tmp_path):
    model_config = pi0_config.Pi0Config(pi05=True)
    config = _config.TrainConfig(
        name="state_dropout_policy",
        model=model_config,
        data=_config.DroidInferenceDataConfig(base_config=_config.DataConfig(state_dropout_prob=1.0)),
    )
    monkeypatch.setattr(_model, "restore_params", lambda *args, **kwargs: None)
    monkeypatch.setattr(pi0_config.Pi0Config, "load", lambda self, params: None)
    monkeypatch.setattr(
        _policy_config._policy, "Policy", lambda model, **kwargs: _transforms.compose(kwargs["transforms"])
    )

    preprocess = _policy_config.create_trained_policy(config, tmp_path, norm_stats={})
    example = droid_policy.make_droid_example()
    data = preprocess(example)

    state = np.concatenate([example["observation/joint_position"], example["observation/gripper_position"]])
    tokenizer = _tokenizer.PaligemmaTokenizer(model_config.max_token_len)
    tokens, mask = tokenizer.tokenize(example["prompt"], state)
    np.testing.assert_array_equal(data["tokenized_prompt"], tokens)
    np.testing.assert_array_equal(data["tokenized_prompt_mask"], mask)
    assert "state_mask" not in data


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_droid")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_droid")

    example = droid_policy.make_droid_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 8)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_droid")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_droid")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = droid_policy.make_droid_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (8,)
