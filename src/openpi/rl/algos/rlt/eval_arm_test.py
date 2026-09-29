import jax.numpy as jnp
import numpy as np
import pytest

from openpi.rl.algos.rlt import collector_test
from openpi.rl.algos.rlt import eval_arm
from openpi.rl.algos.rlt import learner as _learner


def test_rlt_arm_drives_only_open_windows_and_vla_arm_never(tmp_path):
    collector, robot, learner = collector_test._setup([])
    learner.save(tmp_path / "0", {"binding": {}})
    actor, _ = _learner.load_actor(tmp_path / "0", learner.config, learner.space, z_dim=collector_test.Z)
    obs = robot._obs()
    features = collector.extractor.extract(obs)
    collector.extractor.extract = lambda _: features
    rlt = eval_arm.RLTArm(collector.extractor, actor, horizon=collector_test.C)
    vla = eval_arm.RLTArm(collector.extractor, None, horizon=collector_test.C)
    assert (rlt.name, vla.name) == ("rlt", "vla")

    # Inside a window the rlt arm runs the learner's deterministic mean chunk, decoded to absolute actions.
    actions, source = rlt.act(obs, window_open=True)
    expected = learner.space.decode(jnp.asarray(learner.mean(features))[None], jnp.asarray(features["state"])[None])[0]
    np.testing.assert_allclose(actions, expected, atol=1e-6)
    assert source == "actor"

    # Everywhere else both arms run the first C steps of the VLA reference, as training does.
    for arm, window_open in ((rlt, False), (vla, True), (vla, False)):
        actions, source = arm.act(obs, window_open=window_open)
        np.testing.assert_array_equal(actions, features["ref_exec"][: collector_test.C])
        assert source == "vla"


def test_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="Unknown RLT arm"):
        eval_arm.create_arm(None, "expo")
