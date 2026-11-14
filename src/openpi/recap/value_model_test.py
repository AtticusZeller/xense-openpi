import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.recap import value_model as _value_model


class _FakeVision(nn.Module):
    @nn.compact
    def __call__(self, image, *, train=False):
        batch = image.shape[0]
        patches = image.reshape(batch, 4, -1)
        return nn.Dense(6)(patches), {}


class _FakeLanguage(nn.Module):
    width: int = 8

    def setup(self):
        self.embedding = nn.Embed(32, self.width)
        self.mix = nn.Dense(self.width)

    def embed(self, tokens):
        return self.embedding(tokens)

    def encode(self, embeddings, mask):
        # Causal mean over the real tokens seen so far.
        weights = jnp.tril(jnp.ones((mask.shape[1], mask.shape[1]))) * mask[:, None, :]
        pooled = jnp.einsum("bij,bjd->bid", weights, embeddings) / jnp.maximum(weights.sum(-1, keepdims=True), 1)
        return jnp.tanh(self.mix(pooled))


def _model() -> _value_model.ValueModel:
    return _value_model.ValueModel(vision=_FakeVision(), language=_FakeLanguage(), width=8)


def _batch(batch: int = 4):
    rng = np.random.default_rng(0)
    images = jnp.asarray(rng.uniform(-1, 1, (batch, 3, 4, 4, 3)), dtype=jnp.float32)
    tokens = jnp.asarray(rng.integers(0, 32, (batch, 5)))
    mask = jnp.asarray([[True, True, True, False, False]] * batch)
    targets = jnp.asarray(rng.uniform(-0.5, 0, batch), dtype=jnp.float32)
    return images, tokens, mask, targets


def test_two_hot_preserves_the_expectation():
    values = jnp.asarray([-1.0, -0.5, -0.4975, -0.123456, -0.0025, 0.0])
    probs = _value_model.two_hot(values)
    centers = jnp.linspace(-1.0, 0.0, 201)

    assert probs.shape == (6, 201)
    np.testing.assert_allclose(probs.sum(-1), 1.0, atol=1e-6)
    np.testing.assert_allclose((probs * centers).sum(-1), values, atol=1e-6)
    assert int((probs > 0).sum(-1).max()) <= 2
    # -0.4975 lies halfway between the bins at -0.5 and -0.495.
    np.testing.assert_allclose(probs[2, 100:102], [0.5, 0.5], atol=1e-4)


def test_two_hot_clips_out_of_range_values():
    probs = _value_model.two_hot(jnp.asarray([-3.0, 2.0]))

    assert float(probs[0, 0]) == 1.0
    assert float(probs[1, -1]) == 1.0


def test_expected_value_inverts_two_hot():
    values = jnp.asarray([-0.9, -0.31, 0.0])
    logits = jnp.log(_value_model.two_hot(values) + 1e-30)

    np.testing.assert_allclose(_value_model.expected_value(logits), values, atol=1e-5)


def test_value_loss_is_minimal_at_the_target():
    targets = jnp.asarray([-0.31])
    at_target = jnp.log(_value_model.two_hot(targets) + 1e-30)
    elsewhere = jnp.log(_value_model.two_hot(jnp.asarray([-0.7])) + 1e-30)

    assert float(_value_model.value_loss(at_target, targets)[0]) < float(_value_model.value_loss(elsewhere, targets)[0])


def test_model_output_shape_and_padding_invariance():
    model = _model()
    images, tokens, mask, _ = _batch()
    params = model.init(jax.random.key(0), images, tokens, mask)

    logits = model.apply(params, images, tokens, mask)
    assert logits.shape == (4, 201)

    # Tokens under the padding mask must not change the output.
    changed = tokens.at[:, 3:].set(7)
    np.testing.assert_allclose(model.apply(params, images, changed, mask), logits, atol=1e-6)


def test_one_training_step_reduces_the_loss():
    model = _model()
    images, tokens, mask, targets = _batch()
    params = model.init(jax.random.key(0), images, tokens, mask)
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)

    def loss_fn(p):
        return _value_model.value_loss(model.apply(p, images, tokens, mask), targets).mean()

    before, grads = jax.value_and_grad(loss_fn)(params)
    updates, _ = optimizer.update(grads, opt_state)
    after = loss_fn(optax.apply_updates(params, updates))

    assert float(after) < float(before)
