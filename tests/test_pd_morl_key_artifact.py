from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evorl.types import PyTreeDict
from scripts.train_brax_key_solutions import (
    evaluation_preference,
    expected_update_counts,
    replay_next_obs_and_done,
    sample_training_preference,
)


def test_training_preference_is_nonnegative_and_unit_l1():
    preference = sample_training_preference(
        jax.random.PRNGKey(0), (32,), jnp.array([0.0, 1.0])
    )
    assert jnp.all(preference >= 0)
    np.testing.assert_allclose(preference.sum(axis=-1), 1.0, atol=2e-7)


def test_evaluation_preference_is_the_original_key():
    key = jnp.array([0.5, 0.5])
    np.testing.assert_array_equal(
        evaluation_preference(key, (7,)), jnp.tile(key, (7, 1))
    )


def test_replay_uses_true_next_obs_and_combines_boundaries():
    obs = jnp.zeros((2, 2, 3))
    ori_obs = jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3) + 1
    trajectory = SimpleNamespace(
        obs=obs,
        extras=PyTreeDict(
            env_extras=PyTreeDict(
                ori_obs=ori_obs,
                termination=jnp.array([[0, 1], [0, 0]]),
                truncation=jnp.array([[0, 0], [1, 0]]),
            )
        ),
    )
    next_obs, done = replay_next_obs_and_done(trajectory, obs_dim=3)
    np.testing.assert_array_equal(next_obs, ori_obs.reshape(4, 3))
    np.testing.assert_array_equal(done, jnp.array([0, 1, 1, 0]))


def test_replay_refuses_missing_true_successor():
    trajectory = SimpleNamespace(
        extras=PyTreeDict(
            env_extras=PyTreeDict(
                termination=jnp.zeros((1, 1)), truncation=jnp.zeros((1, 1))
            )
        )
    )
    with pytest.raises(ValueError, match="ori_obs"):
        replay_next_obs_and_done(trajectory, obs_dim=3)


def test_vectorized_budget_and_key_order():
    counts = expected_update_counts()
    assert counts["total_chunks"] == 7813
    assert counts["random_action_chunks"] == 98
    assert counts["critic_optimizer_steps"] == 1_999_929
    assert counts["actor_optimizer_steps"] == 999_964
