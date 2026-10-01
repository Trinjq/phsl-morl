from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from evorl.types import PyTreeDict
from scripts.train_brax_key_solutions import replay_next_obs_and_done


def test_key_replay_uses_true_next_obs_and_combined_done():
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

    assert jnp.array_equal(next_obs, ori_obs.reshape(4, 3))
    assert not jnp.array_equal(next_obs, obs.reshape(4, 3))
    assert jnp.array_equal(done, jnp.array([0, 1, 1, 0]))


def test_key_replay_refuses_missing_successor_observation():
    trajectory = SimpleNamespace(
        extras=PyTreeDict(
            env_extras=PyTreeDict(
                termination=jnp.zeros((1, 1)),
                truncation=jnp.zeros((1, 1)),
            )
        )
    )

    with pytest.raises(ValueError, match="ori_obs"):
        replay_next_obs_and_done(trajectory, obs_dim=3)
