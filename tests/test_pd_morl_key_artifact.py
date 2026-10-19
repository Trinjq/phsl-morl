from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization

from evorl.types import PyTreeDict
from scripts.train_brax_key_solutions import (
    evaluation_preference,
    expected_update_counts,
    replay_next_obs_and_done,
    sample_training_preference,
    save_key_networks,
    updates_for_chunk,
    write_artifact,
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
    assert counts["chunk_size"] == 64
    assert counts["total_chunks"] == 78125
    assert counts["random_action_chunks"] == 392
    assert counts["initial_no_update_chunks"] == 7
    assert counts["first_learning_chunk_updates"] == 8
    assert counts["critic_optimizer_steps"] == 624_944
    assert counts["actor_optimizer_steps"] == 312_472
    previous = expected_update_counts(2_000_128, 25_088)
    assert previous["critic_optimizer_steps"] == 249_960
    assert previous["actor_optimizer_steps"] == 124_980
    # The first seven blocks only fill replay; the eighth and later use 8/4 updates.
    actual = [updates_for_chunk(i) for i in range(16)]
    assert actual == [0] * 7 + [8] * 9
    assert expected_update_counts(448, 448)["critic_optimizer_steps"] == 0
    assert expected_update_counts(512, 512)["critic_optimizer_steps"] == 8
    smoke = expected_update_counts(1024, 768)
    assert sum(actual) == smoke["critic_optimizer_steps"] == 72
    assert smoke["actor_optimizer_steps"] == 36


def test_artifact_refuses_mixed_sampling_schedules_before_writing(tmp_path):
    entries = [{"seed": i, "preference": w,
                "sampling_schedule": {"num_envs": 64, "rollout_length": 1,
                                      "critic_updates_per_chunk": 8}}
               for i, w in enumerate([[0., 1.], [.5, .5], [1., 0.]])]
    entries[2]["sampling_schedule"]["critic_updates_per_chunk"] = 64
    with pytest.raises(ValueError, match="same sampling schedule"):
        write_artifact(tmp_path, entries)
    assert not list(tmp_path.iterdir())


def test_saved_key_weights_preserve_actor_and_critic_predictions(tmp_path):
    from evorl.algorithms.mo_td3 import PreferenceActor, TwinVectorCritic

    actor, critic = PreferenceActor(action_size=6), TwinVectorCritic()
    obs, w = jnp.linspace(-0.5, 0.5, 17), jnp.array([0.5, 0.5])
    ap = actor.init(jax.random.PRNGKey(1), obs, w)
    action = actor.apply(ap, obs, w)
    cp = critic.init(jax.random.PRNGKey(2), obs, w, action)
    q = critic.apply(cp, obs, w, action)
    path = tmp_path / "best_networks.msgpack"
    save_key_networks(path, (None, None, ap, cp, ap, cp), 1, w, 512, [10., 20.])
    restored = serialization.msgpack_restore(path.read_bytes())
    restored_action = actor.apply(restored["actor_params"], obs, w)
    restored_q = critic.apply(restored["critic_params"], obs, w, restored_action)
    np.testing.assert_array_equal(action, restored_action)
    np.testing.assert_array_equal(q, restored_q)
    assert restored["seed"] == 1 and restored["step"] == 512
    np.testing.assert_array_equal(restored["preference"], w)
