import jax
import jax.numpy as jnp

from evorl.envs import create_brax_env, create_wrapped_brax_env
from evorl.replay_buffers import ReplayBuffer
from evorl.rollout import rollout
from evorl.types import PyTreeDict
from evorl.utils.rl_toolkits import flatten_rollout_trajectory


def _zero_actions(agent_state, sample_batch, key):
    del agent_state, key
    return jnp.zeros((*sample_batch.obs.shape[:-1], 6)), PyTreeDict()


def test_mo_walker2d_single_transition():
    env = create_brax_env("walker2d", vector_reward=True)
    state = env.reset(jax.random.PRNGKey(0))
    action = jnp.array([-2.0, -0.5, 0.0, 0.5, 1.0, 2.0])

    next_state = jax.jit(env.step)(state, action)
    expected = jnp.array(
        [
            next_state.info.metrics.x_velocity + 1.0,
            5.0 - jnp.sum(jnp.square(jnp.clip(action, -1.0, 1.0))),
        ]
    )

    assert next_state.reward.shape == (2,)
    assert jnp.allclose(next_state.reward, expected)
    assert jnp.isfinite(next_state.reward).all()


def test_mo_walker2d_batched_rollout_and_replay():
    batch_size = 3
    rollout_length = 4
    env = create_wrapped_brax_env(
        "walker2d", parallel=batch_size, vector_reward=True
    )
    state = env.reset(jax.random.PRNGKey(1))
    actions = jnp.array(
        [
            [0.0] * 6,
            [0.5] * 6,
            [1.0] * 6,
        ]
    )

    next_state = jax.jit(env.step)(state, actions)
    expected_energy = 5.0 - jnp.sum(jnp.square(actions), axis=-1)
    assert next_state.reward.shape == (batch_size, 2)
    assert jnp.allclose(next_state.reward[:, 1], expected_energy)
    assert jnp.unique(next_state.reward[:, 1]).shape[0] == batch_size

    trajectory, _ = jax.jit(
        lambda env_state: rollout(
            env.step,
            _zero_actions,
            env_state,
            None,
            jax.random.PRNGKey(2),
            rollout_length,
        )
    )(state)
    assert trajectory.rewards.shape == (rollout_length, batch_size, 2)

    flat_trajectory = flatten_rollout_trajectory(trajectory)
    replay = ReplayBuffer(capacity=32, sample_batch_size=5)
    replay_state = replay.init(flat_trajectory.take(0))
    replay_state = jax.jit(replay.add)(replay_state, flat_trajectory)
    sample = jax.jit(replay.sample)(replay_state, jax.random.PRNGKey(3))
    assert flat_trajectory.rewards.shape == (rollout_length * batch_size, 2)
    assert sample.rewards.shape == (5, 2)


def test_scalar_walker2d_reward_regression():
    env = create_wrapped_brax_env("walker2d", parallel=2)
    state = env.reset(jax.random.PRNGKey(4))
    next_state = jax.jit(env.step)(state, jnp.zeros((2, 6)))

    assert state.reward.shape == (2,)
    assert next_state.reward.shape == (2,)


if __name__ == "__main__":
    print("backend:", jax.default_backend())
    print("devices:", jax.devices())
    test_mo_walker2d_single_transition()
    test_mo_walker2d_batched_rollout_and_replay()
    test_scalar_walker2d_reward_regression()
    print("all vector-reward checks passed")
