import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from evorl.algorithms.mo_td3 import (
    add_target_policy_smoothing,
    make_mo_td3_agent,
    scalarize,
    select_pessimistic_q_vector,
    twin_smooth_l1_loss,
    vector_bellman_target,
)
from evorl.algorithms.td3 import make_mlp_td3_agent
from evorl.distributed.gradients import agent_gradient_update
from evorl.envs import Box, create_wrapped_brax_env
from evorl.replay_buffers import ReplayBuffer
from evorl.rollout import rollout
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.utils.pd_morl_interpolator import fit_interpolator_state
from evorl.utils.rl_toolkits import (
    flatten_rollout_trajectory,
    soft_target_update,
)

OBS_SPACE = Box(low=-jnp.ones(17), high=jnp.ones(17))
ACTION_SPACE = Box(low=-jnp.ones(6), high=jnp.ones(6))
# Deterministic test fixture; these are not official Walker key solutions.
INTERPOLATOR_STATE = fit_interpolator_state(
    [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]],
    [[120.0, 900.0], [550.0, 620.0], [980.0, 140.0]],
    "initial",
)


def _batch(batch_size=4):
    preference = jnp.tile(jnp.array([[0.25, 0.75]]), (batch_size, 1))
    return SampleBatch(
        obs=jnp.zeros((batch_size, 17)),
        actions=jnp.zeros((batch_size, 6)),
        rewards=jnp.ones((batch_size, 2)),
        extras=PyTreeDict(
            policy_extras=PyTreeDict(preference=preference),
            env_extras=PyTreeDict(
                ori_obs=jnp.ones((batch_size, 17)),
                termination=jnp.zeros(batch_size),
                truncation=jnp.zeros(batch_size),
            ),
        ),
    )


def _tree_changed(before, after):
    return any(
        bool(jnp.any(x != y))
        for x, y in zip(jtu.tree_leaves(before), jtu.tree_leaves(after))
    )


def test_mo_actor_critic_shapes_architecture_and_independence():
    agent = make_mo_td3_agent(
        ACTION_SPACE, interpolator_state=INTERPOLATOR_STATE
    )
    state = agent.init(OBS_SPACE, ACTION_SPACE, jax.random.PRNGKey(0))
    batch = _batch()
    preference = batch.extras.policy_extras.preference

    actions = jax.jit(agent.actor_network.apply)(
        state.params.actor_params, batch.obs, preference
    )
    q_values = jax.jit(agent.critic_network.apply)(
        state.params.critic_params, batch.obs, preference, actions
    )

    assert actions.shape == (4, 6)
    assert q_values.shape == (4, 2, 2)
    assert jnp.all(actions >= -1) and jnp.all(actions <= 1)

    actor_params = state.params.actor_params["params"]["actor"]
    assert actor_params["hidden_0"]["kernel"].shape == (19, 400)
    assert actor_params["hidden_1"]["kernel"].shape == (400, 400)
    assert actor_params["hidden_2"]["kernel"].shape == (400, 6)
    critic_params = state.params.critic_params["params"]
    for critic_id in (0, 1):
        params = critic_params[f"critic_{critic_id}"]
        assert params["hidden_0"]["kernel"].shape == (25, 400)
        assert params["hidden_1"]["kernel"].shape == (400, 400)
        assert params["hidden_2"]["kernel"].shape == (400, 2)
        assert all(jnp.all(layer["bias"] == 0) for layer in params.values())
    assert _tree_changed(critic_params["critic_0"], critic_params["critic_1"])


def test_scalarization_and_whole_vector_target_selection():
    q_values = jnp.array([[1.0, 10.0], [5.0, 2.0]])
    preferences = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    assert jnp.array_equal(scalarize(q_values, preferences), jnp.array([1.0, 2.0]))

    twin_q = jnp.array(
        [
            [[1.0, 10.0], [5.0, 2.0]],
            [[1.0, 10.0], [5.0, 2.0]],
        ]
    )
    scalarized = scalarize(twin_q, preferences)
    selected = select_pessimistic_q_vector(twin_q, preferences)

    assert scalarized.shape == (2, 2)
    assert jnp.array_equal(selected[0], jnp.array([1.0, 10.0]))
    assert jnp.array_equal(selected[1], jnp.array([5.0, 2.0]))
    assert not jnp.any(jnp.all(selected == jnp.array([1.0, 2.0]), axis=-1))


def test_vector_target_done_mask_smoothing_and_losses_jit():
    rewards = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    next_q = jnp.array([[10.0, 20.0], [30.0, 40.0]])
    target = jax.jit(vector_bellman_target, static_argnums=3)(
        rewards, jnp.array([1.0, 0.0]), next_q, 0.5
    )
    assert target.shape == (2, 2)
    assert jnp.array_equal(target[0], rewards[0])
    assert jnp.array_equal(target[1], jnp.array([18.0, 24.0]))
    assert twin_smooth_l1_loss(jnp.zeros((1, 2, 2)), jnp.ones((1, 2))) == 1

    smoothed = jax.jit(add_target_policy_smoothing, static_argnums=(2, 3))(
        jnp.zeros((8, 6)), jax.random.PRNGKey(1), 0.2, 0.5
    )
    assert smoothed.shape == (8, 6)
    assert jnp.all(smoothed >= -1) and jnp.all(smoothed <= 1)

    agent = make_mo_td3_agent(
        ACTION_SPACE, interpolator_state=INTERPOLATOR_STATE
    )
    state = agent.init(OBS_SPACE, ACTION_SPACE, jax.random.PRNGKey(2))
    batch = _batch(2)
    batch = batch.replace(
        extras=batch.extras.replace(
            env_extras=batch.extras.env_extras.replace(
                termination=jnp.array([1.0, 0.0]),
                truncation=jnp.array([0.0, 1.0]),
            )
        )
    )
    critic_loss = jax.jit(agent.critic_loss)(state, batch, jax.random.PRNGKey(3))
    actor_loss = jax.jit(agent.actor_loss)(state, batch, jax.random.PRNGKey(4))
    assert critic_loss.q_target.shape == (2, 2)
    assert jnp.array_equal(critic_loss.q_target, batch.rewards)
    assert jnp.isfinite(critic_loss.critic_loss)
    assert jnp.isfinite(actor_loss.actor_loss)


def test_scalar_td3_networks_remain_scalar():
    agent = make_mlp_td3_agent(ACTION_SPACE)
    state = agent.init(OBS_SPACE, ACTION_SPACE, jax.random.PRNGKey(5))
    obs = jnp.zeros((3, 17))
    actions = agent.actor_network.apply(state.params.actor_params, obs)
    q_values = agent.critic_network.apply(state.params.critic_params, obs, actions)
    assert actions.shape == (3, 6)
    assert q_values.shape == (3, 2)


def test_real_walker_replay_and_delayed_updates_on_gpu():
    assert jax.default_backend() == "gpu"
    assert jax.devices()[0].platform == "gpu"
    assert "CudaDevice" in repr(jax.devices()[0])

    env = create_wrapped_brax_env(
        "walker2d",
        episode_length=2,
        parallel=10,
        vector_reward=True,
        episode_preference=True,
        process_count=10,
        record_ori_obs=True,
    )
    agent = make_mo_td3_agent(
        env.action_space, interpolator_state=INTERPOLATOR_STATE
    )
    agent_state = agent.init(env.obs_space, env.action_space, jax.random.PRNGKey(6))
    env_state = env.reset(jax.random.PRNGKey(7))
    trajectory, _ = jax.jit(
        lambda state: rollout(
            env.step,
            agent.compute_actions,
            state,
            agent_state,
            jax.random.PRNGKey(8),
            2,
            env_extra_fields=("ori_obs", "termination", "truncation"),
        )
    )(env_state)
    flat = flatten_rollout_trajectory(trajectory).replace(
        next_obs=None, dones=None
    )
    replay = ReplayBuffer(capacity=64, sample_batch_size=10)
    replay_state = replay.init(flat.take(0))
    replay_state = jax.jit(replay.add)(replay_state, flat)
    sample = jax.jit(replay.sample)(replay_state, jax.random.PRNGKey(9))

    optimizer = optax.chain(optax.clip_by_global_norm(100), optax.adam(3e-4))
    critic_opt_state = optimizer.init(agent_state.params.critic_params)
    actor_opt_state = optimizer.init(agent_state.params.actor_params)

    critic_update = agent_gradient_update(
        lambda state, batch, key: (
            agent.critic_loss(state, batch, key).critic_loss,
            agent.critic_loss(state, batch, key),
        ),
        optimizer,
        has_aux=True,
        attach_fn=lambda state, params: state.replace(
            params=state.params.replace(critic_params=params)
        ),
        detach_fn=lambda state: state.params.critic_params,
    )
    actor_update = agent_gradient_update(
        lambda state, batch, key: (
            agent.actor_loss(state, batch, key).actor_loss,
            agent.actor_loss(state, batch, key),
        ),
        optimizer,
        has_aux=True,
        attach_fn=lambda state, params: state.replace(
            params=state.params.replace(actor_params=params)
        ),
        detach_fn=lambda state: state.params.actor_params,
    )
    critic_update = jax.jit(critic_update)
    actor_update = jax.jit(actor_update)

    initial_actor = agent_state.params.actor_params
    initial_critic = agent_state.params.critic_params
    (critic_loss, _), agent_state, critic_opt_state = critic_update(
        critic_opt_state, agent_state, sample, jax.random.PRNGKey(10)
    )
    assert _tree_changed(initial_critic, agent_state.params.critic_params)
    assert not _tree_changed(initial_actor, agent_state.params.actor_params)

    (critic_loss_2, _), agent_state, critic_opt_state = critic_update(
        critic_opt_state, agent_state, sample, jax.random.PRNGKey(11)
    )
    (actor_loss, _), agent_state, actor_opt_state = actor_update(
        actor_opt_state, agent_state, sample, jax.random.PRNGKey(12)
    )
    assert _tree_changed(initial_actor, agent_state.params.actor_params)

    old_target_actor = agent_state.params.target_actor_params
    old_target_critic = agent_state.params.target_critic_params
    target_actor = soft_target_update(
        old_target_actor, agent_state.params.actor_params, 0.005
    )
    target_critic = soft_target_update(
        old_target_critic, agent_state.params.critic_params, 0.005
    )
    assert _tree_changed(old_target_actor, target_actor)
    assert _tree_changed(old_target_critic, target_critic)
    assert jnp.isfinite(critic_loss)
    assert jnp.isfinite(critic_loss_2)
    assert jnp.isfinite(actor_loss)
    assert sample.rewards.shape == (10, 2)
    assert sample.extras.policy_extras.preference.shape == (10, 2)
