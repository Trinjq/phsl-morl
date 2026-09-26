from pathlib import Path

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from evorl.algorithms.mo_td3 import (
    make_mo_td3_agent,
    parallel_actor_update_mask,
    pd_morl_actor_loss,
    pd_morl_critic_loss,
)
from evorl.distributed.gradients import agent_gradient_update
from evorl.envs import Box
from evorl.replay_buffers import ReplayBuffer
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.utils.morl_math import directional_angle, scalarize
from evorl.utils.pd_morl_interpolator import (
    fit_interpolator_state,
    interpolate,
)
from omegaconf import OmegaConf

# Deterministic test fixture; these are not official Walker key solutions.
INTERPOLATOR_STATE = fit_interpolator_state(
    [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]],
    [[120.0, 900.0], [550.0, 620.0], [980.0, 140.0]],
    "initial",
)


def _all_finite(tree):
    return all(bool(jnp.all(jnp.isfinite(x))) for x in jtu.tree_leaves(tree))


def _tree_changed(before, after):
    return any(
        bool(jnp.any(x != y))
        for x, y in zip(jtu.tree_leaves(before), jtu.tree_leaves(after))
    )


def _batch(batch_size=4):
    preference = jnp.array(
        [[0.1, 0.9], [0.3, 0.7], [0.6, 0.4], [0.9, 0.1]],
        dtype=jnp.float32,
    )[:batch_size]
    return SampleBatch(
        obs=jnp.arange(batch_size * 3, dtype=jnp.float32).reshape(batch_size, 3) / 10,
        actions=jnp.zeros((batch_size, 2), dtype=jnp.float32),
        rewards=jnp.ones((batch_size, 2), dtype=jnp.float32),
        extras=PyTreeDict(
            policy_extras=PyTreeDict(preference=preference),
            env_extras=PyTreeDict(
                ori_obs=jnp.ones((batch_size, 3), dtype=jnp.float32),
                termination=jnp.zeros(batch_size, dtype=jnp.float32),
                truncation=jnp.zeros(batch_size, dtype=jnp.float32),
            ),
        ),
    )


def test_directional_angle_boundaries_batch_jit_vmap_float32():
    wp = jnp.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 2.0]],
        dtype=jnp.float32,
    )
    q = jnp.array(
        [[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [20.0, 40.0]],
        dtype=jnp.float32,
    )
    eager = directional_angle(wp, q)
    compiled = jax.jit(directional_angle)(wp, q)
    vmapped = jax.vmap(directional_angle)(wp, q)

    assert eager.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(eager))
    assert jnp.allclose(eager, compiled)
    assert jnp.allclose(eager, vmapped)
    assert jnp.isclose(eager[0], 0.810291, atol=1e-3)
    assert jnp.allclose(eager[1:3], 90.0)
    assert jnp.isclose(eager[3], eager[0], atol=1e-4)


def test_critic_loss_matches_official_terms_without_actor_coefficient():
    wp = jnp.array([[0.8, 0.2], [0.3, 0.7]])
    q = jnp.array([[[1.0, 0.5], [0.2, 1.4]], [[0.4, 1.1], [1.3, 0.1]]])
    target = jnp.array([[0.6, 0.9], [0.7, 0.8]])
    smooth_l1 = optax.huber_loss(q, target[:, None, :]).mean(axis=(0, 2)).sum()
    angles = directional_angle(wp[:, None, :], q).mean(axis=0).sum()
    expected = smooth_l1 + angles

    actual = jax.jit(pd_morl_critic_loss)(q, target, wp)
    assert jnp.allclose(actual, expected)
    assert not jnp.allclose(actual, smooth_l1 + 10 * angles)


def test_actor_loss_sign_roles_and_coefficient_from_config():
    config = OmegaConf.load(
        Path(__file__).parents[1] / "configs" / "agent" / "mo-td3.yaml"
    )
    w = jnp.array([[0.2, 0.8], [0.7, 0.3]])
    wp = jnp.array([[0.4, 0.6], [0.8, 0.2]])
    q1 = jnp.array([[1.5, 0.5], [0.3, 1.7]])
    angle_mean = directional_angle(wp, q1).mean()
    scalar_term = -scalarize(q1, w).mean()

    loss_10 = jax.jit(pd_morl_actor_loss, static_argnums=3)(q1, w, wp, 10.0)
    loss_3 = pd_morl_actor_loss(q1, w, wp, 3.0)
    assert config.actor_loss_coeff == 10
    assert jnp.allclose(loss_10, scalar_term + 10 * angle_mean)
    assert jnp.allclose(loss_10 - loss_3, 7 * angle_mean)


def test_alignment_gradients_finite_at_source_boundaries():
    wp = jnp.array([1.0, 0.0], dtype=jnp.float32)
    q_cases = jnp.array(
        [[0.7, 0.4], [2.0, 0.0], [-2.0, 0.0], [1e-10, -1e-10]],
        dtype=jnp.float32,
    )
    gradients = jax.vmap(jax.grad(lambda q: directional_angle(wp, q)))(q_cases)
    assert jnp.all(jnp.isfinite(gradients))


def test_agent_losses_and_parameter_gradients_jit():
    obs_space = Box(low=-jnp.ones(3), high=jnp.ones(3))
    action_space = Box(low=-jnp.ones(2), high=jnp.ones(2))
    agent = make_mo_td3_agent(
        action_space,
        actor_hidden_layer_sizes=(8,),
        critic_hidden_layer_sizes=(8,),
        actor_loss_coeff=10,
        interpolator_state=INTERPOLATOR_STATE,
    )
    state = agent.init(obs_space, action_space, jax.random.PRNGKey(0))
    batch = _batch()
    replay = ReplayBuffer(capacity=8, sample_batch_size=4)
    replay_state = jax.jit(replay.add)(replay.init(batch.take(0)), batch)
    batch = jax.jit(replay.sample)(replay_state, jax.random.PRNGKey(4))

    def critic_loss(params):
        critic_state = state.replace(params=state.params.replace(critic_params=params))
        return agent.critic_loss(critic_state, batch, jax.random.PRNGKey(1)).critic_loss

    def actor_loss(params):
        actor_state = state.replace(params=state.params.replace(actor_params=params))
        return agent.actor_loss(actor_state, batch, jax.random.PRNGKey(2)).actor_loss

    critic_value, critic_grad = jax.jit(jax.value_and_grad(critic_loss))(
        state.params.critic_params
    )
    actor_value, actor_grad = jax.jit(jax.value_and_grad(actor_loss))(
        state.params.actor_params
    )
    wp = jax.jit(interpolate)(
        state.extra_state.interpolator,
        batch.extras.policy_extras.preference,
    )

    assert wp.shape == (4, 2)
    assert jnp.isfinite(critic_value) and jnp.isfinite(actor_value)
    assert _all_finite(critic_grad)
    assert _all_finite(actor_grad)

    optimizer = optax.chain(optax.clip_by_global_norm(100), optax.adam(3e-4))
    critic_update = jax.jit(
        agent_gradient_update(
            lambda current, sample, key: (
                agent.critic_loss(current, sample, key).critic_loss
            ),
            optimizer,
            attach_fn=lambda current, params: current.replace(
                params=current.params.replace(critic_params=params)
            ),
            detach_fn=lambda current: current.params.critic_params,
        )
    )
    actor_update = jax.jit(
        agent_gradient_update(
            lambda current, sample, key: (
                agent.actor_loss(current, sample, key).actor_loss
            ),
            optimizer,
            attach_fn=lambda current, params: current.replace(
                params=current.params.replace(actor_params=params)
            ),
            detach_fn=lambda current: current.params.actor_params,
        )
    )
    initial_actor = state.params.actor_params
    initial_critic = state.params.critic_params
    _, state, _ = critic_update(
        optimizer.init(initial_critic), state, batch, jax.random.PRNGKey(5)
    )
    actor_due = parallel_actor_update_mask(jnp.array(0), 10, 10)[-1]
    assert actor_due
    _, state, _ = actor_update(
        optimizer.init(initial_actor), state, batch, jax.random.PRNGKey(6)
    )
    assert _tree_changed(initial_critic, state.params.critic_params)
    assert _tree_changed(initial_actor, state.params.actor_params)


def test_missing_interpolator_fails_instead_of_using_fake_data():
    agent = make_mo_td3_agent(
        Box(low=-jnp.ones(2), high=jnp.ones(2)),
        actor_hidden_layer_sizes=(8,),
        critic_hidden_layer_sizes=(8,),
    )
    state = agent.init(
        Box(low=-jnp.ones(3), high=jnp.ones(3)),
        Box(low=-jnp.ones(2), high=jnp.ones(2)),
        jax.random.PRNGKey(3),
    )
    try:
        agent.actor_loss(state, _batch(), jax.random.PRNGKey(4))
    except ValueError as error:
        assert "interpolator state" in str(error)
    else:
        raise AssertionError("missing interpolator state was silently accepted")
