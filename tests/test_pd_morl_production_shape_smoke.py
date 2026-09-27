"""GPU/slow/integration gate for the production-shape PD-MORL baseline."""

from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest
from omegaconf import OmegaConf

from evorl.algorithms.mo_td3 import (
    PDMORLWorkflow,
    parallel_actor_update_mask,
)
from evorl.distributed.gradients import agent_gradient_update
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.utils.orbax_utils import load, save
from evorl.utils.pd_morl_interpolator import interpolate, key_preferences
from evorl.utils.rl_toolkits import soft_target_update


def _tree_changed(before, after):
    return any(
        bool(jnp.any(x != y))
        for x, y in zip(jtu.tree_leaves(before), jtu.tree_leaves(after))
    )


def _tree_equal(left, right):
    return all(
        bool(jnp.array_equal(x, y))
        for x, y in zip(jtu.tree_leaves(left), jtu.tree_leaves(right))
    )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.integration
def test_pd_morl_walker_production_shape_single_gpu(tmp_path, request):
    if request.config.getoption("--device") != "gpu":
        pytest.skip("production-shape smoke requires --device gpu")

    assert jax.default_backend() == "gpu"
    assert jax.devices()[0].platform == "gpu"
    root = Path(__file__).parents[1]
    production = OmegaConf.load(root / "configs" / "agent" / "mo-td3.yaml")

    # TEST OVERRIDE: only runtime length and host-trigger timing are shortened.
    config = OmegaConf.merge(
        production,
        {
            "checkpoint": {"enable": False},
            "output_dir": str(tmp_path),
            "env": {
                "env_name": "walker2d",
                "env_type": "brax",
                "max_episode_steps": 2,
            },
            "random_timesteps": 0,
            "learning_start_timesteps": 64,
            "learner_start_threshold": 256,
            "random_action_warmup": 0,
            "her_start_timesteps": 0,
            "eval_freq": 1,
            "eval_episodes": 1,
            "training_eval_preference_step": 0.5,
            "offline_eval_preference_step": 0.5,
        },
    )

    # Prohibited overrides remain byte-for-value identical to production config.
    for name in (
        "num_objectives",
        "process_count",
        "batch_size",
        "replay_capacity",
        "actor_loss_coeff",
        "gamma",
        "tau",
        "policy_freq",
        "exploration_noise",
        "target_policy_noise",
        "noise_clip",
        "gradient_clip_norm",
        "num_relabel_preferences",
        "matmul_precision",
    ):
        assert config[name] == production[name], name
    assert config.agent_network == production.agent_network
    assert config.batch_size == 256
    assert config.process_count == config.num_envs == 10
    assert config.matmul_precision == "highest"

    SmokeWorkflow = type("ProductionShapeSmokeWorkflow", (PDMORLWorkflow,), {})
    workflow = SmokeWorkflow.build_from_config(config, enable_jit=True)
    state = workflow.setup(jax.random.PRNGKey(71))
    assert jax.config.jax_default_matmul_precision == "highest"

    params = state.agent_state.params
    actor = params.actor_params["params"]["actor"]
    critics = params.critic_params["params"]
    expected = {
        "actor layer1": ((19, 400), actor["hidden_0"]["kernel"].shape),
        "actor layer2": ((400, 400), actor["hidden_1"]["kernel"].shape),
        "actor out": ((400, 6), actor["hidden_2"]["kernel"].shape),
        "q1 layer1": ((25, 400), critics["critic_0"]["hidden_0"]["kernel"].shape),
        "q1 layer2": ((400, 400), critics["critic_0"]["hidden_1"]["kernel"].shape),
        "q1 out": ((400, 2), critics["critic_0"]["hidden_2"]["kernel"].shape),
        "q2 layer1": ((25, 400), critics["critic_1"]["hidden_0"]["kernel"].shape),
        "q2 layer2": ((400, 400), critics["critic_1"]["hidden_1"]["kernel"].shape),
        "q2 out": ((400, 2), critics["critic_1"]["hidden_2"]["kernel"].shape),
    }
    assert all(actual == wanted for wanted, actual in expected.values())
    assert _tree_changed(critics["critic_0"], critics["critic_1"])
    assert all(x.dtype == jnp.float32 for x in jtu.tree_leaves(params))

    sample = workflow.replay_buffer.sample(
        state.replay_buffer_state, jax.random.PRNGKey(72)
    )
    preference = sample.extras.policy_extras.preference
    policy_action = workflow.agent.actor_network.apply(
        params.actor_params, sample.obs, preference
    )
    twin_q = workflow.agent.critic_network.apply(
        params.critic_params, sample.obs, preference, policy_action
    )
    wp = interpolate(state.agent_state.extra_state.interpolator, preference)
    critic_input = jnp.concatenate((sample.obs, preference, sample.actions), axis=-1)
    loss = workflow.agent.critic_loss(
        state.agent_state, sample, jax.random.PRNGKey(73)
    )
    assert sample.obs.shape == (256, 17)
    assert preference.shape == (256, 2)
    assert sample.actions.shape == policy_action.shape == (256, 6)
    assert critic_input.shape == (256, 25)
    assert twin_q.shape == (256, 2, 2)
    assert twin_q[:, 0, :].shape == twin_q[:, 1, :].shape == (256, 2)
    assert wp.shape == loss.q_target.shape == (256, 2)

    def critic_loss_fn(agent_state, batch, key):
        values = workflow.agent.critic_loss(agent_state, batch, key)
        return values.critic_loss, values

    def actor_loss_fn(agent_state, batch, key):
        values = workflow.agent.actor_loss(agent_state, batch, key)
        return values.actor_loss, values

    critic_update = agent_gradient_update(
        critic_loss_fn,
        workflow.optimizer,
        has_aux=True,
        grad_norm_key="critic_grad_norm",
        attach_fn=lambda agent_state, value: agent_state.replace(
            params=agent_state.params.replace(critic_params=value)
        ),
        detach_fn=lambda agent_state: agent_state.params.critic_params,
    )
    actor_update = agent_gradient_update(
        actor_loss_fn,
        workflow.optimizer,
        has_aux=True,
        grad_norm_key="actor_grad_norm",
        attach_fn=lambda agent_state, value: agent_state.replace(
            params=agent_state.params.replace(actor_params=value)
        ),
        detach_fn=lambda agent_state: agent_state.params.actor_params,
    )
    manual = state.agent_state
    actor_before = manual.params.actor_params
    critic_before = manual.params.critic_params
    (_, _), manual, critic_opt = critic_update(
        state.opt_state.critic, manual, sample, jax.random.PRNGKey(74)
    )
    assert _tree_changed(critic_before, manual.params.critic_params)
    assert not _tree_changed(actor_before, manual.params.actor_params)
    due = parallel_actor_update_mask(jnp.uint32(0), 10, 10)
    assert not bool(jnp.any(due[:-1])) and bool(due[-1])
    (_, _), manual, actor_opt = actor_update(
        state.opt_state.actor, manual, sample, jax.random.PRNGKey(75)
    )
    assert _tree_changed(actor_before, manual.params.actor_params)
    target_actor = soft_target_update(
        manual.params.target_actor_params, manual.params.actor_params, config.tau
    )
    target_critic = soft_target_update(
        manual.params.target_critic_params, manual.params.critic_params, config.tau
    )
    assert _tree_changed(manual.params.target_actor_params, target_actor)
    assert _tree_changed(manual.params.target_critic_params, target_critic)
    assert critic_opt is not None and actor_opt is not None

    initial = state.agent_state.params
    initial_interpolator = state.agent_state.extra_state.interpolator
    first_start = perf_counter()
    metrics, state = workflow._multi_steps(state)
    jax.block_until_ready(metrics.critic_loss)
    first_jit_seconds = perf_counter() - first_start
    first_cache_size = getattr(workflow._multi_steps, "_cache_size", lambda: None)()
    # The first compiled output commits initially uncommitted setup arrays. Allow
    # that one input-sharding specialization, then require a stable cache.
    _, state = workflow._multi_steps(state)
    second_cache_size = getattr(workflow._multi_steps, "_cache_size", lambda: None)()
    state, control = workflow._after_multi_steps(state)
    steady_start = perf_counter()
    _, state = workflow._multi_steps(state)
    jax.block_until_ready(state.agent_state.extra_state.total_it)
    steady_seconds = perf_counter() - steady_start
    steady_cache_size = getattr(workflow._multi_steps, "_cache_size", lambda: None)()
    if second_cache_size is not None:
        assert steady_cache_size == second_cache_size

    diagnostic = metrics.raw_loss_dict
    for name in (
        "critic_total_loss",
        "critic_smooth_l1_q1",
        "critic_smooth_l1_q2",
        "critic_angle_q1",
        "critic_angle_q2",
        "actor_scalarized_term",
        "actor_angle_term",
        "actor_total_loss",
        "critic_grad_norm",
        "actor_grad_norm",
        "q1_min",
        "q1_mean",
        "q1_max",
        "q2_min",
        "q2_mean",
        "q2_max",
        "wp_min",
        "wp_mean",
        "wp_max",
        "q_target",
    ):
        assert np.isfinite(np.asarray(diagnostic[name])).all(), name
    td_target = np.asarray(diagnostic.q_target)
    assert np.isfinite([td_target.min(), td_target.mean(), td_target.max()]).all()
    assert _tree_changed(initial.critic_params, state.agent_state.params.critic_params)
    assert _tree_changed(initial.actor_params, state.agent_state.params.actor_params)
    assert _tree_changed(
        initial.target_actor_params, state.agent_state.params.target_actor_params
    )
    assert int(state.agent_state.extra_state.total_it) == 30
    assert int(state.replay_buffer_state.buffer_size) >= 256
    assert bool(control["morl/train/her_active"])
    assert control["morl/train/episode_count_min"] > 1
    assert control["morl/train/episode_count_max"] >= control[
        "morl/train/episode_count_min"
    ]
    assert int(state.agent_state.extra_state.key_update_count) == 1
    assert int(state.agent_state.extra_state.interpolator_refit_count) == 1
    assert "morl/eval/hv" in control
    assert np.isfinite(control["morl/eval/hv"])
    assert np.isfinite(control["morl/eval/sparsity"])
    assert control["morl/eval/num_pareto_points"] > 0
    assert _tree_changed(
        initial_interpolator, state.agent_state.extra_state.interpolator
    )

    preferences = jnp.asarray(key_preferences(2), dtype=jnp.float32)
    controlled = SampleBatch(
        obs=jnp.broadcast_to(state.env_state.obs[0], (3, 17)),
        extras=PyTreeDict(policy_extras=PyTreeDict(preference=preferences)),
    )
    actions, _ = workflow.agent.evaluate_actions(
        state.agent_state, controlled, jax.random.PRNGKey(76)
    )
    assert float(jnp.max(jnp.ptp(actions, axis=0))) > 1e-7

    checkpoint = tmp_path / "production_shape_state"
    save(checkpoint, state)
    restored = load(checkpoint, state)
    query = jnp.asarray([[0.25, 0.75]], dtype=jnp.float32)
    np.testing.assert_array_equal(
        interpolate(restored.agent_state.extra_state.interpolator, query),
        interpolate(state.agent_state.extra_state.interpolator, query),
    )
    assert _tree_equal(restored.agent_state.params, state.agent_state.params)
    assert _tree_equal(restored.opt_state, state.opt_state)
    assert _tree_equal(restored.key, state.key)
    assert _tree_equal(
        restored.agent_state.extra_state.raw_key_solutions,
        state.agent_state.extra_state.raw_key_solutions,
    )
    assert _tree_equal(
        restored.agent_state.extra_state.worker_steps,
        state.agent_state.extra_state.worker_steps,
    )
    for name in (
        "total_it",
        "episode_count",
        "eval_cnt",
        "eval_cnt_ep",
        "key_update_count",
        "interpolator_refit_count",
    ):
        assert _tree_equal(
            restored.agent_state.extra_state[name],
            state.agent_state.extra_state[name],
        )
    assert int(restored.replay_buffer_state.buffer_size) == int(
        state.replay_buffer_state.buffer_size
    )

    print("PRODUCTION NETWORK SHAPE TABLE")
    for name, (_, actual) in expected.items():
        print(f"{name}: {actual}")
    print(
        {
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
            "dtype": "float32",
            "matmul_precision": config.matmul_precision,
            "batch_size": sample.obs.shape[0],
            "K": config.process_count,
            "B": config.num_envs,
            "first_jit_seconds": first_jit_seconds,
            "steady_update_seconds": steady_seconds,
            "first_cache_size": first_cache_size,
            "steady_cache_size": steady_cache_size,
            "unexpected_recompilation": False,
            "oom": False,
            "td_target_min_mean_max": (
                td_target.min(),
                td_target.mean(),
                td_target.max(),
            ),
        }
    )
