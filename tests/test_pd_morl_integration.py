"""TEST / INTEGRATION INFRASTRUCTURE for the complete PD-MORL workflow."""

from pathlib import Path

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from omegaconf import OmegaConf

from evorl.algorithms.mo_td3 import PDMORLWorkflow, make_pd_morl_agent
from evorl.envs import Box, EnvState
from evorl.evaluators.pd_morl import (
    KeyInterpolatorUpdateController,
    PDMORLEvaluator,
)
from evorl.replay_buffers import ReplayBuffer
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.utils.orbax_utils import load, save
from evorl.utils.pd_morl_interpolator import (
    fit_interpolator_state,
    interpolate,
    key_preferences,
)
from evorl.envs.wrappers import official_preference_subspaces


class ToyMOEnv:
    """TEST FIXTURE: two-objective continuous, vectorized environment."""

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.obs_space = Box(low=-jnp.ones(3), high=jnp.ones(3))
        self.action_space = Box(low=-jnp.ones(1), high=jnp.ones(1))

    def reset(self, key):
        del key
        first = jnp.linspace(0, 1, self.num_envs)
        preference = jnp.stack((first, 1 - first), axis=-1)
        obs = jnp.zeros((self.num_envs, 3))
        return EnvState(
            env_state=PyTreeDict(step=jnp.zeros(self.num_envs, dtype=jnp.uint32)),
            obs=obs,
            reward=jnp.zeros((self.num_envs, 2)),
            done=jnp.zeros(self.num_envs, dtype=jnp.bool_),
            info=PyTreeDict(
                preference=preference,
                ori_obs=obs,
                termination=jnp.zeros(self.num_envs),
                truncation=jnp.zeros(self.num_envs),
            ),
        )

    def step(self, state, action):
        step = state.env_state.step + 1
        done = step % 2 == 0
        obs = jnp.where(done[:, None], 0.0, state.obs + 0.1)
        reward = jnp.concatenate((1 + action, 1 - action), axis=-1)
        return state.replace(
            env_state=state.env_state.replace(step=step),
            obs=obs,
            reward=reward,
            done=done,
            info=state.info.replace(
                ori_obs=obs,
                termination=done.astype(jnp.float32),
                truncation=jnp.zeros(done.shape),
            ),
        )


def _config():
    # TEST OVERRIDE: short warmup, episode length, and evaluation threshold.
    return OmegaConf.create(
        {
            "checkpoint": {"enable": False},
            "output_dir": "",
            "num_envs": 2,
            "num_objectives": 2,
            "rollout_length": 1,
            "random_timesteps": 0,
            "learning_start_timesteps": 4,
            "num_relabel_preferences": 1,
            "her_start_timesteps": 0,
            "process_count": 2,
            "fold_iters": 1,
            "num_updates_per_iter": 1,
            "policy_freq": 2,
            "tau": 0.005,
            "key_update_enabled": True,
            "eval_freq": 1,
            "eval_episodes": 3,
            "training_eval_preference_step": 0.5,
            "offline_eval_preference_step": 0.5,
        }
    )


def _workflow():
    config = _config()
    env = ToyMOEnv(2)
    keys = key_preferences(2)
    solutions = np.array([[1.0, 4.0], [3.0, 3.0], [4.0, 1.0]])
    interpolator = fit_interpolator_state(keys, solutions, "initial")
    agent = make_pd_morl_agent(
        env.action_space,
        actor_hidden_layer_sizes=(8,),
        critic_hidden_layer_sizes=(8,),
        process_count=2,
        start_timesteps=0,
        actor_loss_coeff=10,
        interpolator_state=interpolator,
        initial_key_solutions=jnp.asarray(solutions, dtype=jnp.float32),
    )
    optimizer = optax.chain(optax.clip_by_global_norm(100), optax.adam(3e-4))
    replay = ReplayBuffer(capacity=64, min_sample_timesteps=4, sample_batch_size=4)
    workflow = PDMORLWorkflow(env, agent, optimizer, None, replay, config)
    workflow.morl_evaluator = PDMORLEvaluator(ToyMOEnv(1), agent, 2)
    workflow.key_update_controller = KeyInterpolatorUpdateController(2, repeats=3)
    return workflow


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


def test_toy_end_to_end_control_checkpoint_and_preference_sensitivity(tmp_path):
    workflow = _workflow()
    state = workflow.setup(jax.random.PRNGKey(0))
    before = state.agent_state.params
    metrics, state = jax.jit(workflow.step)(state)

    assert int(state.replay_buffer_state.buffer_size) > 4
    assert int(state.agent_state.extra_state.total_it) == 2
    assert _tree_changed(before.critic_params, state.agent_state.params.critic_params)
    assert _tree_changed(before.actor_params, state.agent_state.params.actor_params)
    assert _tree_changed(
        before.target_actor_params, state.agent_state.params.target_actor_params
    )
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
        "q1_mean",
        "q2_mean",
        "wp_mean",
    ):
        assert jnp.all(jnp.isfinite(diagnostic[name])), name
    np.testing.assert_allclose(
        diagnostic.critic_total_loss,
        diagnostic.critic_smooth_l1_q1
        + diagnostic.critic_smooth_l1_q2
        + diagnostic.critic_angle_q1
        + diagnostic.critic_angle_q2,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        diagnostic.actor_total_loss,
        diagnostic.actor_scalarized_term + 10 * diagnostic.actor_angle_term,
        rtol=1e-5,
    )

    preferences = jnp.asarray(key_preferences(2), dtype=jnp.float32)
    batch = SampleBatch(
        obs=jnp.zeros((3, 3)),
        extras=PyTreeDict(policy_extras=PyTreeDict(preference=preferences)),
    )
    actions, _ = workflow.agent.evaluate_actions(
        state.agent_state, batch, jax.random.PRNGKey(1)
    )
    assert float(jnp.ptp(actions)) > 1e-7
    sensitivity = workflow.morl_evaluator.evaluate(
        state.agent_state, np.asarray(preferences), repeats=1
    )
    assert sensitivity.mean_returns.shape == (3, 2)
    assert np.isfinite(sensitivity.mean_returns).all()

    state = state.replace(
        agent_state=state.agent_state.replace(
            extra_state=state.agent_state.extra_state.replace(
                episode_count=jnp.full((2,), 2, dtype=jnp.uint32)
            )
        )
    )
    old_wp = interpolate(state.agent_state.extra_state.interpolator, preferences)
    state, control = workflow._after_multi_steps(state)
    new_wp = interpolate(state.agent_state.extra_state.interpolator, preferences)
    assert int(state.agent_state.extra_state.key_update_count) == 1
    assert int(state.agent_state.extra_state.interpolator_refit_count) == 1
    assert "morl/eval/hv" in control and np.isfinite(control["morl/eval/hv"])
    assert "morl/eval/sparsity" in control
    assert _tree_changed(old_wp, new_wp)

    checkpoint = tmp_path / "pd_morl_state"
    save(checkpoint, state)
    restored = load(checkpoint, state)
    restored_wp = interpolate(
        restored.agent_state.extra_state.interpolator, preferences
    )
    np.testing.assert_array_equal(restored_wp, new_wp)
    assert _tree_equal(restored.agent_state.params, state.agent_state.params)
    assert _tree_equal(restored.opt_state, state.opt_state)
    assert _tree_equal(restored.key, state.key)
    assert _tree_equal(
        restored.agent_state.extra_state.worker_steps,
        state.agent_state.extra_state.worker_steps,
    )
    assert _tree_equal(
        restored.agent_state.extra_state.raw_key_solutions,
        state.agent_state.extra_state.raw_key_solutions,
    )
    assert int(restored.agent_state.extra_state.total_it) == 2
    assert int(restored.agent_state.extra_state.eval_cnt_ep) == 2
    assert int(restored.agent_state.extra_state.eval_cnt) == 2
    assert int(restored.replay_buffer_state.buffer_size) == int(
        state.replay_buffer_state.buffer_size
    )


def test_real_walker_single_gpu_end_to_end(tmp_path):
    assert jax.default_backend() == "gpu"
    root = Path(__file__).parents[1]
    production = OmegaConf.load(root / "configs" / "agent" / "mo-td3.yaml")
    assert production.process_count == 10
    assert production.random_action_warmup == 10000
    assert production.eval_freq == 100
    assert production.training_eval_preference_step == 0.005
    assert production.offline_eval_preference_step == 0.001

    # TEST OVERRIDE: exercise every integration hook in a short real Brax run.
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
            "random_timesteps": 10,
            "learning_start_timesteps": 10,
            "learner_start_threshold": 10,
            "random_action_warmup": 0,
            "her_start_timesteps": 0,
            "batch_size": 10,
            "replay_capacity": 256,
            "eval_freq": 1,
            "training_eval_preference_step": 0.5,
            "offline_eval_preference_step": 0.5,
        },
    )
    assert config.agent_network == production.agent_network
    SmokeWorkflow = type("SmokeWorkflow", (PDMORLWorkflow,), {})
    workflow = SmokeWorkflow.build_from_config(config, enable_jit=True)
    state = workflow.setup(jax.random.PRNGKey(10))
    assert workflow.env.num_envs == 10
    actor = state.agent_state.params.actor_params["params"]["actor"]
    critics = state.agent_state.params.critic_params["params"]
    assert actor["hidden_0"]["kernel"].shape == (19, 400)
    assert actor["hidden_1"]["kernel"].shape == (400, 400)
    assert actor["hidden_2"]["kernel"].shape == (400, 6)
    for critic in (critics["critic_0"], critics["critic_1"]):
        assert critic["hidden_0"]["kernel"].shape == (25, 400)
        assert critic["hidden_1"]["kernel"].shape == (400, 400)
        assert critic["hidden_2"]["kernel"].shape == (400, 2)
    preferences = np.asarray(state.env_state.info.preference)
    for worker_id, subspace in enumerate(official_preference_subspaces(10)):
        assert np.any(np.all(np.asarray(subspace) == preferences[worker_id], axis=1))

    initial = state.agent_state.params
    control = {}
    metrics = None
    for _ in range(5):
        metrics, state = workflow._multi_steps(state)
        state, current = workflow._after_multi_steps(state)
        control.update(current)
        if int(state.agent_state.extra_state.key_update_count) > 0:
            break

    assert metrics is not None
    assert int(state.agent_state.extra_state.total_it) >= 10
    assert int(state.replay_buffer_state.buffer_size) > 10
    assert _tree_changed(initial.critic_params, state.agent_state.params.critic_params)
    assert _tree_changed(initial.actor_params, state.agent_state.params.actor_params)
    assert _tree_changed(
        initial.target_critic_params, state.agent_state.params.target_critic_params
    )
    assert np.all(np.asarray(state.agent_state.extra_state.episode_count) > 1)
    assert int(state.agent_state.extra_state.key_update_count) > 0
    assert int(state.agent_state.extra_state.interpolator_refit_count) > 0
    assert "morl/eval/hv" in control
    assert np.isfinite(control["morl/eval/hv"])
    assert np.isfinite(control["morl/eval/sparsity"])
    assert control["morl/eval/num_pareto_points"] > 0
    assert all(
        np.isfinite(np.asarray(value)).all()
        for value in metrics.raw_loss_dict.values()
    )

    sensitivity = workflow.morl_evaluator.evaluate(
        state.agent_state, key_preferences(2), repeats=1
    )
    assert np.isfinite(sensitivity.mean_returns).all()
    batch = SampleBatch(
        obs=jnp.broadcast_to(state.env_state.obs[0], (3, state.env_state.obs.shape[-1])),
        extras=PyTreeDict(
            policy_extras=PyTreeDict(
                preference=jnp.asarray(key_preferences(2), dtype=jnp.float32)
            )
        ),
    )
    actions, _ = workflow.agent.evaluate_actions(
        state.agent_state, batch, jax.random.PRNGKey(11)
    )
    assert float(jnp.max(jnp.ptp(actions, axis=0))) > 1e-7

    checkpoint = tmp_path / "walker_state"
    save(checkpoint, state)
    restored = load(checkpoint, state)
    query = jnp.asarray([[0.25, 0.75]], dtype=jnp.float32)
    np.testing.assert_array_equal(
        interpolate(restored.agent_state.extra_state.interpolator, query),
        interpolate(state.agent_state.extra_state.interpolator, query),
    )
    assert _tree_equal(restored.agent_state.params, state.agent_state.params)
    assert _tree_equal(restored.opt_state, state.opt_state)
