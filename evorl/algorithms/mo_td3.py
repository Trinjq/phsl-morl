import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from omegaconf import DictConfig

from evorl.agent import AgentState
from evorl.envs import AutoresetMode, Box, Space, create_env
from evorl.evaluators import BatchedPDMORLEvaluator, Evaluator, PDMORLEvaluator
from evorl.evaluators.pd_morl import (
    full_evaluation_due,
    key_update_due,
    load_key_solution_artifact,
    preference_grid,
    update_key_solutions_jax,
)
from evorl.networks import MLP
from evorl.recorders import add_prefix
from evorl.replay_buffers import ReplayBuffer
from evorl.replay_buffers.her import add_her_transitions
from evorl.sample_batch import SampleBatch
from evorl.types import (
    Action,
    LossDict,
    PolicyExtraInfo,
    PyTreeDict,
    State,
    pytree_field,
)
from evorl.utils import running_statistics
from evorl.utils.jax_utils import tree_get
from evorl.utils.morl_math import directional_angle, scalarize
from evorl.utils.pd_morl_convergence import (
    append_convergence_result,
    plot_convergence_csv,
    prepare_convergence_history,
)
from evorl.utils.pd_morl_interpolator import (
    PDMORLInterpolatorState,
    fit_interpolator_state,
    interpolate,
    key_preferences,
)

from .offpolicy_utils import skip_replay_buffer_state
from .td3 import TD3Agent, TD3NetworkParams, TD3Workflow

HV_HISTORY_FIELDS = (
    "seed",
    "run_id",
    "artifact_sha256",
    "kind",
    "preference_count",
    "repeats",
    "actual_env_transitions",
    "iteration",
    "source_hv",
    "mean_repeat_hv",
    "source_sparsity",
    "source_pareto_point_count",
    "wall_clock_seconds",
)

KEY_DIAGNOSTIC_FIELDS = (
    "seed",
    "run_id",
    "artifact_sha256",
    "iteration",
    "actual_env_transitions",
    "event_id",
    "eval_cnt_ep_before",
    "episode_count_raw_10",
    "episode_count_logical_10",
    "key_index",
    "w_0",
    "w_1",
    "stored_r0_before",
    "stored_r1_before",
    "candidate_r0",
    "candidate_r1",
    "candidate_repeat_scores_3",
    "stored_score",
    "candidate_score",
    "score_gap",
    "relative_gap_pct",
    "replaced",
    "replaced_expected",
    "stored_r0_after",
    "stored_r1_after",
    "evaluation_seconds",
    "logging_seconds",
)


def append_key_diagnostics(path: Path, rows: list[dict[str, Any]]) -> int:
    """Append unseen event/key rows; resume-safe without touching learner state."""
    seen = set()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            seen = {
                (row["run_id"], int(row["event_id"]), int(row["key_index"]))
                for row in csv.DictReader(handle)
            }
    rows = [
        row
        for row in rows
        if (row["run_id"], row["event_id"], row["key_index"]) not in seen
    ]
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=KEY_DIAGNOSTIC_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    return len(rows)


def build_key_diagnostic_rows(
    *,
    seed: int,
    run_id: str,
    artifact_sha256: str,
    iteration: int,
    actual_env_transitions: int,
    event_id: int,
    eval_cnt_ep_before: int,
    episode_count_raw: np.ndarray,
    episode_count_logical: np.ndarray,
    keys: np.ndarray,
    old_solutions: np.ndarray,
    returns_per_repeat: np.ndarray,
    improved: np.ndarray,
    new_solutions: np.ndarray,
    evaluation_seconds: float,
    logging_seconds: float,
) -> list[dict[str, Any]]:
    candidates = np.asarray(returns_per_repeat).mean(axis=0)
    old = np.asarray(old_solutions)
    new = np.asarray(new_solutions)
    weights = np.asarray(keys, dtype=old.dtype)
    stored_scores = np.sum(weights * old, axis=1)
    candidate_scores = np.sum(weights * candidates, axis=1)
    repeat_scores = np.sum(np.asarray(returns_per_repeat) * weights[None, :, :], axis=2)
    gaps = candidate_scores - stored_scores
    expected = gaps > 0
    if not np.array_equal(np.asarray(improved, dtype=bool), expected):
        raise AssertionError("diagnostic replacement decision disagrees with learner")
    common = {
        "seed": int(seed),
        "run_id": run_id,
        "artifact_sha256": artifact_sha256,
        "iteration": int(iteration),
        "actual_env_transitions": int(actual_env_transitions),
        "event_id": int(event_id),
        "eval_cnt_ep_before": int(eval_cnt_ep_before),
        "episode_count_raw_10": json.dumps(np.asarray(episode_count_raw).tolist()),
        "episode_count_logical_10": json.dumps(
            np.asarray(episode_count_logical).tolist()
        ),
        "evaluation_seconds": float(evaluation_seconds),
        "logging_seconds": float(logging_seconds),
    }
    return [
        {
            **common,
            "key_index": i,
            "w_0": float(weights[i, 0]),
            "w_1": float(weights[i, 1]),
            "stored_r0_before": float(old[i, 0]),
            "stored_r1_before": float(old[i, 1]),
            "candidate_r0": float(candidates[i, 0]),
            "candidate_r1": float(candidates[i, 1]),
            "candidate_repeat_scores_3": json.dumps(repeat_scores[:, i].tolist()),
            "stored_score": float(stored_scores[i]),
            "candidate_score": float(candidate_scores[i]),
            "score_gap": float(gaps[i]),
            "relative_gap_pct": float(
                100.0 * gaps[i] / max(abs(float(stored_scores[i])), 1e-8)
            ),
            "replaced": bool(improved[i]),
            "replaced_expected": bool(expected[i]),
            "stored_r0_after": float(new[i, 0]),
            "stored_r1_after": float(new[i, 1]),
        }
        for i in range(len(weights))
    ]


def append_hv_history(
    path: Path,
    *,
    seed: int,
    run_id: str,
    artifact_sha256: str,
    iteration: int,
    actual_env_transitions: int,
    result: Any,
    wall_clock_seconds: float | None,
) -> bool:
    """Append one existing full-evaluation result, once."""
    source_hv = float(getattr(result, "source_hv", result.mean_hv))
    source_sparsity = float(getattr(result, "source_sparsity", result.mean_sparsity))
    source_count = int(
        getattr(result, "source_pareto_point_count", len(result.pareto_returns))
    )
    key = (run_id, "training_full", int(actual_env_transitions))
    seen = set()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                seen.add(
                    (
                        row.get("run_id", ""),
                        row.get("kind", ""),
                        int(row["actual_env_transitions"]),
                    )
                )
    if key in seen:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    row = {
        "seed": int(seed),
        "run_id": run_id,
        "artifact_sha256": artifact_sha256,
        "kind": "training_full",
        "preference_count": len(result.preferences),
        "repeats": int(result.returns_per_repeat.shape[0]),
        "actual_env_transitions": int(actual_env_transitions),
        "iteration": int(iteration),
        "source_hv": source_hv,
        "mean_repeat_hv": float(getattr(result, "mean_repeat_hv", result.mean_hv)),
        "source_sparsity": source_sparsity,
        "source_pareto_point_count": source_count,
        "wall_clock_seconds": wall_clock_seconds,
    }
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HV_HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
    return True


def select_pessimistic_q_vector(
    twin_q_values: chex.Array, preference: chex.Array
) -> chex.Array:
    """Select one complete critic vector using its scalarized value."""
    critic_index = jnp.argmin(scalarize(twin_q_values, preference), axis=-1)
    return jnp.take_along_axis(twin_q_values, critic_index[..., None, None], axis=-2)[
        ..., 0, :
    ]


def vector_bellman_target(
    rewards: chex.Array,
    done: chex.Array,
    next_q_values: chex.Array,
    discount: float,
) -> chex.Array:
    return rewards + discount * (1 - done)[..., None] * next_q_values


def twin_smooth_l1_loss(
    twin_q_values: chex.Array, target_q_values: chex.Array
) -> chex.Array:
    """Sum the mean Smooth-L1 loss of both critics."""
    losses = optax.huber_loss(twin_q_values, target_q_values[..., None, :])
    return losses.mean(axis=(0, 2)).sum()


def pd_morl_critic_loss(
    twin_q_values: chex.Array,
    target_q_values: chex.Array,
    projected_preference: chex.Array,
) -> chex.Array:
    """Official two Smooth-L1 means plus two unweighted angle means."""
    angles = directional_angle(projected_preference[..., None, :], twin_q_values)
    return (
        twin_smooth_l1_loss(twin_q_values, target_q_values) + angles.mean(axis=0).sum()
    )


def pd_morl_actor_loss(
    q_values: chex.Array,
    preference: chex.Array,
    projected_preference: chex.Array,
    actor_loss_coeff: float,
) -> chex.Array:
    """Official scalar Q objective plus weighted directional-angle mean."""
    return -scalarize(q_values, preference).mean() + actor_loss_coeff * (
        directional_angle(projected_preference, q_values).mean()
    )


def add_target_policy_smoothing(
    actions: chex.Array,
    key: chex.PRNGKey,
    policy_noise: float,
    clip_policy_noise: float,
) -> chex.Array:
    noise = jnp.clip(
        jax.random.normal(key, actions.shape) * policy_noise,
        -clip_policy_noise,
        clip_policy_noise,
    )
    return jnp.clip(actions + noise, -1.0, 1.0)


def parallel_actor_update_mask(
    round_index: chex.Array, process_count: int, policy_freq: int
) -> chex.Array:
    """Actor-update positions for K source-faithful learner calls."""
    update_ids = round_index * process_count + jnp.arange(1, process_count + 1)
    return update_ids % policy_freq == 0


def periodic_update_count(start: int, count: int, interval: int) -> int:
    """Count interval boundaries crossed by ``count`` consecutive updates."""
    return (start + count) // interval - start // interval


def training_chunk_plan(
    total_timesteps: int,
    sampled_timesteps: int,
    start_iteration: int,
    transitions_per_chunk: int,
    fold_iters: int,
) -> tuple[int, int]:
    """Return host chunks and the matching final rollout iteration."""
    remaining = total_timesteps - sampled_timesteps
    if remaining < 0:
        raise ValueError("sampled_timesteps exceeds total_timesteps")
    if transitions_per_chunk <= 0 or fold_iters <= 0:
        raise ValueError("chunk size and fold_iters must be positive")
    chunks = math.ceil(remaining / transitions_per_chunk) if remaining else 0
    return chunks, start_iteration + chunks * fold_iters


def select_warmup_actions(
    policy_actions: chex.Array,
    random_actions: chex.Array,
    worker_steps: chex.Array,
    start_timesteps: int,
) -> chex.Array:
    """Select random actions for logical workers still in source warm-up."""
    worker_ids = jnp.arange(policy_actions.shape[0]) % worker_steps.shape[0]
    random_mask = worker_steps[worker_ids] < start_timesteps
    return jnp.where(random_mask[..., None], random_actions, policy_actions)


def completed_episodes_by_worker(dones: chex.Array, process_count: int) -> chex.Array:
    """Count completed episodes for each logical worker in a rollout."""
    dones = jnp.asarray(dones, dtype=jnp.uint32)
    if dones.ndim == 1:
        worker_ids = jnp.arange(dones.shape[0]) % process_count
        return jnp.zeros((process_count,), dtype=jnp.uint32).at[worker_ids].add(dones)
    completed_per_lane = dones.sum(axis=0)
    worker_ids = jnp.arange(completed_per_lane.shape[0]) % process_count
    return (
        jnp.zeros((process_count,), dtype=jnp.uint32)
        .at[worker_ids]
        .add(completed_per_lane)
    )


class PreferenceActor(nn.Module):
    action_size: int
    hidden_layer_sizes: tuple[int, ...] = (400, 400)
    max_action: float = 1.0

    @nn.compact
    def __call__(self, obs: chex.Array, preference: chex.Array) -> chex.Array:
        inputs = jnp.concatenate((obs, preference), axis=-1)
        return self.max_action * MLP(
            layer_sizes=(*self.hidden_layer_sizes, self.action_size),
            activation=nn.relu,
            activation_final=nn.tanh,
            kernel_init=jax.nn.initializers.xavier_normal(),
            name="actor",
        )(inputs)


class TwinVectorCritic(nn.Module):
    reward_size: int = 2
    hidden_layer_sizes: tuple[int, ...] = (400, 400)

    @nn.compact
    def __call__(
        self,
        obs: chex.Array,
        preference: chex.Array,
        actions: chex.Array,
    ) -> chex.Array:
        inputs = jnp.concatenate((obs, preference, actions), axis=-1)
        q_values = [
            MLP(
                layer_sizes=(*self.hidden_layer_sizes, self.reward_size),
                activation=nn.relu,
                kernel_init=jax.nn.initializers.xavier_normal(),
                name=f"critic_{critic_id}",
            )(inputs)
            for critic_id in range(2)
        ]
        return jnp.stack(q_values, axis=-2)


class MOTD3Agent(TD3Agent):
    """Preference-conditioned vector-Q extension of EvoRL TD3."""

    obs_preprocessor: Any = pytree_field(default=None, static=True)
    reward_size: int = pytree_field(default=2, static=True)
    process_count: int = pytree_field(default=1, static=True)
    start_timesteps: int = pytree_field(default=10000, static=True)
    actor_loss_coeff: float = pytree_field(default=10.0, static=True)
    interpolator_state: PDMORLInterpolatorState | None = None
    initial_key_solutions: chex.Array | None = None
    action_low: chex.Array | None = None
    action_high: chex.Array | None = None

    def init(
        self, obs_space: Space, action_space: Space, key: chex.PRNGKey
    ) -> AgentState:
        key, critic_key, actor_key = jax.random.split(key, num=3)
        dummy_obs = jtu.tree_map(lambda x: x[None, ...], obs_space.sample(key))
        dummy_action = action_space.sample(key)[None, ...]
        dummy_preference = jnp.full((1, self.reward_size), 1 / self.reward_size)

        critic_params = self.critic_network.init(
            critic_key, dummy_obs, dummy_preference, dummy_action
        )
        actor_params = self.actor_network.init(actor_key, dummy_obs, dummy_preference)
        params_state = TD3NetworkParams(
            critic_params=critic_params,
            actor_params=actor_params,
            target_critic_params=critic_params,
            target_actor_params=actor_params,
        )

        obs_preprocessor_state = (
            running_statistics.init_state(tree_get(dummy_obs, 0))
            if self.normalize_obs
            else None
        )
        return AgentState(
            params=params_state,
            obs_preprocessor_state=obs_preprocessor_state,
            extra_state=PyTreeDict(
                worker_steps=jnp.zeros((self.process_count,), dtype=jnp.uint32),
                episode_count=jnp.zeros((self.process_count,), dtype=jnp.uint32),
                total_it=jnp.uint32(0),
                eval_cnt_ep=jnp.uint32(1),
                eval_cnt=jnp.uint32(1),
                raw_key_solutions=self.initial_key_solutions,
                interpolator=self.interpolator_state,
            ),
        )

    @staticmethod
    def _preference(sample_batch: SampleBatch) -> chex.Array:
        return sample_batch.extras.policy_extras.preference

    @staticmethod
    def _project_preference(
        agent_state: AgentState, preference: chex.Array
    ) -> chex.Array:
        if agent_state.extra_state.interpolator is None:
            raise ValueError("PD-MORL losses require an interpolator state")
        return interpolate(agent_state.extra_state.interpolator, preference)

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        preference = self._preference(sample_batch)
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        policy_actions = self.actor_network.apply(
            agent_state.params.actor_params, obs, preference
        )
        noise_key, random_key = jax.random.split(key)
        policy_actions += (
            jax.random.normal(noise_key, policy_actions.shape)
            * self.exploration_epsilon
        )
        policy_actions = jnp.clip(policy_actions, self.action_low, self.action_high)
        random_actions = jax.random.uniform(
            random_key,
            policy_actions.shape,
            minval=self.action_low,
            maxval=self.action_high,
        )
        actions = select_warmup_actions(
            policy_actions,
            random_actions,
            agent_state.extra_state.worker_steps,
            self.start_timesteps,
        )
        return actions, PyTreeDict(preference=preference)

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        del key
        obs = sample_batch.obs
        preference = self._preference(sample_batch)
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
        actions = self.actor_network.apply(
            agent_state.params.actor_params, obs, preference
        )
        return actions, PyTreeDict(preference=preference)

    def target_actions(
        self,
        agent_state: AgentState,
        next_obs: chex.Array,
        preference: chex.Array,
        key: chex.PRNGKey,
    ) -> chex.Array:
        actions = self.actor_network.apply(
            agent_state.params.target_actor_params, next_obs, preference
        )
        return add_target_policy_smoothing(
            actions, key, self.policy_noise, self.clip_policy_noise
        )

    def critic_loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        obs = sample_batch.obs
        next_obs = sample_batch.extras.env_extras.ori_obs
        preference = self._preference(sample_batch)
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
            next_obs = self.obs_preprocessor(
                next_obs, agent_state.obs_preprocessor_state
            )

        next_actions = self.target_actions(agent_state, next_obs, preference, key)
        next_twin_q = self.critic_network.apply(
            agent_state.params.target_critic_params,
            next_obs,
            preference,
            next_actions,
        )
        next_q = select_pessimistic_q_vector(next_twin_q, preference)
        env_extras = sample_batch.extras.env_extras
        done = jnp.maximum(env_extras.termination, env_extras.truncation)
        q_target = vector_bellman_target(
            sample_batch.rewards, done, next_q, self.discount
        )
        q_target = jax.lax.stop_gradient(q_target)

        q_values = self.critic_network.apply(
            agent_state.params.critic_params,
            obs,
            preference,
            sample_batch.actions,
        )
        projected_preference = self._project_preference(agent_state, preference)
        smooth_l1 = twin_smooth_l1_loss(q_values, q_target)
        angles = directional_angle(projected_preference[..., None, :], q_values).mean(
            axis=0
        )
        critic_loss = smooth_l1 + angles.sum()
        return PyTreeDict(
            critic_loss=critic_loss,
            critic_total_loss=critic_loss,
            critic_smooth_l1=smooth_l1,
            q_value=scalarize(q_values, preference).mean(),
            q_target=q_target,
            critic_angle=angles,
            q1_min=q_values[..., 0, :].min(),
            q1_mean=q_values[..., 0, :].mean(),
            q1_max=q_values[..., 0, :].max(),
            q2_min=q_values[..., 1, :].min(),
            q2_mean=q_values[..., 1, :].mean(),
            q2_max=q_values[..., 1, :].max(),
            target_min=q_target.min(),
            target_mean=q_target.mean(),
            target_max=q_target.max(),
            wp_min=projected_preference.min(),
            wp_mean=projected_preference.mean(),
            wp_max=projected_preference.max(),
        )

    def actor_loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        del key
        obs = sample_batch.obs
        preference = self._preference(sample_batch)
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        actions = self.actor_network.apply(
            agent_state.params.actor_params, obs, preference
        )
        q_values = self.critic_network.apply(
            agent_state.params.critic_params, obs, preference, actions
        )
        q1 = q_values[..., 0, :]
        projected_preference = self._project_preference(agent_state, preference)
        scalarized = scalarize(q1, preference).mean()
        angle = directional_angle(projected_preference, q1).mean()
        actor_loss = -scalarized + self.actor_loss_coeff * angle
        return PyTreeDict(
            actor_loss=actor_loss,
            actor_total_loss=actor_loss,
            actor_scalarized_term=scalarized,
            actor_angle=angle,
        )


def make_mo_td3_agent(
    action_space: Space,
    actor_hidden_layer_sizes: tuple[int, ...] = (400, 400),
    critic_hidden_layer_sizes: tuple[int, ...] = (400, 400),
    reward_size: int = 2,
    discount: float = 0.995,
    exploration_epsilon: float = 0.1,
    policy_noise: float = 0.2,
    clip_policy_noise: float = 0.5,
    normalize_obs: bool = False,
    process_count: int = 1,
    start_timesteps: int = 10000,
    actor_loss_coeff: float = 10.0,
    interpolator_state: PDMORLInterpolatorState | None = None,
    initial_key_solutions: chex.Array | None = None,
) -> MOTD3Agent:
    assert isinstance(action_space, Box), "Only continuous action spaces are supported."
    max_action = float(jnp.max(action_space.high))
    actor_network = PreferenceActor(
        action_size=action_space.shape[0],
        hidden_layer_sizes=tuple(actor_hidden_layer_sizes),
        max_action=max_action,
    )
    critic_network = TwinVectorCritic(
        reward_size=reward_size,
        hidden_layer_sizes=tuple(critic_hidden_layer_sizes),
    )
    return MOTD3Agent(
        critic_network=critic_network,
        actor_network=actor_network,
        obs_preprocessor=(running_statistics.normalize if normalize_obs else None),
        reward_size=reward_size,
        process_count=process_count,
        start_timesteps=start_timesteps,
        actor_loss_coeff=actor_loss_coeff,
        interpolator_state=interpolator_state,
        initial_key_solutions=initial_key_solutions,
        action_low=action_space.low,
        action_high=action_space.high,
        discount=discount,
        exploration_epsilon=exploration_epsilon,
        policy_noise=policy_noise,
        clip_policy_noise=clip_policy_noise,
        critics_in_actor_loss="first",
    )


class MOTD3Workflow(TD3Workflow):
    env_extra_fields = ("ori_obs", "termination", "truncation")
    critic_raw_grad_norm_key = "critic_raw_grad_norm"
    actor_raw_grad_norm_key = "actor_raw_grad_norm"

    @staticmethod
    def _load_configured_key_solutions(config, keys):
        artifact_path = Path(config.interp_artifact_path)
        if not artifact_path.is_absolute():
            artifact_path = Path(__file__).parents[2] / artifact_path
        brax_artifacts = {
            "interp_objs_walker2d_brax.txt",
            "interp_objs_walker2d_brax_legacy.txt",
            "interp_objs_walker2d_brax_v2.txt",
            "interp_objs_walker2d_brax_v3.txt",
        }
        if config.env.env_type == "brax" and artifact_path.name not in brax_artifacts:
            raise ValueError(
                "Brax PD-MORL runs must use a versioned "
                "configs/artifacts/interp_objs_walker2d_brax artifact"
            )
        key_solutions, artifact = load_key_solution_artifact(artifact_path, keys)
        print(f"[KeyArtifact] path={artifact.path}")
        print(f"[KeyArtifact] sha256={artifact.sha256}")
        print(f"[KeyArtifact] values=\n{key_solutions}")
        return key_solutions, artifact

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.runtime_counters = {
            "host_chunk_count": 0,
            "inner_rollout_count": 0,
            "environment_step_count": 0,
            "critic_optimizer_step_count": 0,
            "actor_optimizer_step_count": 0,
            "target_update_count": 0,
            "replay_sample_call_count": 0,
            "key_evaluation_count": 0,
            "key_replacement_count": 0,
            "rbf_refit_count": 0,
        }
        self.runtime_timing = {"host_chunk_time": 0.0}

    def setup(self, key: chex.PRNGKey) -> State:
        state = super().setup(key)
        self.runtime_counters["environment_step_count"] = int(
            jax.device_get(state.metrics.sampled_timesteps)
        )
        return state

    @classmethod
    def name(cls):
        return "MO-TD3"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        jax.config.update("jax_default_matmul_precision", config.matmul_precision)
        if config.rollout_length != 1 or config.num_envs != config.process_count:
            raise ValueError(
                "source-faithful control requires rollout_length=1 and "
                "num_envs=process_count"
            )
        keys = key_preferences(config.reward_size)
        key_solutions, artifact = cls._load_configured_key_solutions(config, keys)
        interpolator_state = fit_interpolator_state(keys, key_solutions, "initial")
        env_kwargs = {
            "episode_length": config.env.max_episode_steps,
            "autoreset_mode": AutoresetMode.NORMAL,
            "record_ori_obs": True,
            "vector_reward": True,
            "episode_preference": True,
            "process_count": config.process_count,
        }
        env = create_env(config.env, parallel=config.num_envs, **env_kwargs)
        agent = make_mo_td3_agent(
            action_space=env.action_space,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            critic_hidden_layer_sizes=config.agent_network.critic_hidden_layer_sizes,
            reward_size=config.reward_size,
            discount=config.discount,
            exploration_epsilon=config.exploration_epsilon,
            policy_noise=config.policy_noise,
            clip_policy_noise=config.clip_policy_noise,
            normalize_obs=config.normalize_obs,
            process_count=config.process_count,
            start_timesteps=config.start_timesteps,
            actor_loss_coeff=config.actor_loss_coeff,
            interpolator_state=interpolator_state,
            initial_key_solutions=jnp.asarray(key_solutions, dtype=jnp.float32),
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.optimizer.grad_clip_norm),
            optax.adam(config.optimizer.lr),
        )
        replay_buffer = ReplayBuffer(
            capacity=config.replay_buffer_capacity,
            min_sample_timesteps=max(
                config.batch_size, config.learner_start_replay_entries
            ),
            sample_batch_size=config.batch_size,
        )
        eval_env = create_env(
            config.env,
            parallel=config.num_eval_envs,
            episode_length=config.env.max_episode_steps,
            autoreset_mode=AutoresetMode.DISABLED,
            vector_reward=True,
            episode_preference=True,
            process_count=config.process_count,
        )
        evaluator = Evaluator(
            env=eval_env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=config.env.max_episode_steps,
        )
        workflow = cls(env, agent, optimizer, evaluator, replay_buffer, config)
        control_eval_env = create_env(
            config.env,
            parallel=1,
            episode_length=config.env.max_episode_steps,
            autoreset_mode=AutoresetMode.DISABLED,
            vector_reward=True,
        )
        workflow.morl_evaluator = PDMORLEvaluator(
            env=control_eval_env,
            agent=agent,
            max_episode_steps=config.env.max_episode_steps,
        )
        workflow.interpolator_artifact = artifact
        return workflow

    def _setup_replaybuffer(self, key: chex.PRNGKey):
        dummy_obs = self.env.obs_space.sample(key)
        dummy_done = jnp.zeros(())
        return self.replay_buffer.init(
            SampleBatch(
                obs=dummy_obs,
                actions=jnp.zeros(self.env.action_space.shape),
                rewards=jnp.zeros((self.config.reward_size,)),
                extras=PyTreeDict(
                    policy_extras=PyTreeDict(
                        preference=jnp.zeros((self.config.reward_size,))
                    ),
                    env_extras=PyTreeDict(
                        ori_obs=dummy_obs,
                        termination=dummy_done,
                        truncation=dummy_done,
                    ),
                ),
            )
        )

    def _add_to_replay_buffer(self, replay_buffer_state, trajectory, key):
        return add_her_transitions(
            self.replay_buffer,
            replay_buffer_state,
            trajectory,
            jax.random.fold_in(key, 0x484552),
            self.config.num_relabel_preferences,
            self.config.her_start_timesteps,
            self.config.process_count,
        )

    def _postsetup_replaybuffer(self, state: State) -> State:
        state = super()._postsetup_replaybuffer(state)
        prefill_steps = (
            self.config.learning_start_timesteps + self.config.num_envs - 1
        ) // self.config.num_envs
        return state.replace(
            agent_state=state.agent_state.replace(
                extra_state=state.agent_state.extra_state.replace(
                    worker_steps=jnp.full(
                        (self.config.process_count,),
                        prefill_steps,
                        dtype=jnp.uint32,
                    )
                )
            )
        )

    def _on_prefill_trajectory(self, agent_state, trajectory):
        return agent_state.replace(
            extra_state=agent_state.extra_state.replace(
                episode_count=agent_state.extra_state.episode_count
                + completed_episodes_by_worker(
                    trajectory.dones, self.config.process_count
                )
            )
        )

    def _critic_updates_per_rollout(self) -> int:
        return self.config.process_count

    def _after_rollout_agent_state(self, agent_state, trajectory_dones):
        return agent_state.replace(
            extra_state=agent_state.extra_state.replace(
                worker_steps=(
                    agent_state.extra_state.worker_steps
                    + jnp.uint32(self.config.rollout_length)
                ),
                episode_count=(
                    agent_state.extra_state.episode_count
                    + completed_episodes_by_worker(
                        trajectory_dones, self.config.process_count
                    )
                ),
                total_it=(
                    agent_state.extra_state.total_it
                    + jnp.uint32(self._critic_updates_per_rollout())
                ),
            )
        )

    def _control_episode_count(self, extra):
        return np.asarray(jax.device_get(extra.episode_count))

    def _her_threshold_multiplier(self) -> int:
        return self.config.process_count

    def _her_base_transition_threshold(self) -> int:
        return int(self.config.her_start_timesteps) * int(
            self._her_threshold_multiplier()
        )

    def _after_multi_steps(self, state: State):
        """Run source-faithful key/full evaluation at the host boundary."""
        extra = state.agent_state.extra_state
        episode_count = self._control_episode_count(extra)
        control_metrics = {}
        history = self.config.get("hv_history", {})
        eval_cnt_ep_before = int(jax.device_get(extra.eval_cnt_ep))

        if key_update_due(episode_count, eval_cnt_ep_before):
            diagnostics_enabled = bool(
                self.config.get("key_diagnostics", {}).get("enable", False)
            )
            old_key_solutions = extra.raw_key_solutions
            raw_episode_count = extra.episode_count
            eval_cnt_ep = extra.eval_cnt_ep + jnp.uint32(1)
            self.runtime_counters["key_evaluation_count"] += 1
            self.runtime_counters["rbf_refit_count"] += 1
            state = state.replace(
                agent_state=state.agent_state.replace(
                    extra_state=extra.replace(eval_cnt_ep=eval_cnt_ep)
                )
            )
            evaluation_started_at = time.perf_counter() if diagnostics_enabled else None
            key_returns = self.morl_evaluator.evaluate_keys_device(
                state.agent_state, repeats=3
            )
            if diagnostics_enabled:
                jax.block_until_ready(key_returns)
            evaluation_seconds = (
                time.perf_counter() - evaluation_started_at
                if evaluation_started_at is not None
                else 0.0
            )
            keys = jnp.asarray(
                key_preferences(self.config.reward_size),
                dtype=extra.raw_key_solutions.dtype,
            )
            solutions, improved, interpolator = update_key_solutions_jax(
                extra.raw_key_solutions,
                key_returns,
                keys,
            )
            extra = state.agent_state.extra_state.replace(
                raw_key_solutions=solutions,
                interpolator=interpolator,
            )
            state = state.replace(
                agent_state=state.agent_state.replace(extra_state=extra)
            )
            num_replacements = int(improved.sum())
            control_metrics["control/key_replacements"] = num_replacements
            self.key_replacement_count = (
                getattr(self, "key_replacement_count", 0) + num_replacements
            )
            self.runtime_counters["key_replacement_count"] += num_replacements
            if diagnostics_enabled:
                logging_started_at = time.perf_counter()
                host_values = jax.device_get(
                    (
                        raw_episode_count,
                        old_key_solutions,
                        key_returns,
                        improved,
                        solutions,
                        keys,
                    )
                )
                raw_counts, old, returns, host_improved, new, host_keys = host_values
                rows = build_key_diagnostic_rows(
                    seed=int(self.config.seed),
                    run_id=str(
                        self.config.get("run_id", Path(self.config.output_dir).name)
                    ),
                    artifact_sha256=self.interpolator_artifact.sha256,
                    iteration=int(jax.device_get(state.metrics.iterations)),
                    actual_env_transitions=int(
                        jax.device_get(state.metrics.sampled_timesteps)
                    ),
                    event_id=eval_cnt_ep_before,
                    eval_cnt_ep_before=eval_cnt_ep_before,
                    episode_count_raw=raw_counts,
                    episode_count_logical=episode_count,
                    keys=host_keys,
                    old_solutions=old,
                    returns_per_repeat=returns,
                    improved=host_improved,
                    new_solutions=new,
                    evaluation_seconds=evaluation_seconds,
                    logging_seconds=time.perf_counter() - logging_started_at,
                )
                append_key_diagnostics(
                    Path(self.config.output_dir) / "key_replacement_diagnostics.csv",
                    rows,
                )

        extra = state.agent_state.extra_state
        sampled_timesteps = int(jax.device_get(state.metrics.sampled_timesteps))
        full_eval_due = full_evaluation_due(
            episode_count, int(jax.device_get(extra.eval_cnt)), eval_freq=100
        )

        if full_eval_due:
            eval_cnt = extra.eval_cnt + jnp.uint32(1)
            state = state.replace(
                agent_state=state.agent_state.replace(
                    extra_state=extra.replace(eval_cnt=eval_cnt)
                )
            )
            result = self.morl_evaluator.evaluate(
                state.agent_state, preference_grid(0.005), repeats=3
            )
            if bool(history.get("enable", False)) and bool(
                history.get("write_csv", True)
            ):
                append_hv_history(
                    Path(self.config.output_dir) / "hv_history.csv",
                    seed=int(self.config.seed),
                    run_id=str(
                        self.config.get("run_id", Path(self.config.output_dir).name)
                    ),
                    artifact_sha256=self.interpolator_artifact.sha256,
                    iteration=int(jax.device_get(state.metrics.iterations)),
                    actual_env_transitions=sampled_timesteps,
                    result=result,
                    wall_clock_seconds=(
                        time.perf_counter() - self._learning_started_at
                        if hasattr(self, "_learning_started_at")
                        else None
                    ),
                )
            self._write_evaluation_snapshot(
                "training_full",
                result,
                int(jax.device_get(state.metrics.iterations)),
                sampled_timesteps=sampled_timesteps,
            )
            if bool(history.get("console_output", False)):
                control_metrics.update(
                    {
                        "eval/hypervolume": float(
                            getattr(result, "source_hv", result.mean_hv)
                        ),
                        "eval/sparsity": float(
                            getattr(result, "source_sparsity", result.mean_sparsity)
                        ),
                    }
                )

        control_metrics.update(self._maybe_evaluate_convergence(state))
        return state, control_metrics

    def _convergence_run_id(self) -> str:
        configured = self.config.get("run_id")
        if configured:
            return str(configured)
        path = Path(self.config.output_dir) / "hv_convergence.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("kind") == "convergence" and row.get("run_id"):
                        return row["run_id"]
        return Path(self.config.output_dir).name

    def _prepare_convergence_schedule(self, state: State) -> None:
        config = self.config.get("convergence_eval", {})
        if not bool(config.get("enable", False)):
            return
        interval = int(config.get("interval_transitions", 0))
        sampled = int(jax.device_get(state.metrics.sampled_timesteps))
        self._next_convergence_threshold = prepare_convergence_history(
            Path(self.config.output_dir) / "hv_convergence.csv",
            run_id=self._convergence_run_id(),
            sampled_timesteps=sampled,
            interval=interval,
        )

    def _maybe_evaluate_convergence(self, state: State) -> dict[str, float]:
        config = self.config.get("convergence_eval", {})
        if not bool(config.get("enable", False)):
            return {}
        interval = int(config.get("interval_transitions", 0))
        if interval <= 0:
            raise ValueError("convergence_eval.interval_transitions must be positive")
        sampled = int(jax.device_get(state.metrics.sampled_timesteps))
        target = getattr(self, "_next_convergence_threshold", None)
        if target is None:
            self._prepare_convergence_schedule(state)
            target = self._next_convergence_threshold
        if sampled < target:
            return {}

        started_at = time.perf_counter()
        result = self.morl_evaluator.evaluate(
            state.agent_state,
            preference_grid(float(config.get("preference_step", 0.02))),
            repeats=int(config.get("repeats", 3)),
        )
        evaluation_seconds = time.perf_counter() - started_at
        if bool(config.get("write_csv", True)):
            append_convergence_result(
                Path(self.config.output_dir) / "hv_convergence.csv",
                seed=int(self.config.seed),
                run_id=self._convergence_run_id(),
                artifact_sha256=self.interpolator_artifact.sha256,
                target_transition_threshold=target,
                actual_env_transitions=sampled,
                iteration=int(jax.device_get(state.metrics.iterations)),
                result=result,
                evaluation_seconds=evaluation_seconds,
                wall_clock_seconds=(
                    time.perf_counter() - self._learning_started_at
                    if hasattr(self, "_learning_started_at")
                    else None
                ),
            )
        self._next_convergence_threshold = (sampled // interval + 1) * interval
        if not bool(config.get("console_output", False)):
            return {}
        return {
            "convergence/source_hv": float(
                getattr(result, "source_hv", result.mean_hv)
            ),
            "convergence/evaluation_seconds": evaluation_seconds,
        }

    def _observe_diagnostics(self, iteration, train_metrics, state):
        """Write periodic formal-run diagnostics without touching learner state."""
        raw = train_metrics["raw_loss_dict"]
        finite = all(np.isfinite(np.asarray(value)).all() for value in raw.values())
        self.diagnostics_finite = getattr(self, "diagnostics_finite", True) and finite
        sampled = int(jax.device_get(state.metrics.sampled_timesteps))
        her_threshold = self._her_base_transition_threshold()
        worker_steps_arr = np.asarray(
            jax.device_get(state.agent_state.extra_state.worker_steps)
        ).astype(np.int64)
        warmup_limit = int(self.config.start_timesteps)
        random_counts = np.minimum(worker_steps_arr, warmup_limit)
        policy_counts = np.maximum(worker_steps_arr - warmup_limit, 0)
        payload = {
            "iteration": int(iteration),
            **{key: np.asarray(value).tolist() for key, value in raw.items()},
            "finite": bool(finite),
            "replay_size": int(jax.device_get(state.replay_buffer_state.buffer_size)),
            "base_inserts": sampled,
            "her_inserts": max(sampled - her_threshold, 0)
            * int(self.config.num_relabel_preferences),
            "her_active": sampled > her_threshold,
            "worker_steps": worker_steps_arr.tolist(),
            "random_transition_count_per_group": random_counts.tolist(),
            "policy_transition_count_per_group": policy_counts.tolist(),
            "global_random_transition_count": int(random_counts.sum()),
            "global_policy_transition_count": int(policy_counts.sum()),
            "random_action_fraction": float(np.mean(worker_steps_arr < warmup_limit)),
            "episode_count": np.asarray(
                jax.device_get(state.agent_state.extra_state.episode_count)
            ).tolist(),
            "eval_cnt_ep": int(
                jax.device_get(state.agent_state.extra_state.eval_cnt_ep)
            ),
            "eval_cnt": int(jax.device_get(state.agent_state.extra_state.eval_cnt)),
            "key_evaluation_count": int(
                jax.device_get(state.agent_state.extra_state.eval_cnt_ep)
            )
            - 1,
            "key_replacement_count": getattr(self, "key_replacement_count", 0),
            "interpolator_refit_count": int(
                jax.device_get(state.agent_state.extra_state.eval_cnt_ep)
            )
            - 1,
            "full_evaluation_count": int(
                jax.device_get(state.agent_state.extra_state.eval_cnt)
            )
            - 1,
        }
        path = Path(self.config.output_dir) / "training_diagnostics.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _write_evaluation_snapshot(
        self, kind, result, iteration, *, sampled_timesteps=None
    ):
        """Persist HV/Pareto progression without changing learner state."""
        path = Path(self.config.output_dir) / "pd_morl_evaluations.jsonl"
        mean_returns = np.asarray(result.mean_returns)
        obj_stats = {}
        for l in range(mean_returns.shape[-1]):
            obj_stats[f"obj{l + 1}_min"] = float(mean_returns[:, l].min())
            obj_stats[f"obj{l + 1}_max"] = float(mean_returns[:, l].max())
            obj_stats[f"obj{l + 1}_mean"] = float(mean_returns[:, l].mean())

        pareto_counts = (
            result.pareto_counts_per_repeat.tolist()
            if hasattr(result, "pareto_counts_per_repeat")
            and len(result.pareto_counts_per_repeat) > 0
            else [len(result.pareto_returns)] * len(result.hv_per_repeat)
        )
        mean_pareto_count = (
            float(result.mean_pareto_count)
            if hasattr(result, "mean_pareto_count")
            else float(len(result.pareto_returns))
        )
        pref_return_map = [
            {"preference": pref.tolist(), "mean_return": ret.tolist()}
            for pref, ret in zip(result.preferences, result.mean_returns)
        ]

        source_hv = float(getattr(result, "source_hv", result.mean_hv))
        source_sparsity = float(
            getattr(result, "source_sparsity", result.mean_sparsity)
        )
        source_count = int(
            getattr(result, "source_pareto_point_count", len(result.pareto_returns))
        )
        mean_rep_hv = float(getattr(result, "mean_repeat_hv", result.mean_hv))
        mean_rep_sp = float(
            getattr(result, "mean_repeat_sparsity", result.mean_sparsity)
        )
        mean_rep_cnt = float(
            getattr(result, "mean_repeat_pareto_count", mean_pareto_count)
        )

        payload = {
            "kind": kind,
            "iteration": int(iteration),
            "environment_transitions": sampled_timesteps,
            "actual_env_transitions": sampled_timesteps,
            "wall_clock_seconds": (
                time.perf_counter() - self._learning_started_at
                if hasattr(self, "_learning_started_at")
                else None
            ),
            "source_hv": source_hv,
            "source_sparsity": source_sparsity,
            "source_pareto_point_count": source_count,
            "mean_repeat_hv": mean_rep_hv,
            "mean_repeat_sparsity": mean_rep_sp,
            "mean_repeat_pareto_count": mean_rep_cnt,
            "repeat_hv": result.hv_per_repeat.tolist(),
            "repeat_sparsity": result.sparsity_per_repeat.tolist(),
            "repeat_pareto_point_count": pareto_counts,
            "final_hv": source_hv,
            "final_sparsity": source_sparsity,
            "final_pareto_point_count": source_count,
            "mean_pareto_point_count": mean_rep_cnt,
            "preferences": result.preferences.tolist(),
            "returns_per_repeat": result.returns_per_repeat.tolist(),
            "mean_returns": result.mean_returns.tolist(),
            "hv_per_repeat": result.hv_per_repeat.tolist(),
            "sparsity_per_repeat": result.sparsity_per_repeat.tolist(),
            "pareto_counts_per_repeat": pareto_counts,
            "mean_hv": mean_rep_hv,
            "mean_sparsity": mean_rep_sp,
            "pareto_indices": result.pareto_indices.tolist(),
            "pareto_returns": result.pareto_returns.tolist(),
            **obj_stats,
            "preference_return_map": pref_return_map,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        if kind == "training_final":
            target = (
                Path(self.config.output_dir) / "final_training_eval" / "results.json"
            )
            npz_target = (
                Path(self.config.output_dir)
                / "final_training_eval"
                / "pareto_artifacts.npz"
            )
            npz_target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                npz_target,
                preferences=result.preferences,
                returns_per_repeat=result.returns_per_repeat,
                mean_returns=result.mean_returns,
                pareto_returns=result.pareto_returns,
                pareto_indices=result.pareto_indices,
                hv_per_repeat=result.hv_per_repeat,
                sparsity_per_repeat=result.sparsity_per_repeat,
                pareto_counts_per_repeat=np.asarray(pareto_counts),
                source_hv=source_hv,
                source_sparsity=source_sparsity,
                source_pareto_point_count=source_count,
                mean_repeat_hv=mean_rep_hv,
                mean_repeat_sparsity=mean_rep_sp,
                mean_repeat_pareto_count=mean_rep_cnt,
            )
        else:
            target = (
                Path(self.config.output_dir)
                / "pareto_fronts"
                / f"{kind}_{iteration}.json"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def evaluate_offline(self, state: State):
        """Run the official 1001-preference, six-repeat offline benchmark."""
        return self.morl_evaluator.evaluate(
            state.agent_state, preference_grid(0.001), repeats=6
        )

    def _periodic_evaluation_due(self, iterations, final_iteration):
        del iterations, final_iteration
        return False

    def _after_learning(self, state: State):
        """Run the source training-final 1001-preference, three-repeat evaluation."""
        iterations = int(jax.device_get(state.metrics.iterations))
        if bool(self.config.get("checkpoint", {}).get("enable", False)):
            saved_state = state
            if (
                hasattr(self.config, "save_replay_buffer")
                and not self.config.save_replay_buffer
            ):
                saved_state = skip_replay_buffer_state(saved_state)
            if self.checkpoint_manager.latest_step() != iterations:
                self.checkpoint_manager.save(iterations, saved_state, force=True)
            self.checkpoint_manager.wait_until_finished()

        if getattr(self.config, "run_final_evaluation", True):
            result = self.morl_evaluator.evaluate(
                state.agent_state, preference_grid(0.001), repeats=3
            )
            self._write_evaluation_snapshot(
                "training_final",
                result,
                iterations,
                sampled_timesteps=int(jax.device_get(state.metrics.sampled_timesteps)),
            )
            if bool(self.config.get("hv_history", {}).get("console_output", False)):
                self.recorder.write(
                    {
                        "eval/final_hypervolume": float(
                            getattr(result, "source_hv", result.mean_hv)
                        ),
                        "eval/final_sparsity": float(
                            getattr(result, "source_sparsity", result.mean_sparsity)
                        ),
                        "eval/final_pareto_point_count": int(
                            getattr(
                                result,
                                "source_pareto_point_count",
                                len(result.pareto_returns),
                            )
                        ),
                    },
                    iterations,
                )
        history = self.config.get("hv_history", {})
        if bool(history.get("enable", False)) and bool(
            history.get("plot_on_finish", True)
        ):
            csv_path = Path(self.config.output_dir) / "hv_history.csv"
            if csv_path.exists():
                import subprocess
                import sys

                subprocess.run(
                    [
                        sys.executable,
                        str(
                            Path(__file__).parents[2] / "scripts" / "plot_pd_morl_hv.py"
                        ),
                        "--input",
                        str(csv_path),
                        "--output",
                        str(Path(self.config.output_dir) / "hv_curve.png"),
                    ],
                    check=True,
                    cwd=Path(__file__).parents[2],
                )

        convergence = self.config.get("convergence_eval", {})
        convergence_csv = Path(self.config.output_dir) / "hv_convergence.csv"
        if (
            bool(convergence.get("enable", False))
            and bool(convergence.get("plot_on_finish", True))
        ):
            try:
                plot_convergence_csv(
                    convergence_csv,
                    Path(self.config.output_dir) / "hv_convergence.png",
                )
            except Exception as error:  # noqa: BLE001 - post-processing is noncritical
                (
                    Path(self.config.output_dir) / "hv_convergence_plot_error.log"
                ).write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")

        return state

    def _parallel_actor_update_mask(self, state):
        update_ids = state.agent_state.extra_state.total_it + jnp.arange(
            1, self._critic_updates_per_rollout() + 1
        )
        return update_ids % self.config.actor_update_interval == 0

    def _empty_actor_loss_dict(self):
        zero = jnp.zeros(())
        return PyTreeDict(
            actor_loss=zero,
            actor_total_loss=zero,
            actor_scalarized_term=zero,
            actor_angle=zero,
            actor_raw_grad_norm=zero,
        )

    def learn(self, state: State) -> State:
        self._learning_started_at = time.perf_counter()
        self.runtime_timing["host_chunk_time"] = 0.0
        self._prepare_convergence_schedule(state)
        num_devices = jax.device_count()
        one_step_timesteps = self.config.rollout_length * self.config.num_envs
        sampled_timesteps = int(jax.device_get(state.metrics.sampled_timesteps))
        start_iteration = int(jax.device_get(state.metrics.iterations))
        fold_iters = int(self.config.fold_iters)
        transitions_per_chunk = one_step_timesteps * fold_iters * num_devices
        num_chunks, final_iteration = training_chunk_plan(
            int(self.config.total_timesteps),
            sampled_timesteps,
            start_iteration,
            transitions_per_chunk,
            fold_iters,
        )

        critic_per_rollout = int(self._critic_updates_per_rollout())
        actor_update_interval = int(self.config.actor_update_interval)

        for chunk_index in range(num_chunks):
            is_last_chunk = chunk_index == num_chunks - 1
            self.runtime_counters["host_chunk_count"] += 1
            chunk_started_at = time.perf_counter()
            train_metrics, state = self._multi_steps(state)
            jax.block_until_ready(state.agent_state.params)
            self.runtime_timing["host_chunk_time"] += (
                time.perf_counter() - chunk_started_at
            )
            self.last_train_metrics = train_metrics

            self.runtime_counters["inner_rollout_count"] += fold_iters
            self.runtime_counters["environment_step_count"] += int(
                fold_iters * self.config.rollout_length * self.config.num_envs
            )
            critic_before = self.runtime_counters["critic_optimizer_step_count"]
            critic_after = critic_before + fold_iters * critic_per_rollout
            actor_updates = periodic_update_count(
                critic_before, fold_iters * critic_per_rollout, actor_update_interval
            )
            self.runtime_counters["critic_optimizer_step_count"] = critic_after
            self.runtime_counters["replay_sample_call_count"] += (
                fold_iters * critic_per_rollout
            )
            self.runtime_counters["actor_optimizer_step_count"] += actor_updates
            self.runtime_counters["target_update_count"] += actor_updates

            state, control_metrics = self._after_multi_steps(state)
            workflow_metrics = state.metrics

            iterations = state.metrics.iterations.tolist()
            if (
                iterations % int(self.config.get("log_interval", 1)) == 0
                or is_last_chunk
            ):
                train_metrics_dict = train_metrics.to_local_dict()
                workflow_metrics_dict = workflow_metrics.to_local_dict()
                counter_metrics = {
                    f"counters/{k}": v for k, v in self.runtime_counters.items()
                }
                self.recorder.write(train_metrics_dict, iterations)
                self.recorder.write(workflow_metrics_dict, iterations)
                self.recorder.write(counter_metrics, iterations)
                observe = getattr(self, "_observe_diagnostics", None)
                if observe is not None:
                    observe(iterations, train_metrics_dict, state)
            if control_metrics:
                self.recorder.write(control_metrics, iterations)

            if self._periodic_evaluation_due(iterations, final_iteration):
                eval_metrics, state = self.evaluate(state)
                self.recorder.write(
                    add_prefix(eval_metrics.to_local_dict(), "eval"), iterations
                )

            if bool(self.config.get("checkpoint", {}).get("enable", False)):
                saved_state = state
                if not self.config.save_replay_buffer:
                    saved_state = skip_replay_buffer_state(saved_state)
                self.checkpoint_manager.save(
                    iterations, saved_state, force=is_last_chunk
                )

        state = self._after_learning(state)
        session_transitions = int(
            jax.device_get(state.metrics.sampled_timesteps)
        ) - sampled_timesteps
        session_seconds = self.runtime_timing["host_chunk_time"]

        summary_path = Path(self.config.output_dir) / "counters_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            **self.runtime_counters,
            "timing": {
                **self.runtime_timing,
                "transitions_per_host_chunk": transitions_per_chunk,
                "session_host_chunk_count": num_chunks,
                "session_train_transitions": session_transitions,
                "total_measured_transitions": session_transitions,
                "steady_state_transitions_per_second": (
                    session_transitions / session_seconds
                    if session_seconds > 0
                    else None
                ),
            },
            "derived_verification": {
                "expected_critic_steps_from_rollouts": self.runtime_counters[
                    "inner_rollout_count"
                ]
                * critic_per_rollout,
                "matches_critic_optimizer_steps": (
                    self.runtime_counters["inner_rollout_count"] * critic_per_rollout
                    == self.runtime_counters["critic_optimizer_step_count"]
                ),
                "deprecated_alias_62240_explained": "Erroneous alias 62,240 was produced by double-multiplying inner_rollouts (1556) by fold_iters * K (40). The real physical optimizer count is 15,560.",
            },
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        return state


class PDMORLGPUWorkflow(MOTD3Workflow):
    """GPU-native batched PD-MORL; not optimizer-trajectory equivalent."""

    @classmethod
    def name(cls):
        return "GPU-native PD-MORL"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        jax.config.update("jax_default_matmul_precision", config.matmul_precision)
        if config.process_count != 10:
            raise ValueError("GPU-native PD-MORL keeps exactly 10 preference groups")
        expected_envs = config.process_count * config.envs_per_preference_group
        if config.num_envs != expected_envs:
            raise ValueError(
                "num_envs must equal process_count * envs_per_preference_group: "
                f"expected {expected_envs}, got {config.num_envs}"
            )
        if config.critic_updates_per_rollout < 1:
            raise ValueError("critic_updates_per_rollout must be positive")
        if config.random_timesteps > config.learning_start_timesteps:
            raise ValueError("random_timesteps cannot exceed learning_start_timesteps")
        random_prefill = (config.random_timesteps // config.num_envs) * config.num_envs
        policy_prefill = (
            math.ceil(
                (config.learning_start_timesteps - random_prefill) / config.num_envs
            )
            * config.num_envs
        )
        prefill_transitions = random_prefill + policy_prefill
        chunk_transitions = config.num_envs * config.rollout_length * config.fold_iters
        remaining = config.total_timesteps - prefill_transitions
        if remaining < 0 or remaining % chunk_transitions:
            raise ValueError(
                "GPU-native total_timesteps must include prefill and leave a whole "
                "number of static training chunks: "
                f"total={config.total_timesteps}, prefill={prefill_transitions}, "
                f"chunk={chunk_transitions}"
            )

        keys = key_preferences(config.reward_size)
        key_solutions, artifact = cls._load_configured_key_solutions(config, keys)
        interpolator_state = fit_interpolator_state(keys, key_solutions, "initial")

        env_kwargs = {
            "episode_length": config.env.max_episode_steps,
            "autoreset_mode": AutoresetMode.NORMAL,
            "record_ori_obs": True,
            "vector_reward": True,
            "episode_preference": True,
            "process_count": config.process_count,
        }
        env = create_env(config.env, parallel=config.num_envs, **env_kwargs)
        agent = make_mo_td3_agent(
            action_space=env.action_space,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            critic_hidden_layer_sizes=config.agent_network.critic_hidden_layer_sizes,
            reward_size=config.reward_size,
            discount=config.discount,
            exploration_epsilon=config.exploration_epsilon,
            policy_noise=config.policy_noise,
            clip_policy_noise=config.clip_policy_noise,
            normalize_obs=config.normalize_obs,
            process_count=config.process_count,
            start_timesteps=config.start_timesteps,
            actor_loss_coeff=config.actor_loss_coeff,
            interpolator_state=interpolator_state,
            initial_key_solutions=jnp.asarray(key_solutions, dtype=jnp.float32),
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.optimizer.grad_clip_norm),
            optax.adam(config.optimizer.lr),
        )
        replay_buffer = ReplayBuffer(
            capacity=config.replay_buffer_capacity,
            min_sample_timesteps=max(
                config.replay_batch_size, config.learner_start_replay_entries
            ),
            sample_batch_size=config.replay_batch_size,
        )
        eval_env = create_env(
            config.env,
            parallel=config.num_eval_envs,
            episode_length=config.env.max_episode_steps,
            autoreset_mode=AutoresetMode.DISABLED,
            vector_reward=True,
            episode_preference=True,
            process_count=config.process_count,
        )
        evaluator = Evaluator(
            env=eval_env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=config.env.max_episode_steps,
        )
        workflow = cls(env, agent, optimizer, evaluator, replay_buffer, config)

        control_kwargs = {
            "episode_length": config.env.max_episode_steps,
            "autoreset_mode": AutoresetMode.DISABLED,
            "vector_reward": True,
        }
        key_eval_env = create_env(config.env, parallel=9, **control_kwargs)
        full_eval_env = create_env(
            config.env, parallel=config.eval_batch_size, **control_kwargs
        )
        workflow.morl_evaluator = BatchedPDMORLEvaluator(
            key_env=key_eval_env,
            env=full_eval_env,
            agent=agent,
            max_episode_steps=config.env.max_episode_steps,
        )
        workflow.interpolator_artifact = artifact
        return workflow

    def _postsetup_replaybuffer(self, state: State) -> State:
        state = super()._postsetup_replaybuffer(state)
        prefill_per_group = (
            self.config.learning_start_timesteps + self.config.process_count - 1
        ) // self.config.process_count
        return state.replace(
            agent_state=state.agent_state.replace(
                extra_state=state.agent_state.extra_state.replace(
                    worker_steps=jnp.full(
                        (self.config.process_count,),
                        prefill_per_group,
                        dtype=jnp.uint32,
                    )
                )
            )
        )

    def _after_rollout_agent_state(self, agent_state, trajectory_dones):
        group_transitions = int(self.config.envs_per_preference_group) * int(
            self.config.rollout_length
        )
        return agent_state.replace(
            extra_state=agent_state.extra_state.replace(
                worker_steps=(
                    agent_state.extra_state.worker_steps + jnp.uint32(group_transitions)
                ),
                episode_count=(
                    agent_state.extra_state.episode_count
                    + completed_episodes_by_worker(
                        trajectory_dones, self.config.process_count
                    )
                ),
                total_it=(
                    agent_state.extra_state.total_it
                    + jnp.uint32(self._critic_updates_per_rollout())
                ),
            )
        )

    def _critic_updates_per_rollout(self) -> int:
        return self.config.critic_updates_per_rollout

    def _add_to_replay_buffer(self, replay_buffer_state, trajectory, key):
        return add_her_transitions(
            self.replay_buffer,
            replay_buffer_state,
            trajectory,
            jax.random.fold_in(key, 0x484552),
            self.config.num_relabel_preferences,
            self.config.her_start_timesteps,
            self.config.num_envs,
            base_transition_threshold=self.config.her_start_base_transitions,
        )

    def _control_episode_count(self, extra):
        completed = np.asarray(jax.device_get(extra.episode_count))
        return completed // int(self.config.envs_per_preference_group)

    def _her_threshold_multiplier(self) -> int:
        return self.config.num_envs

    def _her_base_transition_threshold(self) -> int:
        return int(self.config.her_start_base_transitions)

    def _after_learning(self, state: State):
        return super()._after_learning(state)
