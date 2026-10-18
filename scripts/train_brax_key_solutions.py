"""Train the three PD-MORL Brax Walker2d key policies."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if __name__ == "__main__":
    # Set before imports initialize the backend: a fixed key does not stabilize GPU reductions.
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
    ).strip()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from evorl.algorithms.mo_td3 import (
    PreferenceActor,
    TwinVectorCritic,
    twin_smooth_l1_loss,
)
from evorl.envs.brax import create_wrapped_brax_env
from evorl.envs.wrappers.training_wrapper import AutoresetMode
from evorl.replay_buffers import ReplayBuffer
from evorl.rollout import fast_eval_rollout_episode, rollout
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.utils.pd_morl_interpolator import key_preferences


SEED = 42
NUM_ENVS = 64
ROLLOUT_LENGTH = 1
CHUNK_SIZE = NUM_ENVS * ROLLOUT_LENGTH
CRITIC_UPDATES_PER_CHUNK = 8
TOTAL_ENV_STEPS = 5_000_000
RANDOM_ACTION_STEPS = 25_088
BATCH_SIZE = 256
SMOKE_ENV_STEPS = 1024
SMOKE_RANDOM_ACTION_STEPS = 768
REPLAY_CAPACITY = 500_000
POLICY_FREQ = 2
EVAL_EPISODES = 10
HORIZON = 500
REWARD_SIZE = 2
HIDDEN_SIZES = (400, 400)
LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005
EXPLORATION_NOISE = 0.1
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5


def sample_training_preference(
    key: jax.Array, batch_shape: tuple[int, ...], base_preference: jax.Array
) -> jax.Array:
    """Sample the source-style non-negative, unit-L1 training preference."""
    noise = jnp.clip(
        jax.random.normal(key, (*batch_shape, REWARD_SIZE)) * 0.05,
        -0.05,
        0.05,
    )
    preference = jnp.abs(jnp.asarray(base_preference) + noise)
    return preference / jnp.maximum(
        preference.sum(axis=-1, keepdims=True),
        jnp.finfo(preference.dtype).eps,
    )


def evaluation_preference(
    base_preference: jax.Array, batch_shape: tuple[int, ...]
) -> jax.Array:
    """Broadcast the unperturbed key used for deterministic evaluation."""
    return jnp.broadcast_to(base_preference, (*batch_shape, REWARD_SIZE))


def replay_next_obs_and_done(trajectory: SampleBatch, obs_dim: int):
    """Return true successor observations and both episode-boundary signals."""
    extras = trajectory.extras.env_extras
    required = ("ori_obs", "termination", "truncation")
    missing = [name for name in required if name not in extras]
    if missing:
        raise ValueError(f"rollout is missing required env extras: {missing}")
    next_obs = extras.ori_obs.reshape(-1, obs_dim)
    done = jnp.maximum(extras.termination, extras.truncation).reshape(-1)
    return next_obs, done


def updates_for_chunk(chunk_index: int) -> int:
    """Perform a fixed update block once replay reaches two minibatches."""
    return CRITIC_UPDATES_PER_CHUNK if (chunk_index + 1) * CHUNK_SIZE >= 2 * BATCH_SIZE else 0


def expected_update_counts(
    total_env_steps: int = TOTAL_ENV_STEPS,
    random_action_steps: int = RANDOM_ACTION_STEPS,
) -> dict[str, int]:
    """Return the static update schedule required by the vectorized protocol."""
    if total_env_steps % CHUNK_SIZE or random_action_steps % CHUNK_SIZE:
        raise ValueError("budgets must be divisible by num_envs * rollout_length")
    total_chunks = total_env_steps // CHUNK_SIZE
    random_chunks = random_action_steps // CHUNK_SIZE
    if total_chunks < 1 or random_chunks < 1 or random_chunks > total_chunks:
        raise ValueError("invalid training budget")
    initial_no_update_chunks = (2 * BATCH_SIZE - 1) // CHUNK_SIZE
    critic = max(0, total_chunks - initial_no_update_chunks) * CRITIC_UPDATES_PER_CHUNK
    return {
        "chunk_size": CHUNK_SIZE,
        "total_chunks": total_chunks,
        "random_action_chunks": random_chunks,
        "initial_no_update_chunks": initial_no_update_chunks,
        "first_learning_chunk_updates": updates_for_chunk(initial_no_update_chunks),
        "critic_optimizer_steps": critic,
        "actor_optimizer_steps": critic // POLICY_FREQ,
    }


def train_single_key(
    preference: np.ndarray,
    seed: int,
    *,
    total_env_steps: int = TOTAL_ENV_STEPS,
    random_action_steps: int = RANDOM_ACTION_STEPS,
    smoke: bool = False,
    output_dir: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Train one key and return its best fixed-key evaluation result."""
    counts = expected_update_counts(total_env_steps, random_action_steps)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=False)
    base_preference = jnp.asarray(preference, dtype=jnp.float32)
    rng = jax.random.PRNGKey(seed)

    train_env = create_wrapped_brax_env(
        "walker2d",
        episode_length=HORIZON,
        parallel=NUM_ENVS,
        autoreset_mode=AutoresetMode.NORMAL,
        record_ori_obs=True,
        vector_reward=True,
    )
    eval_env = create_wrapped_brax_env(
        "walker2d",
        episode_length=HORIZON,
        parallel=EVAL_EPISODES,
        autoreset_mode=AutoresetMode.DISABLED,
        vector_reward=True,
    )
    obs_dim = train_env.obs_space.shape[0]
    act_dim = train_env.action_space.shape[0]
    actor = PreferenceActor(action_size=act_dim, hidden_layer_sizes=HIDDEN_SIZES)
    critic = TwinVectorCritic(reward_size=REWARD_SIZE, hidden_layer_sizes=HIDDEN_SIZES)

    rng, actor_key, critic_key = jax.random.split(rng, 3)
    zero_obs = jnp.zeros((obs_dim,))
    zero_action = jnp.zeros((act_dim,))
    actor_params = actor.init(actor_key, zero_obs, base_preference)
    critic_params = critic.init(critic_key, zero_obs, base_preference, zero_action)
    target_actor_params = actor_params
    target_critic_params = critic_params
    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(100.0), optax.adam(LEARNING_RATE)
    )
    critic_optimizer = optax.chain(
        optax.clip_by_global_norm(100.0), optax.adam(LEARNING_RATE)
    )
    actor_optimizer_state = actor_optimizer.init(actor_params)
    critic_optimizer_state = critic_optimizer.init(critic_params)

    replay = ReplayBuffer(capacity=REPLAY_CAPACITY, sample_batch_size=BATCH_SIZE)
    replay_state = replay.init(
        SampleBatch(
            obs=zero_obs,
            actions=zero_action,
            rewards=jnp.zeros((REWARD_SIZE,)),
            dones=jnp.zeros(()),
            next_obs=zero_obs,
            extras=PyTreeDict(preference=base_preference),
        )
    )

    def warmup_action_fn(_params, state_batch, key):
        action_key, preference_key = jax.random.split(key)
        preference_batch = sample_training_preference(
            preference_key, state_batch.obs.shape[:-1], base_preference
        )
        actions = jax.random.uniform(
            action_key,
            (*state_batch.obs.shape[:-1], act_dim),
            minval=-1.0,
            maxval=1.0,
        )
        return actions, PyTreeDict(preference=preference_batch)

    def train_action_fn(params, state_batch, key):
        action_key, preference_key = jax.random.split(key)
        preference_batch = sample_training_preference(
            preference_key, state_batch.obs.shape[:-1], base_preference
        )
        actions = actor.apply(params, state_batch.obs, preference_batch)
        actions = actions + jax.random.normal(action_key, actions.shape) * EXPLORATION_NOISE
        return jnp.clip(actions, -1.0, 1.0), PyTreeDict(preference=preference_batch)

    def eval_action_fn(params, state_batch, _key):
        preference_batch = evaluation_preference(
            base_preference, state_batch.obs.shape[:-1]
        )
        return actor.apply(params, state_batch.obs, preference_batch), PyTreeDict()

    def make_train_chunk(random_actions: bool, update_count: int):
        @jax.jit
        def train_chunk(carry, _unused):
            (
                key,
                env_state,
                current_actor,
                current_critic,
                current_target_actor,
                current_target_critic,
                current_actor_opt,
                current_critic_opt,
                current_replay,
                critic_step,
            ) = carry
            key, rollout_key, update_key = jax.random.split(key, 3)
            action_fn = (
                warmup_action_fn
                if random_actions
                else lambda _state, batch, step_key: train_action_fn(
                    current_actor, batch, step_key
                )
            )
            trajectory, env_state = rollout(
                train_env.step,
                action_fn,
                env_state,
                None,
                rollout_key,
                ROLLOUT_LENGTH,
                env_extra_fields=("ori_obs", "termination", "truncation"),
            )
            flat_obs = trajectory.obs.reshape(-1, obs_dim)
            flat_actions = trajectory.actions.reshape(-1, act_dim)
            flat_rewards = trajectory.rewards.reshape(-1, REWARD_SIZE)
            flat_preferences = trajectory.extras.policy_extras.preference.reshape(
                -1, REWARD_SIZE
            )
            flat_next_obs, flat_done = replay_next_obs_and_done(trajectory, obs_dim)
            current_replay = replay.add(
                current_replay,
                SampleBatch(
                    obs=flat_obs,
                    actions=flat_actions,
                    rewards=flat_rewards,
                    dones=flat_done,
                    next_obs=flat_next_obs,
                    extras=PyTreeDict(preference=flat_preferences),
                ),
            )
            completed_episodes = jnp.sum(trajectory.dones, dtype=jnp.uint32)

            def update_step(inner, step_key):
                (
                    inner_critic,
                    inner_critic_opt,
                    inner_actor,
                    inner_actor_opt,
                    inner_target_actor,
                    inner_target_critic,
                    inner_step,
                    inner_last_actor_loss,
                ) = inner
                sample_key, noise_key = jax.random.split(step_key)
                sample = replay.sample(current_replay, sample_key)
                sampled_preference = sample.extras.preference
                next_actions = actor.apply(
                    inner_target_actor, sample.next_obs, sampled_preference
                )
                next_actions = next_actions + jnp.clip(
                    jax.random.normal(noise_key, next_actions.shape) * POLICY_NOISE,
                    -NOISE_CLIP,
                    NOISE_CLIP,
                )
                next_actions = jnp.clip(next_actions, -1.0, 1.0)
                twin_q = critic.apply(
                    inner_target_critic,
                    sample.next_obs,
                    sampled_preference,
                    next_actions,
                )
                q1_scalar = jnp.sum(twin_q[..., 0, :] * sampled_preference, axis=-1)
                q2_scalar = jnp.sum(twin_q[..., 1, :] * sampled_preference, axis=-1)
                target_q = jnp.where(
                    q1_scalar[..., None] <= q2_scalar[..., None],
                    twin_q[..., 0, :],
                    twin_q[..., 1, :],
                )
                target_q = jax.lax.stop_gradient(
                    sample.rewards
                    + GAMMA * (1.0 - sample.dones[..., None]) * target_q
                )

                def critic_loss(params):
                    values = critic.apply(
                        params, sample.obs, sampled_preference, sample.actions
                    )
                    return twin_smooth_l1_loss(values, target_q)

                critic_loss_value, critic_grad = jax.value_and_grad(critic_loss)(
                    inner_critic
                )
                critic_updates, inner_critic_opt = critic_optimizer.update(
                    critic_grad, inner_critic_opt
                )
                inner_critic = optax.apply_updates(inner_critic, critic_updates)
                inner_step = inner_step + 1
                should_update_actor = inner_step % POLICY_FREQ == 0

                def update_actor(params, params_opt, target_actor, target_critic):
                    def actor_loss(actor_params):
                        actions = actor.apply(
                            actor_params, sample.obs, sampled_preference
                        )
                        values = critic.apply(
                            inner_critic, sample.obs, sampled_preference, actions
                        )
                        return -jnp.mean(
                            jnp.sum(values[..., 0, :] * sampled_preference, axis=-1)
                        )

                    actor_loss_value, actor_grad = jax.value_and_grad(actor_loss)(params)
                    actor_updates, params_opt = actor_optimizer.update(
                        actor_grad, params_opt
                    )
                    params = optax.apply_updates(params, actor_updates)
                    target_actor = jax.tree_util.tree_map(
                        lambda new, old: TAU * new + (1.0 - TAU) * old,
                        params,
                        target_actor,
                    )
                    target_critic = jax.tree_util.tree_map(
                        lambda new, old: TAU * new + (1.0 - TAU) * old,
                        inner_critic,
                        target_critic,
                    )
                    return params, params_opt, target_actor, target_critic, actor_loss_value

                def skip_actor(params, params_opt, target_actor, target_critic):
                    return params, params_opt, target_actor, target_critic, jnp.zeros(())

                (
                    inner_actor,
                    inner_actor_opt,
                    inner_target_actor,
                    inner_target_critic,
                    actor_loss_value,
                ) = jax.lax.cond(
                    should_update_actor,
                    update_actor,
                    skip_actor,
                    inner_actor,
                    inner_actor_opt,
                    inner_target_actor,
                    inner_target_critic,
                )
                inner_last_actor_loss = jnp.where(
                    should_update_actor, actor_loss_value, inner_last_actor_loss
                )
                return (
                    inner_critic,
                    inner_critic_opt,
                    inner_actor,
                    inner_actor_opt,
                    inner_target_actor,
                    inner_target_critic,
                    inner_step,
                    inner_last_actor_loss,
                ), critic_loss_value

            inner = (
                current_critic,
                current_critic_opt,
                current_actor,
                current_actor_opt,
                current_target_actor,
                current_target_critic,
                critic_step,
                jnp.asarray(jnp.nan, dtype=jnp.float32),
            )
            if update_count:
                update_keys = jax.random.split(update_key, update_count)
                inner, losses = jax.lax.scan(update_step, inner, update_keys)
                last_critic_loss = losses[-1]
            else:
                last_critic_loss = jnp.asarray(jnp.nan, dtype=jnp.float32)
            (
                current_critic,
                current_critic_opt,
                current_actor,
                current_actor_opt,
                current_target_actor,
                current_target_critic,
                critic_step,
                last_actor_loss,
            ) = inner
            return (
                key,
                env_state,
                current_actor,
                current_critic,
                current_target_actor,
                current_target_critic,
                current_actor_opt,
                current_critic_opt,
                current_replay,
                critic_step,
            ), PyTreeDict(
                completed_episodes=completed_episodes,
                critic_loss=last_critic_loss,
                actor_loss=last_actor_loss,
            )

        return train_chunk

    chunk_fns = {}
    rng, reset_key = jax.random.split(rng)
    env_state = train_env.reset(reset_key)
    carry = (
        rng,
        env_state,
        actor_params,
        critic_params,
        target_actor_params,
        target_critic_params,
        actor_optimizer_state,
        critic_optimizer_state,
        replay_state,
        jnp.int32(0),
    )

    best_scalar = -np.inf
    best_vector = None
    best_std = None
    best_step = 0
    completed_episodes_total = 0
    evaluation_history = []
    next_evaluation_episode = 100
    start_time = time.time()

    @jax.jit
    def evaluate(params, eval_key):
        eval_key, reset_key = jax.random.split(eval_key)
        eval_state = eval_env.reset(reset_key)
        metrics, _ = fast_eval_rollout_episode(
            eval_env.step,
            lambda _state, batch, key: eval_action_fn(params, batch, key),
            eval_state,
            None,
            eval_key,
            HORIZON,
        )
        returns = metrics.episode_returns
        mean_return = jnp.mean(returns, axis=0)
        return mean_return, jnp.std(returns, axis=0), jnp.dot(mean_return, base_preference)

    def evaluate_and_record(step: int, phase: str):
        nonlocal best_scalar, best_vector, best_std, best_step
        mean_return, std_return, scalar_return = evaluate(carry[2], jax.random.PRNGKey(0))
        mean_np = np.asarray(mean_return, dtype=np.float64)
        std_np = np.asarray(std_return, dtype=np.float64)
        scalar_value = float(scalar_return)
        if not np.isfinite(mean_np).all() or not np.isfinite(std_np).all():
            raise FloatingPointError(f"non-finite evaluation at step {step}")
        if scalar_value > best_scalar:
            best_scalar = scalar_value
            best_vector = mean_np.copy()
            best_std = std_np.copy()
            best_step = step
            if output_dir is not None:
                save_key_networks(output_dir / "best_networks.msgpack", carry, seed,
                                  preference, step, mean_np)
        evaluation_history.append(
            {
                "step": step,
                "phase": phase,
                "completed_episodes": completed_episodes_total,
                "mean_vector_return": mean_np.tolist(),
                "std_vector_return": std_np.tolist(),
                "scalarized_return": scalar_value,
            }
        )

        if output_dir is not None:
            _write_metadata(output_dir / "progress.json", {
                "status": "running", "seed": seed, "preference": preference.tolist(),
                "actual_environment_steps": step, "total_environment_steps": total_env_steps,
                "completed_episodes": completed_episodes_total,
                "best_step": best_step, "best_scalarized_return": best_scalar,
                "best_vector_return": best_vector.tolist(),
                "last_evaluation": evaluation_history[-1],
            })
            print(f"seed={seed} steps={step}/{total_env_steps} "
                  f"score={scalar_value:.3f} best={best_scalar:.3f}", flush=True)
            if phase == "training_final":
                save_key_networks(output_dir / "final_networks.msgpack", carry, seed,
                                  preference, step, mean_np)

    for chunk_index in range(counts["total_chunks"]):
        schedule = (chunk_index < counts["random_action_chunks"], updates_for_chunk(chunk_index))
        if schedule not in chunk_fns:
            chunk_fns[schedule] = make_train_chunk(*schedule)
        chunk_fn = chunk_fns[schedule]
        carry, chunk_metrics = jax.lax.scan(chunk_fn, carry, None, length=1)
        completed_episodes_total += int(np.asarray(chunk_metrics.completed_episodes[0]))
        while completed_episodes_total >= next_evaluation_episode:
            evaluate_and_record((chunk_index + 1) * CHUNK_SIZE, "periodic")
            next_evaluation_episode += 100

    evaluate_and_record(total_env_steps, "training_final")
    elapsed = time.time() - start_time
    final_metrics = {
        "actual_environment_steps": counts["total_chunks"] * CHUNK_SIZE,
        "random_action_steps": random_action_steps,
        "critic_optimizer_step_count": int(np.asarray(carry[9])),
        "actor_optimizer_step_count": counts["actor_optimizer_steps"],
        "replay_size": min(REPLAY_CAPACITY, total_env_steps),
        "completed_episodes": completed_episodes_total,
        "last_critic_loss": float(np.asarray(chunk_metrics.critic_loss[0])),
        "last_actor_loss": float(np.asarray(chunk_metrics.actor_loss[0])),
    }
    if final_metrics["critic_optimizer_step_count"] != counts["critic_optimizer_steps"]:
        raise AssertionError(f"critic count mismatch: {final_metrics}")
    if not np.isfinite(final_metrics["last_critic_loss"]):
        raise FloatingPointError("non-finite critic loss")
    if not np.isfinite(final_metrics["last_actor_loss"]):
        raise FloatingPointError("non-finite actor loss")
    if best_vector is None or not np.isfinite(best_vector).all():
        raise FloatingPointError("no finite evaluation result")
    return best_vector, {
        "seed": seed,
        "preference": np.asarray(preference, dtype=np.float64).tolist(),
        "best_step": best_step,
        "best_vector_return": best_vector.tolist(),
        "best_std_vector_return": best_std.tolist(),
        "best_scalarized_return": float(best_scalar),
        "evaluation_history": evaluation_history,
        "elapsed_seconds": elapsed,
        "counts": final_metrics,
        "expected_counts": counts,
        "sampling_schedule": {
            "num_envs": NUM_ENVS,
            "rollout_length": ROLLOUT_LENGTH,
            "chunk_size": CHUNK_SIZE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "critic_updates_per_chunk": CRITIC_UPDATES_PER_CHUNK,
            "actor_updates_per_chunk": CRITIC_UPDATES_PER_CHUNK // POLICY_FREQ,
            "learning_starts": 2 * BATCH_SIZE,
            "initial_no_update_chunks": counts["initial_no_update_chunks"],
            "first_learning_chunk_updates": counts["first_learning_chunk_updates"],
        },
        "smoke": smoke,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_version": jax.__version__,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_metadata(path: Path, metadata: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_key_networks(path, carry, seed, preference, step, returns):
    """Save weights for independent evaluation; this is not a full resume state."""
    names = ("actor_params", "critic_params", "target_actor_params", "target_critic_params")
    snapshot = {**dict(zip(names, carry[2:6])), "seed": seed, "step": step,
                "preference": np.asarray(preference), "vector_return": np.asarray(returns)}
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(serialization.to_bytes(jax.device_get(snapshot)))
    temporary.replace(path)


def run_smoke(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    key = np.asarray(key_preferences(2)[1], dtype=np.float32)
    result, run_metadata = train_single_key(
        key, SEED, total_env_steps=SMOKE_ENV_STEPS,
        random_action_steps=SMOKE_RANDOM_ACTION_STEPS, smoke=True
    )
    assert np.isfinite(result).all()
    assert run_metadata["counts"]["critic_optimizer_step_count"] == expected_update_counts(
        SMOKE_ENV_STEPS, SMOKE_RANDOM_ACTION_STEPS
    )["critic_optimizer_steps"]
    assert np.isfinite(run_metadata["counts"]["last_critic_loss"])
    assert np.isfinite(run_metadata["counts"]["last_actor_loss"])
    assert run_metadata["counts"]["last_actor_loss"] != 0.0
    _write_metadata(output_dir / "smoke.json", run_metadata)


def run_full(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    keys = np.asarray(key_preferences(2), dtype=np.float32)
    seeds = [SEED + 17 * index for index in range(len(keys))]
    per_key = []
    for key, key_seed in zip(keys, seeds):
        _, metadata = train_single_key(key, key_seed)
        per_key.append(metadata)

    write_artifact(output_dir, per_key)


def write_artifact(output_dir: Path, per_key: list[dict]) -> None:
    keys = np.asarray([item["preference"] for item in per_key])
    np.testing.assert_array_equal(keys, key_preferences(2))
    seeds = [item["seed"] for item in per_key]
    if len(set(seeds)) != 3:
        raise ValueError("three independent key seeds are required")
    if any("sampling_schedule" not in item for item in per_key):
        raise ValueError("key metadata lacks sampling_schedule; assemble with its original training script")
    sampling_schedule = per_key[0]["sampling_schedule"]
    if any(item["sampling_schedule"] != sampling_schedule for item in per_key):
        raise ValueError("all three keys must use the same sampling schedule")
    if not {"batch_size", "learning_rate", "critic_updates_per_chunk",
            "actor_updates_per_chunk"}.issubset(sampling_schedule):
        raise ValueError("key metadata lacks update configuration; assemble with its original training script")
    results = [item["best_vector_return"] for item in per_key]

    candidate_path = output_dir / "interp_objs_walker2d_brax.candidate.txt"
    if candidate_path.exists() or (output_dir / "interp_objs_walker2d_brax.metadata.json").exists():
        raise FileExistsError("refusing to overwrite an existing key artifact")
    np.savetxt(candidate_path, np.asarray(results), delimiter=",")
    artifact_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    metadata = {
        "artifact": candidate_path.name,
        "artifact_sha256": artifact_sha256,
        "keys": keys.tolist(),
        "seeds": seeds,
        "per_key": per_key,
        "hyperparameters": {
            "gamma": GAMMA,
            "replay_capacity": REPLAY_CAPACITY,
            "policy_freq": POLICY_FREQ,
            "critic_loss_aggregation": "sum_of_per_critic_mean_smooth_l1",
            "evaluation_episodes": EVAL_EPISODES,
            **sampling_schedule,
            "total_env_steps": per_key[0]["counts"]["actual_environment_steps"],
            "random_action_steps": per_key[0]["counts"]["random_action_steps"],
            "first_chunk_updates": (sampling_schedule["critic_updates_per_chunk"]
                                    if sampling_schedule["initial_no_update_chunks"] == 0 else 0),
        },
        "vectorization_note": (
            f"Each block collects {sampling_schedule['chunk_size']} transitions. "
            f"Updates start at replay size {sampling_schedule['learning_starts']}; "
            f"skip {sampling_schedule['initial_no_update_chunks']} blocks, then perform "
            f"{sampling_schedule['first_learning_chunk_updates']} critic updates. "
            f"Subsequent blocks perform {sampling_schedule['critic_updates_per_chunk']} critic "
            f"and {sampling_schedule['actor_updates_per_chunk']} actor updates."
        ),
        "git_commit": _git_commit(),
        "command": sys.argv,
        "jax_version": jax.__version__,
        "matmul_precision": str(jax.config.jax_default_matmul_precision),
        "jax_backend": per_key[0]["jax_backend"],
        "jax_devices": per_key[0]["jax_devices"],
        "smoke": all(item["smoke"] for item in per_key),
    }
    _write_metadata(output_dir / "interp_objs_walker2d_brax.metadata.json", metadata)
    print(f"Saved artifact: {candidate_path}; seeds={seeds}", flush=True)


def main() -> None:
    jax.config.update("jax_default_matmul_precision", "highest")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/key_retrain"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--key-index", type=int, choices=range(3))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--assemble-seeds", nargs=3, type=int)
    args = parser.parse_args()
    if args.assemble_seeds is not None:
        per_key = [json.loads((args.output_dir / f"key_{i}_seed_{seed}" / "metadata.json").read_text())
                   for i, seed in enumerate(args.assemble_seeds)]
        write_artifact(args.output_dir, per_key)
        return
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.key_index is not None:
        if args.seed is None or args.seed < 0:
            parser.error("--key-index requires a nonnegative --seed")
        if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
            raise RuntimeError("single-key jobs require exactly one visible GPU")
        key = np.asarray(key_preferences(2)[args.key_index], dtype=np.float32)
        kwargs = {"total_env_steps": SMOKE_ENV_STEPS,
                  "random_action_steps": SMOKE_RANDOM_ACTION_STEPS} if args.smoke else {}
        _, metadata = train_single_key(key, args.seed, smoke=args.smoke,
                                       output_dir=args.output_dir, **kwargs)
        _write_metadata(args.output_dir / "metadata.json", metadata)
        progress = json.loads((args.output_dir / "progress.json").read_text())
        _write_metadata(args.output_dir / "progress.json", {**progress, "status": "complete"})
    elif args.smoke:
        run_smoke(args.output_dir)
    else:
        run_full(args.output_dir)


if __name__ == "__main__":
    main()
