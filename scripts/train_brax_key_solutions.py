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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from evorl.algorithms.mo_td3 import PreferenceActor, TwinVectorCritic
from evorl.envs.brax import create_wrapped_brax_env
from evorl.envs.wrappers.training_wrapper import AutoresetMode
from evorl.replay_buffers import ReplayBuffer
from evorl.rollout import fast_eval_rollout_episode, rollout
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.utils.pd_morl_interpolator import key_preferences


SEED = 42
NUM_ENVS = 64
ROLLOUT_LENGTH = 4
CHUNK_SIZE = NUM_ENVS * ROLLOUT_LENGTH
TOTAL_ENV_STEPS = 2_000_128
RANDOM_ACTION_STEPS = 25_088
FIRST_CHUNK_UPDATES = 57
BATCH_SIZE = 100
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
    critic = FIRST_CHUNK_UPDATES + (total_chunks - 1) * CHUNK_SIZE
    return {
        "chunk_size": CHUNK_SIZE,
        "total_chunks": total_chunks,
        "random_action_chunks": random_chunks,
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
) -> tuple[np.ndarray, dict]:
    """Train one key and return its best fixed-key evaluation result."""
    counts = expected_update_counts(total_env_steps, random_action_steps)
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
                    return jnp.mean(optax.huber_loss(values, target_q[..., None, :]))

                critic_loss_value, critic_grad = jax.value_and_grad(critic_loss)(
                    inner_critic
                )
                critic_updates, inner_critic_opt = critic_optimizer.update(
                    critic_grad, inner_critic_opt
                )
                inner_critic = optax.apply_updates(inner_critic, critic_updates)
                inner_step = inner_step + 1

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
                    inner_step % POLICY_FREQ == 0,
                    update_actor,
                    skip_actor,
                    inner_actor,
                    inner_actor_opt,
                    inner_target_actor,
                    inner_target_critic,
                )
                return (
                    inner_critic,
                    inner_critic_opt,
                    inner_actor,
                    inner_actor_opt,
                    inner_target_actor,
                    inner_target_critic,
                    inner_step,
                ), (critic_loss_value, actor_loss_value)

            inner = (
                current_critic,
                current_critic_opt,
                current_actor,
                current_actor_opt,
                current_target_actor,
                current_target_critic,
                critic_step,
            )
            update_keys = jax.random.split(update_key, update_count)
            inner, losses = jax.lax.scan(update_step, inner, update_keys)
            (
                current_critic,
                current_critic_opt,
                current_actor,
                current_actor_opt,
                current_target_actor,
                current_target_critic,
                critic_step,
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
                critic_loss=losses[0][-1],
                actor_loss=jnp.max(losses[1]),
            )

        return train_chunk

    first_random_chunk = make_train_chunk(True, FIRST_CHUNK_UPDATES)
    later_random_chunk = make_train_chunk(True, CHUNK_SIZE)
    policy_chunk = make_train_chunk(False, CHUNK_SIZE)
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

    for chunk_index in range(counts["total_chunks"]):
        if chunk_index == 0:
            chunk_fn = first_random_chunk
        elif chunk_index < counts["random_action_chunks"]:
            chunk_fn = later_random_chunk
        else:
            chunk_fn = policy_chunk
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
        "smoke": smoke,
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_smoke(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    key = np.asarray(key_preferences(2)[1], dtype=np.float32)
    result, run_metadata = train_single_key(
        key, SEED, total_env_steps=512, random_action_steps=256, smoke=True
    )
    assert np.isfinite(result).all()
    assert run_metadata["counts"]["critic_optimizer_step_count"] == 313
    assert np.isfinite(run_metadata["counts"]["last_critic_loss"])
    _write_metadata(output_dir / "smoke.json", run_metadata)


def run_full(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    keys = np.asarray(key_preferences(2), dtype=np.float32)
    seeds = [SEED + 17 * index for index in range(len(keys))]
    results = []
    per_key = []
    for key, key_seed in zip(keys, seeds):
        result, metadata = train_single_key(key, key_seed)
        results.append(result)
        per_key.append(metadata)

    candidate_path = output_dir / "interp_objs_walker2d_brax.candidate.txt"
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
            "batch_size": BATCH_SIZE,
            "replay_capacity": REPLAY_CAPACITY,
            "policy_freq": POLICY_FREQ,
            "evaluation_episodes": EVAL_EPISODES,
            "num_envs": NUM_ENVS,
            "rollout_length": ROLLOUT_LENGTH,
            "total_env_steps": TOTAL_ENV_STEPS,
            "random_action_steps": RANDOM_ACTION_STEPS,
            "first_chunk_updates": FIRST_CHUNK_UPDATES,
        },
        "vectorization_note": (
            "Budgets are rounded up to multiples of num_envs * rollout_length = 256; "
            "the first 256-transition block performs 57 updates after replay reaches 200 entries."
        ),
        "git_commit": _git_commit(),
        "command": sys.argv,
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [repr(device) for device in jax.devices()],
    }
    _write_metadata(output_dir / "interp_objs_walker2d_brax.metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/key_retrain"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.smoke:
        run_smoke(args.output_dir)
    else:
        run_full(args.output_dir)


if __name__ == "__main__":
    main()
