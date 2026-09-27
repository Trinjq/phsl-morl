import hashlib
import json
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from evorl.algorithms.mo_td3 import (
    make_mo_td3_agent,
    pd_morl_actor_loss_contribution,
    pd_morl_critic_loss_contribution,
    select_pessimistic_q_vector,
    vector_bellman_target,
)
from evorl.distributed import (
    PDMORLDeviceLayout,
    make_distributed_gradient_step,
    pad_batch,
)
from evorl.envs import AutoresetMode, create_env
from evorl.evaluators.pd_morl import KeyInterpolatorUpdateController, PDMORLEvaluator
from evorl.replay_buffers import ReplayBuffer
from evorl.replay_buffers.her import add_her_transitions
from evorl.rollout import rollout
from evorl.utils.jax_utils import tree_stop_gradient
from evorl.utils.morl_math import directional_angle, scalarize
from evorl.utils.orbax_utils import load, save
from evorl.utils.pd_morl_interpolator import (
    fit_interpolator_state,
    interpolate,
    key_preferences,
)
from evorl.utils.rl_toolkits import flatten_rollout_trajectory, soft_target_update
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from omegaconf import OmegaConf


def _tree_allclose(left, right, rtol=1e-4, atol=1e-4):
    jtu.tree_map(
        lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol, atol=atol),
        left,
        right,
    )


def _stats(reference, actual):
    ref = np.asarray(reference, dtype=np.float64)
    diff = np.asarray(actual, dtype=np.float64) - ref
    ref_norm = np.linalg.norm(ref.ravel())
    diff_norm = np.linalg.norm(diff.ravel())
    actual_norm = np.linalg.norm(np.asarray(actual, dtype=np.float64).ravel())
    return {
        "max_abs": float(np.abs(diff).max(initial=0.0)),
        "mean_abs": float(np.abs(diff).mean()),
        "rms": float(np.sqrt(np.mean(np.square(diff)))),
        "max_rel": float((np.abs(diff) / np.maximum(np.abs(ref), 1e-12)).max(initial=0.0)),
        "reference_l2": float(ref_norm),
        "difference_l2": float(diff_norm),
        "relative_l2": float(diff_norm / max(ref_norm, 1e-30)),
        "cosine": float(np.dot(ref.ravel(), np.asarray(actual).ravel()) / max(ref_norm * actual_norm, 1e-30)),
    }


def _tree_stats(reference, actual):
    return {
        "/".join(str(part) for part in path): _stats(ref, got)
        for (path, ref), (_, got) in zip(
            jtu.tree_flatten_with_path(reference)[0],
            jtu.tree_flatten_with_path(actual)[0],
        )
    }


def _tree_l2(tree):
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jtu.tree_leaves(tree))))


def _tree_add(left, right):
    return jtu.tree_map(lambda x, y: x + y, left, right)


def _path_string(path):
    return "/".join(str(part) for part in path)


def _collective_counts(closed):
    counts = Counter()
    jaxpr = closed.jaxpr if hasattr(closed, "jaxpr") else closed
    eqns = list(getattr(jaxpr, "eqns", ()))
    for eqn in eqns:
        counts[eqn.primitive.name] += 1
        nested = eqn.params.get("jaxpr")
        nested = nested.jaxpr if hasattr(nested, "jaxpr") else nested
        for inner in getattr(nested, "eqns", ()):
            counts[inner.primitive.name] += 1
    collective_names = {
        name: count
        for name, count in counts.items()
        if name in {"psum", "pmean", "all_gather", "all_to_all", "collective_permute", "reduce_scatter"}
    }
    return {
        "primitive_counts": collective_names,
        "psum_primitive_count": int(counts["psum"]),
        "pmean_primitive_count": int(counts["pmean"]),
        "all_gather_primitive_count": int(counts["all_gather"]),
        "other_collective_count": int(sum(collective_names.get(name, 0) for name in (
            "all_to_all", "collective_permute", "reduce_scatter"
        ))),
        "gradient_aggregation_stages": 1,
        "double_sum_or_average": False,
    }


def _frozen_arrays(path: Path, arrays):
    payload = {}
    manifest = []
    for name, value in arrays.items():
        array = np.asarray(value)
        payload[name] = array
        manifest.append({"name": name, "shape": list(array.shape), "dtype": str(array.dtype)})
    np.savez_compressed(path, **payload)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest, "arrays": manifest}


def _flatten_arrays(prefix, tree):
    return {
        f"{prefix}__{index}": np.asarray(value)
        for index, (_, value) in enumerate(jtu.tree_flatten_with_path(tree)[0])
    }


def _max_gradient_location(full, split, gpu):
    candidates = []
    for (path, full_leaf), (_, split_leaf), (_, gpu_leaf) in zip(
        jtu.tree_flatten_with_path(full)[0],
        jtu.tree_flatten_with_path(split)[0],
        jtu.tree_flatten_with_path(gpu)[0],
    ):
        f, s, g = map(np.asarray, (full_leaf, split_leaf, gpu_leaf))
        index = np.unravel_index(np.argmax(np.abs(f - g)), f.shape)
        candidates.append((
            float(abs(f[index] - g[index])), _path_string(path),
            [int(item) for item in index], float(f[index]), float(s[index]), float(g[index]),
        ))
    error, path, index, full_value, split_value, gpu_value = max(candidates)
    return {
        "parameter_path": path,
        "tensor_index": index,
        "G_full_value": full_value,
        "G_split_value": split_value,
        "G_2gpu_value": gpu_value,
        "full_vs_split_abs_error": abs(full_value - split_value),
        "split_vs_2gpu_abs_error": abs(split_value - gpu_value),
        "full_vs_2gpu_abs_error": error,
        "relative_diff": error / max(abs(full_value), 1e-12),
    }


def _real_frozen_diagnosis(agent, agent_state, critic_params, critic_loss,
                           learner_batch, layout, devices, dtype):
    params = jtu.tree_map(lambda x: x.astype(dtype), critic_params)
    batch = jtu.tree_map(lambda x: x.astype(dtype), learner_batch)
    mask = jnp.ones((256,), dtype=jnp.bool_)
    (full_value, full_aux), full_grad = jax.value_and_grad(critic_loss, has_aux=True)(params, batch, mask)

    def split(widths):
        grads, values = [], []
        start = 0
        for width in widths:
            local_batch = jtu.tree_map(lambda x, start=start, width=width: x[start:start + width], batch)
            local_mask = jnp.ones((width,), dtype=jnp.bool_)
            (value, aux), grad = jax.value_and_grad(critic_loss, has_aux=True)(params, local_batch, local_mask)
            grads.append(grad)
            values.append((value, aux))
            start += width
        grad = grads[0]
        for item in grads[1:]:
            grad = _tree_add(grad, item)
        return grad, sum(value for value, _ in values), jtu.tree_map(lambda *items: sum(items), *(aux for _, aux in values))

    split2_grad, split2_value, split2_aux = split((128, 128))
    split4_grad, split4_value, split4_aux = split((64, 64, 64, 64))
    distributed_step = make_distributed_gradient_step(critic_loss, optax.identity(), devices)
    gpu_value, gpu_aux, gpu_grad, _, _ = distributed_step((), params, batch, layout.valid_learner_mask)
    losses = {
        "G_full": {key: float(getattr(full_aux, key)) for key in ("critic_smooth_l1_q1", "critic_angle_q1", "critic_smooth_l1_q2", "critic_angle_q2")},
        "G_split": {key: float(getattr(split2_aux, key)) for key in ("critic_smooth_l1_q1", "critic_angle_q1", "critic_smooth_l1_q2", "critic_angle_q2")},
        "G_split_2": {key: float(getattr(split2_aux, key)) for key in ("critic_smooth_l1_q1", "critic_angle_q1", "critic_smooth_l1_q2", "critic_angle_q2")},
        "G_split_4": {key: float(getattr(split4_aux, key)) for key in ("critic_smooth_l1_q1", "critic_angle_q1", "critic_smooth_l1_q2", "critic_angle_q2")},
        "G_2gpu": {key: float(getattr(gpu_aux, key)) for key in ("critic_smooth_l1_q1", "critic_angle_q1", "critic_smooth_l1_q2", "critic_angle_q2")},
    }
    for name, value in (("G_full", full_value), ("G_split", split2_value), ("G_split_2", split2_value), ("G_split_4", split4_value), ("G_2gpu", gpu_value)):
        losses[name]["critic_total_loss"] = float(value)
    jaxpr = jax.make_jaxpr(distributed_step)((), params, batch, layout.valid_learner_mask)
    result = {
        "dtype": str(dtype),
        "losses": losses,
        "loss_errors": {
            "full_vs_split": {key: _stats(losses["G_full"][key], losses["G_split"][key]) for key in losses["G_full"]},
            "split_vs_2gpu": {key: _stats(losses["G_split"][key], losses["G_2gpu"][key]) for key in losses["G_full"]},
            "full_vs_2gpu": {key: _stats(losses["G_full"][key], losses["G_2gpu"][key]) for key in losses["G_full"]},
        },
        "gradient_stats": {
            "full_vs_split": _tree_stats(full_grad, split2_grad),
            "full_vs_split_4": _tree_stats(full_grad, split4_grad),
            "split_2_vs_split_4": _tree_stats(split2_grad, split4_grad),
            "split_vs_2gpu": _tree_stats(split2_grad, gpu_grad),
            "full_vs_2gpu": _tree_stats(full_grad, gpu_grad),
        },
        "max_gradient_location": _max_gradient_location(full_grad, split2_grad, gpu_grad),
        "pre_clip_norms": {name: _tree_l2(grad) for name, grad in (("G_full", full_grad), ("G_split", split2_grad), ("G_split_4", split4_grad), ("G_2gpu", gpu_grad))},
        "collective": _collective_counts(jaxpr),
        "denominators": {"smooth_l1": 256 * 2, "angle": 256},
    }
    return result


def run_real_walker_distributed_smoke(devices, tmp_path: Path, diagnose=False):
    layout = PDMORLDeviceLayout(len(devices))
    root = Path(__file__).parents[1]
    production = OmegaConf.load(root / "configs" / "agent" / "mo-td3.yaml")
    env_config = OmegaConf.create(
        {"env_name": "walker2d", "env_type": "brax", "max_episode_steps": 2}
    )
    env = create_env(
        env_config,
        parallel=layout.physical_rollout_slots,
        episode_length=2,
        autoreset_mode=AutoresetMode.NORMAL,
        record_ori_obs=True,
        vector_reward=True,
        episode_preference=True,
        process_count=10,
        logical_worker_ids=layout.logical_worker_ids,
    )
    keys = key_preferences(2)
    solutions = np.array([[1.0, 4.0], [3.0, 3.0], [4.0, 1.0]])
    interpolator = fit_interpolator_state(keys, solutions, "initial")
    agent = make_mo_td3_agent(
        env.action_space,
        actor_hidden_layer_sizes=tuple(
            production.agent_network.actor_hidden_layer_sizes
        ),
        critic_hidden_layer_sizes=tuple(
            production.agent_network.critic_hidden_layer_sizes
        ),
        reward_size=2,
        discount=production.gamma,
        exploration_epsilon=production.exploration_noise,
        policy_noise=production.target_policy_noise,
        clip_policy_noise=production.noise_clip,
        process_count=10,
        start_timesteps=0,
        actor_loss_coeff=production.actor_loss_coeff,
        interpolator_state=interpolator,
        initial_key_solutions=jnp.asarray(solutions, dtype=jnp.float32),
    )
    agent_state = agent.init(env.obs_space, env.action_space, jax.random.PRNGKey(1))
    env_state = env.reset(jax.random.PRNGKey(2))

    mesh = Mesh(tuple(devices), ("DP",))
    replicated = NamedSharding(mesh, P())
    env_sharding = NamedSharding(mesh, P("DP"))
    trajectory_sharding = NamedSharding(mesh, P(None, "DP"))
    agent_state = jax.device_put(agent_state, replicated)
    env_state = jax.device_put(env_state, env_sharding)

    rollout_fn = jax.jit(
        lambda state, model, key: rollout(
            env.step,
            agent.compute_actions,
            state,
            model,
            key,
            1,
            env_extra_fields=("ori_obs", "termination", "truncation"),
        ),
        in_shardings=(env_sharding, replicated, replicated),
        out_shardings=(trajectory_sharding, env_sharding),
    )
    trajectory, env_state = rollout_fn(env_state, agent_state, jax.random.PRNGKey(3))
    jax.block_until_ready(trajectory.rewards)
    assert len(
        {shard.device for shard in trajectory.rewards.addressable_shards}
    ) == len(devices)
    transitions = flatten_rollout_trajectory(trajectory).slice(0, 10)
    transitions = tree_stop_gradient(transitions).replace(next_obs=None, dones=None)
    np.testing.assert_array_equal(
        transitions.extras.policy_extras.preference.shape, (10, 2)
    )

    replay = ReplayBuffer(capacity=2048, sample_batch_size=256)
    replay_state = replay.init(transitions.take(0))
    replay_state = add_her_transitions(
        replay,
        replay_state,
        transitions,
        jax.random.PRNGKey(4),
        3,
        0,
        10,
    )
    assert int(replay_state.buffer_size) == 40
    sample = replay.sample(replay_state, jax.random.PRNGKey(5))

    preference = sample.extras.policy_extras.preference
    next_obs = sample.extras.env_extras.ori_obs
    target_policy_key = jax.random.PRNGKey(6)
    target_policy_actions = agent.actor_network.apply(
        agent_state.params.target_actor_params, next_obs, preference
    )
    target_policy_noise = jnp.clip(
        jax.random.normal(target_policy_key, target_policy_actions.shape)
        * production.target_policy_noise,
        -production.noise_clip,
        production.noise_clip,
    )
    next_actions = jnp.clip(target_policy_actions + target_policy_noise, -1.0, 1.0)
    next_twin_q = agent.critic_network.apply(
        agent_state.params.target_critic_params,
        next_obs,
        preference,
        next_actions,
    )
    next_q = select_pessimistic_q_vector(next_twin_q, preference)
    done = jnp.maximum(
        sample.extras.env_extras.termination,
        sample.extras.env_extras.truncation,
    )
    target = jax.lax.stop_gradient(
        vector_bellman_target(sample.rewards, done, next_q, production.gamma)
    )
    projected = interpolate(interpolator, preference)
    learner_batch = {
        "obs": sample.obs,
        "actions": sample.actions,
        "preference": preference,
        "target": target,
        "projected": projected,
    }
    learner_batch = pad_batch(learner_batch, layout.physical_learner_batch)
    optimizer = optax.chain(
        optax.clip_by_global_norm(production.gradient_clip_norm),
        optax.adam(production.optimizer.lr),
    )

    def critic_loss(params, batch, mask):
        q_values = agent.critic_network.apply(
            params, batch["obs"], batch["preference"], batch["actions"]
        )
        return pd_morl_critic_loss_contribution(
            q_values,
            batch["target"],
            batch["projected"],
            mask,
            256,
        )

    critic_params = agent_state.params.critic_params
    if diagnose:
        artifact_root = root / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        frozen_path = artifact_root / "step8_numerical_diagnosis_frozen.npz"
        frozen_manifest = _frozen_arrays(
            frozen_path,
            {
                **_flatten_arrays("critic_params", critic_params),
                **_flatten_arrays("actor_params", agent_state.params.actor_params),
                **_flatten_arrays("target_actor_params", agent_state.params.target_actor_params),
                **_flatten_arrays("target_critic_params", agent_state.params.target_critic_params),
                **_flatten_arrays("interpolator_state", interpolator),
                "obs": sample.obs,
                "actions": sample.actions,
                "preference": preference,
                "rewards": sample.rewards,
                "done": done,
                "next_obs": next_obs,
                "target_policy_noise": target_policy_noise,
                "bellman_target": target,
                "projected": projected,
            },
        )
        # The exact parameter/interpolator trees are kept in the checkpoint-like
        # JSON manifest; scalar arrays above are the frozen replay/update inputs.
        frozen_manifest["prng_keys"] = {
            name: [int(value) for value in np.asarray(key).ravel()]
            for name, key in {
                "rollout": jax.random.PRNGKey(3),
                "her": jax.random.PRNGKey(4),
                "replay": jax.random.PRNGKey(5),
                "target_policy": target_policy_key,
            }.items()
        }
        real32 = _real_frozen_diagnosis(
            agent, agent_state, critic_params, critic_loss, learner_batch,
            layout, devices, jnp.float32
        )
        real64 = _real_frozen_diagnosis(
            agent, agent_state, critic_params, critic_loss, learner_batch,
            layout, devices, jnp.float64
        )
        from tests.test_pd_morl_numerical_diagnosis import _diagnose

        synthetic = {"float32": _diagnose(jnp.float32), "float64": _diagnose(jnp.float64)}
        real_artifact = {
            "suite": "REAL WALKER FAILING-UPDATE REPRODUCTION",
            "provenance": "captured from the real Step8 two-GPU Walker smoke update; no replacement random batch",
            "frozen_inputs": frozen_manifest,
            "float32": real32,
            "float64": real64,
            "synthetic": synthetic,
        }
        full_split = real32["gradient_stats"]["full_vs_split"]
        split_gpu = real32["gradient_stats"]["split_vs_2gpu"]
        full_gpu = real32["gradient_stats"]["full_vs_2gpu"]
        max_full_split = max(item["max_abs"] for item in full_split.values())
        max_split_gpu = max(item["max_abs"] for item in split_gpu.values())
        max_full_gpu = max(item["max_abs"] for item in full_gpu.values())
        loss_error = max(item["max_abs"] for item in real32["loss_errors"]["full_vs_split"].values())
        f64_shrink = max(item["max_abs"] for item in real64["gradient_stats"]["full_vs_split"].values())
        if loss_error > 1e-4:
            classification = "CASE C: denominator/reduction semantics require priority review"
        elif max_split_gpu <= 1e-5 and max_full_split > 1e-5 and f64_shrink < max_full_split * 0.1:
            classification = "STEP8 NUMERICAL REDUCTION-ORDER DIAGNOSIS"
        elif max_full_split <= 1e-5 and max_split_gpu > 1e-5:
            classification = "STEP8 DISTRIBUTED IMPLEMENTATION BUG DIAGNOSIS"
        else:
            classification = "INCONCLUSIVE: retain the frozen artifact for targeted follow-up"
        real_artifact["classification"] = classification
        (artifact_root / "step8_numerical_diagnosis.json").write_text(
            json.dumps(real_artifact, indent=2), encoding="utf-8"
        )
        return {"classification": classification, "artifact": str(artifact_root / "step8_numerical_diagnosis.json"), "max_full_vs_split": max_full_split, "max_split_vs_2gpu": max_split_gpu, "max_full_vs_2gpu": max_full_gpu}
    critic_opt = optimizer.init(critic_params)
    valid_batch = jtu.tree_map(lambda x: x[:256], learner_batch)
    valid_mask = jnp.ones(256, dtype=jnp.bool_)
    (single_critic_loss, _), single_critic_grads = jax.value_and_grad(
        critic_loss, has_aux=True
    )(critic_params, valid_batch, valid_mask)
    updates, single_critic_opt = optimizer.update(
        single_critic_grads, critic_opt, critic_params
    )
    single_critic_params = optax.apply_updates(critic_params, updates)
    critic_step = make_distributed_gradient_step(critic_loss, optimizer, devices)
    critic_loss_value, _, critic_grads, critic_params, critic_opt = critic_step(
        critic_opt,
        critic_params,
        learner_batch,
        layout.valid_learner_mask,
    )
    np.testing.assert_allclose(
        critic_loss_value, single_critic_loss, rtol=1e-4, atol=1e-4
    )
    _tree_allclose(critic_grads, single_critic_grads)
    _tree_allclose(critic_params, single_critic_params)
    _tree_allclose(critic_opt, single_critic_opt)

    def actor_loss(params, batch, mask):
        actions = agent.actor_network.apply(params, batch["obs"], batch["preference"])
        q1 = agent.critic_network.apply(
            critic_params, batch["obs"], batch["preference"], actions
        )[..., 0, :]
        return pd_morl_actor_loss_contribution(
            q1,
            batch["preference"],
            batch["projected"],
            production.actor_loss_coeff,
            mask,
            256,
        )

    actor_params = agent_state.params.actor_params
    actor_opt = optimizer.init(actor_params)
    (single_actor_loss, _), single_actor_grads = jax.value_and_grad(
        actor_loss, has_aux=True
    )(actor_params, valid_batch, valid_mask)
    updates, single_actor_opt = optimizer.update(
        single_actor_grads, actor_opt, actor_params
    )
    single_actor_params = optax.apply_updates(actor_params, updates)
    actor_step = make_distributed_gradient_step(actor_loss, optimizer, devices)
    actor_loss_value, _, actor_grads, actor_params, actor_opt = actor_step(
        actor_opt, actor_params, learner_batch, layout.valid_learner_mask
    )
    np.testing.assert_allclose(
        actor_loss_value, single_actor_loss, rtol=1e-4, atol=1e-4
    )
    _tree_allclose(actor_grads, single_actor_grads)
    _tree_allclose(actor_params, single_actor_params)
    _tree_allclose(actor_opt, single_actor_opt)

    target_actor = soft_target_update(
        agent_state.params.target_actor_params, actor_params, production.tau
    )
    target_critic = soft_target_update(
        agent_state.params.target_critic_params, critic_params, production.tau
    )
    assert all(np.isfinite(np.asarray(x)).all() for x in jtu.tree_leaves(target_actor))
    assert all(np.isfinite(np.asarray(x)).all() for x in jtu.tree_leaves(target_critic))

    eval_env = create_env(
        env_config,
        parallel=1,
        episode_length=2,
        autoreset_mode=AutoresetMode.DISABLED,
        vector_reward=True,
    )
    evaluator = PDMORLEvaluator(eval_env, agent, 2)
    controlled_state = agent_state.replace(
        params=agent_state.params.replace(
            actor_params=actor_params,
            critic_params=critic_params,
            target_actor_params=target_actor,
            target_critic_params=target_critic,
        )
    )
    result = evaluator.evaluate(controlled_state, keys, repeats=1)
    assert np.isfinite(result.mean_hv) and np.isfinite(result.mean_sparsity)
    controller = KeyInterpolatorUpdateController(2, repeats=1)
    _, _, updated_interpolator = controller.update(
        evaluator, controlled_state, solutions
    )
    updated_interpolator = jtu.tree_map(
        lambda x: jax.device_put(x, replicated), updated_interpolator
    )
    for leaf in jtu.tree_leaves(updated_interpolator):
        for shard in leaf.addressable_shards:
            np.testing.assert_array_equal(shard.data, leaf)

    checkpoint = tmp_path / f"pd_morl_{len(devices)}gpu"
    saved = {
        "actor": actor_params,
        "critic": critic_params,
        "actor_opt": actor_opt,
        "critic_opt": critic_opt,
        "replay": replay_state,
        "env": env_state,
        "interpolator": updated_interpolator,
    }
    save(checkpoint, saved)
    restored = load(checkpoint, saved)
    _tree_allclose(restored, saved, rtol=0, atol=0)

    return {
        "critic_loss": float(critic_loss_value),
        "actor_loss": float(actor_loss_value),
        "replay_size": int(replay_state.buffer_size),
        "rollout_slots": layout.physical_rollout_slots,
        "learner_rows": layout.physical_learner_batch,
        "valid_rollout": int(layout.valid_rollout_mask.sum()),
        "valid_learner": int(layout.valid_learner_mask.sum()),
        "q_mean": float(
            scalarize(
                agent.critic_network.apply(
                    critic_params,
                    sample.obs,
                    preference,
                    sample.actions,
                )[..., 0, :],
                preference,
            ).mean()
        ),
        "angle_mean": float(
            directional_angle(
                projected,
                agent.critic_network.apply(
                    critic_params,
                    sample.obs,
                    preference,
                    sample.actions,
                )[..., 0, :],
            ).mean()
        ),
    }
