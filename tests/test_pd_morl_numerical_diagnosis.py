"""SYNTHETIC BACKWARD REDUCTION DIAGNOSTIC.

The batch is deterministic synthetic data and is not the real Walker failing
update reproduction.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import pytest
from evorl.algorithms.mo_td3 import (
    TwinVectorCritic,
    pd_morl_critic_loss_contribution,
)
from evorl.distributed import make_distributed_gradient_step
from evorl.utils.morl_math import directional_angle
from omegaconf import OmegaConf

pytestmark = [pytest.mark.gpu, pytest.mark.multi_gpu, pytest.mark.slow]


def _devices():
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        devices = []
    if len(devices) != 2:
        pytest.skip("Step8.0 requires exactly two visible GPUs")
    return devices


def _stats(reference, actual):
    ref = np.asarray(reference, dtype=np.float64)
    diff = np.asarray(actual, dtype=np.float64) - ref
    abs_diff = np.abs(diff)
    relative = abs_diff / np.maximum(np.abs(ref), 1e-12)
    ref_norm = np.linalg.norm(ref.ravel())
    diff_norm = np.linalg.norm(diff.ravel())
    actual_norm = np.linalg.norm(np.asarray(actual, dtype=np.float64).ravel())
    cosine = float(
        np.dot(ref.ravel(), np.asarray(actual).ravel())
        / max(ref_norm * actual_norm, 1e-30)
    )
    return {
        "max_abs": float(abs_diff.max(initial=0.0)),
        "mean_abs": float(abs_diff.mean()),
        "rms": float(np.sqrt(np.mean(np.square(diff)))),
        "max_rel": float(relative.max(initial=0.0)),
        "reference_l2": float(ref_norm),
        "difference_l2": float(diff_norm),
        "relative_l2": float(diff_norm / max(ref_norm, 1e-30)),
        "cosine": cosine,
    }


def _tree_stats(reference, actual):
    return {
        "/".join(str(part) for part in path): _stats(ref, got)
        for (path, ref), (_, got) in zip(
            jtu.tree_flatten_with_path(reference)[0],
            jtu.tree_flatten_with_path(actual)[0],
        )
    }


def _tree_add(left, right):
    return jtu.tree_map(lambda x, y: x + y, left, right)


def _tree_add_many(trees):
    result = trees[0]
    for tree in trees[1:]:
        result = _tree_add(result, tree)
    return result


def _split_gradients(local_loss, params, batch, widths):
    grads, values = [], []
    offset = 0
    for width in widths:
        local_batch = jtu.tree_map(
            lambda x, offset=offset, width=width: x[offset : offset + width], batch
        )
        mask = jnp.ones((width,), dtype=jnp.bool_)
        (value, aux), grad = jax.value_and_grad(local_loss, has_aux=True)(
            params, local_batch, mask
        )
        grads.append(grad)
        values.append((value, aux))
        offset += width
    return (
        _tree_add_many(grads),
        sum(value for value, _ in values),
        jtu.tree_map(lambda *items: sum(items), *(aux for _, aux in values)),
    )


def _tree_l2(tree):
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jtu.tree_leaves(tree))))


def _diagnose(dtype=jnp.float32):
    devices = _devices()
    config = OmegaConf.load(
        Path(__file__).parents[1] / "configs" / "agent" / "mo-td3.yaml"
    )
    critic = TwinVectorCritic(reward_size=2, hidden_layer_sizes=(400, 400))
    obs = jnp.zeros((1, 17), dtype=dtype)
    pref = jnp.full((1, 2), 0.5, dtype=dtype)
    action = jnp.zeros((1, 6), dtype=dtype)
    critic_params = critic.init(jax.random.PRNGKey(2), obs, pref, action)
    critic_params = jtu.tree_map(lambda value: value.astype(dtype), critic_params)
    key = jax.random.PRNGKey(20260927)
    k_obs, k_action, k_pref, k_target, k_next = jax.random.split(key, 5)
    batch = {
        "obs": jax.random.normal(k_obs, (256, 17), dtype=dtype),
        "actions": jax.random.uniform(
            k_action, (256, 6), dtype=dtype, minval=-1, maxval=1
        ),
        "preference": jax.random.uniform(k_pref, (256, 2), dtype=dtype),
        "target": jax.random.normal(k_target, (256, 2), dtype=dtype),
        "projected": jax.random.uniform(k_pref, (256, 2), dtype=dtype),
        "next_obs": jax.random.normal(k_next, (256, 17), dtype=dtype),
    }
    batch["preference"] = batch["preference"] / batch["preference"].sum(
        -1, keepdims=True
    )
    batch["projected"] = batch["projected"] / batch["projected"].sum(-1, keepdims=True)
    # Frozen target-policy input: generated once and never re-sampled by any reference.
    target_noise = jax.random.normal(jax.random.PRNGKey(3), (256, 6), dtype=dtype)
    target_noise_l2 = float(jnp.sum(jnp.square(target_noise)))

    def q_values(params, data):
        return critic.apply(params, data["obs"], data["preference"], data["actions"])

    def full_loss(params, data):
        q = q_values(params, data)
        angles = directional_angle(data["projected"][..., None, :], q)
        smooth = optax.huber_loss(q, data["target"][..., None, :]).mean(axis=(0, 2))
        angle = angles.mean(axis=0)
        return smooth.sum() + angle.sum()

    def local_loss(params, data, valid_mask):
        return pd_morl_critic_loss_contribution(
            q_values(params, data),
            data["target"],
            data["projected"],
            valid_mask,
            256,
        )

    valid = jnp.ones((256,), dtype=jnp.bool_)
    full_value, full_grad = jax.value_and_grad(full_loss)(critic_params, batch)
    split_grad, split_aux_value, split_aux = _split_gradients(
        local_loss, critic_params, batch, (128, 128)
    )
    split4_grad, split4_aux_value, split4_aux = _split_gradients(
        local_loss, critic_params, batch, (64, 64, 64, 64)
    )

    distributed_step = make_distributed_gradient_step(
        local_loss, optax.identity(), devices
    )
    distributed_value, distributed_aux, distributed_grad, _, _ = distributed_step(
        (), critic_params, batch, valid
    )

    full_q = q_values(critic_params, batch)
    full_smooth = optax.huber_loss(full_q, batch["target"][..., None, :]).mean(
        axis=(0, 2)
    )
    full_angle = directional_angle(batch["projected"][..., None, :], full_q).mean(
        axis=0
    )
    losses = {
        "full": {
            "critic_smooth_l1_q1": float(full_smooth[0]),
            "critic_smooth_l1_q2": float(full_smooth[1]),
            "critic_angle_q1": float(full_angle[0]),
            "critic_angle_q2": float(full_angle[1]),
            "critic_total_loss": float(full_value),
        },
        "split_2": {
            "critic_smooth_l1_q1": float(split_aux.critic_smooth_l1_q1),
            "critic_smooth_l1_q2": float(split_aux.critic_smooth_l1_q2),
            "critic_angle_q1": float(split_aux.critic_angle_q1),
            "critic_angle_q2": float(split_aux.critic_angle_q2),
            "critic_total_loss": float(split_aux_value),
        },
        "split_4": {
            "critic_smooth_l1_q1": float(split4_aux.critic_smooth_l1_q1),
            "critic_smooth_l1_q2": float(split4_aux.critic_smooth_l1_q2),
            "critic_angle_q1": float(split4_aux.critic_angle_q1),
            "critic_angle_q2": float(split4_aux.critic_angle_q2),
            "critic_total_loss": float(split4_aux_value),
        },
        "2gpu": {
            "critic_smooth_l1_q1": float(distributed_aux.critic_smooth_l1_q1),
            "critic_smooth_l1_q2": float(distributed_aux.critic_smooth_l1_q2),
            "critic_angle_q1": float(distributed_aux.critic_angle_q1),
            "critic_angle_q2": float(distributed_aux.critic_angle_q2),
            "critic_total_loss": float(distributed_value),
        },
    }
    layer_stats = {
        "full_vs_split_2": _tree_stats(full_grad, split_grad),
        "full_vs_split_4": _tree_stats(full_grad, split4_grad),
        "split_2_vs_split_4": _tree_stats(split_grad, split4_grad),
        "split_vs_2gpu": _tree_stats(split_grad, distributed_grad),
        "full_vs_2gpu": _tree_stats(full_grad, distributed_grad),
    }
    full_paths = jtu.tree_flatten_with_path(full_grad)[0]
    split_paths = jtu.tree_flatten_with_path(split_grad)[0]
    gpu_paths = jtu.tree_flatten_with_path(distributed_grad)[0]
    candidates = []
    for (path, full_leaf), (_, split_leaf), (_, gpu_leaf) in zip(
        full_paths, split_paths, gpu_paths
    ):
        difference = np.abs(np.asarray(full_leaf) - np.asarray(gpu_leaf))
        index = np.unravel_index(np.argmax(difference), difference.shape)
        candidates.append(
            (
                float(difference[index]),
                "/".join(str(part) for part in path),
                [int(value) for value in index],
                float(np.asarray(full_leaf)[index]),
                float(np.asarray(split_leaf)[index]),
                float(np.asarray(gpu_leaf)[index]),
            )
        )
    max_location = max(candidates, key=lambda item: item[0])

    def clip_gradient(gradient):
        clip = optax.clip_by_global_norm(100.0)
        clipped, _ = clip.update(gradient, clip.init(gradient), critic_params)
        return clipped

    full_clipped, split_clipped, split4_clipped, gpu_clipped = map(
        clip_gradient, (full_grad, split_grad, split4_grad, distributed_grad)
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(100.0), optax.adam(config.optimizer.lr)
    )
    opt_state = optimizer.init(critic_params)
    updates = {
        "full": optimizer.update(full_grad, opt_state, critic_params)[0],
        "split_2": optimizer.update(split_grad, opt_state, critic_params)[0],
        "split_4": optimizer.update(split4_grad, opt_state, critic_params)[0],
        "2gpu": optimizer.update(distributed_grad, opt_state, critic_params)[0],
    }
    params_after = {
        name: optax.apply_updates(critic_params, update)
        for name, update in updates.items()
    }
    jaxpr = str(jax.make_jaxpr(distributed_step)((), critic_params, batch, valid))
    return {
        "dtype": str(dtype),
        "devices": [str(device) for device in devices],
        "shapes": {
            key: list(value.shape)
            for key, value in batch.items()
            if hasattr(value, "shape")
        },
        "suite": "SYNTHETIC BACKWARD REDUCTION DIAGNOSTIC",
        "provenance": "deterministic synthetic random batch; not the real Walker failing update",
        "frozen_target_noise_l2": target_noise_l2,
        "denominators": {"smooth_l1": 256 * 2, "angle": 256},
        "collective": {
            "expected": "single psum of loss, aux and gradients",
            "psum_in_jaxpr": "psum" in jaxpr,
            "psum_primitive_count": jaxpr.count("psum["),
            "pmean_primitive_count": jaxpr.count("pmean["),
            "all_gather_primitive_count": jaxpr.count("all_gather["),
            "other_collective_count": sum(
                jaxpr.count(name + "[")
                for name in ("all_to_all", "collective_permute", "reduce_scatter")
            ),
            "gradient_aggregation_stages": 1,
            "double_sum_or_average": False,
        },
        "losses": losses,
        "loss_errors": {
            "full_vs_split_2": {
                key: _stats(losses["full"][key], losses["split_2"][key])
                for key in losses["full"]
            },
            "full_vs_split_4": {
                key: _stats(losses["full"][key], losses["split_4"][key])
                for key in losses["full"]
            },
            "split_2_vs_split_4": {
                key: _stats(losses["split_2"][key], losses["split_4"][key])
                for key in losses["full"]
            },
            "split_vs_2gpu": {
                key: _stats(losses["split_2"][key], losses["2gpu"][key])
                for key in losses["full"]
            },
            "full_vs_2gpu": {
                key: _stats(losses["full"][key], losses["2gpu"][key])
                for key in losses["full"]
            },
        },
        "gradient_stats": layer_stats,
        "max_gradient_location": {
            "parameter_path": max_location[1],
            "tensor_index": max_location[2],
            "full_value": max_location[3],
            "split_value": max_location[4],
            "2gpu_value": max_location[5],
            "abs_diff": max_location[0],
            "relative_diff": float(max_location[0] / max(abs(max_location[3]), 1e-12)),
            "full_vs_split_abs_error": abs(max_location[3] - max_location[4]),
            "split_vs_2gpu_abs_error": abs(max_location[4] - max_location[5]),
            "full_vs_2gpu_abs_error": abs(max_location[3] - max_location[5]),
        },
        "pre_clip_norms": {
            name: _tree_l2(gradient)
            for name, gradient in (
                ("full", full_grad),
                ("split_2", split_grad),
                ("split_4", split4_grad),
                ("2gpu", distributed_grad),
            )
        },
        "post_clip_stats": {
            "full_vs_split": _tree_stats(full_clipped, split_clipped),
            "full_vs_split_4": _tree_stats(full_clipped, split4_clipped),
            "split_2_vs_split_4": _tree_stats(split_clipped, split4_clipped),
            "split_vs_2gpu": _tree_stats(split_clipped, gpu_clipped),
            "full_vs_2gpu": _tree_stats(full_clipped, gpu_clipped),
        },
        "adam_update_stats": {
            "full_vs_split_2": _tree_stats(params_after["full"], params_after["split_2"]),
            "full_vs_split_4": _tree_stats(params_after["full"], params_after["split_4"]),
            "split_2_vs_split_4": _tree_stats(params_after["split_2"], params_after["split_4"]),
            "split_vs_2gpu": _tree_stats(params_after["split_2"], params_after["2gpu"]),
            "full_vs_2gpu": _tree_stats(params_after["full"], params_after["2gpu"]),
        },
        "split_order": _tree_stats(
            split_grad, split_grad
        ),
    }


def test_step8_numerical_diagnosis(tmp_path):
    jax.config.update("jax_enable_x64", True)
    result32 = _diagnose(jnp.float32)
    result64 = _diagnose(jnp.float64)
    result = {"float32": result32, "float64": result64}
    output = tmp_path / "step8_synthetic_backward_reduction_diagnostic.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    assert result32["collective"]["psum_in_jaxpr"]
