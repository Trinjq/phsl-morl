from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from evorl.algorithms.mo_td3 import (
    PreferenceActor,
    TwinVectorCritic,
    pd_morl_actor_loss,
    pd_morl_critic_loss,
    select_pessimistic_q_vector,
    vector_bellman_target,
)
from evorl.utils.morl_math import cosine_similarity, directional_angle, scalarize
from evorl.utils.pd_morl_interpolator import fit_interpolator_state, interpolate
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
FIXTURE = np.load(HERE / "pd_morl_walker_input.npz")
REFERENCE = np.load(HERE / "pd_morl_pytorch_reference.npz")
DTYPES = ("float64", "float32")
TOLERANCES = {
    "float64": {
        "exact": (2e-14, 2e-14),
        "forward": (2e-13, 2e-13),
        "angle": (5e-12, 5e-12),
        "gradient": (5e-12, 1e-9),
    },
    "float32": {
        "exact": (2e-6, 2e-6),
        "forward": (2e-6, 1e-5),
        "angle": (2e-4, 2e-6),
        "gradient": (1e-4, 1e-4),
    },
}


def torch_to_jax_params(prefix: str, root: str, dtype: jnp.dtype):
    """Map canonical PyTorch [out,in] weights to Flax [in,out] kernels."""
    return {
        "params": {
            root: {
                f"hidden_{index}": {
                    "kernel": jnp.asarray(FIXTURE[f"{prefix}_w{index}"].T, dtype=dtype),
                    "bias": jnp.asarray(FIXTURE[f"{prefix}_b{index}"], dtype=dtype),
                }
                for index in range(3)
            }
        }
    }


def torch_to_jax_critic_params(prefix: str, dtype: jnp.dtype):
    params = {}
    for critic_index in range(2):
        source = f"{prefix}_q{critic_index + 1}"
        params[f"critic_{critic_index}"] = {
            f"hidden_{layer_index}": {
                "kernel": jnp.asarray(
                    FIXTURE[f"{source}_w{layer_index}"].T, dtype=dtype
                ),
                "bias": jnp.asarray(FIXTURE[f"{source}_b{layer_index}"], dtype=dtype),
            }
            for layer_index in range(3)
        }
    return {"params": params}


def _error(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    absolute = np.abs(actual - expected)
    relative = absolute / np.maximum(np.abs(expected), np.finfo(expected.dtype).eps)
    return float(absolute.max(initial=0.0)), float(relative.max(initial=0.0))


def _assert_close(name, dtype_name, category="forward"):
    actual = golden(dtype_name)[name]
    expected = REFERENCE[f"{name}_{dtype_name}"]
    atol, rtol = TOLERANCES[dtype_name][category]
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


@lru_cache(maxsize=2)
def golden(dtype_name: str):
    with (
        jax.enable_x64(dtype_name == "float64"),
        jax.default_matmul_precision("highest"),
    ):
        return _golden(dtype_name)


def _golden(dtype_name: str):
    dtype = jnp.float64 if dtype_name == "float64" else jnp.float32
    value = lambda name: jnp.asarray(FIXTURE[name], dtype=dtype)
    actor = PreferenceActor(action_size=6)
    critic = TwinVectorCritic()
    actor_params = torch_to_jax_params("actor", "actor", dtype)
    target_actor_params = torch_to_jax_params("target_actor", "actor", dtype)
    critic_params = torch_to_jax_critic_params("critic", dtype)
    target_critic_params = torch_to_jax_critic_params("target_critic", dtype)
    s, a, s_next, w = map(value, ("s", "a", "s_next", "w"))
    r_vec, done, target_noise = map(value, ("r_vec", "done", "target_noise"))

    actor_output = actor.apply(actor_params, s, w)
    target_actor_output = actor.apply(target_actor_params, s_next, w)
    smoothed_target_action = jnp.clip(target_actor_output + target_noise, -1.0, 1.0)
    twins = critic.apply(critic_params, s, w, a)
    target_twins = critic.apply(target_critic_params, s_next, w, smoothed_target_action)
    state = fit_interpolator_state(
        FIXTURE["interpolator_keys"],
        FIXTURE["interpolator_key_solutions"],
        "initial",
        dtype=np.dtype(dtype_name),
    )
    wp = interpolate(state, w)
    selected_q = select_pessimistic_q_vector(target_twins, w)
    td_target = vector_bellman_target(r_vec, done, selected_q, 0.995)

    def critic_loss(params):
        return pd_morl_critic_loss(critic.apply(params, s, w, a), td_target, wp)

    actor_q1 = critic.apply(critic_params, s, w, actor_output)[..., 0, :]

    def actor_loss(params):
        actions = actor.apply(params, s, w)
        q1 = critic.apply(critic_params, s, w, actions)[..., 0, :]
        return pd_morl_actor_loss(q1, w, wp, 10.0)

    critic_grad = jax.grad(critic_loss)(critic_params)
    actor_grad = jax.grad(actor_loss)(actor_params)
    return {
        "actor_output": actor_output,
        "target_actor_output": target_actor_output,
        "smoothed_target_action": smoothed_target_action,
        "critic_q1": twins[..., 0, :],
        "critic_q2": twins[..., 1, :],
        "target_q1": target_twins[..., 0, :],
        "target_q2": target_twins[..., 1, :],
        "scalar_q1": scalarize(twins[..., 0, :], w),
        "scalar_q2": scalarize(twins[..., 1, :], w),
        "wp": wp,
        "cosine_q1": cosine_similarity(wp, twins[..., 0, :]),
        "cosine_q2": cosine_similarity(wp, twins[..., 1, :]),
        "angle_q1": directional_angle(wp, twins[..., 0, :]),
        "angle_q2": directional_angle(wp, twins[..., 1, :]),
        "selected_target_critic_index": jnp.argmin(scalarize(target_twins, w), axis=-1),
        "selected_target_q_vector": selected_q,
        "td_target_y": td_target,
        "smooth_l1_q1": jnp.mean(
            jnp.where(
                jnp.abs(twins[..., 0, :] - td_target) < 1,
                0.5 * (twins[..., 0, :] - td_target) ** 2,
                jnp.abs(twins[..., 0, :] - td_target) - 0.5,
            )
        ),
        "smooth_l1_q2": jnp.mean(
            jnp.where(
                jnp.abs(twins[..., 1, :] - td_target) < 1,
                0.5 * (twins[..., 1, :] - td_target) ** 2,
                jnp.abs(twins[..., 1, :] - td_target) - 0.5,
            )
        ),
        "critic_angle_q1": directional_angle(wp, twins[..., 0, :]).mean(),
        "critic_angle_q2": directional_angle(wp, twins[..., 1, :]).mean(),
        "critic_total_loss": critic_loss(critic_params),
        "actor_scalarized_q": scalarize(actor_q1, w).mean(),
        "actor_angle": directional_angle(wp, actor_q1).mean(),
        "actor_total_loss": actor_loss(actor_params),
        "critic_grad": critic_grad,
        "actor_grad": actor_grad,
        "actor": actor,
        "critic": critic,
        "actor_params": actor_params,
        "critic_params": critic_params,
        "s": s,
        "w": w,
        "a": a,
    }


@pytest.mark.parametrize("dtype_name", DTYPES)
def test_parameter_mapping(dtype_name):
    dtype = jnp.float64 if dtype_name == "float64" else jnp.float32
    with jax.enable_x64(dtype_name == "float64"):
        params = torch_to_jax_params("actor", "actor", dtype)
    for index in range(3):
        restored = np.asarray(params["params"]["actor"][f"hidden_{index}"]["kernel"]).T
        expected = FIXTURE[f"actor_w{index}"].astype(dtype_name)
        assert np.max(np.abs(restored - expected)) == 0


def test_precision_policy_config():
    config = OmegaConf.load(HERE.parents[1] / "configs" / "agent" / "mo-td3.yaml")
    assert config.matmul_precision == "highest"


@pytest.mark.parametrize("dtype_name", DTYPES)
def test_actor_forward(dtype_name):
    _assert_close("actor_output", dtype_name)
    _assert_close("target_actor_output", dtype_name)


@pytest.mark.parametrize("dtype_name", DTYPES)
def test_vector_q(dtype_name):
    for name in ("critic_q1", "critic_q2", "target_q1", "target_q2"):
        _assert_close(name, dtype_name)
    assert golden(dtype_name)["critic_q1"].shape == (4, 2)
    assert golden(dtype_name)["critic_q2"].shape == (4, 2)


@pytest.mark.parametrize("dtype_name", DTYPES)
def test_scalarization(dtype_name):
    data = golden(dtype_name)
    for index in (1, 2):
        name = f"scalar_q{index}"
        _assert_close(name, dtype_name, "exact")
        hand = np.sum(np.asarray(data[f"critic_q{index}"]) * FIXTURE["w"], axis=-1)
        np.testing.assert_allclose(
            data[name],
            hand,
            **dict(zip(("atol", "rtol"), TOLERANCES[dtype_name]["exact"])),
        )


@pytest.mark.parametrize("dtype_name", DTYPES)
def test_interpolator_wp(dtype_name):
    _assert_close("wp", dtype_name, "forward")


@pytest.mark.parametrize("dtype_name", DTYPES)
def test_cosine_similarity(dtype_name):
    _assert_close("cosine_q1", dtype_name, "forward")
    _assert_close("cosine_q2", dtype_name, "forward")


def test_directional_angle():
    with jax.enable_x64():
        x = jnp.array([[1.0, 0.0]] * 5)
        y = jnp.array([[2.0, 0.0], [0.0, 3.0], [-1.0, 0.0], [10.0, 0.0], [0.5, 0.0]])
        np.testing.assert_allclose(
            directional_angle(x, y),
            [0.81029144, 90.0, 90.0, 0.81029144, 0.81029144],
            atol=1e-5,
        )
    for dtype_name in DTYPES:
        _assert_close("angle_q1", dtype_name, "angle")
        _assert_close("angle_q2", dtype_name, "angle")


@pytest.mark.parametrize("dtype_name", DTYPES)
def test_target_critic_selection(dtype_name):
    _assert_close("selected_target_critic_index", dtype_name, "exact")
    _assert_close("selected_target_q_vector", dtype_name)
    data = golden(dtype_name)
    elementwise_min = jnp.minimum(data["target_q1"], data["target_q2"])
    assert not np.allclose(data["selected_target_q_vector"], elementwise_min)


@pytest.mark.parametrize("dtype_name", DTYPES)
def test_td_target(dtype_name):
    _assert_close("td_target_y", dtype_name)


@pytest.mark.parametrize("dtype_name", DTYPES)
def test_critic_loss(dtype_name):
    for name in (
        "smooth_l1_q1",
        "smooth_l1_q2",
        "critic_angle_q1",
        "critic_angle_q2",
        "critic_total_loss",
    ):
        _assert_close(name, dtype_name, "angle")
    data = golden(dtype_name)
    with jax.enable_x64(dtype_name == "float64"):
        expected_sum = sum(
            data[name]
            for name in (
                "smooth_l1_q1",
                "smooth_l1_q2",
                "critic_angle_q1",
                "critic_angle_q2",
            )
        )
        np.testing.assert_allclose(data["critic_total_loss"], expected_sum)


@pytest.mark.parametrize("dtype_name", DTYPES)
def test_actor_objective(dtype_name):
    for name in ("actor_scalarized_q", "actor_angle", "actor_total_loss"):
        _assert_close(name, dtype_name, "angle")
    data = golden(dtype_name)
    with jax.enable_x64(dtype_name == "float64"):
        np.testing.assert_allclose(
            data["actor_total_loss"],
            -data["actor_scalarized_q"] + 10 * data["actor_angle"],
        )


@pytest.mark.parametrize("dtype_name", DTYPES)
def test_critic_raw_gradient(dtype_name):
    gradients = golden(dtype_name)["critic_grad"]["params"]
    for critic_index in range(2):
        for layer_index in range(3):
            current = gradients[f"critic_{critic_index}"][f"hidden_{layer_index}"]
            prefix = f"critic_q{critic_index + 1}_raw_gradient"
            atol, rtol = TOLERANCES[dtype_name]["gradient"]
            np.testing.assert_allclose(
                current["kernel"],
                REFERENCE[f"{prefix}_w{layer_index}_{dtype_name}"].T,
                atol=atol,
                rtol=rtol,
            )
            np.testing.assert_allclose(
                current["bias"],
                REFERENCE[f"{prefix}_b{layer_index}_{dtype_name}"],
                atol=atol,
                rtol=rtol,
            )


@pytest.mark.parametrize("dtype_name", DTYPES)
def test_actor_raw_gradient(dtype_name):
    gradients = golden(dtype_name)["actor_grad"]["params"]["actor"]
    for layer_index in range(3):
        current = gradients[f"hidden_{layer_index}"]
        prefix = "actor_raw_gradient"
        atol, rtol = TOLERANCES[dtype_name]["gradient"]
        np.testing.assert_allclose(
            current["kernel"],
            REFERENCE[f"{prefix}_w{layer_index}_{dtype_name}"].T,
            atol=atol,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            current["bias"],
            REFERENCE[f"{prefix}_b{layer_index}_{dtype_name}"],
            atol=atol,
            rtol=rtol,
        )


def test_jit_compile():
    data = golden("float32")
    with jax.default_matmul_precision("highest"):
        eager = data["critic"].apply(
            data["critic_params"], data["s"], data["w"], data["a"]
        )
        compiled = jax.jit(data["critic"].apply)(
            data["critic_params"], data["s"], data["w"], data["a"]
        )
    np.testing.assert_array_equal(eager, compiled)


def test_vmap_batch():
    data = golden("float32")
    with jax.default_matmul_precision("highest"):
        batched = data["actor"].apply(data["actor_params"], data["s"], data["w"])
        mapped = jax.vmap(
            lambda state, preference: data["actor"].apply(
                data["actor_params"], state, preference
            )
        )(data["s"], data["w"])
    np.testing.assert_allclose(batched, mapped, atol=2e-6, rtol=2e-6)


def measured_errors():
    """Return compact values used to freeze the documented tolerances."""
    metrics = (
        "actor_output",
        "critic_q1",
        "critic_q2",
        "scalar_q1",
        "wp",
        "cosine_q1",
        "angle_q1",
        "selected_target_q_vector",
        "td_target_y",
        "critic_total_loss",
        "actor_total_loss",
    )
    return {
        dtype_name: {
            name: _error(golden(dtype_name)[name], REFERENCE[f"{name}_{dtype_name}"])
            for name in metrics
        }
        for dtype_name in DTYPES
    }
