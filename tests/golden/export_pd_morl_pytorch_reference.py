"""Export deterministic PD-MORL PyTorch golden inputs and outputs.

The network and loss operations below mirror official commit
35aa1bccb31c149c3f3bb895c7c6ffa674e4b461.  The interpolator data is a
synthetic TEST FIXTURE, not an official Walker result.
"""

from __future__ import annotations

import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.interpolate import RBFInterpolator
from torch import nn
from torch.nn import functional

HERE = Path(__file__).resolve().parent
FIXTURE_PATH = HERE / "pd_morl_walker_input.npz"
REFERENCE_PATH = HERE / "pd_morl_pytorch_reference.npz"
METADATA_PATH = HERE / "pd_morl_golden_metadata.json"
DTYPES = {
    "float64": (np.float64, torch.float64),
    "float32": (np.float32, torch.float32),
}
OBS_DIM, ACTION_DIM, OBJECTIVE_DIM, HIDDEN_DIM, BATCH_SIZE = 17, 6, 2, 400, 4
GAMMA, ACTOR_LOSS_COEFF, POLICY_FREQ = 0.995, 10.0, 10


class Actor(nn.Module):
    """Official Actor layer order with canonical parameters supplied externally."""

    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(19, HIDDEN_DIM, dtype=dtype),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM, dtype=dtype),
                nn.Linear(HIDDEN_DIM, ACTION_DIM, dtype=dtype),
            ]
        )

    def forward(self, state, preference):
        x = torch.cat((state, preference), dim=1)
        x = functional.relu(self.layers[0](x))
        x = functional.relu(self.layers[1](x))
        return torch.tanh(self.layers[2](x))


class Critic(nn.Module):
    """Official independent vector-Q1/vector-Q2 architecture."""

    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.q1 = nn.ModuleList(
            [
                nn.Linear(25, HIDDEN_DIM, dtype=dtype),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM, dtype=dtype),
                nn.Linear(HIDDEN_DIM, OBJECTIVE_DIM, dtype=dtype),
            ]
        )
        self.q2 = nn.ModuleList(
            [
                nn.Linear(25, HIDDEN_DIM, dtype=dtype),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM, dtype=dtype),
                nn.Linear(HIDDEN_DIM, OBJECTIVE_DIM, dtype=dtype),
            ]
        )

    @staticmethod
    def _forward(layers, x):
        x = functional.relu(layers[0](x))
        x = functional.relu(layers[1](x))
        return layers[2](x)

    def forward(self, state, preference, action):
        x = torch.cat((state, preference, action), dim=1)
        return self._forward(self.q1, x), self._forward(self.q2, x)

    def Q1(self, state, preference, action):
        return self._forward(self.q1, torch.cat((state, preference, action), dim=1))


def _weights(rng, fan_in, fan_out):
    return rng.normal(0.0, np.sqrt(2.0 / (fan_in + fan_out)), (fan_out, fan_in))


def make_fixture() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260306)
    data = {
        "s": rng.normal(size=(BATCH_SIZE, OBS_DIM)),
        "a": np.tanh(rng.normal(size=(BATCH_SIZE, ACTION_DIM))),
        "r_vec": rng.normal(0.2, 0.5, size=(BATCH_SIZE, OBJECTIVE_DIM)),
        "s_next": rng.normal(size=(BATCH_SIZE, OBS_DIM)),
        "done": np.array([0.0, 1.0, 0.0, 1.0]),
        "w": np.array([[0.1, 0.9], [0.35, 0.65], [0.7, 0.3], [0.95, 0.05]]),
        "target_noise": np.clip(
            rng.normal(0.0, 0.2, size=(BATCH_SIZE, ACTION_DIM)), -0.5, 0.5
        ),
        "interpolator_keys": np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]),
        "interpolator_key_solutions": np.array([[1.0, 4.0], [3.0, 3.0], [5.0, 1.0]]),
    }
    specs = {
        "actor": (19, ACTION_DIM),
        "critic_q1": (25, OBJECTIVE_DIM),
        "critic_q2": (25, OBJECTIVE_DIM),
        "target_actor": (19, ACTION_DIM),
        "target_critic_q1": (25, OBJECTIVE_DIM),
        "target_critic_q2": (25, OBJECTIVE_DIM),
    }
    for prefix, (input_dim, output_dim) in specs.items():
        dims = (
            (input_dim, HIDDEN_DIM),
            (HIDDEN_DIM, HIDDEN_DIM),
            (HIDDEN_DIM, output_dim),
        )
        for index, (fan_in, fan_out) in enumerate(dims):
            data[f"{prefix}_w{index}"] = _weights(rng, fan_in, fan_out)
            data[f"{prefix}_b{index}"] = rng.normal(0.0, 0.02, fan_out)
    return data


def _load_layers(layers, data, prefix, dtype):
    with torch.no_grad():
        for index, layer in enumerate(layers):
            layer.weight.copy_(torch.as_tensor(data[f"{prefix}_w{index}"], dtype=dtype))
            layer.bias.copy_(torch.as_tensor(data[f"{prefix}_b{index}"], dtype=dtype))


def _angle(x, y):
    cosine = functional.cosine_similarity(x, y, dim=-1)
    return torch.rad2deg(torch.acos(torch.clamp(cosine, 0.0, 0.9999))), cosine


def _export_dtype(data, torch_dtype):
    tensor = lambda name: torch.as_tensor(data[name], dtype=torch_dtype)
    actor, target_actor = Actor(torch_dtype), Actor(torch_dtype)
    critic, target_critic = Critic(torch_dtype), Critic(torch_dtype)
    _load_layers(actor.layers, data, "actor", torch_dtype)
    _load_layers(target_actor.layers, data, "target_actor", torch_dtype)
    for model, q1_prefix, q2_prefix in (
        (critic, "critic_q1", "critic_q2"),
        (target_critic, "target_critic_q1", "target_critic_q2"),
    ):
        _load_layers(model.q1, data, q1_prefix, torch_dtype)
        _load_layers(model.q2, data, q2_prefix, torch_dtype)

    s, a, s_next, w = map(tensor, ("s", "a", "s_next", "w"))
    r_vec, done, target_noise = map(tensor, ("r_vec", "done", "target_noise"))
    actor_output = actor(s, w)
    target_actor_output = target_actor(s_next, w)
    smoothed_target_action = torch.clamp(target_actor_output + target_noise, -1.0, 1.0)
    q1, q2 = critic(s, w, a)
    target_q1, target_q2 = target_critic(s_next, w, smoothed_target_action)
    scalar_q1 = torch.sum(w * q1, dim=-1)
    scalar_q2 = torch.sum(w * q2, dim=-1)

    solutions = data["interpolator_key_solutions"]
    solutions = solutions / np.linalg.norm(solutions, axis=1, keepdims=True)
    wp_np = RBFInterpolator(data["interpolator_keys"], solutions, kernel="linear")(
        data["w"]
    )
    wp = torch.as_tensor(wp_np, dtype=torch_dtype)
    angle_q1, cosine_q1 = _angle(wp, q1)
    angle_q2, cosine_q2 = _angle(wp, q2)

    target_scores = torch.stack(
        (torch.sum(w * target_q1, dim=-1), torch.sum(w * target_q2, dim=-1)),
        dim=-1,
    )
    selected_index = torch.argmin(target_scores, dim=-1)
    target_twins = torch.stack((target_q1, target_q2), dim=1)
    selected_q = torch.gather(
        target_twins, 1, selected_index[:, None, None].expand(-1, 1, 2)
    ).squeeze(1)
    td_target = r_vec + GAMMA * (1.0 - done[:, None]) * selected_q

    smooth1 = functional.smooth_l1_loss(q1, td_target)
    smooth2 = functional.smooth_l1_loss(q2, td_target)
    critic_angle1, critic_angle2 = angle_q1.mean(), angle_q2.mean()
    critic_total = smooth1 + smooth2 + critic_angle1 + critic_angle2
    critic.zero_grad(set_to_none=True)
    critic_total.backward()
    critic_gradients = {
        f"{q_name}_raw_gradient_{kind}{index}": gradient.detach().clone()
        for q_name, layers in (("critic_q1", critic.q1), ("critic_q2", critic.q2))
        for index, layer in enumerate(layers)
        for kind, gradient in (("w", layer.weight.grad), ("b", layer.bias.grad))
    }

    actor_q1 = critic.Q1(s, w, actor_output)
    actor_scalarized_q = torch.sum(w * actor_q1, dim=-1).mean()
    actor_angles, _ = _angle(wp, actor_q1)
    actor_angle = actor_angles.mean()
    actor_total = -actor_scalarized_q + ACTOR_LOSS_COEFF * actor_angle
    actor.zero_grad(set_to_none=True)
    actor_total.backward()

    values = {
        "actor_output": actor_output,
        "target_actor_output": target_actor_output,
        "smoothed_target_action": smoothed_target_action,
        "critic_q1": q1,
        "critic_q2": q2,
        "target_q1": target_q1,
        "target_q2": target_q2,
        "scalar_q1": scalar_q1,
        "scalar_q2": scalar_q2,
        "wp": wp,
        "cosine_q1": cosine_q1,
        "cosine_q2": cosine_q2,
        "angle_q1": angle_q1,
        "angle_q2": angle_q2,
        "selected_target_critic_index": selected_index,
        "selected_target_q_vector": selected_q,
        "td_target_y": td_target,
        "smooth_l1_q1": smooth1,
        "smooth_l1_q2": smooth2,
        "critic_angle_q1": critic_angle1,
        "critic_angle_q2": critic_angle2,
        "critic_total_loss": critic_total,
        "actor_scalarized_q": actor_scalarized_q,
        "actor_angle": actor_angle,
        "actor_total_loss": actor_total,
    }
    values.update(critic_gradients)
    for index, layer in enumerate(actor.layers):
        values[f"actor_raw_gradient_w{index}"] = layer.weight.grad
        values[f"actor_raw_gradient_b{index}"] = layer.bias.grad
    return {
        name: value.detach().cpu().numpy() if torch.is_tensor(value) else value
        for name, value in values.items()
    }


def main():
    data = make_fixture()
    np.savez(FIXTURE_PATH, **data)
    reference = {}
    for dtype_name, (_, torch_dtype) in DTYPES.items():
        reference.update(
            {
                f"{name}_{dtype_name}": value
                for name, value in _export_dtype(data, torch_dtype).items()
            }
        )
    np.savez(REFERENCE_PATH, **reference)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    METADATA_PATH.write_text(
        json.dumps(
            {
                "fixture": "TEST FIXTURE; not an official Walker result",
                "pd_morl_commit": "35aa1bccb31c149c3f3bb895c7c6ffa674e4b461",
                "evorl_commit_at_export": revision,
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "scipy": scipy.__version__,
                "numpy": np.__version__,
                "jax": version("jax"),
                "jax_gpu_matmul_precision_policy": "highest",
                "dtypes": list(DTYPES),
                "walker_dimensions": {"obs": 17, "action": 6, "objective": 2},
                "batch_size": BATCH_SIZE,
                "actor_loss_coeff": ACTOR_LOSS_COEFF,
                "gamma": GAMMA,
                "policy_freq": POLICY_FREQ,
                "seed": 20260306,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
