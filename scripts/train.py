import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import hydra
from hydra_utils import (
    get_output_dir,
    set_absl_log_level,
    set_omegaconf_resolvers,
)
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("train")

set_absl_log_level("warning")
set_omegaconf_resolvers()


def _json_default(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _evaluation_payload(result):
    pareto_counts = getattr(result, "pareto_counts_per_repeat", None)
    if pareto_counts is None or len(pareto_counts) == 0:
        pareto_counts = [len(result.pareto_returns)]
    return {
        "evaluation_seeds": [0, 11, 22, 33, 44, 55],
        "preferences": result.preferences,
        "returns_per_repeat": result.returns_per_repeat,
        "mean_returns": result.mean_returns,
        "hv_per_repeat": result.hv_per_repeat,
        "hv_mean": result.mean_hv,
        "hv_std": result.hv_per_repeat.std(),
        "sparsity_per_repeat": result.sparsity_per_repeat,
        "sparsity_mean": result.mean_sparsity,
        "sparsity_std": result.sparsity_per_repeat.std(),
        "pareto_indices": result.pareto_indices,
        "pareto_returns": result.pareto_returns,
        "pareto_point_count": len(result.pareto_returns),
        "source_hv": getattr(result, "source_hv", result.mean_hv),
        "source_sparsity": getattr(result, "source_sparsity", result.mean_sparsity),
        "source_pareto_point_count": getattr(result, "source_pareto_point_count", len(result.pareto_returns)),
        "mean_repeat_hv": getattr(result, "mean_repeat_hv", result.mean_hv),
        "mean_repeat_sparsity": getattr(result, "mean_repeat_sparsity", result.mean_sparsity),
        "mean_repeat_pareto_count": getattr(result, "mean_repeat_pareto_count", float(np.mean(pareto_counts))),
        "pareto_counts_per_repeat": np.asarray(pareto_counts),
    }


def _gpu_metadata():
    physical_id = os.environ.get("CUDA_VISIBLE_DEVICES", "unknown").split(",")[0]
    try:
        fields = "name,uuid,driver_version,memory.total"
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
                "-i",
                physical_id,
            ],
            text=True,
        ).strip()
        name, uuid, driver, memory = (part.strip() for part in output.split(","))
        return {
            "physical_gpu_id": physical_id,
            "gpu_model": name,
            "gpu_uuid": uuid,
            "nvidia_driver": driver,
            "gpu_memory_mib": int(memory),
        }
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return {"physical_gpu_id": physical_id, "gpu_model": "unknown", "gpu_uuid": "unknown"}


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _parameter_shapes(state):
    import jax.tree_util as jtu
    import numpy as np

    params = state.agent_state.params
    shapes = {
        jtu.keystr(path): list(np.shape(leaf))
        for path, leaf in jtu.tree_leaves_with_path(params)
    }
    actor_kernels = [
        tuple(shape)
        for path, shape in shapes.items()
        if "actor_params" in path and path.endswith("['kernel']")
    ]
    critic_kernels = [
        tuple(shape)
        for path, shape in shapes.items()
        if "critic_params" in path and path.endswith("['kernel']")
    ]
    # Online and target copies must share the production architecture.
    expected_actor = [(19, 400), (400, 400), (400, 6)] * 2
    expected_critic = sorted([(25, 400), (400, 400), (400, 2)] * 4)
    if sorted(actor_kernels) != sorted(expected_actor) or sorted(critic_kernels) != expected_critic:
        raise RuntimeError(
            f"production network mismatch: actor={actor_kernels}, critic={critic_kernels}"
        )
    q1 = [leaf for path, leaf in jtu.tree_leaves_with_path(params.critic_params) if "critic_0" in jtu.keystr(path)]
    q2 = [leaf for path, leaf in jtu.tree_leaves_with_path(params.critic_params) if "critic_1" in jtu.keystr(path)]
    if not q1 or not q2 or _fingerprint(q1) == _fingerprint(q2):
        raise RuntimeError("Q1/Q2 parameters are not independently initialized")
    return shapes


def _deterministic_probe(workflow, state):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from evorl.sample_batch import SampleBatch
    from evorl.types import PyTreeDict

    preferences = jnp.asarray([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    obs = jnp.broadcast_to(state.env_state.obs[0], (3, state.env_state.obs.shape[-1]))
    batch = SampleBatch(
        obs=obs,
        extras=PyTreeDict(policy_extras=PyTreeDict(preference=preferences)),
    )
    actions, _ = workflow.agent.evaluate_actions(
        state.agent_state, batch, jax.random.PRNGKey(0)
    )
    return np.asarray(jax.device_get(actions))


def _directory_size(path):
    return sum(file.stat().st_size for file in Path(path).rglob("*") if file.is_file())


def _source_fingerprint():
    root = Path(__file__).parents[1]
    digest = hashlib.sha256()
    files = sorted((root / "evorl").rglob("*.py")) + [Path(__file__)]
    for file in files:
        digest.update(file.relative_to(root).as_posix().encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def _resolve_resume_checkpoint(config: DictConfig) -> Path | None:
    raw = config.get("resume_from_checkpoint")
    if raw:
        path = Path(str(raw)).expanduser()
    elif config.get("resume_latest", False):
        path = Path(config.output_dir) / "checkpoints"
    else:
        return None
    if path.name.isdigit() and path.is_dir():
        return path
    if not bool(config.get("resume_latest", False)):
        raise ValueError(
            "resume_from_checkpoint must name a checkpoint step, or set resume_latest=true"
        )
    candidates = [
        child for child in path.iterdir() if child.is_dir() and child.name.isdigit()
    ]
    if not candidates:
        raise FileNotFoundError(f"no checkpoint steps found in {path}")
    return max(candidates, key=lambda child: int(child.name))


def _restore_checkpoint(config, workflow, state, metadata):
    checkpoint_path = _resolve_resume_checkpoint(config)
    if checkpoint_path is None:
        return state
    if not bool(config.get("save_replay_buffer", False)):
        raise ValueError(
            "resume requires save_replay_buffer=true; incomplete checkpoints are rejected"
        )
    output_root = checkpoint_path.parent.parent
    saved_metadata_path = output_root / "run_metadata.json"
    if saved_metadata_path.exists() and metadata:
        saved_metadata = json.loads(saved_metadata_path.read_text(encoding="utf-8"))
        saved_sha = saved_metadata.get("key_artifact_sha256")
        current_sha = metadata.get("key_artifact_sha256")
        if saved_sha and current_sha and saved_sha != current_sha:
            raise ValueError(
                f"checkpoint Artifact SHA mismatch: saved={saved_sha} current={current_sha}"
            )
    source_history = output_root / "hv_history.csv"
    target_history = Path(config.output_dir) / "hv_history.csv"
    if source_history.exists() and source_history.resolve() != target_history.resolve():
        if not target_history.exists():
            target_history.write_bytes(source_history.read_bytes())
    from evorl.utils.orbax_utils import load

    restored = load(str(checkpoint_path), state)
    sampled = int(restored.metrics.sampled_timesteps)
    if sampled <= 0:
        raise ValueError("checkpoint has no sampled transitions")
    if hasattr(workflow, "runtime_counters"):
        extra = restored.agent_state.extra_state
        critic_steps = int(extra.total_it)
        iterations = int(restored.metrics.iterations)
        actor_steps = critic_steps // int(config.actor_update_interval)
        workflow.runtime_counters.update(
            {
                "host_chunk_count": iterations,
                "inner_rollout_count": iterations * int(config.fold_iters),
                "environment_step_count": sampled,
                "critic_optimizer_step_count": critic_steps,
                "actor_optimizer_step_count": actor_steps,
                "target_update_count": actor_steps,
                "replay_sample_call_count": critic_steps,
                "key_evaluation_count": max(int(extra.eval_cnt_ep) - 1, 0),
                "rbf_refit_count": max(int(extra.eval_cnt_ep) - 1, 0),
            }
        )
        diagnostics_path = output_root / "training_diagnostics.jsonl"
        if diagnostics_path.exists():
            rows = [
                json.loads(line)
                for line in diagnostics_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            if rows:
                workflow.runtime_counters["key_replacement_count"] = int(
                    rows[-1].get("key_replacement_count", 0)
                )
    if metadata:
        metadata["resumed_from_checkpoint"] = str(checkpoint_path)
    return restored


def setup_recorders(config: DictConfig, workflow_name: str):
    output_dir = Path(config.output_dir)

    from evorl.recorders import LogRecorder, WandbRecorder

    recorders = []
    tags = OmegaConf.to_container(config.tags, resolve=True)
    exp_name = f"{workflow_name}_{config.env.env_name}_{config.env.env_type}"
    if len(tags) > 0:
        exp_name = exp_name + "|" + ",".join(tags)

    for rec in config.recorders:
        match rec:
            case "wandb":
                wandb_tags = [
                    workflow_name,
                    config.env.env_name,
                    config.env.env_type,
                ] + tags

                wandb_recorder = WandbRecorder(
                    project=config.project,
                    name=exp_name,
                    group="dev",
                    config=OmegaConf.to_container(
                        config, resolve=True
                    ),  # save the unrescaled config
                    tags=wandb_tags,
                    path=output_dir,
                )
                recorders.append(wandb_recorder)
            case "log":
                log_recorder = LogRecorder(
                    log_path=output_dir / f"{exp_name}.log", console=True
                )
                recorders.append(log_recorder)
            case _:
                raise ValueError(f"Unknown recorder: {rec}")

    return recorders


def _fingerprint(tree):
    import jax.tree_util as jtu
    import numpy as np

    digest = hashlib.sha256()
    for leaf in jtu.tree_leaves(tree):
        digest.update(np.asarray(leaf).tobytes())
    return digest.hexdigest()


def _smoke_report(config, workflow, state):
    import jax
    import numpy as np

    iterations = int(jax.device_get(state.metrics.iterations))
    process_count = int(config.get("process_count", 1))
    policy_freq = int(config.get("policy_freq", config.get("actor_update_interval", 1)))
    replay_state = state.replay_buffer_state
    replay_size = int(jax.device_get(replay_state.buffer_size))
    sampled_timesteps = int(jax.device_get(state.metrics.sampled_timesteps))
    rollout_length = int(config.get("rollout_length", 1))
    post_prefill_transitions = iterations * rollout_length * process_count
    prefill_env_transitions = sampled_timesteps - post_prefill_transitions
    extra_state = state.agent_state.extra_state
    total_it = int(jax.device_get(getattr(extra_state, "total_it", 0)))
    worker_steps = np.asarray(jax.device_get(extra_state.worker_steps))
    her_start = int(config.get("her_start_timesteps", 0))
    try:
        physical_id = os.environ.get("CUDA_VISIBLE_DEVICES", "unknown").split(",")[0]
        uuid = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", physical_id],
            text=True,
        ).strip().splitlines()[0]
    except (FileNotFoundError, IndexError, subprocess.CalledProcessError, ValueError):
        uuid = "unknown"
    last_metrics = getattr(workflow, "last_train_metrics", None)
    finite_loss = True
    if last_metrics is not None:
        finite_loss = all(
            bool(np.isfinite(np.asarray(leaf)).all())
            for leaf in jax.tree_util.tree_leaves(last_metrics)
            if hasattr(leaf, "shape")
        )
    latest_step = workflow.checkpoint_manager.latest_step()
    checkpoint_dir = Path(str(workflow.checkpoint_manager.directory))
    checkpoint_path = checkpoint_dir / str(latest_step) if latest_step is not None else checkpoint_dir
    report = {
        "training_seed": int(config.seed),
        "physical_gpu_uuid": uuid,
        "jax_visible_device_count": len(jax.devices("gpu")),
        "jax_visible_devices": [str(device) for device in jax.devices("gpu")],
        "actor_hidden_layer_sizes": list(config.agent_network.actor_hidden_layer_sizes),
        "critic_hidden_layer_sizes": list(config.agent_network.critic_hidden_layer_sizes),
        "logical_worker_count_K": process_count,
        "batch_size": int(config.batch_size),
        "workflow_rounds": iterations,
        "source_worker_steps": int(worker_steps.min()),
        "valid_env_transitions": sampled_timesteps,
        "prefill_env_transitions": prefill_env_transitions,
        "sampled_timesteps": sampled_timesteps,
        "critic_update_count": total_it,
        "actor_update_count": total_it // policy_freq,
        "target_update_count": total_it // policy_freq,
        "total_it": total_it,
        "replay_sampling": iterations > 0,
        "replay_size": replay_size,
        "her_active": replay_size >= her_start * process_count,
        "loss_finite": finite_loss,
        "model_fingerprint": _fingerprint(state.agent_state.params),
        "optimizer_fingerprint": _fingerprint(state.opt_state),
        "replay_fingerprint": _fingerprint(replay_state),
        "interpolator_fingerprint": _fingerprint(
            getattr(state.agent_state.extra_state, "interpolator", state.agent_state.extra_state)
        ),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_latest_step": latest_step,
        "output_dir": str(config.output_dir),
    }
    Path(config.output_dir, "smoke_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _source_budget(config):
    """Return the source-equivalent budget, counting prefill exactly once."""
    source_worker_steps = config.get("source_worker_steps")
    if source_worker_steps is None:
        return None
    source_worker_steps = int(source_worker_steps)
    process_count = int(config.get("source_process_count", config.process_count))
    rollout_length = int(config.rollout_length)
    if process_count != int(config.num_envs):
        raise ValueError("source_process_count must equal num_envs")
    if int(config.get("num_updates_per_iter", 1)) != 1:
        raise ValueError("PD-MORL source accounting requires num_updates_per_iter=1")
    valid_env_transitions = source_worker_steps * process_count
    prefill_env_transitions = int(
        config.get("source_prefill_env_transitions", config.learning_start_timesteps)
    )
    if prefill_env_transitions % process_count:
        raise ValueError("prefill transitions must divide evenly across workers")
    if prefill_env_transitions != int(config.learning_start_timesteps):
        raise ValueError("source prefill must match the sampled prefill budget")
    prefill_steps_per_worker = prefill_env_transitions // process_count
    remaining_worker_steps = source_worker_steps - prefill_steps_per_worker
    if remaining_worker_steps < 0 or remaining_worker_steps % rollout_length:
        raise ValueError("source budget must contain an integral post-prefill rollout")
    post_prefill_workflow_rounds = remaining_worker_steps // rollout_length
    policy_freq = int(config.get("policy_freq", config.get("actor_update_interval", 1)))
    critic_updates = post_prefill_workflow_rounds * process_count
    actor_updates = critic_updates // policy_freq
    budget = {
        "source_N": source_worker_steps,
        "K": process_count,
        "prefill_global_transitions": prefill_env_transitions,
        "prefill_steps_per_worker": prefill_steps_per_worker,
        "post_prefill_workflow_rounds": post_prefill_workflow_rounds,
        "expected_final_worker_steps": source_worker_steps,
        "expected_valid_env_transitions": valid_env_transitions,
        "expected_critic_optimizer_steps": critic_updates,
        "expected_actor_optimizer_steps": actor_updates,
        "expected_target_updates": actor_updates,
        "expected_total_timesteps": valid_env_transitions,
    }
    expected = {
        key: value for key, value in budget.items() if key.startswith("expected_")
    }
    expected.update(
        expected_prefill_steps_per_worker=prefill_steps_per_worker,
        expected_post_prefill_workflow_rounds=post_prefill_workflow_rounds,
    )
    for key, value in expected.items():
        configured = config.get(key)
        if configured is not None and int(configured) != value:
            raise ValueError(f"{key}={configured} does not match source mapping {value}")
    if int(config.total_timesteps) != valid_env_transitions:
        raise ValueError(
            f"total_timesteps={config.total_timesteps} double-counts or omits prefill; "
            f"expected {valid_env_transitions}"
        )
    return budget


def _log_source_budget(config):
    budget = _source_budget(config)
    if budget is None:
        return
    logger.info(" ".join(f"{key.upper()}={value}" for key, value in budget.items()))


def _assert_source_budget(config, state):
    """Guard the completed formal run against prefill double counting."""
    budget = _source_budget(config)
    if budget is None:
        return
    import jax
    import numpy as np

    worker_steps = np.asarray(jax.device_get(state.agent_state.extra_state.worker_steps))
    sampled_timesteps = int(jax.device_get(state.metrics.sampled_timesteps))
    post_prefill_transitions = (
        int(jax.device_get(state.metrics.iterations))
        * int(config.rollout_length)
        * budget["K"]
    )
    assert np.all(worker_steps == budget["expected_final_worker_steps"])
    assert sampled_timesteps == budget["expected_valid_env_transitions"]
    assert (
        budget["prefill_global_transitions"] + post_prefill_transitions
        == budget["expected_valid_env_transitions"]
    )


def _write_seed_report(output_dir, metadata, summary):
    report_path = Path(__file__).parents[1] / "docs" / "PD_MORL_WALKER_SEED1_REPRODUCTION.md"
    budget = summary["training_budget"]
    offline = summary["offline_eval"]
    text = f"""# PD-MORL Walker Seed 1 Reproduction

## 1. Run identity

Source-faithful PD-MORL reproduction with Brax environment adaptation; experiment seed 1 (not an official paper-seed assignment).

## 2. Exact config

The resolved configuration is stored at `{output_dir}/resolved_config.yaml`.

## 3. Hardware/software

GPU: {metadata['gpu_model']} ({metadata['gpu_uuid']}); JAX {metadata['jax_version']}; Brax {metadata['brax_version']}; Python {metadata['python_version']}.

## 4. Training budget

N={budget['source_N']:,} per worker, K={budget['K']}, prefill={budget['prefill_global_transitions']:,}, post-prefill rounds={budget['post_prefill_workflow_rounds']:,}.

## 5. Final counters

Worker steps: {summary['worker_steps']}; valid transitions: {summary['valid_env_transitions']:,}; critic/actor/target updates: {summary['critic_optimizer_steps']:,}/{summary['actor_optimizer_steps']:,}/{summary['target_updates']:,}.

## 6. Runtime

{summary['runtime_seconds']:.1f} seconds ({summary['runtime_hours']:.3f} hours).

## 7. Loss stability

Periodic and final diagnostics finite: {summary['loss_finite']}.

## 8. Replay/HER behavior

One global replay, final size {summary['replay_size']:,}; base inserts {summary['base_inserts']:,}; HER inserts {summary['her_inserts']:,}; HER active: {summary['her_active']}.

## 9. Key/interpolator behavior

Key evaluations {summary['key_evaluation_count']}, replacements {summary['key_replacement_count']}, interpolator refits {summary['interpolator_refit_count']}, full evaluations {summary['full_evaluation_count']}.

## 10. HV progression

Stored in `pd_morl_evaluations.jsonl` and `pareto_fronts/` using the frozen online triggers.

## 11. Sparsity progression

Stored per repeat after non-dominated filtering in the same progression artifacts.

## 12. Pareto progression

Early, middle, late, and final snapshots are retained when emitted by the source-faithful trigger.

## 13. Training-final 1001x3 result

Stored in `final_training_eval/results.json`.

## 14. Offline 1001x6 result

HV mean/std: {offline['hv_mean']:.9g}/{offline['hv_std']:.9g}; sparsity mean/std: {offline['sparsity_mean']:.9g}/{offline['sparsity_std']:.9g}.

## 15. Final Pareto front

Non-dominated points: {offline['pareto_point_count']}; stored in `offline_eval/results.json`.

## 16. Preference sensitivity

Probe actions differ across [0,1], [0.5,0.5], [1,0]: {summary['preference_sensitive']}; objective trade-off observed: {summary['objective_tradeoff']}.

## 17. Checkpoint restore

Latest/final checkpoint restore verification: {summary['checkpoint_restore_pass']}.

## 18. Comparison with paper reference

Paper HV 5.41 ± 0.004 × 10^6 and sparsity 0.03 ± 0.005 × 10^4 are reference-only, not pass/fail thresholds.

## 19. Known Brax/MuJoCo differences

Physics, reset, and termination use the current Brax Walker adapter rather than the original MuJoCo benchmark.

## 20. Classification

SOURCE-FAITHFUL: PD-MORL math, K, replay/HER, interpolator, schedule, metrics, and budget. FRAMEWORK-ADAPTATION: JAX, Brax, single GPU, PRNG mapping, Optax clipping, checkpoint/logger. ENVIRONMENT DEVIATION: Brax dynamics/reset/termination. EXPERIMENT SEED: 1.

## 21. Conclusion

Classification: **{summary['classification']}**.
"""
    report_path.write_text(text, encoding="utf-8")
    return str(report_path)


def _finish_formal_run(config, workflow, state, metadata, started_at):
    import jax
    import numpy as np

    output_dir = Path(config.output_dir)
    budget = _source_budget(config)
    extra = state.agent_state.extra_state
    worker_steps = np.asarray(jax.device_get(extra.worker_steps))
    sampled = int(jax.device_get(state.metrics.sampled_timesteps))
    total_it = int(jax.device_get(extra.total_it))
    policy_freq = int(config.actor_update_interval)

    workflow.checkpoint_manager.wait_until_finished()
    latest_step = workflow.checkpoint_manager.latest_step()
    if latest_step is None:
        raise RuntimeError("final checkpoint is missing")
    before_actions = _deterministic_probe(workflow, state)
    restore_state = state
    if not bool(config.get("save_replay_buffer", False)):
        from evorl.algorithms.offpolicy_utils import skip_replay_buffer_state

        restore_state = skip_replay_buffer_state(state)
    restored = workflow.checkpoint_manager.restore(latest_step, restore_state)
    after_actions = _deterministic_probe(workflow, restored)
    restore_checks = {
        "actor_outputs": np.array_equal(before_actions, after_actions),
        "worker_steps": np.array_equal(
            np.asarray(extra.worker_steps),
            np.asarray(restored.agent_state.extra_state.worker_steps),
        ),
        "counters": all(
            int(jax.device_get(getattr(extra, name)))
            == int(jax.device_get(getattr(restored.agent_state.extra_state, name)))
            for name in ("total_it", "eval_cnt_ep", "eval_cnt")
        ),
        "interpolator": _fingerprint(extra.interpolator)
        == _fingerprint(restored.agent_state.extra_state.interpolator),
    }
    checkpoint_restore_pass = all(restore_checks.values())
    if not checkpoint_restore_pass:
        raise RuntimeError(f"checkpoint restore mismatch: {restore_checks}")

    offline = workflow.evaluate_offline(state)
    offline_payload = _evaluation_payload(offline)
    offline_dir = output_dir / "offline_eval"
    offline_dir.mkdir(parents=True, exist_ok=True)
    (offline_dir / "results.json").write_text(
        json.dumps(offline_payload, indent=2, default=_json_default), encoding="utf-8"
    )

    diagnostics_path = output_dir / "training_diagnostics.jsonl"
    diagnostics = [
        json.loads(line)
        for line in diagnostics_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    final_diag = diagnostics[-1]
    preference_sensitive = not np.all(before_actions == before_actions[0])
    probe_returns = np.asarray(offline.mean_returns)[[0, 500, 1000]]
    objective_tradeoff = bool(np.any(np.ptp(probe_returns, axis=0) > 0))
    finite_metrics = bool(
        np.isfinite(offline.hv_per_repeat).all()
        and np.isfinite(offline.sparsity_per_repeat).all()
        and len(offline.pareto_returns) > 0
    )
    counters_pass = bool(
        np.all(worker_steps == budget["expected_final_worker_steps"])
        and sampled == budget["expected_valid_env_transitions"]
        and total_it == budget["expected_critic_optimizer_steps"]
        and total_it // policy_freq == budget["expected_actor_optimizer_steps"]
    )
    classification = (
        "STEP9.1 ONE FULL WALKER SEED PASS"
        if counters_pass
        and getattr(workflow, "diagnostics_finite", True)
        and finite_metrics
        and preference_sensitive
        and objective_tradeoff
        and checkpoint_restore_pass
        and final_diag["her_active"]
        else "STEP9.2 ONE FULL WALKER SEED FAILED"
    )
    runtime = time.time() - started_at
    summary = {
        "classification": classification,
        "training_budget": budget,
        "worker_steps": worker_steps.tolist(),
        "valid_env_transitions": sampled,
        "critic_optimizer_steps": total_it,
        "actor_optimizer_steps": total_it // policy_freq,
        "target_updates": total_it // policy_freq,
        "runtime_seconds": runtime,
        "runtime_hours": runtime / 3600,
        "loss_finite": bool(getattr(workflow, "diagnostics_finite", True)),
        "replay_size": int(jax.device_get(state.replay_buffer_state.buffer_size)),
        "base_inserts": sampled,
        "her_inserts": max(sampled - int(config.her_start_timesteps) * int(config.process_count), 0)
        * int(config.num_relabel_preferences),
        "her_active": final_diag["her_active"],
        "key_evaluation_count": final_diag["key_evaluation_count"],
        "key_replacement_count": final_diag["key_replacement_count"],
        "interpolator_refit_count": final_diag["interpolator_refit_count"],
        "full_evaluation_count": final_diag["full_evaluation_count"],
        "offline_eval": {
            "hv_mean": offline.mean_hv,
            "hv_std": float(offline.hv_per_repeat.std()),
            "sparsity_mean": offline.mean_sparsity,
            "sparsity_std": float(offline.sparsity_per_repeat.std()),
            "pareto_point_count": len(offline.pareto_returns),
        },
        "preference_probe_actions": before_actions.tolist(),
        "preference_sensitive": bool(preference_sensitive),
        "objective_tradeoff": objective_tradeoff,
        "checkpoint_latest_step": latest_step,
        "checkpoint_restore_checks": restore_checks,
        "checkpoint_restore_pass": checkpoint_restore_pass,
        "checkpoint_disk_bytes": _directory_size(output_dir / "checkpoints"),
    }
    summary["report_path"] = _write_seed_report(output_dir, metadata, summary)
    summary["final_output_disk_bytes"] = _directory_size(output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    logger.info("%s", classification)
    return summary


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(config: DictConfig) -> None:
    import jax
    import numpy as np
    from evorl.workflows import Workflow

    # jax.config.update("jax_threefry_partitionable", True)

    output_dir = get_output_dir()
    config.output_dir = str(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir, "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(config, resolve=True), encoding="utf-8"
    )
    started_at = time.time()

    logger.info("config:\n" + OmegaConf.to_yaml(config, resolve=True))
    try:
        _log_source_budget(config)
    except ValueError:
        logger.exception("STEP9.1 PRE-RUN CONFIG MISMATCH")
        raise

    workflow_cls = hydra.utils.get_class(config.workflow_cls)
    workflow_cls = type(workflow_cls.__name__, (workflow_cls,), {})

    devices = jax.local_devices()
    is_pd_morl = "mo_td3" in str(config.workflow_cls).lower()
    if is_pd_morl:
        gpu_devices = [device for device in devices if device.platform == "gpu"]
        allow_cpu_smoke = bool(config.get("allow_cpu_smoke", False))
        if (len(gpu_devices) != 1 or jax.local_device_count() != 1) and not (
            allow_cpu_smoke and not gpu_devices
        ):
            raise RuntimeError(
                "PD-MORL production requires exactly one visible GPU; "
                f"distributed_data_parallel=false, devices={gpu_devices}"
            )
        if gpu_devices:
            logger.info(
                "distributed_data_parallel=false physical_gpu=%s visible_devices=%s",
                gpu_devices[0],
                gpu_devices,
            )
            gpu = _gpu_metadata()
        else:
            logger.warning("CPU-only PD-MORL smoke; not valid for performance claims")
            gpu = {
                "physical_gpu_id": None,
                "gpu_model": None,
                "gpu_uuid": None,
                "nvidia_driver": None,
                "gpu_memory_mib": None,
            }
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).parents[1],
                text=True,
            ).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            git_commit = str(config.get("source_git_commit", "unavailable"))
        artifact_path = Path(config.interp_artifact_path)
        if not artifact_path.is_absolute():
            artifact_path = Path(__file__).parents[1] / artifact_path
        metadata = {
            "algorithm": "PD-MORL",
            "training_seed": int(config.seed),
            "environment": "Walker Brax adapter",
            "backend": "Brax",
            "run_name": (
                "GPU-native PD-MORL"
                if "PDMORLGPUWorkflow" in str(config.workflow_cls)
                else "source-faithful PD-MORL reproduction with Brax environment adaptation"
            ),
            "interpolator_backend": "jax",
            "key_objective_normalization": "l2",
            "key_artifact_path": str(artifact_path.resolve()),
            "key_artifact_sha256": hashlib.sha256(
                artifact_path.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest(),
            "git_commit": git_commit,
            "runtime_source_sha256": _source_fingerprint(),
            "date_utc": datetime.now(timezone.utc).isoformat(),
            **gpu,
            "jax_version": jax.__version__,
            "brax_version": _package_version("brax"),
            "python_version": platform.python_version(),
            "cuda_version": jax.devices()[0].client.platform_version,
            "jax_backend": jax.default_backend(),
            "visible_gpu_count": len(gpu_devices),
            "jax_devices": [str(device) for device in devices],
            "dtype": str(jax.config.jax_enable_x64 and np.dtype("float64") or np.dtype("float32")),
            "matmul_precision": str(config.matmul_precision),
            "frozen_hyperparameters": OmegaConf.to_container(config, resolve=True),
        }
        workflow: Workflow = workflow_cls.build_from_config(
            config, enable_jit=config.enable_jit
        )
    elif len(devices) > 1:
        logger.info(f"Enable Multiple Devices: {devices}")
        workflow: Workflow = workflow_cls.build_from_config(
            config, enable_multi_devices=True
        )
    else:
        workflow: Workflow = workflow_cls.build_from_config(
            config, enable_jit=config.enable_jit
        )

    recorders = setup_recorders(config, workflow_cls.name())
    workflow.add_recorders(recorders)

    try:
        state = workflow.init(jax.random.PRNGKey(config.seed))
        if is_pd_morl:
            state = _restore_checkpoint(config, workflow, state, metadata)
            import jax.tree_util as jtu

            digest = hashlib.sha256()
            for leaf in jtu.tree_leaves(
                (state.agent_state.params, state.agent_state.extra_state)
            ):
                digest.update(np.asarray(leaf).tobytes())
            logger.info("initial_pd_morl_state_sha256=%s", digest.hexdigest())
            metadata["parameter_shapes"] = _parameter_shapes(state)
            logger.info("production_parameter_shapes=%s", metadata["parameter_shapes"])
            Path(output_dir, "run_metadata.json").write_text(
                json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8"
            )
        state = workflow.learn(state)
        _assert_source_budget(config, state)
        if is_pd_morl and config.get("formal_reproduction", False):
            _finish_formal_run(config, workflow, state, metadata, started_at)
        if config.get("independent_seed_smoke", False):
            logger.info("independent_seed_smoke_report=%s", _smoke_report(config, workflow, state))
    except Exception as e:
        logger.error(f"Exception: {e}")
        raise
    finally:
        workflow.close()


if __name__ == "__main__":
    train()
