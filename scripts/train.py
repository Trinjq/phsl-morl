import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path

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
    num_updates = int(config.get("num_updates_per_iter", 1))
    policy_freq = int(config.get("policy_freq", config.get("actor_update_interval", 1)))
    replay_state = state.replay_buffer_state
    replay_size = int(jax.device_get(replay_state.buffer_size))
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
        "critic_update_count": iterations * num_updates * policy_freq,
        "actor_update_count": iterations * num_updates,
        "target_update_count": iterations * num_updates,
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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(config: DictConfig) -> None:
    import jax
    from evorl.workflows import Workflow

    # jax.config.update("jax_threefry_partitionable", True)

    output_dir = get_output_dir()
    config.output_dir = str(output_dir)

    logger.info("config:\n" + OmegaConf.to_yaml(config, resolve=True))

    workflow_cls = hydra.utils.get_class(config.workflow_cls)
    workflow_cls = type(workflow_cls.__name__, (workflow_cls,), {})

    devices = jax.local_devices()
    is_pd_morl = "mo_td3" in str(config.workflow_cls).lower()
    if is_pd_morl:
        gpu_devices = jax.devices("gpu")
        if len(gpu_devices) != 1 or jax.local_device_count() != 1:
            raise RuntimeError(
                "PD-MORL production requires exactly one visible GPU; "
                f"distributed_data_parallel=false, devices={gpu_devices}"
            )
        logger.info(
            "distributed_data_parallel=false physical_gpu=%s visible_devices=%s",
            gpu_devices[0],
            gpu_devices,
        )
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
            import jax.tree_util as jtu
            import numpy as np

            digest = hashlib.sha256()
            for leaf in jtu.tree_leaves(
                (state.agent_state.params, state.agent_state.extra_state)
            ):
                digest.update(np.asarray(leaf).tobytes())
            logger.info("initial_pd_morl_state_sha256=%s", digest.hexdigest())
        state = workflow.learn(state)
        if config.get("independent_seed_smoke", False):
            logger.info("independent_seed_smoke_report=%s", _smoke_report(config, workflow, state))
    except Exception as e:
        logger.error(f"Exception: {e}")
        raise
    finally:
        workflow.close()


if __name__ == "__main__":
    train()
