from omegaconf import OmegaConf
from evorl.utils.orbax_utils import setup_checkpoint_manager

cfg = OmegaConf.create({
    "output_dir": "/home/qiuquanj/projects/evorl/outputs/train/2026-09-28_11-52-59",
    "checkpoint": {
        "enable": True,
        "save_interval_steps": 1,
        "max_to_keep": 5,
    },
})
mgr = setup_checkpoint_manager(cfg)
latest = mgr.latest_step()
print(f"Latest step: {latest}")
assert latest == 8, f"Expected latest step 8, got {latest}"
mgr.close()
print("Checkpoint restore validation: SUCCESS")
