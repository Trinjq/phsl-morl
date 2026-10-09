import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from omegaconf import OmegaConf

from train import _resolve_resume_checkpoint, _restore_checkpoint


def test_resume_latest_selects_highest_numeric_checkpoint(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    (checkpoints / "100").mkdir(parents=True)
    (checkpoints / "200").mkdir()
    (checkpoints / "not-a-step").mkdir()
    config = OmegaConf.create(
        {
            "output_dir": str(tmp_path),
            "resume_from_checkpoint": None,
            "resume_latest": True,
        }
    )
    assert _resolve_resume_checkpoint(config) == checkpoints / "200"


def test_resume_rejects_checkpoint_without_replay_buffer(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "100"
    checkpoint.mkdir(parents=True)
    config = OmegaConf.create(
        {
            "output_dir": str(tmp_path),
            "resume_from_checkpoint": str(checkpoint),
            "resume_latest": False,
            "save_replay_buffer": False,
        }
    )
    with pytest.raises(ValueError, match="save_replay_buffer=true"):
        _restore_checkpoint(config, object(), object(), {})
