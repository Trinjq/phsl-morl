import csv
import json
from pathlib import Path

import numpy as np
from evorl.algorithms.mo_td3 import (
    append_key_diagnostics,
    build_key_diagnostic_rows,
)
from hydra import compose, initialize_config_dir


def _rows():
    keys = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]], dtype=np.float32)
    old = np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 1.0]], dtype=np.float32)
    candidates = np.array([[99.0, 1.0], [1.0, 3.0], [4.0, -99.0]], dtype=np.float32)
    repeats = np.stack((candidates - 0.5, candidates, candidates + 0.5))
    improved = np.array([False, False, True])
    new = old.copy()
    new[improved] = candidates[improved]
    return build_key_diagnostic_rows(
        seed=15,
        run_id="smoke",
        artifact_sha256="abc",
        iteration=10,
        actual_env_transitions=260160,
        event_id=7,
        eval_cnt_ep_before=7,
        episode_count_raw=np.full(10, 128),
        episode_count_logical=np.full(10, 8),
        keys=keys,
        old_solutions=old,
        returns_per_repeat=repeats,
        improved=improved,
        new_solutions=new,
        evaluation_seconds=1.25,
        logging_seconds=0.01,
    )


def test_key_diagnostic_rows_match_learner_decision():
    rows = _rows()
    assert len(rows) == 3
    assert [row["key_index"] for row in rows] == [0, 1, 2]
    assert [row["replaced"] for row in rows] == [False, False, True]
    assert [row["score_gap"] for row in rows] == [-1.0, 0.0, 1.0]
    assert all(len(json.loads(row["candidate_repeat_scores_3"])) == 3 for row in rows)
    assert rows[0]["stored_r1_after"] == rows[0]["stored_r1_before"]
    assert rows[2]["stored_r0_after"] == rows[2]["candidate_r0"]


def test_key_diagnostic_append_is_resume_safe(tmp_path):
    path = tmp_path / "key_replacement_diagnostics.csv"
    assert append_key_diagnostics(path, _rows()) == 3
    assert append_key_diagnostics(path, _rows()) == 0
    with path.open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == 3
    assert {(row["run_id"], row["event_id"], row["key_index"]) for row in saved} == {
        ("smoke", "7", "0"),
        ("smoke", "7", "1"),
        ("smoke", "7", "2"),
    }


def test_diagnostic_experiment_is_explicitly_enabled():
    with initialize_config_dir(
        version_base=None,
        config_dir=str((Path(__file__).parents[1] / "configs").resolve()),
    ):
        config = compose(
            config_name="experiment/pd_morl_key_replacement_diagnostics",
            overrides=["seed=15"],
        )
    assert config.key_diagnostics.enable is True
    assert config.interp_artifact_path.endswith("walker2d_brax_v3.txt")
