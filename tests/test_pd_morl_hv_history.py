from types import SimpleNamespace

import numpy as np

from evorl.algorithms.mo_td3 import append_hv_history


def _result():
    return SimpleNamespace(
        preferences=np.zeros((201, 2)),
        returns_per_repeat=np.zeros((3, 201, 2)),
        mean_hv=4.0,
        source_hv=5.0,
        mean_repeat_hv=4.5,
        mean_sparsity=1.0,
        source_sparsity=2.0,
        pareto_returns=np.zeros((3, 2)),
        source_pareto_point_count=3,
    )


def test_hv_history_appends_and_deduplicates(tmp_path):
    path = tmp_path / "hv_history.csv"
    result = _result()
    assert append_hv_history(
        path,
        seed=15,
        run_id="seed15",
        artifact_sha256="abc",
        iteration=7,
        actual_env_transitions=123,
        result=result,
        wall_clock_seconds=8.5,
    )
    assert not append_hv_history(
        path,
        seed=15,
        run_id="seed15",
        artifact_sha256="abc",
        iteration=7,
        actual_env_transitions=123,
        result=result,
        wall_clock_seconds=8.5,
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "source_hv" in lines[0]
    assert ",5.0," in lines[1]
