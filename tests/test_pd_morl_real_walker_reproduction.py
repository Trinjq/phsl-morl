"""REAL WALKER FAILING-UPDATE REPRODUCTION; intentionally two-GPU only."""

import jax
import pytest

from tests.pd_morl_multi_gpu_smoke import run_real_walker_distributed_smoke

pytestmark = [pytest.mark.gpu, pytest.mark.multi_gpu, pytest.mark.slow]


def test_real_walker_failing_update_reproduction(tmp_path):
    devices = jax.devices("gpu")
    if len(devices) != 2:
        pytest.skip("real Step8 reproduction requires exactly two visible GPUs")
    result = run_real_walker_distributed_smoke(devices, tmp_path, diagnose=True)
    print(result)
    assert result["classification"] in {
        "STEP8 NUMERICAL REDUCTION-ORDER DIAGNOSIS",
        "STEP8 DISTRIBUTED IMPLEMENTATION BUG DIAGNOSIS",
    }
