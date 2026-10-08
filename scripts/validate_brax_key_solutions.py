"""Validate and optionally promote a completed Brax key artifact."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import numpy as np

from evorl.utils.pd_morl_interpolator import (
    fit_interpolator_state,
    fit_reference_interpolator,
    interpolate,
    key_preferences,
    normalize_key_solutions,
)


EXPECTED_KEYS = np.asarray([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
EXPECTED_CRITIC_STEPS = 1_999_929
EXPECTED_ACTOR_STEPS = 999_964


def validate(artifact_path: Path, metadata_path: Path) -> dict:
    artifact = np.loadtxt(artifact_path, delimiter=",")
    assert artifact.shape == (3, 2), artifact.shape
    assert np.isfinite(artifact).all()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifact_bytes = artifact_path.read_bytes().replace(b"\r\n", b"\n")
    expected_hash = hashlib.sha256(artifact_bytes).hexdigest()
    assert metadata["artifact_sha256"] == expected_hash
    np.testing.assert_array_equal(np.asarray(metadata["keys"]), EXPECTED_KEYS)
    assert metadata["hyperparameters"]["gamma"] == 0.99
    assert metadata["hyperparameters"]["batch_size"] == 100
    assert metadata["hyperparameters"]["replay_capacity"] == 500_000
    assert metadata["hyperparameters"]["policy_freq"] == 2
    assert metadata["hyperparameters"]["evaluation_episodes"] == 10

    for row, per_key in zip(artifact, metadata["per_key"]):
        np.testing.assert_allclose(row, per_key["best_vector_return"])
        assert per_key["counts"]["actual_environment_steps"] == 2_000_128
        assert per_key["counts"]["random_action_steps"] == 25_088
        assert per_key["counts"]["critic_optimizer_step_count"] == EXPECTED_CRITIC_STEPS
        assert per_key["counts"]["actor_optimizer_step_count"] == EXPECTED_ACTOR_STEPS
        assert per_key["expected_counts"]["critic_optimizer_steps"] == EXPECTED_CRITIC_STEPS
        assert per_key["expected_counts"]["actor_optimizer_steps"] == EXPECTED_ACTOR_STEPS
        assert per_key["evaluation_history"][-1]["phase"] == "training_final"
        np.testing.assert_allclose(
            per_key["best_scalarized_return"], np.dot(per_key["preference"], row)
        )

    assert np.argmax(artifact[:, 1]) == 0
    assert np.argmax(artifact[:, 0]) == 2
    assert not np.any(
        np.all(artifact[[0, 2]] >= artifact[1], axis=1)
        & np.any(artifact[[0, 2]] > artifact[1], axis=1)
    )

    checks = {}
    keys = key_preferences(2)
    for normalization in ("initial", "online"):
        reference = fit_reference_interpolator(keys, artifact, normalization)
        state = fit_interpolator_state(keys, artifact, normalization)
        np.testing.assert_allclose(reference(keys), normalize_key_solutions(artifact, normalization))
        np.testing.assert_allclose(
            np.asarray(interpolate(state, keys)),
            normalize_key_solutions(artifact, normalization),
            rtol=2e-5,
            atol=2e-5,
        )
        values = np.asarray(interpolate(state, np.asarray([[0.25, 0.75], [0.75, 0.25]])))
        assert np.isfinite(values).all()
        checks[normalization] = {"finite": True}
    return {"artifact": str(artifact_path), "sha256": expected_hash, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    metadata_path = args.metadata or args.artifact.with_name(
        "interp_objs_walker2d_brax.metadata.json"
    )
    result = validate(args.artifact, metadata_path)
    if args.promote:
        target = Path("configs/artifacts/interp_objs_walker2d_brax.txt")
        target_metadata = target.with_name("interp_objs_walker2d_brax.metadata.json")
        shutil.copyfile(args.artifact, target)
        shutil.copyfile(metadata_path, target_metadata)
        result["promoted_to"] = str(target)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
