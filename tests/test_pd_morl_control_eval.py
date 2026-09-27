from pathlib import Path

import jax
import numpy as np
import pytest
from evorl.algorithms.mo_td3 import completed_episodes_by_worker
from evorl.evaluators.pd_morl import (
    aggregate_candidate_returns,
    evaluate_morl,
    evaluation_seeds,
    full_evaluation_due,
    hypervolume,
    key_update_due,
    load_key_solution_artifact,
    non_dominated_indices,
    preference_grid,
    replace_key_solutions,
    sparsity,
    update_key_solutions,
)
from evorl.utils.pd_morl_interpolator import key_preferences
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

KEYS = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])


def test_initial_artifact_required(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_key_solution_artifact(tmp_path / "missing.txt", KEYS)


def test_initial_artifact_shape(tmp_path):
    artifact = tmp_path / "bad.txt"
    np.savetxt(artifact, np.ones((2, 2)), delimiter=",")
    with pytest.raises(ValueError, match="shape"):
        load_key_solution_artifact(artifact, KEYS)


def test_initial_artifact_preference_order(tmp_path):
    artifact = tmp_path / "keys.npz"
    np.savez(
        artifact,
        key_solutions=np.ones((3, 2)),
        key_preferences=KEYS[::-1],
    )
    with pytest.raises(ValueError, match="ordering"):
        load_key_solution_artifact(artifact, KEYS)


def test_official_artifact_provenance():
    path = Path("configs/artifacts/interp_objs_walker2d.txt")
    values, provenance = load_key_solution_artifact(path, KEYS)
    assert values.shape == (3, 2)
    assert provenance.shape == (3, 2)
    assert provenance.key_preferences == KEYS.tolist()
    assert (
        provenance.sha256
        == "7b7f701574ced75af9df1fefdece160463643a85df9826544f7fb2449257886b"
    )


def test_key_update_off_by_one():
    assert not key_update_due([2] * 9 + [1], 1)
    assert key_update_due([2] * 10, 1)
    assert not key_update_due([3] * 9 + [2], 2)
    assert key_update_due([3] * 10, 2)


def test_full_evaluation_trigger_is_independent():
    assert not full_evaluation_due([101] * 9 + [100], 1)
    assert full_evaluation_due([101] * 10, 1)
    assert not key_update_due([101] * 10, 101)


def test_worker_episode_counts_are_cumulative():
    dones = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    np.testing.assert_array_equal(completed_episodes_by_worker(dones, 2), [2, 2])


def test_candidate_repeat_mean():
    returns = np.arange(18).reshape(3, 3, 2)
    np.testing.assert_array_equal(aggregate_candidate_returns(returns), returns.mean(0))


@pytest.mark.parametrize(
    ("candidate", "changed"),
    [([1.0, 3.0], True), ([1.0, 2.0], False), ([1.0, 1.0], False)],
)
def test_candidate_replacement_rule(candidate, changed):
    updated, improved = replace_key_solutions([[1.0, 2.0]], [candidate], [[0.0, 1.0]])
    assert improved.tolist() == [changed]
    np.testing.assert_array_equal(updated[0], candidate if changed else [1.0, 2.0])


def test_replacement_is_direct_assignment_and_no_ema():
    candidate = np.array([[9.0, 7.0]])
    updated, _ = replace_key_solutions([[1.0, 2.0]], candidate, [[0.5, 0.5]])
    np.testing.assert_array_equal(updated, candidate)


def test_online_refit_uses_l1():
    old = np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 1.0]])
    updated, _, state = update_key_solutions(old, np.stack((old, old)), KEYS)
    np.testing.assert_allclose(
        state.normalized_key_solutions, updated / updated.sum(1)[:, None]
    )


def test_refit_even_without_replacement():
    old = np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 1.0]])
    updated, improved, state = update_key_solutions(old, np.stack((old, old)), KEYS)
    assert not improved.any()
    assert state.coefficients.shape[0] == len(KEYS) + state.powers.shape[0]
    np.testing.assert_array_equal(updated, old)


def test_interpolator_shape_stable_after_refit():
    old = np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 1.0]])
    _, _, first = update_key_solutions(old, np.stack((old, old)), KEYS)
    _, _, second = update_key_solutions(old, np.stack((old + 1, old + 1)), KEYS)
    assert jax.tree.map(lambda x: x.shape, first) == jax.tree.map(
        lambda x: x.shape, second
    )


def test_training_jit_contains_no_scipy():
    import inspect

    from evorl.algorithms.mo_td3 import MOTD3Workflow

    source = inspect.getsource(MOTD3Workflow.step)
    assert "scipy" not in source.lower()
    assert "RBFInterpolator" not in source


def test_evaluation_seed_formula():
    np.testing.assert_array_equal(evaluation_seeds(4), [0, 11, 22, 33])


def test_training_eval_seed_list():
    np.testing.assert_array_equal(evaluation_seeds(3), [0, 11, 22])


def test_offline_eval_seed_list():
    np.testing.assert_array_equal(evaluation_seeds(6), [0, 11, 22, 33, 44, 55])


def test_deterministic_actor_eval():
    fn = lambda preference, seed, index: preference + seed + index
    first = evaluate_morl(fn, KEYS, 3).returns_per_repeat
    second = evaluate_morl(fn, KEYS, 3).returns_per_repeat
    np.testing.assert_array_equal(first, second)


def test_vector_return_is_undiscounted():
    rewards = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = evaluate_morl(lambda *_: rewards.sum(0), KEYS[:1], 1)
    np.testing.assert_array_equal(result.mean_returns[0], [4.0, 6.0])


def test_episode_limit_and_done():
    rewards = np.array([[1.0, 1.0], [2.0, 2.0], [99.0, 99.0]])
    done_at = 2
    result = evaluate_morl(lambda *_: rewards[:done_at].sum(0), KEYS[:1], 1)
    np.testing.assert_array_equal(result.mean_returns[0], [3.0, 3.0])


def test_training_grid_201():
    grid = preference_grid(0.005)
    assert grid.shape == (201, 2)
    np.testing.assert_array_equal(grid[[0, -1]], [[0.0, 1.0], [1.0, 0.0]])


def test_final_grid_1001():
    assert preference_grid(0.001).shape == (1001, 2)


def test_training_aggregation_order():
    result = evaluate_morl(
        lambda preference, seed, index: preference * (seed + 1), KEYS, 2
    )
    assert result.mean_hv == pytest.approx(result.hv_per_repeat.mean())


def test_offline_aggregation_order():
    result = evaluate_morl(lambda preference, seed, index: preference + seed, KEYS, 6)
    assert result.returns_per_repeat.shape == (6, 3, 2)
    np.testing.assert_array_equal(
        result.pareto_returns, result.mean_returns[result.pareto_indices]
    )


def test_pareto_dominated_and_nondominated_points():
    points = np.array([[1, 5], [2, 4], [3, 3], [2, 2]])
    np.testing.assert_array_equal(non_dominated_indices(points), [0, 1, 2])


def test_pareto_duplicates_preserved():
    points = np.array([[1, 5], [2, 4], [2, 4], [3, 3]])
    np.testing.assert_array_equal(non_dominated_indices(points), [0, 1, 2, 3])


@pytest.mark.parametrize(
    ("points", "expected"),
    [([[1, 1]], [0]), ([[2, 4], [2, 4]], [0, 1]), ([[1, 1], [2, 2]], [1])],
)
def test_pareto_boundaries(points, expected):
    np.testing.assert_array_equal(non_dominated_indices(points), expected)


def test_pareto_tie_on_one_objective():
    np.testing.assert_array_equal(non_dominated_indices([[2, 3], [2, 4]]), [1])


def test_hv_against_pymoo():
    points = np.array([[1.0, 5.0], [2.0, 4.0], [3.0, 3.0]])
    assert hypervolume(points) == pytest.approx(HV(ref_point=np.zeros(2))(-points))


def test_hv_zero_reference_and_sign_conversion():
    assert hypervolume([[2.0, 3.0]]) == pytest.approx(6.0)


def test_sparsity_against_source_formula():
    points = np.array([[1.0, 5.0], [2.0, 4.0], [3.0, 3.0]])
    assert sparsity(points) == pytest.approx(2.0)


def test_sparsity_golden_filters_dominated_points_with_source_sign():
    returns = np.array([[1, 4], [2, 3], [3, 2], [4, 1], [1, 1]], dtype=float)
    objectives = -returns
    front = objectives[
        NonDominatedSorting().do(objectives, only_non_dominated_front=True)
    ]
    expected = np.square(np.diff(np.sort(front, axis=0), axis=0)).sum() / (
        len(front) - 1
    )
    assert sparsity(returns) == pytest.approx(expected, abs=1e-12)


def test_sparsity_dominated_points_change_unfiltered_value():
    returns = np.array(
        [[1, 5], [2, 4], [3, 3], [4, 2], [5, 1], [1, 1], [2, 2]],
        dtype=float,
    )
    unfiltered = np.square(np.diff(np.sort(returns, axis=0), axis=0)).sum() / (
        len(returns) - 1
    )
    assert sparsity(returns) != pytest.approx(unfiltered)


def test_sparsity_empty_front_zero():
    assert sparsity(np.empty((0, 2))) == 0.0


def test_sparsity_single_point_zero():
    assert sparsity([[1.0, 1.0]]) == 0.0


def test_sparsity_duplicates():
    assert sparsity([[1.0, 1.0], [1.0, 1.0]]) == 0.0


def test_official_key_preference_order():
    np.testing.assert_array_equal(key_preferences(2), KEYS)
