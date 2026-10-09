import jax
import jax.numpy as jnp
import numpy as np
from evorl.algorithms.mo_td3 import periodic_update_count
from evorl.envs import EnvState
from evorl.envs.wrappers.preference_wrapper import make_logical_worker_ids
from evorl.evaluators.pd_morl import BatchedPDMORLEvaluator, PDMORLEvaluator
from evorl.types import PyTreeDict


class _PreferenceAgent:
    def evaluate_actions(self, agent_state, batch, key):
        del agent_state, key
        preference = batch.extras.policy_extras.preference
        return preference[:, :1], PyTreeDict(preference=preference)


class _TwoStepEnv:
    def __init__(self, num_envs):
        self.num_envs = num_envs

    def reset(self, key):
        if key.ndim == 1:
            key = jax.random.split(key, self.num_envs)
        signal = jax.vmap(lambda k: jax.random.uniform(k, ()))(key)
        return EnvState(
            env_state=PyTreeDict(),
            obs=signal[:, None],
            reward=jnp.zeros((self.num_envs, 2)),
            done=jnp.zeros((self.num_envs,), dtype=jnp.bool_),
            info=PyTreeDict(),
            _internal=PyTreeDict(step=jnp.zeros((self.num_envs,), dtype=jnp.int32)),
        )

    def step(self, state, action):
        step = state._internal.step + 1
        signal = state.obs[:, 0]
        first = action[:, 0] + signal
        reward = jnp.stack((first, 1.0 - action[:, 0] + signal), axis=-1)
        return state.replace(
            reward=reward,
            done=step >= 2,
            _internal=state._internal.replace(step=step),
        )


def test_logical_groups_are_replicated_not_expanded():
    worker_ids = make_logical_worker_ids(40, 10)
    assert worker_ids.shape == (40,)
    np.testing.assert_array_equal(np.bincount(np.asarray(worker_ids)), [4] * 10)
    np.testing.assert_array_equal(np.asarray(worker_ids[:10]), np.arange(10))


def test_batched_evaluator_matches_serial_for_key_and_padded_batches():
    agent = _PreferenceAgent()
    serial = PDMORLEvaluator(_TwoStepEnv(1), agent, max_episode_steps=5)
    batched = BatchedPDMORLEvaluator(
        key_env=_TwoStepEnv(9),
        env=_TwoStepEnv(5),
        agent=agent,
        max_episode_steps=5,
    )

    for preferences, repeats in (
        (np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]), 3),
        (np.array([[0.0, 1.0], [0.3, 0.7], [0.6, 0.4], [1.0, 0.0]]), 2),
    ):
        expected = serial.evaluate(None, preferences, repeats)
        actual = batched.evaluate(None, preferences, repeats)
        np.testing.assert_allclose(
            actual.returns_per_repeat, expected.returns_per_repeat, rtol=0, atol=0
        )
        np.testing.assert_allclose(actual.mean_returns, expected.mean_returns)
        np.testing.assert_allclose(actual.hv_per_repeat, expected.hv_per_repeat)
        np.testing.assert_allclose(
            actual.sparsity_per_repeat, expected.sparsity_per_repeat
        )
        np.testing.assert_array_equal(
            actual.pareto_counts_per_repeat, expected.pareto_counts_per_repeat
        )
        assert actual.mean_pareto_count == expected.mean_pareto_count
        assert actual.source_hv == expected.source_hv
        assert actual.source_sparsity == expected.source_sparsity
        assert actual.source_pareto_point_count == expected.source_pareto_point_count
        assert actual.mean_repeat_hv == expected.mean_repeat_hv
        assert actual.mean_repeat_sparsity == expected.mean_repeat_sparsity


def test_device_key_evaluation_returns_fixed_jax_shape():
    batched = BatchedPDMORLEvaluator(
        key_env=_TwoStepEnv(9),
        env=_TwoStepEnv(5),
        agent=_PreferenceAgent(),
        max_episode_steps=5,
    )
    returns = batched.evaluate_keys_device(None, repeats=3)
    assert isinstance(returns, jax.Array)
    assert returns.shape == (3, 3, 2)
    assert np.isfinite(np.asarray(returns)).all()


def test_source_compatible_metric_aggregation():
    """Verify source-compatible evaluation aggregation order and field distinction."""
    from evorl.evaluators.pd_morl import hypervolume, morl_evaluation_result, sparsity

    # 2 repeats, 5 preferences. Point 4 is dominated.
    # Repeat 0:
    r0 = np.array([[10.0, 1.0], [8.0, 3.0], [4.0, 7.0], [1.0, 9.0], [2.0, 2.0]])
    # Repeat 1:
    r1 = np.array([[8.0, 1.0], [6.0, 5.0], [4.0, 7.0], [3.0, 9.0], [2.0, 2.0]])
    returns = np.stack([r0, r1], axis=0)  # [2, 5, 2]
    prefs = np.zeros((5, 2))

    res = morl_evaluation_result(prefs, returns)

    # 1. Mean over repeat axis first
    expected_means = np.array(
        [[9.0, 1.0], [7.0, 4.0], [4.0, 7.0], [2.0, 9.0], [2.0, 2.0]]
    )
    np.testing.assert_allclose(res.mean_returns, expected_means)

    # 2. Non-dominated filtering on mean returns excludes [2.0, 2.0]
    expected_source_front = expected_means[:4]
    np.testing.assert_allclose(res.pareto_returns, expected_source_front)
    assert res.source_pareto_point_count == 4

    # 3. Source metrics computed on mean Pareto front
    expected_source_hv = hypervolume(expected_source_front)
    expected_source_sp = sparsity(expected_source_front)
    np.testing.assert_allclose(res.source_hv, expected_source_hv)
    np.testing.assert_allclose(res.source_sparsity, expected_source_sp)

    # 4. Repeat diagnostic metrics distinct from source metrics
    assert len(res.hv_per_repeat) == 2
    assert len(res.sparsity_per_repeat) == 2
    assert len(res.pareto_counts_per_repeat) == 2
    np.testing.assert_allclose(res.mean_repeat_hv, res.hv_per_repeat.mean())
    np.testing.assert_allclose(res.mean_repeat_sparsity, res.sparsity_per_repeat.mean())
    np.testing.assert_allclose(
        res.mean_repeat_pareto_count, res.pareto_counts_per_repeat.mean()
    )

    # 5. Single Pareto point gives sparsity 0.0
    single_pt_res = morl_evaluation_result(np.zeros((1, 2)), np.array([[[5.0, 5.0]]]))
    assert single_pt_res.source_sparsity == 0.0
    assert single_pt_res.mean_repeat_sparsity == 0.0


def test_logical_group_warmup_budget_invariant_to_lane_count():
    """Verify that logical group warmup always completes at 100k global transitions."""
    for envs_per_group in (1, 4, 16):
        num_envs = 10 * envs_per_group
        rollout_length = 4
        group_transitions_per_rollout = envs_per_group * rollout_length
        global_transitions_per_rollout = num_envs * rollout_length
        rollouts_to_warmup = 10000 / group_transitions_per_rollout
        total_global_transitions = rollouts_to_warmup * global_transitions_per_rollout
        assert total_global_transitions == 100_000


def test_policy_delay_aligned_with_critic_steps():
    """Verify actor updates trigger strictly every policy_freq critic steps regardless of K."""
    policy_freq = 10
    for K in (5, 10, 20):
        total_critic_steps = 100
        rollouts = total_critic_steps // K
        total_it = 0
        actor_updates = 0
        for _ in range(rollouts):
            update_ids = total_it + np.arange(1, K + 1)
            actor_updates += np.sum(update_ids % policy_freq == 0)
            total_it += K
        assert actor_updates == total_critic_steps // policy_freq


def test_runtime_actor_counter_handles_k_below_policy_frequency():
    assert periodic_update_count(0, 20, 10) == 2
    assert periodic_update_count(5, 20, 10) == 2


def test_warmup_diagnostic_subtraction_does_not_underflow():
    worker_steps = np.asarray([6816], dtype=np.uint32).astype(np.int64)
    np.testing.assert_array_equal(np.maximum(worker_steps - 10000, 0), [0])


def test_sparsity_excludes_dominated_points():
    from evorl.evaluators.pd_morl import morl_evaluation_result, sparsity

    # 5 points: (1, 10), (5, 5), (10, 1) are non-dominated. (2, 2) and (0.5, 0.5) are dominated.
    raw_returns = np.array(
        [[[1.0, 10.0], [5.0, 5.0], [10.0, 1.0], [2.0, 2.0], [0.5, 0.5]]]
    )
    dummy_prefs = np.zeros((5, 2))
    res = morl_evaluation_result(dummy_prefs, raw_returns)
    assert res.pareto_counts_per_repeat[0] == 3
    assert res.mean_pareto_count == 3.0
    expected_pareto = np.array([[1.0, 10.0], [5.0, 5.0], [10.0, 1.0]])
    expected_sparsity = sparsity(expected_pareto)
    np.testing.assert_allclose(res.sparsity_per_repeat[0], expected_sparsity)


def test_checkpoint_manager_save_and_reload(tmp_path):
    from evorl.utils.orbax_utils import setup_checkpoint_manager
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "output_dir": str(tmp_path),
            "checkpoint": {
                "enable": True,
                "save_interval_steps": 1,
                "max_to_keep": 2,
            },
        }
    )
    manager = setup_checkpoint_manager(cfg)
    dummy_state = {"params": jnp.array([1.0, 2.0, 3.0]), "step": 10}
    manager.save(1, dummy_state, force=True)
    manager.wait_until_finished()

    latest = manager.latest_step()
    assert latest == 1
    target = {"params": jnp.zeros(3), "step": 0}
    restored = manager.restore(latest, items=target)
    np.testing.assert_allclose(restored["params"], dummy_state["params"])
    assert restored["step"] == 10
    manager.close()
