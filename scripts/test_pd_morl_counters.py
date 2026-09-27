"""Deterministic Step9 counter-semantics check; does not launch training."""

from __future__ import annotations

import json


def _replay_accounting(
    base_inserts: int, capacity: int, her_threshold: int, relabel_count: int
) -> tuple[int, int, int]:
    size = 0
    her_inserts = 0
    total_entries_written = 0
    for _ in range(base_inserts):
        active = size >= capacity or size + 1 > her_threshold
        added = 1 + (relabel_count if active else 0)
        her_inserts += relabel_count if active else 0
        total_entries_written += added
        size = min(size + added, capacity)
    return size, her_inserts, total_entries_written


def main() -> dict[str, int | bool]:
    # Production shape and learner constants; only the source-equivalent budget
    # is shortened for this arithmetic test.
    worker_count = 10
    batch_size = 256
    actor_critic_hidden = (400, 400)
    policy_delay = 10
    relabel_count = 3
    source_worker_steps = 100
    prefill_worker_steps = 3
    her_start_worker_steps = 20

    prefill_env_transitions = prefill_worker_steps * worker_count
    workflow_rounds = source_worker_steps - prefill_worker_steps
    post_prefill_env_transitions = workflow_rounds * worker_count
    valid_env_transitions = source_worker_steps * worker_count
    base_replay_inserts = prefill_env_transitions + post_prefill_env_transitions
    critic_optimizer_steps = workflow_rounds * worker_count
    actor_optimizer_steps = workflow_rounds * worker_count // policy_delay
    target_update_steps = actor_optimizer_steps
    replay_size, her_inserts, total_entries_written = _replay_accounting(
        base_replay_inserts,
        capacity=10_000,
        her_threshold=her_start_worker_steps * worker_count,
        relabel_count=relabel_count,
    )

    assert worker_count == 10
    assert batch_size == 256
    assert actor_critic_hidden == (400, 400)
    assert valid_env_transitions == 1_000
    assert prefill_env_transitions == 30
    final_worker_steps = prefill_worker_steps + workflow_rounds
    assert final_worker_steps == source_worker_steps == 100
    assert workflow_rounds == 97
    assert prefill_env_transitions + post_prefill_env_transitions == 1_000
    assert critic_optimizer_steps == 970
    assert actor_optimizer_steps == 97
    assert target_update_steps == 97
    assert her_inserts == 2_400
    assert total_entries_written == base_replay_inserts + her_inserts == 3_400
    assert replay_size == total_entries_written

    result = {
        "status": "PASS",
        "K": worker_count,
        "batch_size": batch_size,
        "network_hidden_layers": list(actor_critic_hidden),
        "source_worker_steps": source_worker_steps,
        "prefill_worker_steps": prefill_worker_steps,
        "final_worker_steps": final_worker_steps,
        "workflow_rounds": workflow_rounds,
        "valid_env_transitions": valid_env_transitions,
        "prefill_env_transitions": prefill_env_transitions,
        "replay_base_inserts": base_replay_inserts,
        "replay_her_inserts": her_inserts,
        "replay_total_entries_written": total_entries_written,
        "replay_size": replay_size,
        "critic_optimizer_steps": critic_optimizer_steps,
        "actor_optimizer_steps": actor_optimizer_steps,
        "target_update_steps": target_update_steps,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
