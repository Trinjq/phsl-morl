import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

CONVERGENCE_FIELDS = (
    "seed",
    "run_id",
    "artifact_sha256",
    "kind",
    "preference_count",
    "repeats",
    "target_transition_threshold",
    "actual_env_transitions",
    "iteration",
    "source_hv",
    "mean_repeat_hv",
    "source_sparsity",
    "source_pareto_point_count",
    "evaluation_seconds",
    "wall_clock_seconds",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONVERGENCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()


def prepare_convergence_history(
    path: Path, *, run_id: str, sampled_timesteps: int, interval: int
) -> int:
    """Clean rollback rows and return the next threshold without backfilling."""
    if interval <= 0:
        raise ValueError("convergence interval must be positive")
    rows = _read_rows(path)
    valid = [
        row
        for row in rows
        if row.get("run_id") == run_id
        and row.get("kind") == "convergence"
        and row.get("actual_env_transitions", "").isdigit()
        and row.get("target_transition_threshold", "").isdigit()
    ]
    future = [
        row for row in valid if int(row["actual_env_transitions"]) > sampled_timesteps
    ]
    if future:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive = path.with_name(f"{path.stem}.rollback-{stamp}{path.suffix}")
        _write_rows(archive, future)
        future_ids = {id(row) for row in future}
        rows = [row for row in rows if id(row) not in future_ids]
        rows.sort(
            key=lambda row: (
                int(row["actual_env_transitions"])
                if row.get("actual_env_transitions", "").isdigit()
                else 0
            )
        )
        _write_rows(path, rows)
        valid = [row for row in valid if id(row) not in future_ids]

    if valid:
        last = max(int(row["target_transition_threshold"]) for row in valid)
        return last + interval
    return (sampled_timesteps // interval + 1) * interval


def append_convergence_result(
    path: Path,
    *,
    seed: int,
    run_id: str,
    artifact_sha256: str,
    target_transition_threshold: int,
    actual_env_transitions: int,
    iteration: int,
    result: Any,
    evaluation_seconds: float,
    wall_clock_seconds: float | None,
) -> bool:
    """Append one independent convergence result, deduplicated by threshold."""
    rows = _read_rows(path)
    key = (run_id, int(target_transition_threshold))
    if any(
        (row.get("run_id"), int(row["target_transition_threshold"])) == key
        for row in rows
        if row.get("target_transition_threshold", "").isdigit()
    ):
        return False

    row = {
        "seed": int(seed),
        "run_id": run_id,
        "artifact_sha256": artifact_sha256,
        "kind": "convergence",
        "preference_count": len(result.preferences),
        "repeats": int(result.returns_per_repeat.shape[0]),
        "target_transition_threshold": int(target_transition_threshold),
        "actual_env_transitions": int(actual_env_transitions),
        "iteration": int(iteration),
        "source_hv": float(getattr(result, "source_hv", result.mean_hv)),
        "mean_repeat_hv": float(getattr(result, "mean_repeat_hv", result.mean_hv)),
        "source_sparsity": float(
            getattr(result, "source_sparsity", result.mean_sparsity)
        ),
        "source_pareto_point_count": int(
            getattr(result, "source_pareto_point_count", len(result.pareto_returns))
        ),
        "evaluation_seconds": float(evaluation_seconds),
        "wall_clock_seconds": wall_clock_seconds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONVERGENCE_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
    return True


def load_convergence_rows(path: Path) -> list[dict[str, str]]:
    rows = [
        row
        for row in _read_rows(path)
        if row.get("kind") == "convergence"
        and row.get("preference_count") == "51"
        and row.get("repeats") == "3"
    ]
    rows.sort(key=lambda row: int(row["actual_env_transitions"]))
    return rows


def plot_convergence_csv(
    input_path: Path, output_path: Path, *, moving_average_window: int = 5
) -> dict[str, Any]:
    """Plot independent HV observations and write an objective tail summary."""
    if moving_average_window < 1:
        raise ValueError("moving_average_window must be positive")
    rows = load_convergence_rows(input_path)
    if not rows:
        summary = {"status": "insufficient_data", "points": 0}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.with_name("hv_convergence_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    transitions = np.asarray(
        [int(row["actual_env_transitions"]) for row in rows], dtype=np.int64
    )
    source_hv = np.asarray([float(row["source_hv"]) for row in rows])
    if not np.isfinite(source_hv).all():
        raise ValueError("convergence history contains non-finite source_hv")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(transitions, source_hv, marker="o", linewidth=1.5, label="source_hv")
    if len(rows) >= moving_average_window:
        kernel = np.ones(moving_average_window) / moving_average_window
        smooth = np.convolve(source_hv, kernel, mode="valid")
        axis.plot(
            transitions[moving_average_window - 1 :],
            smooth,
            linewidth=1.25,
            label=f"{moving_average_window}-point moving average (visual only)",
        )
    axis.set_xlabel("Actual Training Environment Transitions")
    axis.set_ylabel("Hypervolume")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    tail_start = transitions[0] + 0.8 * (transitions[-1] - transitions[0])
    tail_mask = transitions >= tail_start
    tail_x = transitions[tail_mask]
    tail_hv = source_hv[tail_mask]
    slope = (
        float(np.polyfit(tail_x.astype(float), tail_hv, 1)[0])
        if len(tail_hv) >= 2
        else None
    )
    relative_gain = (
        float((tail_hv[-1] - tail_hv[0]) / abs(tail_hv[0]))
        if len(tail_hv) >= 2 and tail_hv[0] != 0
        else None
    )
    volatility = (
        float(np.std(tail_hv) / abs(np.mean(tail_hv)))
        if len(tail_hv) >= 2 and np.mean(tail_hv) != 0
        else None
    )
    if len(tail_hv) < 3 or None in (slope, relative_gain, volatility):
        status = "insufficient_data"
    elif volatility > 0.05:
        status = "volatile"
    elif relative_gain > 0.01 and slope > 0:
        status = "still_rising"
    else:
        status = "approaching_plateau"
    summary = {
        "status": status,
        "kind": "convergence",
        "preference_count": 51,
        "repeats": 3,
        "points": len(rows),
        "tail_points": len(tail_hv),
        "tail_hv_slope_per_transition": slope,
        "tail_relative_gain": relative_gain,
        "tail_relative_volatility": volatility,
        "total_evaluation_seconds": sum(
            float(row["evaluation_seconds"]) for row in rows
        ),
        "first_actual_env_transitions": int(transitions[0]),
        "last_actual_env_transitions": int(transitions[-1]),
    }
    output_path.with_name("hv_convergence_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
