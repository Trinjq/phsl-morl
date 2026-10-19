"""Plot independent 51x3 PD-MORL convergence diagnostics without evaluating."""

import argparse
import json
from pathlib import Path

from evorl.utils.pd_morl_convergence import plot_convergence_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--moving-average-window", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            plot_convergence_csv(
                args.input,
                args.output,
                moving_average_window=args.moving_average_window,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
