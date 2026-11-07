"""Pass/fail check for pilot runs (Catch or the short Pong run).

    python scripts/check_pilot.py runs/pilot_catch_seed1 runs/pilot_catch_seed2 ...
    python scripts/check_pilot.py --glob "runs/pilot_catch*"

For each run it reads eval.csv and checks:
  1. LEARNS     final eval return (mean of last 3 epochs) clearly beats a random policy
  2. NEAR OPT   (Catch only) final return reaches >= 80% of the way from random to optimal
  3. STABLE Q   held-out average max Q is finite and below the bound 1/(1-gamma)
                (rewards are clipped to [-1, 1], so |Q| can never legitimately exceed it)
  4. Q RISES    held-out Q at the end is higher than after the first epoch (paper Fig. 2)
Exit code 0 if every run passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Random-policy returns. Catch is computed from the env; Pong is paper Table 1.
RANDOM = {"Pong": -20.4}
OPTIMAL = {"Pong": 21.0}


def baselines(cfg: dict) -> tuple[float, float | None]:
    if cfg.get("env") == "catch":
        from dqn.catch import random_policy_return

        return random_policy_return(cfg["catch_balls"], cfg["catch_drift"]), float(cfg["catch_balls"])
    return RANDOM.get(cfg["game"], float("nan")), OPTIMAL.get(cfg["game"])


def check_run(run_dir: Path) -> bool:
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    df = pd.read_csv(run_dir / "eval.csv")
    rand, opt = baselines(cfg)
    final = df["eval_mean_return"].tail(3).mean()
    q = df["avg_max_q_holdout"].to_numpy()
    q_bound = 1.0 / (1.0 - cfg["gamma"])

    checks = []
    # 1. clearly better than random: at least 25% of the random-to-optimal gap (or +3 if no optimum)
    gap = (opt - rand) if opt is not None else 12.0
    checks.append(("LEARNS", final >= rand + 0.25 * gap, f"final {final:.2f} vs random {rand:.2f}"))
    if cfg.get("env") == "catch":
        checks.append(("NEAR OPT", final >= rand + 0.8 * gap, f"final {final:.2f}, optimum {opt:.0f}"))
    checks.append(
        ("STABLE Q", bool(np.all(np.isfinite(q)) and np.all(np.abs(q) < q_bound)),
         f"max |Q| {np.nanmax(np.abs(q)):.3f}, bound {q_bound:.0f}")
    )
    checks.append(("Q RISES", len(q) > 1 and q[-1] > q[0], f"Q {q[0]:.3f} -> {q[-1]:.3f}"))

    ok = all(c[1] for c in checks)
    print(f"\n{run_dir.name}  ({len(df)} epochs, {df['agent_step'].iloc[-1]} steps, "
          f"{df['wall_clock_s'].iloc[-1] / 60:.1f} min)  ->  {'PASS' if ok else 'FAIL'}")
    for name, passed, detail in checks:
        print(f"  [{'ok' if passed else 'XX'}] {name:<9} {detail}")
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="*")
    p.add_argument("--glob", default=None)
    args = p.parse_args()
    dirs = [Path(r) for r in args.runs] + [Path(r) for r in sorted(glob.glob(args.glob or ""))]
    dirs = [d for d in dirs if (d / "eval.csv").exists()]
    if not dirs:
        sys.exit("No runs with eval.csv found.")
    results = [check_run(d) for d in dirs]
    print(f"\n{sum(results)}/{len(results)} runs passed.")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
