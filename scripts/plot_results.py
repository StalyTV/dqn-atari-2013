"""Aggregate runs into paper-style figures and a results table.

    python scripts/plot_results.py --runs runs --out figures

Groups runs by the `name` in their config.yaml (e.g. breakout_dqn2013,
spaceinvaders_dqn2013), plots mean +- std over seeds of
  (a) average evaluation return per epoch      (paper Fig. 2, left panels)
  (b) average max Q on the held-out states     (paper Fig. 2, right panels)
and writes results.md comparing the final numbers with paper Table 1.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Mnih et al. (2013), Table 1. Check these against the paper before reporting.
PAPER = {
    "Breakout": {"Random": 1.2, "Sarsa": 5.2, "Contingency": 6, "DQN": 168, "Human": 31, "DQN best": 225},
    "SpaceInvaders": {"Random": 148, "Sarsa": 271, "Contingency": 268, "DQN": 581, "Human": 1652, "DQN best": 1075},
}


def load_runs(runs_dir: Path):
    groups = defaultdict(list)
    for cfg_path in sorted(runs_dir.glob("*/config.yaml")):
        run_dir = cfg_path.parent
        eval_path = run_dir / "eval.csv"
        if not eval_path.exists() or eval_path.stat().st_size == 0:
            continue
        cfg = yaml.safe_load(open(cfg_path))
        df = pd.read_csv(eval_path)
        if df.empty:
            continue
        groups[(cfg["game"], cfg["name"])].append({"cfg": cfg, "df": df, "dir": run_dir.name})
    return groups


def mean_std(runs, column):
    n = min(len(r["df"]) for r in runs)  # align on the shortest run
    arr = np.stack([r["df"][column].to_numpy()[:n] for r in runs])
    epochs = runs[0]["df"]["epoch"].to_numpy()[:n]
    return epochs, arr.mean(0), arr.std(0), arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="figures")
    p.add_argument("--last-k", type=int, default=5, help="epochs averaged for the final score")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    groups = load_runs(Path(args.runs))
    if not groups:
        print("No finished evaluations found.")
        return

    by_game = defaultdict(list)
    for (game, name), runs in groups.items():
        by_game[game].append((name, runs))

    lines = ["# Results\n", f"Final score = mean eval return over the last {args.last_k} epochs, then mean +- std over seeds.\n"]
    for game, entries in by_game.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for name, runs in entries:
            for ax, col, label in [
                (axes[0], "eval_mean_return", "Average reward per episode"),
                (axes[1], "avg_max_q_holdout", "Average action value (Q)"),
            ]:
                ep, m, s, _ = mean_std(runs, col)
                ax.plot(ep, m, label=f"{name} (n={len(runs)})")
                ax.fill_between(ep, m - s, m + s, alpha=0.2)
                ax.set_xlabel("Training epochs (50k updates)")
                ax.set_ylabel(label)
        axes[0].set_title(f"{game}: evaluation return")
        axes[1].set_title(f"{game}: held-out max Q")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{game}_fig2.png", dpi=150)
        plt.close(fig)

        lines.append(f"\n## {game}\n")
        if game in PAPER:
            ref = ", ".join(f"{k} {v}" for k, v in PAPER[game].items())
            lines.append(f"Paper Table 1: {ref}\n")
        lines.append("| config | seeds | epochs | final score | best episode |")
        lines.append("|---|---|---|---|---|")
        for name, runs in entries:
            finals = [r["df"]["eval_mean_return"].tail(args.last_k).mean() for r in runs]
            best = max(r["df"]["eval_max_return"].max() for r in runs)
            n_ep = min(len(r["df"]) for r in runs)
            lines.append(
                f"| {name} | {len(runs)} | {n_ep} | {np.mean(finals):.1f} +- {np.std(finals):.1f} | {best:.0f} |"
            )

    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nFigures and results.md written to {out}/")


if __name__ == "__main__":
    main()
