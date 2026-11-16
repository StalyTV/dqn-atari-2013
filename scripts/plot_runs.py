"""Detailed figure for one or more individual runs (e.g. the seeds of one game).

    python scripts/plot_runs.py runs/breakout_seed1
    python scripts/plot_runs.py --glob "runs/breakout_seed*" --out figures/breakout_runs.png
    python scripts/plot_runs.py --glob "$RESULTS_DIR/runs/breakout_seed*"

Unlike plot_results.py (mean +- std over seeds, paper Fig. 2 style), this shows every
run separately, to judge how a run is going and how much seeds differ:

  (a) evaluation return per epoch, with paper Table 1 lines (DQN, human, random)
  (b) training episode returns, moving average over 50 episodes
  (c) average max Q on the held-out states (paper Fig. 2, right)
  (d) training loss per epoch (log scale)

The grey band marks the exploration phase (epsilon still decaying). All panels share
the x axis: emulator frames, in millions. A short summary table is printed too.
Works with a run folder that is still being written (partial eval.csv is fine).
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_results import PAPER  # noqa: E402  Table 1 of the 2013 paper

# Categorical series colours in fixed order (seed 1, 2, 3, ...), never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#2b2a28"  # primary text
INK_2 = "#52514e"  # secondary text, reference lines
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SMOOTH = 50  # episodes in the moving average of panel (b)


def load_run(d: Path) -> dict | None:
    if not (d / "eval.csv").exists():
        return None
    ev = pd.read_csv(d / "eval.csv")
    tr = pd.read_csv(d / "train_episodes.csv") if (d / "train_episodes.csv").exists() else None
    cfg = yaml.safe_load(open(d / "config.yaml")) if (d / "config.yaml").exists() else {}
    return {"dir": d, "eval": ev, "train": tr, "cfg": cfg, "label": d.name}


def exploration_end_mframes(runs: list[dict]) -> float | None:
    """Frames (millions) at which epsilon reaches its final value."""
    for r in runs:
        if "epsilon_decay_frames" in r["cfg"]:
            return r["cfg"]["epsilon_decay_frames"] / 1e6
        tr = r["train"]
        if tr is not None and len(tr):
            done = tr[tr["epsilon"] <= tr["epsilon"].min() + 1e-9]
            if len(done):
                return done["frames"].iloc[0] / 1e6
    return None


def style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="run folders")
    p.add_argument("--glob", default=None, help='e.g. "runs/breakout_seed*"')
    p.add_argument("--game", default=None, help="game for the paper lines, if config.yaml is missing")
    p.add_argument("--out", default=None, help="output PNG (default: figures/<first run>.png)")
    args = p.parse_args()

    dirs = [Path(r) for r in args.runs] + [Path(r) for r in sorted(glob.glob(args.glob or ""))]
    runs = [r for r in (load_run(d) for d in dirs) if r is not None]
    if not runs:
        sys.exit("No run folders with an eval.csv found.")
    game = args.game or runs[0]["cfg"].get("game")
    out = Path(args.out) if args.out else Path("figures") / f"{runs[0]['label']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, facecolor=SURFACE)
    (a, b), (c, d) = axes
    style(a, "(a) Evaluation return per epoch (ε = 0.05)", "average return per episode")
    style(b, f"(b) Training episodes, moving average over {SMOOTH}", "return per episode")
    style(c, "(c) Held-out average max Q", "average max Q")
    style(d, "(d) Training loss per epoch", "mean loss (log)")
    d.set_yscale("log")

    explore = exploration_end_mframes(runs)
    if explore:
        for ax in axes.flat:
            ax.axvspan(0, explore, color="#efeeea", zorder=0, linewidth=0)


    for i, r in enumerate(runs):
        col = SERIES[i % len(SERIES)] if i < len(SERIES) else None
        if col is None:
            sys.exit("More than 6 runs: plot them in groups (colours are never cycled).")
        ev, tr = r["eval"], r["train"]
        x = ev["frames"] / 1e6
        a.plot(x, ev["eval_mean_return"], color=col, lw=2, marker="o", ms=3.5, label=r["label"])
        c.plot(x, ev["avg_max_q_holdout"], color=col, lw=2, label=r["label"])
        d.plot(x, ev["loss_mean"], color=col, lw=2, label=r["label"])
        if tr is not None and len(tr):
            if len(runs) == 1:  # raw episodes only for a single run, else too busy
                b.scatter(tr["frames"] / 1e6, tr["return"], s=3, color=col, alpha=0.15, linewidths=0)
            sm = tr["return"].rolling(SMOOTH, min_periods=max(5, SMOOTH // 5)).mean()
            b.plot(tr["frames"] / 1e6, sm, color=col, lw=2, label=r["label"])

    # paper reference lines in panel (a): labelled directly, in text ink
    ref = PAPER.get(game, {})
    for key, ls in (("DQN", "--"), ("Human", ":"), ("Random", "-.")):
        if key in ref:
            a.axhline(ref[key], color=INK_2, lw=1, ls=ls, zorder=1)
            a.text(1.0, ref[key], f" paper {key} {ref[key]:g}", transform=a.get_yaxis_transform(),
                   va="center", ha="left", fontsize=8, color=INK_2, clip_on=False)

    for ax in (c, d):
        ax.set_xlabel("emulator frames (millions)", color=INK_2, fontsize=9)
    if len(runs) > 1:
        a.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle(f"{game or 'DQN'}: {', '.join(r['label'] for r in runs)}", x=0.01, y=0.99, ha="left",
                 va="top", fontsize=13, color=INK)
    if explore:
        fig.text(0.01, 0.935, f"Grey band: exploration phase (ε decaying until {explore:g}M frames).",
                 ha="left", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0, 0.94, 0.92))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # summary table
    print(f"{'run':<28}{'epochs':>7}{'last 5':>9}{'best avg':>10}{'at epoch':>10}{'best game':>11}{'hours':>7}")
    for r in runs:
        ev = r["eval"]
        best = ev["eval_mean_return"].idxmax()
        print(f"{r['label']:<28}{len(ev):>7}{ev['eval_mean_return'].tail(5).mean():>9.1f}"
              f"{ev['eval_mean_return'].max():>10.1f}{int(ev['epoch'][best]):>10}"
              f"{ev['eval_max_return'].max():>11.0f}{ev['wall_clock_s'].iloc[-1] / 3600:>7.2f}")
    if ref:
        print("paper Table 1: " + ", ".join(f"{k} {v}" for k, v in ref.items()))
    print(f"\nFigure written to {out}")


if __name__ == "__main__":
    main()