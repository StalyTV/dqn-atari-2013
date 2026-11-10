#!/usr/bin/env bash
# Run a block of experiments one after another, each logging to logs/<name>.log.
#
#   bash scripts/run_experiments.sh pilot_catch # pilot on Catch, seeds 1-3, CPU, ~10-15 min per run
#   bash scripts/run_experiments.sh pilot_pong  # pilot on Pong (shortened), seeds 1-3, GPU, ~1-2 h per run
#   bash scripts/run_experiments.sh repro       # 2013 agent, Breakout + Space Invaders, seeds 1-3
#   SEEDS="4 5" bash scripts/run_experiments.sh repro
#
# Runs are long (hours each on a GPU). Start them inside tmux or with nohup so they
# survive a closed browser tab:
#   nohup bash scripts/run_experiments.sh repro > logs/repro.out 2>&1 &
#
# Persistent copy: set RESULTS_DIR (e.g. a SWITCHdrive data connector on Renku) and
#   runs/<name>/   are mirrored to $RESULTS_DIR/runs/<name>/ after every eval epoch (by train.py)
#   logs/<name>.log is copied to $RESULTS_DIR/logs/ after each run
#   the pilot PASS/FAIL summary goes to $RESULTS_DIR/summaries/
#   export RESULTS_DIR=/home/renku/work/<mount>/dqn-atari-2013
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
if [ -n "${RESULTS_DIR:-}" ]; then
  if [ ! -d "$(dirname "$RESULTS_DIR")" ]; then
    echo "RESULTS_DIR: $(dirname "$RESULTS_DIR") does not exist. Is the drive mounted?"; exit 1
  fi
  mkdir -p "$RESULTS_DIR/runs" "$RESULTS_DIR/logs" "$RESULTS_DIR/summaries"
  echo "Results are copied to $RESULTS_DIR"
else
  echo "RESULTS_DIR not set: results stay only in runs/ and logs/ of this machine"
fi
save_log() { [ -n "${RESULTS_DIR:-}" ] && cp "logs/$1.log" "$RESULTS_DIR/logs/" || true; }

BLOCK="${1:?usage: run_experiments.sh pilot_catch|pilot_pong|repro}"
SEEDS="${SEEDS:-1 2 3}"

case "$BLOCK" in
  pilot_catch) CONFIGS="configs/pilot_catch.yaml" ;;
  pilot_pong)  CONFIGS="configs/pilot_pong.yaml" ;;
  repro)       CONFIGS="configs/breakout.yaml configs/space_invaders.yaml" ;;
  *) echo "unknown block $BLOCK"; exit 1 ;;
esac

for cfg in $CONFIGS; do
  for seed in $SEEDS; do
    name="$(basename "$cfg" .yaml)_seed${seed}"
    if [ -d "runs/$name" ]; then
      echo "skip $name (runs/$name exists)"; continue
    fi
    echo "=== $name  $(date) ==="
    python -m dqn.train --config "$cfg" --seed "$seed" --run-name "$name" 2>&1 | tee "logs/$name.log"
    save_log "$name"
  done
done

case "$BLOCK" in
  pilot_*)
    summary="logs/${BLOCK}_summary_$(date +%Y%m%d-%H%M%S).txt"
    python scripts/check_pilot.py --glob "runs/${BLOCK}_seed*" | tee "$summary" || true
    [ -n "${RESULTS_DIR:-}" ] && cp "$summary" "$RESULTS_DIR/summaries/" || true
    ;;
esac
