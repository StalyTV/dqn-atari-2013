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
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

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
  done
done

case "$BLOCK" in
  pilot_*) python scripts/check_pilot.py --glob "runs/${BLOCK}_seed*" ;;
esac
