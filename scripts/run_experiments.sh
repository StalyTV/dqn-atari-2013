#!/usr/bin/env bash
# Run a block of experiments one after another, each logging to logs/<name>.log.
#
#   bash scripts/run_experiments.sh repro       # 2013 agent, Breakout + Space Invaders, seeds 1-3
#   SEEDS="4 5" bash scripts/run_experiments.sh repro
#
# Runs are long (hours each on a GPU). Start them inside tmux or with nohup so they
# survive a closed browser tab:
#   nohup bash scripts/run_experiments.sh repro > logs/repro.out 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

BLOCK="${1:?usage: run_experiments.sh repro}"
SEEDS="${SEEDS:-1 2 3}"

case "$BLOCK" in
  repro)     CONFIGS="configs/breakout.yaml configs/space_invaders.yaml" ;;
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
