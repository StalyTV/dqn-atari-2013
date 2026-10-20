# Reproducing "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)

UZH Reinforcement Learning, Fall 2026, reproducibility project P01.
Paper: https://arxiv.org/abs/1312.5602

A PyTorch reimplementation of the **2013 NIPS workshop DQN**, not the 2015 Nature
DQN that most repositories implement. The difference matters: the 2013 agent has
**no target network**, a smaller network, and squared-error loss.

## Claims we test

1. **Scores.** The same architecture and hyperparameters learn Breakout and Space
   Invaders from pixels, beating the 2013 baselines; on Breakout DQN exceeds the
   human expert (paper Table 1: DQN 168 vs human 31; Space Invaders DQN 581 vs human 1652).
2. **Stability.** The average max Q on a fixed set of held-out states rises smoothly
   during training while episode reward is noisy, and training did not diverge (paper Fig. 2, Sec. 5.1).

## Repository layout

```
configs/
  base_nips2013.yaml        all hyperparameters, each tagged [paper] / [deep_q_rl] / [ours]
  breakout.yaml             game configs (inherit from the base)
  space_invaders.yaml       frame_skip 3, as in the paper
  smoke_test.yaml           few-minute end-to-end check
dqn/
  env.py      ALE wrapper: frame skip, gray-scale, 110x84 resize, 84x84 crop
  model.py    2013 network
  replay.py   memory-efficient replay buffer (frames stored once, stacks built on sampling)
  train.py    training loop, evaluation, held-out Q tracking, logging
  config.py   YAML inheritance and --set overrides
scripts/
  run_experiments.sh   runs the reproduction block (both games, several seeds)
  plot_results.py      paper-style figures + results table vs Table 1
tests/test_basics.py   checks for config, network, preprocessing, replay boundaries
DEVIATIONS.md          every difference from the paper. Keep it up to date.
```

## Setup

### On RenkuLab

One-time setup (done once for the group):

1. In the RenkuLab project, add this GitHub repository as a **code repository**.
2. Add a **session launcher**, choose **Create from code**, select this repository,
   environment type **Python**, user interface **VSCodium** or **JupyterLab**.
   RenkuLab builds the image from `environment.yml`. Code-based environments
   require a **public** repository; a private one needs RenkuLab's GitHub Actions route.
3. Pick a resource class with a GPU and at least 10 GB RAM once the course staff has
   assigned project compute.
4. After changing `environment.yml`, open the launcher and press **Rebuild**.

In a session terminal (the repo is cloned into the session):

```bash
python -m pytest -q                                              # ~5 s
python -m dqn.train --config configs/smoke_test.yaml --seed 1    # a few minutes
```

### Locally

```bash
conda env create -f environment.yml && conda activate dqn2013
# or, with plain pip (Python 3.10+):
pip install torch "gymnasium[atari]==1.4.0" ale-py==0.12.1 opencv-python-headless numpy pandas matplotlib pyyaml pytest
```

The Atari ROMs ship with `ale-py`, nothing else to download.

If the GPU driver is older than the CUDA version of the default PyTorch wheel,
install PyTorch from https://pytorch.org with the matching CUDA version.

## Running experiments

One run:

```bash
python -m dqn.train --config configs/breakout.yaml --seed 1
python -m dqn.train --config configs/breakout.yaml --seed 1 --set gamma=0.99   # any key can be overridden
```

A block of runs (seeds 1-3, both games), surviving a closed browser tab:

```bash
mkdir -p logs
nohup bash scripts/run_experiments.sh repro > logs/repro.out 2>&1 &
tail -f logs/repro.out
```

Figures and the results table:

```bash
python scripts/plot_results.py --runs runs --out figures
```

## What each run writes (`runs/<name>/`)

| File | Content |
|---|---|
| `config.yaml` | the fully resolved config, enough to rerun |
| `run_info.json` | command, git commit, library versions, device, GPU |
| `train_episodes.csv` | every training episode: raw score, length, epsilon |
| `eval.csv` | every epoch (50k updates): eval mean/max return (epsilon 0.05), held-out avg max Q, loss |
| `model_latest.pt` | weights after the latest evaluation |

`runs/` is git-ignored. Share results through the Renku project storage or a
shared drive, and commit only the final figures and tables.

## Compute

Measured on a 2-core CPU: about 65 agent steps/s, almost all of it in the
gradient update. A full run is 10M frames, i.e. 2.5M agent steps on Breakout and
3.33M on Space Invaders (frame skip 3), so a CPU run takes 11 to 15 hours. A GPU
should be several times faster; **measure it with the smoke test before planning
the sweep**, and put the number here.

- RAM: the 1M-frame replay buffer needs about 7 GB per run, plus ~1 GB overhead.
  Request at least 10 GB per concurrent run, or lower `replay_capacity` and log it as a deviation.
- Reproduction runs: 2 games x 3 seeds = 6 runs.

## Team rules

- Change hyperparameters only through configs, never in code, so every run is reproducible from its `config.yaml`.
- Any difference from the paper goes into `DEVIATIONS.md` in the same commit.
- Commit before launching a long run: `run_info.json` records the commit hash and marks uncommitted changes as `-dirty`.
