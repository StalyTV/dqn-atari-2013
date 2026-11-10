# Pilot and tests: what runs, what it checks, how to read the result

Following the TA's advice, we validate the implementation on cheap setups before
spending compute on the full Atari runs. There are three levels, each cheaper
than the next and each catching a different kind of bug.

| Level | Command | Where | Time | Checks |
|---|---|---|---|---|
| 0. Unit tests | `python -m pytest -q` | anywhere | ~5 s | individual pieces (config, network, preprocessing, replay buffer, Catch) |
| 1. Catch pilot | `bash scripts/run_experiments.sh pilot_catch` | CPU is enough | ~15 min per seed | the whole learning pipeline, against a known optimum |
| 2. Pong pilot | `bash scripts/run_experiments.sh pilot_pong` | GPU | not measured yet (see below) | the Atari side (ALE, preprocessing, frame skip) on the paper's easiest game |

Do them in order. If one fails, do not move on to the next: fix it first.

Neither pilot is a result for the report's claims. They only tell us whether the
code is trustworthy enough that a later Atari result means something.

---

## Level 0: unit tests (`tests/test_basics.py`)

| Test | What it checks | Bug it would catch |
|---|---|---|
| `test_config_inheritance_and_overrides` | game configs inherit the base, `--set` overrides work | wrong hyperparameters silently used |
| `test_network_shapes` | 2013 network output shape and exact parameter count (16x8x8/4, 32x4x4/2, fc 256) | wrong architecture (e.g. the 2015 Nature net) |
| `test_env_preprocessing` | Breakout and Space Invaders give 84x84 uint8 frames | broken resize or crop |
| `test_replay_stacks_respect_episode_boundaries` | 4-frame stacks are zero-padded across episode ends and at the buffer start; `done` is stored | the agent "sees" frames from the previous episode |
| `test_replay_wraparound_never_mixes_old_and_new` | once the buffer wraps, no stack contains an overwritten frame | corrupted states after 1M frames, which would only show up late in a long run |
| `test_catch_env_interface_and_scores` | Catch has the Atari wrapper's interface; a scripted perfect player scores the maximum; a paddle parked in a corner scores below zero | a broken pilot game (the pilot would then prove nothing) |

---

## Level 1: Catch pilot (`configs/pilot_catch.yaml`, `dqn/catch.py`)

### The game

A 21x21 grid drawn at 4x4 pixels per cell, giving **84x84 grayscale frames, the
same format as the Atari wrapper**. A ball falls from the top, one row per step,
and drifts sideways (-1, 0 or +1 columns per step, bouncing off the walls). The
agent moves a 3-cell paddle on the bottom row (left / stay / right). Catch = +1,
miss = -1. One episode = 5 balls.

- best possible return: **+5**
- uniform random policy: **about -3.6**

Because the ball drifts, a single frame does not show where it is going. The agent
must use the frame history, so the 4-frame stacking is actually exercised.

### What code runs

Everything except the Atari emulator is the real code used for the experiments:

| Component | Same as the Atari runs? |
|---|---|
| `dqn/train.py` training loop, epsilon-greedy, evaluation, held-out Q tracking, logging | yes |
| `dqn/replay.py` frame-based replay buffer and 4-frame stacking | yes |
| `dqn/model.py` 2013 network (`nips2013`), uint8 -> [0, 1] scaling | yes |
| Q-learning update **without a target network**, 0.5 x squared TD error | yes |
| RMSProp lr 0.0002, gamma 0.95, batch 32, one update per step, reward clipping | yes |
| epsilon 1.0 -> 0.1, then fixed; eval epsilon 0.05 | yes (decay shortened to 20k steps) |
| environment | **no**: `CatchEnv` instead of `AtariEnv` |
| scale | **no**: 60k steps, 50k replay, eval every 5k steps over 2k steps, 500 held-out states |

Selecting the game is a single config key: `env: catch` (default `env: atari`).

### Pass criteria (`scripts/check_pilot.py`)

Applied to each run's `eval.csv`, using the mean of the last 3 evaluation epochs:

| Check | Condition | Meaning if it fails |
|---|---|---|
| LEARNS | final return >= random + 25% of the gap to optimal (about -1.4) | nothing is learned: broken update, replay, or action selection |
| NEAR OPT | final return >= random + 80% of the gap (about +3.3) | learns something but plateaus, e.g. frame stacking broken, or instability |
| STABLE Q | held-out avg max Q finite and below 1/(1-gamma) = 20 | Q-values diverging, which is the main risk without a target network |
| Q RISES | held-out Q at the last epoch > after the first | no value is being propagated (paper Fig. 2 shows Q rising) |

The script prints PASS/FAIL per run and exits with code 1 if any run fails.

### Results so far (Claude, tested outside Renku, 2-core CPU, 2026-11-07)

Seed 1, unchanged config:

| epoch (5k steps) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eval return | -2.9 | -1.0 | 1.9 | 2.7 | 4.1 | 4.2 | 4.8 | 4.7 | ... | 4.7 | 4.9 | 4.8 |

Result: **PASS**: final 4.80 (optimum 5, random -3.57), max |Q| 0.87, 14 min at ~65-85 steps/s.

Negative control, to show the checker can fail: same config with
`--set learning_rate=0.0 total_frames=15000` -> **FAIL** (final -3.90, Q flat at 0.075).

Still to do: run seeds 1-3 on Renku and record the result here.

### Optional diagnostics, if Catch fails or is unstable

Each one changes a single thing (use `--run-name` so the runs don't collide):

```bash
# Nature-style stabilisers. If this passes and the 2013 setup doesn't,
# the problem is instability, not a bug.
python -m dqn.train --config configs/pilot_catch.yaml --seed 1 --run-name catch_target \
    --set target_update_interval=1000 loss=huber
# no sideways drift. If this passes and the default doesn't, suspect the frame stacking.
python -m dqn.train --config configs/pilot_catch.yaml --seed 1 --run-name catch_nodrift --set catch_drift=false
# deep_q_rl initialisation instead of PyTorch's default
python -m dqn.train --config configs/pilot_catch.yaml --seed 1 --run-name catch_init --set init=normal_0.01
python scripts/check_pilot.py runs/catch_target runs/catch_nodrift runs/catch_init
```

---

## Level 2: Pong pilot (`configs/pilot_pong.yaml`)

Pong is one of the paper's seven games (Table 1: random -20.4, human -3, DQN 20)
and the easiest of them. It is the cheapest check of what Catch cannot test: the
ALE wrapper, the 210x160 -> 110x84 -> 84x84 preprocessing and crop, the frame skip,
and the minimal action set.

- Agent: the unchanged 2013 setup.
- Shortened: **4M frames (1M agent steps, 20 epochs)** instead of 10M, and a replay
  buffer of **500k frames** (about 3.5 GB RAM) instead of 1M.
- Pass: LEARNS = final return >= about -10 (25% of the way from random to 21),
  plus STABLE Q and Q RISES. NEAR OPT is not required at this length.
- **Runtime not measured yet.** Run the smoke test on the Renku GPU first
  (`python -m dqn.train --config configs/smoke_test.yaml --seed 1`), read the
  steps/s, and estimate: 1M steps / (steps/s).
- The -10 threshold is our guess for a shortened run, not a number from a paper.
  A run that is still near -20 after 1M steps is a warning sign, not proof of a bug:
  then try the target-network diagnostic above on Pong.

---

## How to run on Renku

```bash
git pull                              # make sure the session has the latest code
python -m pytest -q
export RESULTS_DIR=/home/renku/work/synchronizing/dqn-atari-2013   # persistent copy, see README
mkdir -p logs
nohup bash scripts/run_experiments.sh pilot_catch > logs/pilot_catch.out 2>&1 &
tail -f logs/pilot_catch.out          # ends with the PASS/FAIL summary
# later, on a GPU session:
nohup bash scripts/run_experiments.sh pilot_pong > logs/pilot_pong.out 2>&1 &
```

Re-check any time: `python scripts/check_pilot.py --glob "runs/pilot_catch_seed*"`.
The PASS/FAIL summary is also saved in `$RESULTS_DIR/summaries/`.

Note: with `nohup ... &`, `RESULTS_DIR` must be exported in the same terminal *before*
starting. A new terminal does not inherit it.
Plots: `python scripts/plot_results.py --runs runs --out figures` (also works for the pilots).

## What a PASS does and does not tell us

It does tell us that the pipeline (replay, stacking, network, the update without a
target network, evaluation, logging) learns a pixel task close to optimally, and
that the Q-values stay bounded.

It does not tell us that the Atari hyperparameters are right: gamma, learning rate,
what counts as a "frame", and init are all still open questions (see
`DEVIATIONS.md`). Catch is far easier than Breakout, so passing it says nothing
about whether we will reach the paper's scores.
