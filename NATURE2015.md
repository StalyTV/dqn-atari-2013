# 2015 (Nature) DQN and the ablation path back to 2013

Branch `2015-nature`. Idea (from our TA): implement the full 2015 agent of Mnih et al.,
*Human-level control through deep reinforcement learning* (Nature 518, 2015), check that it
learns Space Invaders, then switch its components back to the 2013 setup one at a time to
find which change(s) make the difference.

Every 2015 component is a separate config key. The 2013 configs keep their old values for
all of them (checked: 2013 runs give bit-identical `eval.csv` and `train_episodes.csv` to the
code before this branch), so any ablation is a plain `--set key=value`.

## Sources

- **Paper**: Extended Data Table 1 (hyperparameters) and Methods.
- **DQN 3.0**, DeepMind's own code for the paper (github.com/google-deepmind/dqn, archived;
  Lua/Torch, academic-use licence): `run_gpu` (parameters), `NeuralQLearner.lua` (update rule,
  RMSProp, target network, epsilon), `convnet_atari3.lua` (network), and `alewrap`
  `GameEnvironment.lua` (life loss, random starts). Used where the paper is not specific.
- **deep_q_rl** `run_nature.py` (independent reimplementation) as a cross-check.

## The components: 2015 value vs 2013 value

| Component | Config key | 2015 (`base_nature2015.yaml`) | 2013 (`base_nips2013.yaml`) | Source of 2015 value |
|---|---|---|---|---|
| Network | `network` | `nature2015`: conv 32x8x8/4, 64x4x4/2, 64x3x3/1, fc 512 | `nips2013`: 16x8x8/4, 32x4x4/2, fc 256 | paper, dqn3 |
| Frame preprocessing | `resize` | `scale` (whole frame to 84x84) | `crop` | dqn3 (`Scale(84,84)`) |
| Frame max over 2 frames | `max_pool_last_two` | true | false | paper, dqn3 |
| Action repeat | `frame_skip` | 4 for all games | 4 (3 for Space Invaders) | paper, dqn3 |
| Random starts | `noop_max` | 30 NOOP frames (train and eval) | 0 | paper, dqn3 |
| Life loss | `life_loss_terminal` | terminal in training, game continues | no | paper Methods, dqn3 |
| Discount | `gamma` | 0.99 | 0.95 | paper |
| Learning rate | `learning_rate` | 0.00025 | 0.0002 | paper |
| Optimizer | `optimizer` | `rmsprop_deepmind` (centered, see `dqn/optim.py`) | `rmsprop_torch` | paper, dqn3 |
| RMSProp decay / eps | `rmsprop_alpha`, `rmsprop_eps` | 0.95 / 0.01 (inside sqrt) | 0.99 / 1e-6 | paper, dqn3 |
| Error clipping | `loss` | `huber` (TD error clipped to [-1, 1]) | `mse` | paper, dqn3 |
| Minibatch gradient | `loss_reduction` | `sum` | `mean` | dqn3 |
| Update frequency | `update_frequency` | every 4 agent steps | every step | paper |
| Target network | `target_update_interval` | every 10,000 agent steps | none (0) | dqn3 (see below) |
| Replay start | `learning_starts` | 50,000 steps | 100 | paper |
| Epsilon schedule | `epsilon_decay_frames`, `epsilon_decay_after_learning_starts` | 1 -> 0.1 over 1M steps, starting after the 50k random steps | 1 -> 0.1 over 1M frames (or steps), from step 0 | paper, dqn3 |
| Training length | `total_frames` | 10M agent steps (paper: 50M) | 10M frames / 10M steps | ours, see below |

## Where we deviate from the 2015 paper, and why

- **Target network interval.** The paper (ED Table 1) says every 10,000 *parameter updates*.
  DQN 3.0 copies when `numSteps % target_q == 1`, i.e. every 10,000 *agent steps* = 2,500
  updates (deep_q_rl counts updates). We follow DQN 3.0, the code behind the published
  results. Ablation: `--set target_update_interval=40000` gives the paper's reading.
- **Training length.** The paper trains 50M agent steps (200M emulator frames); that is far
  beyond the Renku job deadline. We use 10M agent steps, the length of the paper's own
  ablation table (ED Table 3, "10 million frames", which the 2015 paper counts in agent steps).
  Compare with ED Table 3 (Space Invaders: 1088.9 with replay and target network, 826.3 with
  replay but no target network), not with the 50M-step results of ED Table 2.
- **Evaluation protocol.** We keep the 2013 protocol (epsilon 0.05, 10,000 steps after every
  50,000 steps) so curves are comparable with our 2013 runs. DQN 3.0 evaluates 125,000 steps
  every 250,000 steps, and ED Table 3 reports the highest evaluation score over training.
- **Grayscale.** DQN 3.0 converts RGB to the luminance (Y) channel; we use ALE's grayscale
  (luminance) output. Same idea, possibly slightly different values.
- **Initialisation.** PyTorch default (uniform, about +-1/sqrt(fan_in)); Torch7's default,
  used by DQN 3.0, is similar.
- **RMSProp numerics.** `g2 - g^2` is clamped at 0 before adding eps (guards against tiny
  negative values from floating point). DQN 3.0 does not clamp.

## Ablation path for Space Invaders

Step 0: run the full 2015 agent.

```bash
python -m dqn.train --config configs/space_invaders_nature2015.yaml --seed 1 --run-name si_nature2015_seed1
```

Before planning the sweep, measure the GPU speed with a short run (the logged steps/s); a
10M-step run must fit under the job deadline (about 13 h, so at least about 215 steps/s).
If it does not fit, lower `total_frames` for all ablations alike.

**If it learns** (clearly above "always fire", about 285): switch back one component per run,
each with its own run name. Leave-one-out (each run = full 2015 minus ONE component) shows
which components are necessary; it is easier to interpret than a cumulative path. Suggested
order (most likely culprits for the 2013 failure first):

| Run name | `--set` | Turns off |
|---|---|---|
| `si_n15_no_target` | `target_update_interval=0` | target network |
| `si_n15_no_clip` | `loss=mse` | error clipping |
| `si_n15_torch_rmsprop` | `optimizer=rmsprop_torch rmsprop_alpha=0.99 rmsprop_eps=1e-6 loss_reduction=mean` | DeepMind RMSProp (and sum) |
| `si_n15_update1` | `update_frequency=1` | update every 4 steps |
| `si_n15_no_lifeloss` | `life_loss_terminal=false` | life loss as terminal |
| `si_n15_small_net` | `network=nips2013` | 2015 network |
| `si_n15_gamma095` | `gamma=0.95` | gamma 0.99 |
| `si_n15_preproc2013` | `resize=crop crop_offset=4 max_pool_last_two=false noop_max=0 frame_skip=3 total_frames=30000000 epsilon_decay_frames=3000000` | 2015 preprocessing / random starts |

A cumulative path (switching more and more back until it stops learning) is the alternative
the TA suggested; it ends at our 2013 configuration. Pick one and write it into
`DEVIATIONS.md` before running.

**If the full 2015 agent does not learn either**, the problem is likely in our environment
(emulator version) or shared code rather than in the 2013 settings - also a clear result.

## Results

| Run | Last 10 epochs | Best epoch | Notes |
|---|---|---|---|
| (fill in) | | | |
