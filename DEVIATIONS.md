# Implementation choices and deviations

The report must list every difference from the source paper (Mnih et al., 2013,
arXiv:1312.5602) and say how it could affect the conclusion. This file is the
running log. Add a row whenever you change something, with the date and who.

No official code exists for the 2013 paper. DeepMind's released code (DQN 3.0,
Lua/Torch) belongs to the 2015 Nature paper. Where the 2013 paper is silent we
follow Nathan Sprague's `deep_q_rl` (`run_nips.py`), an independent Theano
reimplementation with an explicit NIPS 2013 configuration
(https://github.com/spragunr/deep_q_rl).

## Taken directly from the paper

| Item | Value |
|---|---|
| Games (our subset) | Breakout, Space Invaders |
| Preprocessing | gray-scale, 110x84 downsample, 84x84 crop, last 4 frames stacked |
| Frame skip | k=4; k=3 for Space Invaders |
| Reward clipping | sign of the reward, training only |
| Network | conv 16x8x8/4, conv 32x4x4/2, fc 256, linear output per action, ReLU |
| Algorithm | Q-learning, experience replay, NO target network |
| Optimizer | RMSProp, minibatch 32 |
| Replay memory | last 1M frames |
| Exploration | epsilon 1.0 to 0.1 linearly over 1M frames, then fixed |
| Training length | 10M frames |
| Evaluation | epsilon = 0.05; epoch = 50k minibatch updates; 10k eval steps per epoch |
| Q tracking | average max Q over a fixed set of states collected by a random policy |

## Not stated in the paper: our choices (open questions for the TA)

| Item | Our value | Source | Possible effect |
|---|---|---|---|
| Discount gamma | 0.95 | deep_q_rl | Scale of Q-values (Fig. 2 right); maybe learning speed. Nature used 0.99 |
| Learning rate | 0.0002 | deep_q_rl | Large; DQN is sensitive to it |
| RMSProp rho, eps | 0.99, 1e-6 | deep_q_rl | PyTorch puts eps outside the sqrt, Theano/Lasagne inside: small numeric difference |
| Replay start size | 100 steps | deep_q_rl | Early updates on a tiny buffer |
| Weight init | PyTorch default | ours | deep_q_rl uses N(0, 0.01), bias 0.1 (`init: normal_0.01`) |
| Crop position | rows 18 to 101 of 110 | deep_q_rl | Paper: "roughly captures the playing area" |
| Loss | 0.5 * squared TD error (mean over batch) | paper implies, deep_q_rl | Huber/error clipping only in 2015 |
| "Frames" | emulator frames (10M frames = 2.5M agent steps at k=4) | ours | deep_q_rl counts agent steps for epsilon decay. Changes exploration length and total training by 4x |
| Episode end | game over, NOT life loss | paper implies, deep_q_rl | Nature-era repos often end on life loss |
| No-op starts | none | paper | Deterministic ALE + no random starts = evaluation can be very repetitive |
| Held-out set size | 1000 states | ours | Only noise in the Q curve |
| Eval episodes that do not finish | ignored; if none finish, the partial return is logged with eval_n_episodes = 0 | ours | |

## Environment differences we cannot avoid

| Item | Paper (2013) | Ours |
|---|---|---|
| Emulator | ALE as of 2013 (Stella) | ALE 0.12.1 via Gymnasium 1.4.0 (`ALE/<Game>-v5`, `frameskip=1`, `repeat_action_probability=0`) |
| Action set | minimal | minimal (`full_action_space=False`) |
| Max episode length | not stated | ALE default (108k frames) |
| Gray-scale conversion | not stated | ALE luminance (`obs_type="grayscale"`) |

## Change log

| Date | Who | Change | Reason |
|---|---|---|---|
| 2026-10-18 | Stanley | Initial setup | |
| 2026-11-07 | Luke | Added Catch/Pong pilots (`TEST.md`), `env` config key, `scripts/check_pilot.py` | TA: validate on a simple setup first. No change to the 2013 agent |
| 2026-11-07 | Luke | Fixed Space Invaders random/human baselines in `plot_results.py` and README to 179 / 3690 | Were the 2015 Nature values (148 / 1652); now match 2013 Table 1 |
| 2026-11-10 | Luke | `RESULTS_DIR` / `--sync-dir`: mirror runs, logs and pilot summaries to a persistent drive | Renku session storage is not permanent. No effect on training |
| 2026-11-17 | Luke | Space Invaders: `crop_offset` 8 -> 4 (run `space_invaders_crop4_seed1`) | With offset 8 the player's cannon is cut to its tip and the floor (incl. the movement-range marker) is lost, so the crop does not capture the playing area as the paper describes. Offset 4 shows cannon and floor; cost: the top invader row is cut at the start of a game. Decided from inspecting frames, not from scores. The original run `space_invaders_seed1` (offset 8, no learning) is kept and reported. |
| 2026-11-18 | Luke | Result `space_invaders_crop4_seed1`: no learning (eval 207 ± 67 after exploration, same as with crop 8). Crop ruled out as the cause; crop_offset 4 kept for Space Invaders | Decision on the crop was made from frames before this run (see 2026-11-17) |
| 2026-11-18 | Luke | Planned diagnostic `space_invaders_target_seed1`: crop_offset 4 + target network (update every 10k steps) + Huber loss, i.e. the 2015 Nature stabilisers; everything else unchanged | Question: is the missing target network why Space Invaders does not learn (loss ~10x Breakout and not falling, training scores rise then decay)? NOT a replacement for the 2013 runs, which remain the main result. Reported regardless of outcome |
| 2026-11-20 | Luke | Planned ablation `breakout_gamma099_seed1`: gamma 0.99 instead of 0.95, everything else unchanged (2013 setup, no target network) | Question: does the flat held-out Q curve (claim 2, paper Fig. 2) depend on gamma? Paper does not state gamma; 0.95 is from deep_q_rl, 0.99 is the 2015 Nature value. Effect on score is a secondary observation. One seed, exploratory. Reported regardless of outcome |
| 2026-11-22 | Luke | Result `breakout_gamma099_seed1`: unstable. Loss ~100 in epoch 1 (vs ~0.007 at gamma 0.95), then a frozen deterministic policy scoring ~11 per game for the rest of the run; held-out Q plateaus at ~0.9. Gamma stays 0.95 for all main runs | Without a target network, gamma 0.99 diverges early on Breakout (1 seed) |
| 2026-11-22 | Luke | Planned: "frames = agent steps" reading. `breakout_steps_seed1` (total_frames 40M, epsilon_decay_frames 4M) and `space_invaders_steps_seed1` (crop_offset 4, total_frames 30M, epsilon_decay_frames 3M): 10M agent steps = 200 epochs, epsilon 1 -> 0.1 over 1M agent steps; replay (1M observations) unchanged; 2013 setup otherwise (gamma 0.95, no target network) | Paper Sec. 5 says 10M "frames", but Fig. 2 shows 100 epochs x 50k updates = 5M updates, impossible under our reading (2.5M updates for Breakout). The replay memory of "1M frames" is 1M stored observations = 1M agent steps. deep_q_rl (run_nips.py) uses EPOCHS=100 and EPSILON_DECAY=1000000 in agent steps. With "frames = agent steps", Sec. 5, Fig. 2 and the replay size are consistent. Reported at epoch 100 (Fig. 2 / deep_q_rl) and epoch 200 (paper text) |
| 2026-11-22 | Luke | Note on timing | This reinterpretation was decided after seeing the first results (Breakout below the paper, Space Invaders not learning); the argument rests on the paper's own Fig. 2 and replay definition, not on scores. The emulator-frame runs remain reported as the first reading |
| 2026-11-24 | Luke | Branch `2015-nature`: full 2015 Nature agent as config `base_nature2015.yaml` (network, scaled frames, frame max, 30 no-op starts, life loss terminal in training, gamma 0.99, DeepMind RMSProp, error clipping, summed gradient, update every 4 steps, target network, 50k replay start). Every component is a separate key; 2013 configs unchanged (verified bit-identical) | TA's suggestion: start from the 2015 agent and ablate back to 2013 to find what makes Space Invaders learn. Details and deviations from the 2015 paper in NATURE2015.md |
| 2026-11-24 | Luke | Planned: `si_nature2015_seed1` - full 2015 agent (configs/space_invaders_nature2015.yaml) on Space Invaders, 10M agent steps (200 epochs), seed 1 | Step 0 of the ablation path (NATURE2015.md): does the full 2015 agent learn Space Invaders in our environment? Compare with 2015 ED Table 3 (10M steps, best evaluation; Space Invaders 1088.9 with target network) and with "always fire" (~285). Reported regardless of outcome |