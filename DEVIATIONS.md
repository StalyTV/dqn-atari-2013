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
