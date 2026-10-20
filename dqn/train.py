"""Train DQN as in Mnih et al. (2013), "Playing Atari with Deep Reinforcement Learning".

Usage:
    python -m dqn.train --config configs/breakout.yaml --seed 1
    python -m dqn.train --config configs/space_invaders.yaml --seed 1
    python -m dqn.train --config configs/breakout.yaml --seed 1 --set total_frames=200000

Outputs go to runs/<run_name>/:
    config.yaml          resolved config (everything needed to rerun)
    run_info.json        library versions, git commit, device, start time
    train_episodes.csv   one row per training episode (raw score)
    eval.csv             one row per evaluation epoch (paper Fig. 2 and Table 1)
    model_latest.pt      network weights after the latest evaluation
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from dqn.config import apply_overrides, load_config
from dqn.env import AtariEnv
from dqn.model import QNetwork
from dqn.replay import STACK, ReplayBuffer


# --------------------------------------------------------------------------- helpers


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def epsilon_at(frames: int, cfg: dict) -> float:
    """Linear annealing from epsilon_start to epsilon_end over epsilon_decay_frames."""
    frac = min(1.0, frames / cfg["epsilon_decay_frames"])
    return cfg["epsilon_start"] + frac * (cfg["epsilon_end"] - cfg["epsilon_start"])


def new_stack(first_frame: np.ndarray) -> np.ndarray:
    """State at episode start: zeros for the missing history (matches the replay buffer)."""
    stack = np.zeros((STACK, *first_frame.shape), dtype=np.uint8)
    stack[-1] = first_frame
    return stack


def push_frame(stack: np.ndarray, frame: np.ndarray) -> np.ndarray:
    stack = np.roll(stack, -1, axis=0)
    stack[-1] = frame
    return stack


def greedy_action(q_net: QNetwork, stack: np.ndarray, device: torch.device) -> int:
    with torch.no_grad():
        q = q_net(torch.as_tensor(stack, device=device).unsqueeze(0))
    return int(q.argmax(dim=1).item())


def make_env(cfg: dict, seed: int) -> AtariEnv:
    return AtariEnv(
        game=cfg["game"],
        frame_skip=cfg["frame_skip"],
        repeat_action_probability=cfg["repeat_action_probability"],
        max_pool_last_two=cfg["max_pool_last_two"],
        crop_offset=cfg["crop_offset"],
        seed=seed,
    )


# --------------------------------------------------------------------------- evaluation


def collect_holdout_states(cfg: dict, seed: int, n: int) -> np.ndarray:
    """Paper Sec. 5.1: a fixed set of states collected with a random policy before
    training, used to track the average maximum predicted Q-value."""
    env = make_env(cfg, seed)
    rng = np.random.default_rng(seed)
    stack = new_stack(env.reset())
    states = []
    while len(states) < 10 * n:
        frame, _, done = env.step(int(rng.integers(env.num_actions)))
        stack = new_stack(env.reset()) if done else push_frame(stack, frame)
        states.append(stack.copy())
    env.close()
    idx = rng.choice(len(states), size=n, replace=False)
    return np.stack([states[i] for i in idx])


def evaluate(q_net: QNetwork, env: AtariEnv, cfg: dict, device, rng) -> dict:
    """Run epsilon-greedy (epsilon_eval) for eval_steps agent steps; report raw scores."""
    q_net.eval()
    returns, current = [], 0.0
    stack = new_stack(env.reset())
    for _ in range(cfg["eval_steps"]):
        if rng.random() < cfg["epsilon_eval"]:
            action = int(rng.integers(env.num_actions))
        else:
            action = greedy_action(q_net, stack, device)
        frame, reward, done = env.step(action)
        current += reward
        if done:
            returns.append(current)
            current = 0.0
            stack = new_stack(env.reset())
        else:
            stack = push_frame(stack, frame)
    q_net.train()
    completed = len(returns)
    if completed == 0:  # no episode finished: report the partial one, flagged by n_episodes=0
        returns = [current]
    return {
        "eval_mean_return": float(np.mean(returns)),
        "eval_max_return": float(np.max(returns)),
        "eval_n_episodes": completed,
    }


def average_max_q(q_net: QNetwork, states: np.ndarray, device) -> float:
    with torch.no_grad():
        q = q_net(torch.as_tensor(states, device=device))
    return float(q.max(dim=1).values.mean().item())


# --------------------------------------------------------------------------- training


def train(cfg: dict, run_dir: Path, device: torch.device) -> None:
    seed = cfg["seed"]
    set_seed(seed)
    rng = np.random.default_rng(seed)

    env = make_env(cfg, seed)
    eval_env = make_env(cfg, seed + 10_000)
    num_actions = env.num_actions

    q_net = QNetwork(cfg["network"], num_actions, cfg["init"]).to(device)
    use_target = cfg["target_update_interval"] > 0
    target_net = copy.deepcopy(q_net) if use_target else q_net
    optimizer = torch.optim.RMSprop(
        q_net.parameters(),
        lr=cfg["learning_rate"],
        alpha=cfg["rmsprop_alpha"],
        eps=cfg["rmsprop_eps"],
        momentum=0.0,
        centered=False,
    )
    replay = ReplayBuffer(cfg["replay_capacity"], seed=seed)
    holdout = collect_holdout_states(cfg, seed + 20_000, cfg["num_holdout_states"])

    episodes_f = open(run_dir / "train_episodes.csv", "w", newline="")
    ep_writer = csv.writer(episodes_f)
    ep_writer.writerow(["episode", "agent_step", "frames", "return", "length", "epsilon"])
    eval_f = open(run_dir / "eval.csv", "w", newline="")
    eval_writer = None

    total_steps = cfg["total_frames"] // cfg["frame_skip"]
    frame = env.reset()
    stack = new_stack(frame)
    episode, ep_return, ep_len = 0, 0.0, 0
    losses, recent_returns = [], []
    t_start = time.time()
    t_last, step_last = t_start, 0

    for step in range(1, total_steps + 1):
        frames = step * cfg["frame_skip"]
        eps = epsilon_at(frames, cfg)

        # act (epsilon-greedy)
        if rng.random() < eps:
            action = int(rng.integers(num_actions))
        else:
            action = greedy_action(q_net, stack, device)
        next_frame, reward, done = env.step(action)
        clipped = float(np.sign(reward)) if cfg["clip_rewards"] else reward
        replay.add(stack[-1], action, clipped, done)
        ep_return += reward
        ep_len += 1

        if done:
            episode += 1
            ep_writer.writerow([episode, step, frames, ep_return, ep_len, round(eps, 4)])
            recent_returns.append(ep_return)
            ep_return, ep_len = 0.0, 0
            stack = new_stack(env.reset())
        else:
            stack = push_frame(stack, next_frame)

        # learn
        if len(replay) >= cfg["learning_starts"] and step % cfg["update_frequency"] == 0:
            obs, act, rew, nxt, dn = replay.sample(cfg["batch_size"])
            obs = torch.as_tensor(obs, device=device)
            nxt = torch.as_tensor(nxt, device=device)
            act = torch.as_tensor(act, device=device)
            rew = torch.as_tensor(rew, device=device)
            dn = torch.as_tensor(dn, device=device)

            q = q_net(obs).gather(1, act.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                # Without a target network (2013 setup) the bootstrap uses the online network.
                next_q = target_net(nxt).max(dim=1).values
                y = rew + cfg["gamma"] * (1.0 - dn) * next_q
            if cfg["loss"] == "mse":
                loss = 0.5 * F.mse_loss(q, y)
            elif cfg["loss"] == "huber":
                loss = F.smooth_l1_loss(q, y)
            else:
                raise ValueError(cfg["loss"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if use_target and step % cfg["target_update_interval"] == 0:
            target_net.load_state_dict(q_net.state_dict())

        # periodic progress line
        if step % cfg["log_every_steps"] == 0:
            now = time.time()
            sps = (step - step_last) / (now - t_last)
            t_last, step_last = now, step
            avg_ret = np.mean(recent_returns[-20:]) if recent_returns else float("nan")
            print(
                f"step {step:>9}/{total_steps}  frames {frames:>10}  eps {eps:.3f}  "
                f"episodes {episode:>5}  avg_return(20) {avg_ret:8.2f}  {sps:6.0f} steps/s",
                flush=True,
            )

        # evaluation epoch (paper: every 50k minibatch updates)
        if step % cfg["eval_every_steps"] == 0 or step == total_steps:
            row = {
                "epoch": step // cfg["eval_every_steps"],
                "agent_step": step,
                "frames": frames,
                "epsilon": round(eps, 4),
                "train_episodes": episode,
                "loss_mean": float(np.mean(losses)) if losses else float("nan"),
                "avg_max_q_holdout": average_max_q(q_net, holdout, device),
                "wall_clock_s": round(time.time() - t_start, 1),
            }
            row.update(evaluate(q_net, eval_env, cfg, device, rng))
            losses = []
            if eval_writer is None:
                eval_writer = csv.DictWriter(eval_f, fieldnames=list(row))
                eval_writer.writeheader()
            eval_writer.writerow(row)
            eval_f.flush()
            episodes_f.flush()
            torch.save(q_net.state_dict(), run_dir / "model_latest.pt")
            print(
                f"[eval] epoch {row['epoch']}  mean_return {row['eval_mean_return']:.2f}  "
                f"max_return {row['eval_max_return']:.1f}  episodes {row['eval_n_episodes']}  "
                f"avg_max_Q {row['avg_max_q_holdout']:.4f}",
                flush=True,
            )

    episodes_f.close()
    eval_f.close()
    env.close()
    eval_env.close()
    print(f"Done in {(time.time() - t_start) / 3600:.2f} h. Results in {run_dir}")


# --------------------------------------------------------------------------- entry point


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--run-name", default=None)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps ...")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config keys")
    args = p.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.set)
    cfg["seed"] = args.seed

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{cfg['name']}_seed{args.seed}_{stamp}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    import ale_py
    import gymnasium

    info = {
        "command": " ".join(sys.argv),
        "started": stamp,
        "git_commit": git_commit(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "ale_py": ale_py.__version__,
        "numpy": np.__version__,
        "host": platform.node(),
    }
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"Run dir: {run_dir}\nDevice: {device}")
    train(cfg, run_dir, device)


if __name__ == "__main__":
    main()
