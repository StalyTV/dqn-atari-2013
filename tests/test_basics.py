"""Fast checks of the pieces that are easy to get subtly wrong. Run: python -m pytest -q"""

import numpy as np
import torch

from dqn.config import apply_overrides, load_config
from dqn.env import AtariEnv
from dqn.model import QNetwork
from dqn.replay import ReplayBuffer


def frame(v):
    return np.full((84, 84), v, dtype=np.uint8)


def test_config_inheritance_and_overrides():
    cfg = load_config("configs/space_invaders.yaml")
    assert cfg["frame_skip"] == 3 and cfg["network"] == "nips2013"
    assert cfg["target_update_interval"] == 0
    cfg = apply_overrides(cfg, ["target_update_interval=10000"])
    assert cfg["target_update_interval"] == 10000
    abl = load_config("configs/ablation_target_net_breakout.yaml")
    assert abl["game"] == "Breakout" and abl["target_update_interval"] == 10000


def test_network_shapes():
    for name in ["nips2013", "nature2015"]:
        net = QNetwork(name, num_actions=6)
        out = net(torch.zeros(2, 4, 84, 84, dtype=torch.uint8))
        assert out.shape == (2, 6)
    # 2013 net: 16*4*8*8+16 + 32*16*4*4+32 + 2592*256+256 + 256*6+6
    n = sum(p.numel() for p in QNetwork("nips2013", 6).parameters())
    assert n == 4112 + 8224 + 663808 + 1542


def test_env_preprocessing():
    for game, k in [("Breakout", 4), ("SpaceInvaders", 3)]:
        env = AtariEnv(game, frame_skip=k, seed=0)
        f = env.reset()
        assert f.shape == (84, 84) and f.dtype == np.uint8
        f, r, done = env.step(0)
        assert f.shape == (84, 84)
        env.close()


def test_replay_stacks_respect_episode_boundaries():
    buf = ReplayBuffer(capacity=100, seed=0)
    # episode A: frames 1..5, ends after slot with frame 5; episode B: frames 11..20
    for v in range(1, 6):
        buf.add(frame(v), 0, 0.0, done=(v == 5))
    for v in range(11, 21):
        buf.add(frame(v), 0, 1.0, done=False)
    # state whose newest frame is 12 (slot 6): history from episode A must be zeroed
    obs = buf._stack(np.array([6]), 0)[0]
    assert [int(x[0, 0]) for x in obs] == [0, 0, 11, 12]
    nxt = buf._stack(np.array([6]), 1)[0]
    assert [int(x[0, 0]) for x in nxt] == [0, 11, 12, 13]
    # first slot of buffer: frames before slot 0 were never written
    obs = buf._stack(np.array([1]), 0)[0]
    assert [int(x[0, 0]) for x in obs] == [0, 0, 1, 2]
    # terminal transition: done flag is returned so the target ignores next state
    assert buf.dones[4]


def test_replay_wraparound_never_mixes_old_and_new():
    cap = 50
    buf = ReplayBuffer(capacity=cap, seed=1)
    for t in range(cap * 3 + 7):
        buf.add(frame(t % 250), 0, 0.0, done=False)
    for _ in range(20):
        obs, _, _, nxt, _ = buf.sample(32)
        vals = obs[:, :, 0, 0].astype(int)
        nvals = nxt[:, :, 0, 0].astype(int)
        # consecutive frames in a stack differ by exactly 1 (mod 250): no overwritten slot inside
        assert np.all((np.diff(vals, axis=1) % 250) == 1)
        assert np.all((nvals[:, -1] - vals[:, -1]) % 250 == 1)
