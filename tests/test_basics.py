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


def test_network_shapes():
    net = QNetwork("nips2013", num_actions=6)
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


def test_catch_env_interface_and_scores():
    from dqn.catch import GRID, CatchEnv

    env = CatchEnv(balls=3, drift=True, seed=0)
    f = env.reset()
    assert f.shape == (84, 84) and f.dtype == np.uint8
    # an oracle that follows the ball catches every ball: return = +balls
    total, done, steps = 0.0, False, 0
    while not done:
        target = min(max(env.ball_c - 1, 0), GRID - 3)
        a = 1 + int(np.sign(target - env.paddle))
        f, r, done = env.step(a)
        total += r
        steps += 1
        assert f.shape == (84, 84)
    assert total == 3 and steps == 3 * (GRID - 1)
    # a paddle parked in a corner misses most balls
    env = CatchEnv(balls=50, drift=False, seed=1)
    env.reset()
    total, done = 0.0, False
    while not done:
        _, r, done = env.step(0)
        total += r
    assert total < 0


# ---------------------------------------------------------------- 2015 (Nature) components


def test_nature2015_network_shape_and_size():
    net = QNetwork("nature2015", num_actions=6)
    out = net(torch.zeros(2, 4, 84, 84, dtype=torch.uint8))
    assert out.shape == (2, 6)
    # 32x8x8/4, 64x4x4/2, 64x3x3/1, fc 512 (Mnih et al. 2015, DQN 3.0 convnet_atari3.lua)
    n = sum(p.numel() for p in net.parameters())
    assert n == (4 * 32 * 64 + 32) + (32 * 64 * 16 + 64) + (64 * 64 * 9 + 64) + (3136 * 512 + 512) + (512 * 6 + 6)


def test_scale_preprocessing_gives_84x84():
    from dqn.env import preprocess

    f = np.random.default_rng(0).integers(0, 255, size=(210, 160), dtype=np.uint8)
    assert preprocess(f, 8, "scale").shape == (84, 84)
    assert preprocess(f, 8, "crop").shape == (84, 84)


def test_deepmind_rmsprop_matches_dqn3_formula():
    from dqn.optim import DeepMindRMSprop

    w = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    opt = DeepMindRMSprop([w], lr=0.1, alpha=0.95, eps=0.01)
    grad = torch.tensor([0.5, -1.0])
    w.grad = grad.clone()
    opt.step()
    g, g2 = 0.05 * grad, 0.05 * grad**2
    expected = torch.tensor([1.0, -2.0]) - 0.1 * grad / torch.sqrt(g2 - g**2 + 0.01)
    assert torch.allclose(w.detach(), expected)


def test_life_loss_is_reported_and_game_continues():
    from dqn.env import AtariEnv

    env = AtariEnv("SpaceInvaders", 4, resize="scale", noop_max=30, max_pool_last_two=True,
                   seed=0, break_on_life_loss=True)
    env.reset()
    lives0 = env.lives
    rng = np.random.default_rng(0)
    for _ in range(5000):
        f, r, done = env.step(int(rng.integers(env.num_actions)))
        if env.life_lost or done:
            break
    assert env.life_lost and not done and env.lives == lives0 - 1
    f = env.noop_start()  # same game continues after random NOOPs
    assert f.shape == (84, 84) and env.lives == lives0 - 1


def test_2013_configs_unchanged_by_2015_switches():
    for name in ["breakout", "space_invaders"]:
        cfg = load_config(f"configs/{name}.yaml")
        assert cfg["resize"] == "crop" and cfg["noop_max"] == 0 and not cfg["life_loss_terminal"]
        assert cfg["optimizer"] == "rmsprop_torch" and cfg["loss_reduction"] == "mean"
        assert cfg["update_frequency"] == 1 and cfg["target_update_interval"] == 0
