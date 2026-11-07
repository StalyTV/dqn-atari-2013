"""Catch: a tiny pixel game for checking the DQN pipeline before spending Atari compute.

Not part of the 2013 paper. It exists so that the *same* code path used for Atari
(84x84 uint8 frames -> 4-frame stacks in the replay buffer -> 2013 conv network ->
Q-learning update without target network -> evaluation and held-out Q tracking)
can be run end to end in minutes on a CPU, against a known optimum.

The game: a ball falls from the top of a 21x21 grid, one row per step; the agent
moves a 3-cell paddle on the bottom row (actions: left, stay, right). Catching the
ball gives +1, missing it -1, every other step 0. One episode = `balls` balls, so
the best possible return is +balls and a random policy scores well below zero.

With `drift=True` the ball also moves sideways (-1, 0 or +1 column per step) and
bounces off the walls. A single frame then does not show where the ball is going:
the agent has to use the frame history, so a broken frame stack shows up as a
score that plateaus below the optimum.

The grid is drawn at 4x4 pixels per cell, giving 84x84 frames like the Atari
wrapper, and the class has the same interface as `AtariEnv`
(reset() -> frame, step(a) -> (frame, reward, done), num_actions, close()).
"""

from __future__ import annotations

import numpy as np

GRID = 21
CELL = 4  # 21 * 4 = 84 pixels
PADDLE = 3  # paddle width in cells
BALL_VALUE = 255
PADDLE_VALUE = 128


class CatchEnv:
    num_actions = 3  # 0 = left, 1 = stay, 2 = right

    def __init__(self, balls: int = 5, drift: bool = True, frame_skip: int = 1, seed: int | None = None):
        if frame_skip != 1:
            raise ValueError("Catch moves one cell per step; use frame_skip=1")
        self.balls = balls
        self.drift = drift
        self.rng = np.random.default_rng(seed)
        self.paddle = GRID // 2  # column of the paddle's left cell
        self.ball_r = self.ball_c = self.ball_v = 0
        self.balls_left = 0

    # ------------------------------------------------------------------ game logic

    def _new_ball(self) -> None:
        self.ball_r = 0
        self.ball_c = int(self.rng.integers(GRID))
        self.ball_v = int(self.rng.integers(-1, 2)) if self.drift else 0

    def _render(self) -> np.ndarray:
        img = np.zeros((GRID, GRID), dtype=np.uint8)
        img[GRID - 1, self.paddle : self.paddle + PADDLE] = PADDLE_VALUE
        img[self.ball_r, self.ball_c] = BALL_VALUE
        return np.kron(img, np.ones((CELL, CELL), dtype=np.uint8))

    def reset(self) -> np.ndarray:
        self.paddle = int(self.rng.integers(GRID - PADDLE + 1))
        self.balls_left = self.balls
        self._new_ball()
        return self._render()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        self.paddle = int(np.clip(self.paddle + (int(action) - 1), 0, GRID - PADDLE))

        # move the ball, bouncing off the side walls
        c = self.ball_c + self.ball_v
        if c < 0 or c >= GRID:
            self.ball_v = -self.ball_v
            c = self.ball_c + self.ball_v
        self.ball_c = c
        self.ball_r += 1

        reward, done = 0.0, False
        if self.ball_r == GRID - 1:  # ball reached the paddle row
            caught = self.paddle <= self.ball_c < self.paddle + PADDLE
            reward = 1.0 if caught else -1.0
            self.balls_left -= 1
            if self.balls_left == 0:
                done = True
            else:
                self._new_ball()
        return self._render(), reward, done

    def close(self) -> None:
        pass


def random_policy_return(balls: int = 5, drift: bool = True, episodes: int = 2000, seed: int = 0) -> float:
    """Average return of the uniform random policy (the baseline the agent must beat)."""
    env = CatchEnv(balls=balls, drift=drift, seed=seed)
    rng = np.random.default_rng(seed + 1)
    total = 0.0
    for _ in range(episodes):
        env.reset()
        done = False
        while not done:
            _, r, done = env.step(int(rng.integers(3)))
            total += r
    return total / episodes
