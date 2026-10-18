"""Frame-based experience replay (Mnih et al. 2013, Algorithm 1).

Each slot i stores the newest frame f_i of state s_i, the action a_i taken in s_i,
the clipped reward r_i and the flag done_i (episode ended after a_i). A state is
the stack of 4 consecutive frames, built on the fly when sampling, so every frame
is stored only once (1M frames of 84x84 uint8 = about 7 GB, as in the paper).

Frames that belong to a previous episode, or that were never written, are
replaced by zeros in the stack. The next state of slot i uses frame i+1, so a
slot can only be sampled once slot i+1 exists.
"""

from __future__ import annotations

import numpy as np

STACK = 4


class ReplayBuffer:
    def __init__(self, capacity: int, frame_shape=(84, 84), seed: int | None = None):
        self.capacity = capacity
        self.frames = np.zeros((capacity, *frame_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.ptr = 0  # next slot to write
        self.count = 0  # number of slots written so far (capped at capacity)
        self.rng = np.random.default_rng(seed)

    def add(self, frame: np.ndarray, action: int, reward: float, done: bool) -> None:
        self.frames[self.ptr] = frame
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def __len__(self) -> int:
        return self.count

    def _valid_indices(self, n: int) -> np.ndarray:
        """Sample n slot indices i whose state stack and next frame are usable."""
        out = np.empty(0, dtype=np.int64)
        while out.size < n:
            if self.count < self.capacity:
                # slots 0 .. count-2 (slot count-1 has no successor yet)
                cand = self.rng.integers(0, self.count - 1, size=2 * n)
            else:
                cand = self.rng.integers(0, self.capacity, size=2 * n)
                # age = how far slot i is from the oldest slot (ptr). The state stack
                # needs 3 older frames that are not overwritten, and slot i+1 must be
                # written, i.e. i != ptr - 1.
                age = (cand - self.ptr) % self.capacity
                cand = cand[(age >= STACK - 1) & (age <= self.capacity - 2)]
            out = np.concatenate([out, cand])
        return out[:n]

    def _stack(self, idx: np.ndarray, last_offset: int) -> np.ndarray:
        """Build 4-frame stacks whose newest frame is at slot idx + last_offset.

        A frame is zeroed if an episode ended between it and the newest frame,
        or if its slot was never written (buffer not yet full)."""
        offsets = np.arange(last_offset - STACK + 1, last_offset + 1)  # oldest .. newest
        raw = idx[:, None] + offsets[None, :]  # (B, 4)
        slots = raw % self.capacity
        frames = self.frames[slots]  # (B, 4, 84, 84)

        # done flags on the 3 slots just before the newest one
        done_before = self.dones[slots[:, :-1]]  # (B, 3): dones at offsets o, for older frames
        # frame at position p is invalid if any done at positions p .. STACK-2
        invalid = np.zeros_like(slots, dtype=bool)
        invalid[:, :-1] = np.flip(np.maximum.accumulate(np.flip(done_before, axis=1), axis=1), axis=1)
        if self.count < self.capacity:
            never_written = raw < 0
            invalid |= never_written
        frames = frames.copy()
        frames[invalid] = 0
        return frames

    def sample(self, batch_size: int):
        idx = self._valid_indices(batch_size)
        obs = self._stack(idx, 0)
        next_obs = self._stack(idx, 1)
        return (
            obs,
            self.actions[idx],
            self.rewards[idx],
            next_obs,
            self.dones[idx].astype(np.float32),
        )
