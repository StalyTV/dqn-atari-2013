"""Q-network.

`nips2013`: the architecture of Mnih et al. (2013), Section 5:
  input 4x84x84 -> conv 16 filters 8x8 stride 4, ReLU
               -> conv 32 filters 4x4 stride 2, ReLU
               -> fully connected 256, ReLU
               -> linear output, one Q-value per action
"""

from __future__ import annotations

import torch
from torch import nn


def build_q_network(name: str, num_actions: int, init: str = "torch_default") -> nn.Module:
    if name == "nips2013":
        net = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )
    else:
        raise ValueError(f"Unknown network '{name}'")

    if init == "normal_0.01":
        # Initialisation used by the deep_q_rl reference implementation (the paper gives none).
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0.1)
    elif init != "torch_default":
        raise ValueError(f"Unknown init '{init}'")
    return net


class QNetwork(nn.Module):
    """Wraps the network and scales uint8 pixels to [0, 1]."""

    def __init__(self, name: str, num_actions: int, init: str = "torch_default"):
        super().__init__()
        self.net = build_q_network(name, num_actions, init)

    def forward(self, obs_uint8: torch.Tensor) -> torch.Tensor:
        return self.net(obs_uint8.float() / 255.0)
