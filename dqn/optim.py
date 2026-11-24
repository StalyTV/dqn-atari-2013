"""RMSProp as used by DeepMind for the 2015 Nature DQN (DQN 3.0, NeuralQLearner.lua).

It differs from torch.optim.RMSprop (used for the 2013 setup) in three ways:
  * it is "centered": it divides by sqrt(E[g^2] - E[g]^2 + eps), i.e. the running
    standard deviation of the gradient rather than its running RMS,
  * eps = 0.01 is added INSIDE the square root (the paper's "min squared gradient"),
  * decay 0.95 for both running averages (the paper's "gradient momentum" and
    "squared gradient momentum"); there is no momentum term on the update itself.

DQN 3.0 update, with dw the gradient of the summed (clipped) TD error:
    g  = 0.95 g  + 0.05 dw
    g2 = 0.95 g2 + 0.05 dw^2
    w -= lr * dw / sqrt(g2 - g^2 + 0.01)
"""

from __future__ import annotations

import torch


class DeepMindRMSprop(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.00025, alpha: float = 0.95, eps: float = 0.01):
        super().__init__(params, dict(lr=lr, alpha=alpha, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, alpha, eps = group["lr"], group["alpha"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                st = self.state[p]
                if not st:
                    st["g"] = torch.zeros_like(p)
                    st["g2"] = torch.zeros_like(p)
                g, g2 = st["g"], st["g2"]
                g.mul_(alpha).add_(grad, alpha=1 - alpha)
                g2.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                # clamp guards against tiny negative values from floating point
                denom = (g2 - g * g).clamp_(min=0).add_(eps).sqrt_()
                p.addcdiv_(grad, denom, value=-lr)
