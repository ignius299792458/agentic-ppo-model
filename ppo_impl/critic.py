import numpy as np
import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: int, n_hidden: int, out_dim: int) -> nn.Sequential:
    layers, last = [], in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(last, hidden), nn.Tanh()]
        last = hidden
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class Critic(nn.Module):
    """Value network. obs -> scalar V(s)."""

    def __init__(
        self, obs_dim: int, hidden: int = 64, n_hidden: int = 2, verbose: bool = False
    ):
        super().__init__()
        self.net = _mlp(obs_dim, hidden, n_hidden, 1)
        self.verbose = verbose

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs (B, obs_dim) -> values (B,)
        return self.net(obs).squeeze(-1)

    @torch.no_grad()
    def value(self, obs_np: np.ndarray) -> float:
        # rollout: single obs (obs_dim,) -> V(s) float
        device = next(self.parameters()).device
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        value_f = self.forward(obs)

        if self.verbose:
            print(f"CRITIC --- obs: {obs}, value: {value_f}")

        return float(value_f.item())
