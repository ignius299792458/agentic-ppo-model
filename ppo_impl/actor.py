import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _mlp(in_dim: int, hidden: int, n_hidden: int, out_dim: int) -> nn.Sequential:
    layers, last = [], in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(last, hidden), nn.Tanh()]
        last = hidden
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Policy network. obs -> logits over discrete actions."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 64,
        n_hidden: int = 2,
        verbose: bool = False,
    ):
        super().__init__()
        self.verbose = verbose
        self.net = _mlp(obs_dim, hidden, n_hidden, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs (B, obs_dim) -> logits (B, act_dim)
        return self.net(obs)

    def distribution(self, obs: torch.Tensor) -> Categorical:
        logits = self.forward(obs)

        if self.verbose:
            print("ACTOR --- (distribution) Logits: ", logits)

        return Categorical(logits=logits)

    @torch.no_grad()
    def act(self, obs_np: np.ndarray):
        # rollout: single obs (obs_dim,) -> (action int, log_prob float)
        device = next(self.parameters()).device
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        dist = self.distribution(obs)
        action = dist.sample()
        action_item, logit_items = int(action.item()), float(dist.log_prob(action).item())

        if self.verbose:
            print(f"ACTOR --- (act) obs: {obs}, logits: {logit_items}, action: {action_item}")

        return action_item, logit_items

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        # update: batched obs (B, obs_dim), actions (B,) -> log_probs (B,), entropy (B,)
        dist = self.distribution(obs)
        return dist.log_prob(actions), dist.entropy()
