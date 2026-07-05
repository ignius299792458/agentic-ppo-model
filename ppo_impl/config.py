from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Config:
    # env
    env_name: str = "CartPole-v1"
    render_mode: Optional[str] = "human"
    max_steps: Optional[int] = None

    # network
    hidden_size: int = 64
    n_hidden: int = 2

    # optimisation
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    max_grad_norm: float = 0.5

    # ppo
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01

    # loop
    rollout_steps: int = 2048
    update_epochs: int = 10
    minibatch_size: int = 64
    total_iterations: int = 50

    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
