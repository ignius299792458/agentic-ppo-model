from dataclasses import dataclass
from typing import Optional

import torch

ENVS = {"cartpole": "CartPole-v1", "mountain_car": "MountainCar-v0"}


@dataclass
class Config:
    # env
    env_name: str = ENVS["mountain_car"]
    render_mode: Optional[str] = "human"
    max_steps: Optional[int] = None

    # episodes
    episodes: int = 200

    # time
    sleep_between_steps: Optional[float] = None  # sec

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
    rollout_steps: int = 2048  # bump to 2048 for real training
    update_epochs: int = 10
    minibatch_size: int = 64
    total_iterations: int = 50

    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # logs
    verbose: bool = True
