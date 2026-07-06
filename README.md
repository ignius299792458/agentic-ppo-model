# PPO (Proximal Policy Optimization) Implementation

A minimal, readable PPO (Proximal Policy Optimization) implementation for
discrete-action Gymnasium environments

The design keeps one responsibility per file. The **actor** turns an
observation into an action distribution; the **critic** turns the same
observation into a value estimate `V(s)`; the **env** is a thin Gymnasium
wrapper; and **main** is the orchestrator that collects rollouts, computes
GAE advantages, and runs the clipped PPO update. Nothing about the RL
algorithm lives inside the networks — they are pure function approximators
that answer "which action?" and "how good is this state?".

## Flowchart of PPO

![Flowchart](./docs/ppo_flowchart.png)
Data shapes: a single observation is `(obs_dim,)`; networks receive it
batched — `(1, obs_dim)` during rollout, `(B, obs_dim)` during the update.
`obs_dim` and `act_dim` are read from the env's spaces, so swapping
environments needs no code changes.

## Demo

[![Program run demo video](./docs/cartpole_policy_comparison_thumnail.png)](./docs/cartpole_policy_comparison_final.mp4)

## How to run

The project is `poetry` manager setup. Therefore, to install all required dependencies, run this

```bash
poetry lock && poetry install
```

> Make sure the specific venv python interpreter is selected.

Run and Train with defaults (CartPole-v1):
Using

1. Directly running the `python main.py`
2. Using Debugger (Any python file), open main.py and run it on debug mode

Tune anything via `config.py` (or construct a `Config(...)` in code and pass
it to `train`). Common overrides: `env_name`, `total_iterations`,
`rollout_steps`, `clip_eps`, `lr_actor`, `device`.

```python
from config import Config
from main import train

train(Config(env_name="CartPole-v1", total_iterations=50))
```
