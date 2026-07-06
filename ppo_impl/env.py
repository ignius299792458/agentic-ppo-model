from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import time


class PlaygroundEnv:
    """Thin wrapper around a Gymnasium env. Env concerns only."""

    def __init__(
        self,
        env_name: str,
        render_mode: Optional[str] = None,
        max_steps: Optional[int] = None,
        verbose: bool = False,
    ):
        self.env_name = env_name
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.verbose = verbose

        self.env: gym.Env = gym.make(id=env_name, render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # shapes the actor/critic size themselves from
        self.obs_dim = int(np.prod(self.observation_space.shape))
        self.act_dim = int(self.action_space.n)

        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []

    def __enter__(self) -> "PlaygroundEnv":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def reset(self, seed: Optional[int] = None):
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(action)

    def run_episode(
        self,
        policy: Optional[Callable[[np.ndarray], int]] = None,
        seed: Optional[int] = None,
        sleep_time: Optional[float] = None,
    ) -> dict:
        """Run one episode. `policy` maps a single obs -> action."""
        obs, _ = self.env.reset(seed=seed)

        observations = [obs]
        actions: list[int] = []
        rewards: list[float] = []
        total_reward, step = 0.0, 0
        done = False

        while True:
            # pass the CURRENT single observation, not the history
            action = policy(obs) if policy is not None else self.action_space.sample()
            obs, reward, terminated, truncated, _ = self.env.step(action)
            if self.verbose:
                print(
                    f"STEP: {step}\naction:{action}\nobs:{obs}\nreward:{reward}\ndone:{done}\n"
                )
            if sleep_time is not None:
                time.sleep(sleep_time)

            observations.append(obs)
            actions.append(action)
            rewards.append(float(reward))
            total_reward += float(reward)
            step += 1

            done = terminated or truncated
            if self.max_steps is not None:
                done = done or (step >= self.max_steps)
            if done:
                break

        self._episode_rewards.append(total_reward)
        self._episode_lengths.append(step)
        return {
            "total_reward": total_reward,
            "steps": step,
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
        }

    def run_episodes(
        self,
        n: int = 5,
        policy=None,
        seed: Optional[int] = None,
        sleep_time: Optional[float] = None,
    ):
        self._episode_rewards.clear()
        self._episode_lengths.clear()
        results = []
        for i in range(n):
            ep_seed = None if seed is None else seed + i
            r = self.run_episode(policy=policy, seed=ep_seed, sleep_time=sleep_time)
            results.append(r)
            if self.verbose:
                label = "random" if policy is None else "policy"
                print(
                    f"  ep {i + 1:>3}/{n} [{label}] reward={r['total_reward']:7.1f} steps={r['steps']}"
                )
        return results

    def stats(self) -> dict:
        if not self._episode_rewards:
            return {}
        arr = np.array(self._episode_rewards)
        return {
            "episodes": len(arr),
            "mean_reward": float(arr.mean()),
            "std_reward": float(arr.std()),
            "min_reward": float(arr.min()),
            "max_reward": float(arr.max()),
            "mean_length": float(np.mean(self._episode_lengths)),
        }

    def close(self) -> None:
        self.env.close()

    def __repr__(self) -> str:
        return f"PlaygroundEnv(env={self.env_name!r}, obs_dim={self.obs_dim}, act_dim={self.act_dim}, verbase={self.verbose})"
