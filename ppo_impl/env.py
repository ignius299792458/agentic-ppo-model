import gymnasium as gym
from typing import Callable, Optional
import numpy as np
import time


class PlaygroundEnv:

    def __init__(
        self,
        env_name: str,
        render_mode: Optional[str] = None,
        max_steps: Optional[int] = None,
    ):
        self.env_name = env_name
        self.render_mode = render_mode
        self.max_steps = max_steps

        # Core env: create once, reuse across episodes
        self.env: gym.Env = gym.make(id=env_name, render_mode=render_mode)

        # Handy shortcuts so callers don't have to dig into self.env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # Internal stats — reset by run_episodes()
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []

    # ── Context Manager Support ─────────────────────────────────────────
    def __enter__(self) -> "PlaygroundEnv":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Core API ────────────────────────────────────────────────────────
    def runnable_env(self) -> gym.Env:
        return self.env

    def run_episode(
        self,
        policy: Optional[Callable[[np.ndarray], int]] = None,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Run ONE episode.

        Parameters
        ----------
        policy : callable(observation) -> action, or None for a random agent
        seed   : optional RNG seed for reproducibility

        Returns
        -------
        dict with keys: total_reward, steps, observations, actions, rewards
        """

        obs, info = self.env.reset(seed=seed)

        observations: list[np.ndarray] = [obs]
        actions: list[int] = []
        rewards: list[float] = []
        total_reward: float = 0.0
        step = 0

        while True:
            # choose action: use policy if given else sampling
            action = (
                policy(observations)
                if policy is not None
                else self.env.action_space.sample()
            )

            obs, reward, terminated, truncated, info = self.env.step(action=action)
            if self.render_mode == "human":
              time.sleep(0.02) # Clear 50 FPS target cap

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

        # Store for aggregate stats
        self._episode_rewards.append(total_reward)
        self._episode_lengths.append(step)

        return {
            "total_reward": total_reward,
            "steps": step,
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
        }

    def stats(self) -> dict:
        """Aggregate stats from the last run_episodes() call."""
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

    def env_info(self) -> dict:
        """Useful metadata about the environment."""
        return {
            "env_name": self.env_name,
            "render_mode": self.render_mode,
            "observation_space": str(self.observation_space),
            "action_space": str(self.action_space),
            "max_steps": self.max_steps,
        }

    def close(self) -> None:
        """Always call this (or use as a context manager) to release resources."""
        self.env.close()

    def run_episodes(
        self,
        n: int = 5,
        policy: Optional[Callable[[np.ndarray], int]] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> list[dict]:
        """
        Run N episodes and print a summary.

        Parameters
        ----------
        n       : number of episodes
        policy  : same as run_episode — None means random agent
        seed    : base seed; episode i gets seed + i for reproducibility
        verbose : print per-episode and summary stats
        """
        self._episode_rewards.clear()
        self._episode_lengths.clear()

        results = []
        for i in range(n):
            ep_seed = None if seed is None else seed + i
            result = self.run_episode(policy=policy, seed=ep_seed)
            results.append(result)

            if verbose:
                label = "random" if policy is None else "policy"
                print(
                    f"  Episode {i+1:>3}/{n} [{label}] "
                    f"reward={result['total_reward']:7.1f}  "
                    f"steps={result['steps']}"
                )

        if verbose:
            self._print_stats()

        return results

    # ── Private helpers ──────────────────────────────────────────────────
    def _print_stats(self) -> None:
        s = self.stats()
        print(
            f"\n  ── Summary ({s['episodes']} episodes) ──────────────────\n"
            f"  mean reward : {s['mean_reward']:.2f} ± {s['std_reward']:.2f}\n"
            f"  min / max   : {s['min_reward']:.1f} / {s['max_reward']:.1f}\n"
            f"  mean steps  : {s['mean_length']:.1f}\n"
        )

    def __repr__(self) -> str:
        return (
            f"PlaygroundEnv(env={self.env_name!r}, "
            f"render={self.render_mode!r}, "
            f"episodes_run={len(self._episode_rewards)})"
        )


if __name__ == "__main__":

    # ── 1. Basic usage — random agent, headless ──────────────────────────
    print("=" * 55)
    print("1.  Random agent on CartPole-v1")
    print("=" * 55)

    with PlaygroundEnv("CartPole-v1") as pg:
        print(pg.env_info())
        pg.run_episodes(n=5, seed=42)
        print("Raw stats dict:", pg.stats())

    # ── 2. runnable_env() — hand off the raw env to external code ─────────
    print("=" * 55)
    print("2.  runnable_env() — raw gym access")
    print("=" * 55)

    pg = PlaygroundEnv("CartPole-v1")
    raw = pg.runnable_env()
    obs, _ = raw.reset()
    print(f"  obs shape={obs.shape}  action_space={raw.action_space}")
    pg.close()

    # ── 3. Custom policy — always push RIGHT (action=1) ───────────────────
    print("=" * 55)
    print("3.  Custom policy: always push RIGHT")
    print("=" * 55)

    def always_right(obs: np.ndarray) -> int:
        return 1

    with PlaygroundEnv("CartPole-v1", max_steps=200) as pg:
        pg.run_episodes(n=5, policy=always_right, seed=0)

    # ── 4. Single episode, inspect trajectory ────────────────────────────
    print("=" * 55)
    print("4.  Inspect single-episode trajectory")
    print("=" * 55)

    with PlaygroundEnv("CartPole-v1") as pg:
        ep = pg.run_episode(seed=7)
        print(f"  total_reward={ep['total_reward']}  steps={ep['steps']}")
        print(f"  first obs : {ep['observations'][0]}")
        print(f"  actions   : {ep['actions'][:10]} ...")

    # ── 5. Different environment — MountainCar ────────────────────────────
    print("=" * 55)
    print("5.  MountainCar-v0 random agent")
    print("=" * 55)

    with PlaygroundEnv("MountainCar-v0", max_steps=300) as pg:
        pg.run_episodes(n=3, seed=1)
    
    # ----- 6. With render mode = rgb_array 
    print("=" * 55)
    print("6.  With render mode = human ")
    print("=" * 55)
    with PlaygroundEnv("CartPole-v1", render_mode="human") as cartpole_env:
      ep = cartpole_env.run_episode(seed=99)
      print(f"  total_reward={ep['total_reward']}  steps={ep['steps']}")
      print(f"  first obs : {ep['observations'][0]}")
      print(f"  actions   : {ep['actions'][:10]} ...")