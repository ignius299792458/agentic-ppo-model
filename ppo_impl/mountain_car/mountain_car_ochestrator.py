import numpy as np
import torch
import torch.nn as nn
from multiprocessing import Process

from ppo_impl.config import Config
from ppo_impl.actor import Actor
from ppo_impl.critic import Critic
from ppo_impl.env import PlaygroundEnv
from ppo_impl.utils import printHeaderInfo


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    # returns advantages (T,) and returns (T,)
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    returns = adv + np.asarray(values, dtype=np.float32)
    return adv, returns


def train(cfg: Config):
    printHeaderInfo(f"Running Env: {cfg.env_name} with training policy (PPO)")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    env = PlaygroundEnv(cfg.env_name, cfg.render_mode, cfg.max_steps)
    actor = Actor(
        env.obs_dim, env.act_dim, cfg.hidden_size, cfg.n_hidden, cfg.verbose
    ).to(device)
    critic = Critic(env.obs_dim, cfg.hidden_size, cfg.n_hidden, cfg.verbose).to(device)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    obs, _ = env.reset(seed=cfg.seed)
    ep_reward, ep_history = 0.0, []

    N = cfg.rollout_steps
    for it in range(cfg.total_iterations):
        obs_buf = np.zeros((N, env.obs_dim), dtype=np.float32)
        act_buf = np.zeros(N, dtype=np.int64)
        logp_buf = np.zeros(N, dtype=np.float32)
        rew_buf = np.zeros(N, dtype=np.float32)
        val_buf = np.zeros(N, dtype=np.float32)
        done_buf = np.zeros(N, dtype=np.float32)

        # ---- rollout: env feeds both actor and critic ----
        for t in range(N):
            action, logp = actor.act(obs)  # obs -> action, log_prob
            value = critic.value(obs)  # same obs -> V(s)
            if cfg.verbose:
                print(f"Rollout: {t} --> action: {action}, value: {value}")
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs_buf[t], act_buf[t], logp_buf[t] = obs, action, logp
            rew_buf[t], val_buf[t], done_buf[t] = reward, value, float(done)

            ep_reward += reward
            obs = next_obs
            if done:

                if cfg.verbose:
                    print(f"Rollout (Done): {t}, ep_reward:{ep_reward}")

                ep_history.append(ep_reward)
                ep_reward = 0.0
                obs, _ = env.reset()

        last_value = critic.value(obs)
        adv, ret = compute_gae(
            rew_buf, val_buf, done_buf, last_value, cfg.gamma, cfg.gae_lambda
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.as_tensor(obs_buf, device=device)
        act_t = torch.as_tensor(act_buf, device=device)
        logp_old_t = torch.as_tensor(logp_buf, device=device)
        adv_t = torch.as_tensor(adv, device=device)
        ret_t = torch.as_tensor(ret, device=device)

        # ---- ppo update: reuse the batch for several epochs ----
        idx = np.arange(N)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for s in range(0, N, cfg.minibatch_size):
                b = idx[s : s + cfg.minibatch_size]
                logp, entropy = actor.evaluate_actions(obs_t[b], act_t[b])
                ratio = torch.exp(logp - logp_old_t[b])
                clipped = (
                    torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t[b]
                )
                actor_loss = -torch.min(ratio * adv_t[b], clipped).mean()
                loss_a = actor_loss - cfg.entropy_coef * entropy.mean()

                opt_actor.zero_grad()
                loss_a.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                opt_actor.step()

                value = critic(obs_t[b])
                loss_c = ((value - ret_t[b]) ** 2).mean()

                opt_critic.zero_grad()
                loss_c.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                opt_critic.step()

        recent = ep_history[-10:] if ep_history else [0.0]
        print(
            f"PPO Training : iter {it + 1:>3}/{cfg.total_iterations}  "
            f"reward(last10)={np.mean(recent):7.1f}  episodes={len(ep_history)}"
        )

    env.close()
    return actor, critic


def without_train(cfg: Config):
    printHeaderInfo(f"Running Env: {cfg.env_name} without training policy.")
    env = PlaygroundEnv(env_name=cfg.env_name, render_mode=cfg.render_mode)
    env.run_episodes(
        n=cfg.episodes,
        policy=None,
        seed=cfg.seed,
        sleep_time=cfg.sleep_between_steps,
    )


if __name__ == "__main__":

    p1 = Process(target=train, args=(Config(),))
    p2 = Process(target=without_train, args=(Config(),))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
