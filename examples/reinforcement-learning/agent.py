"""
PPO agent for CartPole-v1. Agent edits this file.
Baseline: simple PPO with MLP policy and value networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic network."""

    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value

    def get_action(self, obs):
        """Select action and return action, log_prob, value."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            logits, value = self.forward(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item(), value.item()

    def evaluate(self, obs_batch, act_batch):
        """Evaluate actions for PPO update."""
        logits, values = self.forward(obs_batch)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(act_batch)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


class PPOAgent:
    """Proximal Policy Optimization agent."""

    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, epochs=4, batch_size=64, ent_coef=0.01, vf_coef=0.5):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        self.network = ActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def collect_rollout(self, env, steps=2048):
        """Collect a rollout of experience."""
        obs_list, act_list, rew_list, done_list, logp_list, val_list = [], [], [], [], [], []

        obs, _ = env.reset()
        for _ in range(steps):
            action, log_prob, value = self.network.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs_list.append(obs)
            act_list.append(action)
            rew_list.append(reward)
            done_list.append(done)
            logp_list.append(log_prob)
            val_list.append(value)

            obs = next_obs
            if done:
                obs, _ = env.reset()

        # Bootstrap value for last state
        _, _, last_val = self.network.get_action(obs)

        return {
            "obs": np.array(obs_list, dtype=np.float32),
            "actions": np.array(act_list, dtype=np.int64),
            "rewards": np.array(rew_list, dtype=np.float32),
            "dones": np.array(done_list, dtype=np.float32),
            "log_probs": np.array(logp_list, dtype=np.float32),
            "values": np.array(val_list, dtype=np.float32),
            "last_value": last_val,
        }

    def compute_gae(self, rollout):
        """Compute generalized advantage estimation."""
        rewards = rollout["rewards"]
        values = rollout["values"]
        dones = rollout["dones"]
        last_value = rollout["last_value"]

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else values[t + 1]
            next_done = 0.0 if t == n - 1 else dones[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, rollout):
        """PPO update step."""
        advantages, returns = self.compute_gae(rollout)

        obs_t = torch.FloatTensor(rollout["obs"])
        act_t = torch.LongTensor(rollout["actions"])
        old_logp_t = torch.FloatTensor(rollout["log_probs"])
        adv_t = torch.FloatTensor(advantages)
        ret_t = torch.FloatTensor(returns)

        # Normalize advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(obs_t)
        total_loss = 0.0

        for _ in range(self.epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                log_probs, values, entropy = self.network.evaluate(obs_t[idx], act_t[idx])

                # Policy loss (clipped)
                ratio = torch.exp(log_probs - old_logp_t[idx])
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, ret_t[idx])

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()

        return total_loss
