"""
CartPole-v1 training and evaluation harness for autoresearch.
This file is READ-ONLY — the agent edits agent.py only.

Prints key: value pairs for metric extraction.
Usage: uv run train.py
"""

import os
import time

import gymnasium as gym
import numpy as np

from agent import PPOAgent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TIME_BUDGET = int(os.environ.get("AUTORESEARCH_TIME_BUDGET", 180))
EVAL_EPISODES = 100
ROLLOUT_STEPS = 2048

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

agent = PPOAgent(obs_dim, act_dim)

print(f"Training PPO on CartPole-v1 (time budget: {TIME_BUDGET}s)")
print(f"  obs_dim={obs_dim}, act_dim={act_dim}")

t_start = time.time()
iteration = 0

while True:
    elapsed = time.time() - t_start
    if elapsed >= TIME_BUDGET:
        break

    iteration += 1
    rollout = agent.collect_rollout(env, steps=ROLLOUT_STEPS)
    loss = agent.update(rollout)

    # Quick eval every iteration
    eval_rewards = []
    for _ in range(10):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _, _ = agent.network.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        eval_rewards.append(total_reward)

    mean_r = np.mean(eval_rewards)
    print(f"Iteration {iteration}: mean_reward={mean_r:.1f} loss={loss:.4f} elapsed={elapsed:.0f}s")

env.close()

# ---------------------------------------------------------------------------
# Final Evaluation
# ---------------------------------------------------------------------------

print(f"\nRunning final evaluation ({EVAL_EPISODES} episodes)...")
eval_env = gym.make("CartPole-v1")
final_rewards = []

for _ in range(EVAL_EPISODES):
    obs, _ = eval_env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _, _ = agent.network.get_action(obs)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated
    final_rewards.append(total_reward)

eval_env.close()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

total_time = time.time() - t_start
mean_reward = np.mean(final_rewards)
std_reward = np.std(final_rewards)

print("---")
print(f"mean_reward:      {mean_reward:.1f}")
print(f"std_reward:       {std_reward:.1f}")
print(f"min_reward:       {np.min(final_rewards):.1f}")
print(f"max_reward:       {np.max(final_rewards):.1f}")
print(f"training_seconds: {total_time:.1f}")
print(f"num_iterations:   {iteration}")
