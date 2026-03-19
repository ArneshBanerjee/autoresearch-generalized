"""
One-time setup for reinforcement learning example.
Verifies that gymnasium is installed and CartPole-v1 works.

Usage: python prepare.py
"""

import gymnasium as gym

def prepare():
    print("Verifying gymnasium installation...")
    env = gym.make("CartPole-v1")
    obs, _ = env.reset()
    print(f"  CartPole-v1 observation space: {env.observation_space}")
    print(f"  CartPole-v1 action space: {env.action_space}")
    print(f"  Sample observation: {obs}")
    env.close()
    print("\nEnvironment ready!")

if __name__ == "__main__":
    prepare()
