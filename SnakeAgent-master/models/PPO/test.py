import gym
import numpy as np
import envs

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from utils.evaluate import evaluate

env = Monitor(gym.make('SnakeEnv-v0', window_size=256, block_size=16))
model = PPO.load('ppo2', env=env)
env.seed(42)
env.action_space.seed(42)

obs = env.reset()
scores, steps, rewards = evaluate(model, env)
print('Mean Score:', np.mean(scores), 'Mean Steps:', np.mean(steps), 'Mean Rewards:', np.mean(rewards))
env.close()
