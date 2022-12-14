import gym
import envs
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor

from utils.evaluate import evaluate

env = Monitor(gym.make('SnakeEnv-v0', window_size=256, block_size=16))
env.seed(42)
env.action_space.seed(42)
num_episodes = 100

model1 = A2C.load('../models/a2c/a2c1', env=env)
print('Evaluating A2C')
scores1, steps1, rewards1 = evaluate(model1, env, num_episodes=num_episodes, display_on_screen=False)
print('Average Score:', np.mean(scores1), 'Average Steps:', np.mean(steps1), 'Average Rewards:', np.mean(rewards1))
print('Maximum Score:', np.max(scores1))

_ = env.reset()
model2 = DQN.load('../models/dqn/dqn6', env=env)
print('Evaluating DQN')
scores2, steps2, rewards2 = evaluate(model2, env, num_episodes=num_episodes, display_on_screen=False)
print('Average Score:', np.mean(scores2), 'Average Steps:', np.mean(steps2), 'Average Rewards:', np.mean(rewards2))
print('Maximum Score:', np.max(scores2))

_ = env.reset()
model3 = PPO.load('../models/ppo/ppo2', env=env)
print('Evaluating PPO')
scores3, steps3, rewards3 = evaluate(model3, env, num_episodes=num_episodes, display_on_screen=False)
print('Average Score:', np.mean(scores3), 'Average Steps:', np.mean(steps3), 'Average Rewards:', np.mean(rewards3))
print('Maximum Score:', np.max(scores3))

plt.figure(1)
x = np.array(range(1, num_episodes+1))
plt.plot(x, scores1)
plt.plot(x, scores2)
plt.plot(x, scores3)

plt.xlabel('Episodes')
plt.ylabel('Scores')
plt.grid()
plt.legend(('A2C', 'DQN', 'PPO'))
plt.title('Scores')
plt.savefig('Scores.png')

plt.figure(2)
plt.plot(x, steps1)
plt.plot(x, steps2)
plt.plot(x, steps3)

plt.xlabel('Episodes')
plt.ylabel('Steps')
plt.grid()
plt.legend(('A2C', 'DQN', 'PPO'))
plt.title('Steps')
plt.savefig('Steps.png')

plt.figure(3)
plt.plot(x, rewards1)
plt.plot(x, rewards2)
plt.plot(x, rewards3)

plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.grid()
plt.legend(('A2C', 'DQN', 'PPO'))
plt.title('Rewards')
plt.savefig('Rewards.png')
plt.show()
