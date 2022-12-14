import gym
import envs

env = gym.make('SnakeEnv-v0')
obs = env.reset()

env.step(0)
env.reset()
print('Test done')
