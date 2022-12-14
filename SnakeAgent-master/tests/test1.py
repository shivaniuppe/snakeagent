import gym
import envs
import random


# random user-defined policy
def policy(observation):
	return random.randint(0, 3)


env = gym.make('SnakeEnv-v0', block_size=16, window_size=256)
# setting up the seed for reproducibility
env.seed(42)
env.action_space.seed(42)

observation = env.reset()

for _ in range(100):
	env.render()
	action = policy(observation)
	
	observation, reward, done, info = env.step(action)
	if done:
		observation = env.reset()

env.close()
