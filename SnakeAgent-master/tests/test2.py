import gym
from stable_baselines3.common.env_checker import check_env
import envs


env = gym.make('SnakeEnv-v0')
# setting up the seed for reproducibility
env.seed(42)
env.action_space.seed(42)

check_env(env, warn=True)
print('Environment conforms to Stable Baselines and Gym Requirements')
