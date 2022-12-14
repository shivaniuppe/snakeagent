import gym
import torch.cuda

import envs
import os

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from utils.CustomCallback import TensorboardCallback
from utils.CustomBaseFeaturesExtractor import CustomCNN

env = make_vec_env('SnakeEnv-v0', n_envs=4, env_kwargs={'window_size': 256, 'block_size': 16})

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=64)
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = A2C(
    policy='CnnPolicy',
    env=env,
    verbose=1,
    learning_rate=5e-5,
    gae_lambda=0.99,
    n_steps=64,
    ent_coef=0.001,
    normalize_advantage=True,
    tensorboard_log='../../logs/a2c1/log',
    policy_kwargs=policy_kwargs,
    create_eval_env=True,
    device=device
)

if os.path.exists('a2c1.zip'):
    model = A2C.load('a2c1', env, verbose=1, device=device)


checkpoint_callback = CheckpointCallback(save_freq=500000, save_path='../../logs/a2c1/chck/')
eval_callback = EvalCallback(env, best_model_save_path="../../logs/a2c1/best/", eval_freq=500)
tensorboard_callback = TensorboardCallback(env)
callback_list = CallbackList([checkpoint_callback, eval_callback, tensorboard_callback])

model.learn(
    total_timesteps=2000000,
    callback=callback_list,
    tb_log_name='run',
    log_interval=100,
    eval_log_path='../logs/a2c1/eval/',
    reset_num_timesteps=False
)

model.save('a2c1')
print('Model Saved')

mean_reward, std_reward = evaluate_policy(
    model=model,
    env=model.get_env(),
    n_eval_episodes=10
)
print(mean_reward, std_reward)
