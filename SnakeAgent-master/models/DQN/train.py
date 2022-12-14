import gym
import torch.cuda
from stable_baselines3.common.env_util import make_vec_env

import envs
import os

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from utils.CustomBaseFeaturesExtractor import CustomCNN
from utils.CustomCallback import TensorboardCallback


env = make_vec_env('SnakeEnv-v0', n_envs=8, env_kwargs={'window_size': 256, 'block_size': 16})

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=64)
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DQN(
    policy='CnnPolicy',
    env=env,
    learning_rate=1e-4,
    buffer_size=500000,
    tau=0.9,
    batch_size=64,
    verbose=1,
    policy_kwargs=policy_kwargs,
    tensorboard_log='../../logs/dqn6/log',
    exploration_fraction=1,
    exploration_initial_eps=0.5,
    exploration_final_eps=0.375,
    create_eval_env=True,
    device=device
)
if os.path.exists('dqn6.zip'):
    model = DQN.load(
        'dqn6',
        env=env,
        device=device,
        custom_objects={
            'learning_starts': 0,
            'exploration_initial_eps': 0,
            'exploration_final_eps': 0,
            'exploration_fraction': 1
        }
    )

checkpoint_callback = CheckpointCallback(save_freq=500000, save_path='../../logs/dqn6/chck/')
eval_callback = EvalCallback(env, best_model_save_path="../../logs/dqn6/best/", eval_freq=500)
tensorboard_callback = TensorboardCallback(env)
callback_list = CallbackList([checkpoint_callback, eval_callback, tensorboard_callback])

model.learn(
    total_timesteps=2000000,
    log_interval=100,
    callback=callback_list,
    tb_log_name='run',
    eval_log_path='../../logs/dqn6/eval/',
    reset_num_timesteps=False
)

model.save('dqn6')
print('Model Saved')

mean_reward, std_reward = evaluate_policy(
    model=model,
    env=model.get_env(),
    n_eval_episodes=10
)

print(mean_reward, std_reward)
