import gym
import envs

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from utils.CustomBaseFeaturesExtractor import CustomCNN
from utils.CustomCallback import TensorboardCallback


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)
env = Monitor(gym.make('SnakeEnv-v0', window_size=256, block_size=8))

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./dqn/chck/')
eval_callback = EvalCallback(env, best_model_save_path='./dqn/best/', eval_freq=500)
tensorboard_callback = TensorboardCallback(env)
callback = CallbackList([checkpoint_callback, eval_callback, tensorboard_callback])

model = DQN('CnnPolicy', env, verbose=1, buffer_size=10000, policy_kwargs=policy_kwargs, tensorboard_log='./dqn/logs/')
model.learn(total_timesteps=1000, log_interval=4, callback=callback, tb_log_name='Score_logging_test', eval_log_path='./dqn/eval_logs/')
print('Learned Model')

model.save('dqn_model')
print('Model saved')

# evaluation part
model = DQN.load('dqn_model', env=env)
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(mean_reward, std_reward)
print('Model Evaluation Done')

obs = env.reset()
env.seed(42)
env.action_space.seed(42)
print('Beginning working with training agent')
for _ in range(50):
    env.render()
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()

    if done:
        obs = env.reset()

env.close()
print('Rendering done')