import numpy as np


def evaluate(model, env, num_episodes=10, display_on_screen=True):
    episode_count = 0
    rew = 0
    scores = []
    steps = []
    rewards = []

    obs = env.reset()
    while episode_count < num_episodes:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        rew += reward
        if display_on_screen:
            env.render()

        if done:
            score, step_count = env.get_logging_details()
            obs = env.reset()
            episode_count += 1

            scores.append(score)
            steps.append(step_count)
            rewards.append(rew)
            rew = 0

    return np.array(scores), np.array(steps), np.array(rewards)
