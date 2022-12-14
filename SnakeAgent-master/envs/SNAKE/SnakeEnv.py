import cv2
import pygame
import gym
from gym import spaces
from game.snake import *
import numpy as np


class SnakeEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 12
    }

    colors = {
        'black': (0, 0, 0),
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'white': (255, 255, 255),
        'gray': (128, 128, 128)
    }

    def __init__(self, window_size: int = 512, block_size: int = 16):
        """
        :param window_size: type (int): size of the game window

        :param block_size: type (int): size of each block in the game window

        Other objects:
        snake: a Snake object
        fruit: a Fruit object
        score: an attribute to maintain score
        """
        self.step_count = 0
        self.distance = None
        self.window_size = window_size
        self.block_size = block_size
        self.score = 0
        self.last_eaten = 0

        self.snake = Snake(block_size, window_size, color=self.colors['blue'])
        self.walls = Wall(block_size, window_size, color=self.colors['gray'], body=self.snake.body)
        self.fruit = Fruit(block_size, window_size, color=self.colors['red'], body=self.snake.body, walls=self.walls.segments)

        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)

        # window instance and clock are kept as None before initialising for the first time
        self.window = None
        self.clock = None

        self._action_to_direction = {
            0: Direction.RIGHT,
            1: Direction.UP,
            2: Direction.LEFT,
            3: Direction.DOWN
        }

    def step(self, action: int):
        """
        Performs a step according to the action provided.

        :param action: type: (int): An action value from 0, 1, 2 or 3

        :return: a four tuple consisting of- obs: next observation, reward: current reward, done: done condition and
        info: additional info
        """
        direction = self._action_to_direction[action]
        self.snake.change_direction(inp_direction=direction)
        self.snake.move()
        self.step_count += 1

        done = self.snake.is_dead(walls=self.walls.segments) or self.step_count - self.last_eaten >= 300

        if done:
            # negative reward for dying
            reward = -1
            # higher negative reward for entering into a loop
            if self.step_count - self.step_count >= 300:
                reward -= 5
            self.log_score()
        else:
            reward = self._get_reward()

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, done, info

    def _get_reward(self):
        reward = 0
        # adding reward according to direction moved
        reward += self.get_changed_distance() * 0.1

        # reward if food is eaten
        if self.snake.eat_check(fruit=self.fruit):
            # update the last eaten and distance from food for the other negative reward measures
            self.last_eaten = self.step_count
            self.distance = None

            # positive reward for food being eaten
            reward += 1 + (self.score / 10)

            # updating the score and adding another wall segment if the score is a multiple of 5
            self.score += 1

            if self.score > 0 and self.score % 3 == 0:
                self.walls.add_segment(self.snake.body)

            self.fruit.reset(body=self.snake.body, walls=self.walls.segments)

        # increasing negative reward if food is not eaten for a few steps
        steps_since_food = self.step_count - self.last_eaten
        divisor = 15 * self.snake.length
        reward -= 0.1 * (steps_since_food // divisor)

        # reward according to danger ahead
        reward -= 0.1 if self.snake.check_danger_ahead(walls=self.walls.segments) else 0

        return reward

    def get_changed_distance(self):
        head = np.array(self.snake.body[-1])
        fruit = np.array(self.fruit.pos)

        old_distance = self.distance
        self.distance = np.power(np.sum(np.power((fruit-head)//self.block_size, 2)), 0.5)

        if old_distance is not None:
            return old_distance - self.distance
        else:
            return 0

    def reset(self):
        """
        Resets the environment by resetting the score, snake and fruit objects

        :return: obs: next observation
        """
        self.score = 0
        self.step_count = 0
        self.last_eaten = 0
        self.distance = None
        self.snake.reset()
        self.walls.reset(body=self.snake.body)
        self.fruit.reset(body=self.snake.body, walls=self.walls.segments)

        return self._get_obs()

    def _get_info(self):
        """
        Returns additional info which can be used by the reset or step methods

        :return: a dictionary containing the snake's length, it's direction, it's head position and the fruit position
        """
        return {
            'snake_len': self.snake.length,
            'snake_direction': self.snake.direction,
            'head_pos': self.snake.body[-1],
            'fruit_pos': self.fruit.pos
        }

    def _get_obs(self):
        window = self.render(mode='rgb_array')
        window = cv2.resize(window, (64, 64))

        window = window.astype(np.uint8)
        window = np.transpose(
            window,
            axes=(2, 0, 1)
        )

        return window

    def log_score(self):
        """
        A function to log the score
        :return: None
        """
        print('Score: ', self.score, 'Step Count:', self.step_count)

    def get_logging_details(self):
        return self.score, self.step_count

    def render(self, mode='human'):
        """
        Renders the snake and the fruit on the window. Depending on the mode of rendering, creates a window to display
        the output in real-time

        :param mode: mode of rendering; can be one of two values: 'human' or 'rgb_array'

        :return: returns the rbg array representing the game window when used with mode='rgb_array' otherwise simply displays the window
        """
        surface = pygame.Surface((self.window_size, self.window_size))
        surface.fill(self.colors['black'])

        self.fruit.render(surface)
        self.snake.render(surface)
        self.walls.render(surface)

        if mode == 'human':
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.window.blit(surface, surface.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata['render_fps'])
        else:  # 'rgb_array'
            surface = np.array(pygame.surfarray.pixels3d(surface), dtype=np.uint8)
            return surface
