import pygame
import numpy as np
from enum import Enum


class Direction(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class Fruit:
    def __init__(self, block_size: int, window_size: int, color: tuple, body: list, walls: list) -> None:
        self.pos = []
        self.block_size = block_size
        self.window_size = window_size
        self.color = color

        self.reset(body=body, walls=walls)

    def reset(self, body: list, walls: list) -> None:
        num_blocks = self.window_size//self.block_size

        pos = np.random.randint(1, num_blocks-1, size=2, dtype='int') * self.block_size
        body = np.array(body)
        walls = np.array(walls)

        if len(walls) != 0:
            while (pos == body).all(axis=1).any() or (pos == walls).all(axis=1).any():
                pos = np.random.randint(1, num_blocks-1, size=2, dtype='int') * self.block_size

        else:
            while (pos == body).all(axis=1).any():
                pos = np.random.randint(1, num_blocks-1, size=2, dtype='int') * self.block_size

        self.pos = tuple(pos)

    def render(self, surface) -> None:
        pygame.draw.rect(
            surface,
            self.color,
            (self.pos[0], self.pos[1], self.block_size, self.block_size)
        )


class Snake:
    length = None
    direction = None
    body = []

    def __init__(self, block_size: int, window_size: int, color: tuple):

        self.block_size = block_size
        self.window_size = window_size
        self.color = color

        # a dictionary to convert direction to appropriate step size in the x and y direction
        self.direction_to_step = {
            Direction.RIGHT: np.array((self.block_size, 0)),
            Direction.UP: np.array((0, -self.block_size)),
            Direction.LEFT: np.array((-self.block_size, 0)),
            Direction.DOWN: np.array((0, self.block_size))
        }

        self.reset()

    def reset(self):
        assert self.window_size >= 40

        self.length = 2
        self.direction = Direction.DOWN
        num_blocks = self.window_size//self.block_size

        self.body = []

        # adding the tail part of the snake
        x_pos = (num_blocks//2) * self.block_size
        y_pos = self.block_size * 2
        self.body.append((x_pos, y_pos))

        # adding the head part of the snake
        y_pos += self.block_size
        self.body.append((x_pos, y_pos))

    def render(self, surface) -> None:
        for segment in self.body:
            pygame.draw.rect(
                surface,
                self.color,
                (segment[0], segment[1], self.block_size, self.block_size)
            )

    def change_direction(self, inp_direction: Direction) -> None:
        # if this value is 1, then it means inp_direction is not opposite of current direction.
        if (self.direction.value - inp_direction.value) % 2:
            self.direction = inp_direction

    def move(self) -> None:
        head = np.array(self.body[-1])
        step = self.direction_to_step[self.direction]

        next_head = tuple(head + step)
        self.body.append(next_head)
        if self.length < len(self.body):
            self.body.pop(0)

    def eat_check(self, fruit: Fruit) -> bool:
        head = self.body[-1]

        if head[0] == fruit.pos[0] and head[1] == fruit.pos[1]:
            self.length += 1
            return True
        return False

    def bite_check(self) -> bool:
        head = np.array(self.body[-1], dtype='int')
        tail = np.array(self.body[:-1], dtype='int')

        # if head is equal to 'all' the values of 'any' segment of a tail, return True
        if (head == tail).all(axis=1).any():
            return True
        return False

    def border_check(self) -> bool:
        head = self.body[-1]

        if head[0] <= 0 or head[1] <= 0 or head[0] >= self.window_size or head[1] >= self.window_size:
            return True
        return False

    def wall_check(self, walls: list) -> bool:
        head = np.array(self.body[-1])
        walls = np.array(walls)

        if len(walls) != 0 and (head == walls).all(axis=1).any():
            return True
        return False

    def is_dead(self, walls: list) -> bool:
        return self.bite_check() or self.border_check() or self.wall_check(walls)

    def check_danger_ahead(self, walls: list) -> bool:
        head = np.array(self.body[-1], dtype='int')
        next_step = head + self.direction_to_step[self.direction]
        tail = np.array(self.body[:-1], dtype='int')
        walls = np.array(walls, dtype='int')

        if len(walls) != 0 and (next_step == walls).all(axis=1).any():
            return True
        elif next_step[0] <= 0 or next_step[1] <= 0 or next_step[0] >= self.window_size or next_step[1] >= self.window_size:
            return True
        elif (next_step == tail).all(axis=1).any():
            return True
        return False


class Wall:
    segments = []
    length = None
    segment_length = None

    def __init__(self, block_size: int, window_size: int, color: tuple, body: list):
        self.block_size = block_size
        self.window_size = window_size
        self.color = color

        self.reset(body=body)

    def reset(self, body: list) -> None:
        self.length = 0
        self.segments = []

        for i in range(self.length):
            self.add_segment(body)

    def add_segment(self, body) -> None:
        # Note: a segment is two blocks large

        body = np.array(body, dtype='int')
        num_blocks = self.window_size // self.block_size
        pos = np.random.randint(1, num_blocks - 1, size=2, dtype='int') * self.block_size

        direction = np.random.randint(0, 2)
        if direction == 0:  # right
            step = np.array((self.block_size, 0))
        else:  # down
            step = np.array((0, self.block_size))

        new_pos = pos + step

        # if pos or new_pos is equal to any segment of the body, we generate another pos and new_pos
        while (pos == body).all(axis=1).any() or (new_pos == body).all(axis=1).any():
            pos = np.random.randint(1, num_blocks - 1, size=2, dtype='int') * self.block_size
            new_pos = pos + step

        pos = tuple(pos)
        new_pos = tuple(new_pos)
        self.segments.append(pos)
        self.segments.append(new_pos)
        self.length += 1

    def render(self, surface):
        for segment in self.segments:
            pygame.draw.rect(
                surface,
                self.color,
                (segment[0], segment[1], self.block_size, self.block_size)
            )
