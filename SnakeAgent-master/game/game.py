from snake import Direction, Fruit, Snake, Wall
import pygame

pygame.init()
pygame.display.init()
window = pygame.display.set_mode((256, 256))
pygame.display.set_caption('Snake')

snake = Snake(16, 256, (0, 0, 255))
walls = Wall(16, 256, (128, 128, 128), snake.body)
fruit = Fruit(16, 256, (255, 0, 0), snake.body, walls.segments)

run = True
score = 0
while run:
    pygame.time.delay(60)

    for event in pygame.event.get():
        run = not (event.type == pygame.QUIT)

    key = pygame.key.get_pressed()
    if key[pygame.K_LEFT]:
        snake.change_direction(Direction.LEFT)
    elif key[pygame.K_DOWN]:
        snake.change_direction(Direction.DOWN)
    elif key[pygame.K_RIGHT]:
        snake.change_direction(Direction.RIGHT)
    elif key[pygame.K_UP]:
        snake.change_direction(Direction.UP)

    snake.move()
    if snake.eat_check(fruit):
        score += 1
        if score % 3 == 0 and score > 0:
            walls.add_segment(body=snake.body)

        fruit.reset(body=snake.body, walls=walls.segments)

    if snake.is_dead(walls.segments):
        font = pygame.font.SysFont('ariel', 40, True)
        text = font.render('Game Over', True, (0, 255, 255))
        text_rect = text.get_rect(center=(128, 128))
        window.blit(text, text_rect)

        print(score)
        score = 0

        pygame.display.update()
        pygame.time.delay(2000)

        snake.reset()
        walls.reset(snake.body)
        fruit.reset(snake.body, walls.segments)

    window.fill((0, 0, 0))
    snake.render(window)
    walls.render(window)
    fruit.render(window)
    pygame.display.flip()
