import pygame
from enum import Enum
from collections import namedtuple
import random  # for random food placement generation


# Initialize pygame
pygame.init()
font = pygame.font.Font('Lora-Regular.ttf', 20)

# Define COnstants
BLOCKSIZE   = 20
SPEED       = 10
BLACK       = (0, 0, 0)
WHITE       = (255, 255, 255)
RED         = (255, 0, 0)
GREEN       = (0, 255, 0)
BLUE        = (0, 0, 255)
BLUE2       = (0, 100, 255)


# Define Direction Class
class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


# Define Point as NamedTuple
Point = namedtuple('Point', 'x, y')


# Define Snake Class
class SnakeGame:

    def __init__(self, width: int = 640, height: int = 480) -> None:

        # Set Window Size
        self.width = width
        self.height = height

        # Initialize Window
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()

        # Initialize Game State
        # Starting point of sanke is set to middle of the screen and length is set to 3
        self.direction  = Direction.RIGHT
        self.head = Point(self.width // 2, self.height // 2)
        self.snake      = [self.head, Point(self.head.x - BLOCKSIZE, self.head.y), Point(self.head.x - 2 * BLOCKSIZE, self.head.y)]
        self.score      = 0
        self.food       = None
        self._place_food()

    def _place_food(self) -> None:
        """
        Place food randomly on the screen
        """
        x           = random.randint(0, (self.width - BLOCKSIZE) // BLOCKSIZE) * BLOCKSIZE
        y           = random.randint(0, (self.height - BLOCKSIZE) // BLOCKSIZE) * BLOCKSIZE
        self.food   = Point(x, y)

        # If food is placed on snake, place food again
        if self.food in self.snake:
            self._place_food()

    def play_step(self) -> None:
        """
        Play one step of the game
        """

        # Step 1 - Get input from the User
        self._get_input()

        # Step 2 - Move the Snake
        self._move_snake(self.direction)
        self.snake.insert(0, self.head)

        # Step 3 - Check if Game Over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score

        # Step 4 - Place new food or just move
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        # Step 5 - Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # Step 6 - Return Game Over and Score
        return game_over, self.score

    def _get_input(self) -> None:

        for event in pygame.event.get():

            # CHECK IF USER QUITS
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            # CHECK IF USER PRESSES A KEY
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
                elif event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT

    def _move_snake(self, direction: Direction) -> None:

        # Get current head position
        x = self.head.x
        y = self.head.y

        # Move the head
        if direction == Direction.UP:
            y -= BLOCKSIZE
        elif direction == Direction.DOWN:
            y += BLOCKSIZE
        elif direction == Direction.LEFT:
            x -= BLOCKSIZE
        elif direction == Direction.RIGHT:
            x += BLOCKSIZE

        # Update the head
        self.head = Point(x, y)

    def _is_collision(self) -> bool:
        """
        Check if the snake has collided with itself or the wall
        """
        if self.head.x < 0 or self.head.x >= self.width - BLOCKSIZE or self.head.y < 0 or self.head.y >= self.height - BLOCKSIZE:
            return True

        # Check if snake has collided with itself
        for point in self.snake[1:]:
            if self.head == point:
                return True

        return False

    def _update_ui(self) -> None:
        """
        Update the UI after every step
        """
        # Clear the screen
        self.display.fill(BLACK)

        # Draw the food
        pygame.draw.rect(self.display, RED, (self.food.x, self.food.y, BLOCKSIZE, BLOCKSIZE))

        # Draw the snake
        for point in self.snake:
            pygame.draw.rect(self.display, BLUE, (point.x, point.y, BLOCKSIZE, BLOCKSIZE))
            pygame.draw.rect(self.display, BLUE2, (point.x + 4, point.y + 4, 12, 12))

        # Draw the score
        text = font.render(f'Score: {self.score}', True, WHITE)
        self.display.blit(text, (1, 1))

        # Update the display
        pygame.display.update()


if __name__ == '__main__':
    game = SnakeGame()

    while True:
        game_over, score = game.play_step()

        if game_over:
            print(f'Game Over! Your score is {score}')
            break
