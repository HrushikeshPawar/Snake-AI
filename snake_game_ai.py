import pygame
from enum import Enum
from collections import namedtuple
import random  # for random food placement generation
import numpy as np

# Initialize pygame
pygame.init()
font = pygame.font.Font('Lora-Regular.ttf', 20)

# Define Constants
BLOCKSIZE   = 20
SPEED       = 100
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

    def __init__(self, width: int = 1040, height: int = 720, n_games: int = 0) -> None:

        # Set Window Size
        self.width = width
        self.height = height
        self.n_games = n_games

        # Initialize Window
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Snake Game  ({n_games}th Game)")
        self.clock = pygame.time.Clock()

        # Initialize Game State
        self.reset()

    def reset(self) -> None:

        # Starting point of sanke is set to middle of the screen and length is set to 3
        self.direction  = Direction.RIGHT
        self.head       = Point(self.width // 2, self.height // 2)
        self.snake      = [self.head, Point(self.head.x - BLOCKSIZE, self.head.y), Point(self.head.x - 2 * BLOCKSIZE, self.head.y)]

        # Place food
        self.food = None
        self._place_food()

        # Initialize score
        self.score  = 0

        # Initialize Frame Iteration Count
        self.frame_iteration = 0

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

    def play_step(self, action) -> None:
        """
        Play one step of the game
        """

        self.frame_iteration += 1

        # Step 1 - Get input from the User
        self._get_input()

        # Step 2 - Move the Snake
        self._move_snake(action)
        self.snake.insert(0, self.head)

        # Step 3 - Check if Game Over
        game_over = False
        reward    = 0
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over   = True
            reward      = -10
            return reward, game_over, self.score

        # Step 4 - Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward     = 10
            self._place_food()
        else:
            self.snake.pop()

        # Step 5 - Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # Step 6 - Return Game Over and Score
        return reward, game_over, self.score

    def _get_input(self) -> None:

        for event in pygame.event.get():

            # CHECK IF USER QUITS
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

    def _move_snake(self, action) -> None:

        # Define move in terms of [Straight, Left Turn, Right Turn]
        clock_wise  = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx         = clock_wise.index(self.direction)

        # Check the action and move the snake accordingly
        if np.array_equal(action, np.array([1, 0, 0])):
            self.direction = clock_wise[idx]    # Move straight in the same direction
        elif np.array_equal(action, np.array([0, 1, 0])):
            self.direction = clock_wise[(idx + 1) % 4]    # Turn Right
        else:
            self.direction = clock_wise[(idx - 1) % 4]    # Turn Left

        # Get current head position
        x = self.head.x
        y = self.head.y

        # Move the head
        if self.direction == Direction.UP:
            y -= BLOCKSIZE
        elif self.direction == Direction.DOWN:
            y += BLOCKSIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCKSIZE
        elif self.direction == Direction.RIGHT:
            x += BLOCKSIZE

        # Update the head
        self.head = Point(x, y)

    def _is_collision(self, pt: Point = None) -> bool:
        """
        Check if the snake has collided with itself or the wall
        Updated this function to check for dangers for next step as well
        """
        if pt is None:
            pt = self.head

        if pt.x < 0 or pt.x >= self.width - BLOCKSIZE or pt.y < 0 or pt.y >= self.height - BLOCKSIZE:
            return True

        # Check if snake has collided with itself
        for point in self.snake[1:]:
            if pt == point:
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
        pygame.display.set_caption(f"Snake Game  ({self.n_games}th Game)")

        # Update the display
        pygame.display.update()
