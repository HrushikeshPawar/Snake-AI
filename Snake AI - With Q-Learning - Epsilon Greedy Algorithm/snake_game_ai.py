import pygame
import random  # for random food placement generation
import numpy as np
from helper import Direction, Point, VISION4, Vision

# Initialize pygame
pygame.init()
font = pygame.font.Font('Lora-Regular.ttf', 20)

# Define Constants
BLOCKSIZE   = 20
GRID_H      = 10
GRID_W      = 10
SPEED       = 20
BLACK       = (0, 0, 0)
WHITE       = (255, 255, 255)
RED         = (255, 0, 0)
GREEN       = (0, 255, 0)
BLUE        = (0, 0, 255)
BLUE2       = (0, 100, 255)
VISION      = VISION4


# Define Snake Class
class SnakeGame:

    def __init__(self, width: int = BLOCKSIZE * GRID_W, height: int = BLOCKSIZE * GRID_H, n_games: int = 0, speed: int = 20) -> None:

        # Set Window Size
        self.width = width
        self.height = height
        self.n_games = n_games
        self.speed = speed

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
        self.tail_direction = Direction.RIGHT

        # Place food
        self.food = None
        self._place_food()

        # Initialize score
        self.score  = 0

        # Initialize Frame Iteration Count
        self.frame_iteration = 0
        # self._update_vision()

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

    def _moved_closer_to_food(self) -> bool:
        """
        Check if the snake is closer to the food than the previous frame
        """
        return (self.head.x - self.food.x) + (self.head.y - self.food.y) < (self.snake[1].x - self.food.x) + (self.snake[1].y - self.food.y)

    # Check if the point is inside the grid
    def _is_inside_grid(self, point : Point) -> bool:
        """
        Check if the point is inside the grid
        """
        return point.x >= 0 and point.x < self.width - BLOCKSIZE and point.y >= 0 and point.y < self.height - BLOCKSIZE

    # Check if food is in the given direction
    def _is_food_in_direction(self, direction: Direction) -> bool:
        """
        Check if food is in the given direction
        """
        p = self.head.copy()
        p += direction.value
        while self._is_inside_grid(p):
            if p == self.food:
                return True
            p += direction.value

        return False

    # Check if the snake's body is in the given direction
    def _is_body_in_direction(self, direction: Direction) -> bool:
        """
        Check if body of the snake is in the given direction
        """
        p = self.head.copy()
        p += direction.value
        while self._is_inside_grid(p):
            if p in self.snake[1:]:
                return True
            p += direction.value

        return False

    # Find distance from head to wall in given direction
    def _distance_to_wall(self, direction: Direction) -> float:
        """
        Find distance from head to wall in given direction
        """
        p = self.head.copy()
        p += direction.value
        distance = 0
        while self._is_inside_grid(p):
            distance += 1
            p += direction.value
        if distance == 0:
            return 1.0
        return 1 / distance

    # Update Vision of the snake
    def _update_vision(self) -> None:
        """
        Update the vision of the snake
        """
        self.vision = []
        for dir in VISION:
            v = Vision(
                direction=dir,
                dist_to_wall=self._distance_to_wall(Direction(dir)),
                is_food_visible=self._is_food_in_direction(Direction(dir)),
                is_self_visible=self._is_body_in_direction(Direction(dir))
            )
            self.vision.append(v)

    def play_step(self, action) -> tuple:
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
        if self._is_collision() or self.frame_iteration > GRID_H * GRID_W:
            game_over   = True
            reward      = -100
            self.frame_iteration = 0
            return reward, game_over, self.score

        # Step 4 - Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward     = 10
            # reward     = 100
            self._place_food()
        else:

            if self._moved_closer_to_food():
                reward = 1
            else:
                reward = -10
            self.snake.pop()

        # Step 5 - Update UI and clock
        self._update_ui()
        self.clock.tick(self.speed)

        # Step 6 - Return Game Over and Score
        return (reward, game_over, self.score)

    def _get_input(self) -> None:

        for event in pygame.event.get():

            # CHECK IF USER QUITS
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

    def _get_directions(self, point: Point) -> Direction:

        directions = {
            Point(0, -1): Direction.UP,
            Point(0, 1): Direction.DOWN,
            Point(-1, 0): Direction.LEFT,
            Point(1, 0): Direction.RIGHT
        }
        return directions[point]

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

        # Update the tail direction
        if len(self.snake) == 1:
            self.tail_direction = self.direction
        else:
            p = self.snake[-2] - self.snake[-1]
            self.tail_direction = self._get_directions(Point(p.x // BLOCKSIZE, p.y // BLOCKSIZE))

    def _is_collision(self, pt: Point = None) -> bool:
        """
        Check if the snake has collided with itself or the wall
        Updated this function to check for dangers for next step as well
        """
        if pt is None:
            pt = self.head

        if pt.x < 0 or pt.x > self.width - BLOCKSIZE or pt.y < 0 or pt.y > self.height - BLOCKSIZE:
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
