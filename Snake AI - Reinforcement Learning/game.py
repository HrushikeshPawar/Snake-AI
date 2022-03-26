import pygame
from settings import settings
from helper import Point, Direction, VISION8, Vision
from random import randint
from time import perf_counter
import os
from glob import glob
from datetime import datetime
import moviepy.editor as mpy


# Initialize the Pygame environment
pygame.init()


# Defining the required Constants
# Game Environment
BLOCKSIZE = settings['block_size']
GRID_H    = settings['grid_height']
GRID_W    = settings['grid_width']
BORDER    = settings['border']
SPEED     = settings['speed']
FONT      = pygame.font.Font(settings['font'], 20)
VISION    = VISION8

# Colors for the Game
BLACK     = settings['black']
WHITE     = settings['white']
GREY      = settings['grey']
RED       = settings['red']
GREEN     = settings['green']
GREEN2    = settings['green2']
BLUE      = settings['blue']
BLUE2     = settings['blue2']


# File Paths
GIF_path          = settings['GIF_path']
Graph_path        = settings['Graph_path']
Model_path        = settings['Model_path']
Checkpoint_path   = settings['Checkpoint_path']
Log_path          = settings['Log_path']


# The Snake Class
class Snake:

    def __init__(self, start_pos: Point = None, start_dir: Direction = None) -> None:

        # Initialize the Snake
        self.head           = start_pos
        self.body           = []
        self.direction      = start_dir
        self.tail_direction = None
        self.length         = 1
        self.vision         = []

    def __str__(self) -> str:
        data  = f'Snake(\n\thead\t  = {self.head},'
        data += f'\n\ttail\t  = {self.body[1:]},'
        data += f'\n\tlength\t  = {self.length},'
        data += f'\n\tdirection\t  = {self.direction},'
        data += f'\n\ttail_direction\t  = {self.tail_direction},'
        data += '\n\tvision\t  :'
        for vis in self.vision:
            data += f'\n\t\t{vis}'

        return data

    __repr__ = __str__

    def _get_directions(self, point: Point) -> Direction:

        directions = {
            Point(0, -1): Direction.UP,
            Point(0, 1): Direction.DOWN,
            Point(-1, 0): Direction.LEFT,
            Point(1, 0): Direction.RIGHT
        }

        return directions[point]

    # Move the Snake
    def move(self, action: Direction) -> None:
        """
        Move the snake in the given direction
        """
        if isinstance(action, Direction):
            self.body.insert(0, self.head)
            self.direction = action
            self.head += self.direction.value * BLOCKSIZE

            # Update the tail direction
            if self.length == 1:
                self.tail_direction = self.direction
            elif self.length == 2:
                p = self.head - self.body[0]

                self.tail_direction = self._get_directions(Point(p.x // BLOCKSIZE, p.y // BLOCKSIZE))
            else:
                p = self.body[-2] - self.body[-1]
                self.tail_direction = self._get_directions(Point(p.x // BLOCKSIZE, p.y // BLOCKSIZE))

            # print(self)
        elif action is None:
            pass
        else:
            raise Exception('Not a valid direction. Pass an element of the Direction Enum')


# The main class for the game
# This class is responsible for the game logic
class SnakeGame:

    def __init__(self, n_game: int = 0, save_gif : bool = False, is_human: bool = True) -> None:

        # Set Window Size
        self.width = GRID_W * BLOCKSIZE
        self.height = GRID_H * BLOCKSIZE
        self.n_games = n_game
        self.speed = SPEED
        self.is_human = is_human

        # Initialize Window
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Snake Game  ({self.n_games}th Game)")
        self.clock = pygame.time.Clock()

        # Initialize Game State
        self.reset()

        # Rest settings
        self.save_gif = save_gif
        self.start_time = perf_counter()
        if self.save_gif:
            pygame.image.save(self.display, os.path.join(GIF_path, "screenshot00.png"))
            self.img_cnt = 1

    # Reset the game state
    def reset(self) -> None:

        # Initialize the Snake
        self.snake           = Snake()
        self.snake.direction = None
        self.snake.length    = 1
        self.snake.tail_direction = None
        self._place_snake()

        # Place food
        self.food = None
        self._place_food()

        # Initialize score
        self.score  = 0

        # Initialize Frame Iteration Count and moves
        self.frame_iteration = 0
        self.moves = 0

        # Initialize the Vision
        self._update_vision()

    # Get a random point on the grid
    def _random_point(self) -> Point:
        """
        Generate a random point
        """
        x = randint(0, GRID_W - 1) * BLOCKSIZE
        y = randint(0, GRID_H - 1) * BLOCKSIZE
        return Point(x, y)

    # Place food on the screen
    def _place_food(self) -> None:
        """
        Place food randomly on the screen
        """
        self.food = self._random_point()

        # If food is placed on snake, place food again
        while self.food in self.snake.body or self.food == self.snake.head:
            self.food = self._random_point()

    # Place snake on Screen
    def _place_snake(self) -> None:
        """
        Place snake on the screen
        """
        self.snake.head = self._random_point()

    # Get input from user if any
    def _get_input(self) -> None:

        for event in pygame.event.get():

            # CHECK IF USER QUITS
            if event.type == pygame.QUIT:
                # pygame.quit()
                # quit()
                raise KeyboardInterrupt

            # CHECK IF USER PRESSES A KEY
            if self.is_human and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.snake.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.snake.direction = Direction.DOWN
                elif event.key == pygame.K_LEFT:
                    self.snake.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.snake.direction = Direction.RIGHT

    # Check if the snake has collided with itself or with wall
    def _is_collision(self, pt: Point = None) -> bool:
        """
        Check if the snake has collided with itself or the wall
        Updated this function to check for dangers for next step as well
        """

        if pt is None:
            pt = self.snake.head

        if pt.x < 0 or pt.x > self.width - BLOCKSIZE or pt.y < 0 or pt.y > self.height - BLOCKSIZE:
            return True
        return (pt in self.snake.body)

    # Create the pygame window
    def _update_ui(self) -> None:
        """
        Update the UI
        """
        # Clear the screen
        self.display.fill(BLACK)

        # Draw the food
        pygame.draw.rect(self.display, RED, (self.food.x, self.food.y, BLOCKSIZE - BORDER, BLOCKSIZE - BORDER))

        # Draw the snake
        pygame.draw.rect(self.display, GREY, (self.snake.head.x, self.snake.head.y, BLOCKSIZE - BORDER, BLOCKSIZE - BORDER))
        pygame.draw.rect(self.display, WHITE, (self.snake.head.x + 4, self.snake.head.y + 4, 12 - BORDER, 12 - BORDER))
        try:
            for point in self.snake.body:
                pygame.draw.rect(self.display, BLUE, (point.x, point.y, BLOCKSIZE - BORDER, BLOCKSIZE - BORDER))
                pygame.draw.rect(self.display, BLUE2, (point.x + 4, point.y + 4, 12 - BORDER, 12 - BORDER))
            point = self.snake.body[-1]
            pygame.draw.rect(self.display, GREEN, (point.x, point.y, BLOCKSIZE - BORDER, BLOCKSIZE - BORDER))
            pygame.draw.rect(self.display, GREEN2, (point.x + 4, point.y + 4, 12 - BORDER, 12 - BORDER))
        except IndexError:
            pass

        # Draw the score
        text = FONT.render(f'Score: {self.score}', True, WHITE)
        self.display.blit(text, (1, 1))

        # Update the display
        pygame.display.update()

    # Save the whole game as gif
    def _save_gif(self) -> None:
        """
        Save the game as a gif
        """
        end_time  = perf_counter()
        duration  = int(end_time - self.start_time)
        print(f'Game Time : {end_time - self.start_time}')
        print('\nMaking GIF...')

        imgs = glob(os.path.join(GIF_path, "*.png"))
        list.sort(imgs, key=lambda x: int(x.split('screenshot')[1].split('.png')[0]))

        txt_path = os.path.join(GIF_path, 'Image_List.txt')
        with open(txt_path, 'w') as file:
            for item in imgs:
                file.write(f"{item}\n")

        FPS = len(imgs) // duration
        now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        n = len(glob(os.path.join(GIF_path, "*.gif")))
        filename = os.path.join(GIF_path, f'Game {n+1} - {GRID_W} X {GRID_H} Grid - {now}.gif')

        clip = mpy.ImageSequenceClip(imgs, fps=FPS)
        clip.write_gif(filename, fps=FPS)
        # clip.write_videofile(filename, fps=FPS)
        # os.startfile(filename)
        print(f'GIF saved to {filename}')

        # Remove all the images
        print(f'Removing {len(imgs)} images...')
        for img in imgs:
            os.remove(img)
        print(f'Removed {len(imgs)} images.\n')

    def _moved_closer_to_food(self) -> bool:
        """
        Check if the snake is closer to the food than the previous frame
        """
        if self.snake.length < 2:
            return 1.0
        return abs(self.snake.head.x - self.food.x) + abs(self.snake.head.y - self.food.y) < (self.snake.body[0].x - self.food.x) + (self.snake.body[0].y - self.food.y)

    # Calculate the reward
    def _calculate_reward(self, food_eaten : bool = False) -> float:
        """
        Calculate the reward
        """
        # return self.frame_iteration + (2 ** self.score + 500 * self.score ** 2.1) - (0.25 * self.frame_iteration ** 1.3 * self.score ** 1.2)
        if food_eaten:
            return 1000.0
        elif self._is_collision():
            return -100.0
        # else:
        #     # return -self.moves * (self.frame_iteration ** -0.02)
        #     return -1.0
        elif self._moved_closer_to_food():
            return 10.0
        else:
            return 1.0

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
        p = self.snake.head.copy()
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
        p = self.snake.head.copy()
        p += direction.value
        while self._is_inside_grid(p):
            if p in self.snake.body:
                return True
            p += direction.value

        return False

    # Find distance from head to wall in given direction
    def _distance_to_wall(self, direction: Direction) -> float:
        """
        Find distance from head to wall in given direction
        """
        p = self.snake.head.copy()
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
        self.snake.vision = []
        for dir in VISION:
            v = Vision(
                direction=dir,
                dist_to_wall=self._distance_to_wall(Direction(dir)),
                is_food_visible=self._is_food_in_direction(Direction(dir)),
                is_self_visible=self._is_body_in_direction(Direction(dir))
            )
            self.snake.vision.append(v)

    def play_step(self, action: Direction = None) -> None:
        """
        Play a single step of the game
        """

        # Step 1 - Get input from the User
        food_eaten = False
        self.frame_iteration += 1
        self._get_input()

        # Step 2 - Move the Snake
        if self.is_human:
            self.snake.move(self.snake.direction)
        else:
            self.snake.move(action)

        # Save this frame
        if self.save_gif:
            pygame.image.save(self.display, os.path.join(GIF_path, f"screenshot0{self.img_cnt}.png"))
            self.img_cnt += 1

        # Step 3 - Place new food or just move
        if self.snake.head == self.food:
            self.score += 1
            self.snake.length += 1
            self._place_food()
            self.moves = 0
            food_eaten = True
        else:
            self.moves += 1
            try:
                self.snake.body.pop()
            except IndexError:
                pass

        # Step 4 - Check if Game Over
        game_over = False
        if self._is_collision() or self.moves > GRID_H * GRID_W:
            game_over = True
            reward = self._calculate_reward()

            if self.save_gif:
                self._save_gif()

            return game_over, self.score, reward

        # Step 5 - Update UI and clock
        self._update_ui()
        self.clock.tick(self.speed)
        self._update_vision()

        # Step 6 - Return Game Over and Score
        reward = self._calculate_reward(food_eaten)
        return game_over, self.score, reward


if __name__ == '__main__':

    game = SnakeGame(is_human=True, save_gif=False)
    print('\nStarting Game...')

    try:
        while True:
            game_over, score, reward = game.play_step()
            print(f'Game Over: {game_over}, Score: {score}, Reward: {reward}')

            if game_over:
                print(f'Game Over! Your score is {score}.\n')
                break
    except KeyboardInterrupt:
        print('\n\nExiting...')
        print(f'Your score is {score}.\n')
        pygame.quit()
        quit()
