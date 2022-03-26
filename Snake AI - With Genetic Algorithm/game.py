import pygame
from settings import settings
from helper import Point, Direction
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
FONT      = settings['font']

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

    def __init__(self, start_pos: Point, start_dir: Direction = None) -> None:

        # Initialize the Snake
        self.head           = start_pos
        self.body           = [self.head]
        self.direction      = start_dir
        self.tail_direction = None
        self.length         = 1

    def __str__(self) -> str:
        return f'''Snake(\n\thead\t  = {self.head},\n\ttail\t  = {self.tail},\n\tlength\t  = {self.length},\n\tdirection =   {self.direction}\n)'''

    __repr__ = __str__

    # Move the Snake
    def move(self, action: Direction) -> None:
        """
        Move the snake in the given direction
        """
        if isinstance(action, Direction):
            self.direction = action
            self.head += self.direction.value
        else:
            # pass
            raise('Not a valid direction. Pass an element of the Direction Enum')


# The main class for the game
# This class is responsible for the game logic
class SnakeGame:

    def __init__(self, save_gif : bool = False, is_human: bool = True) -> None:

        # Set Window Size
        self.width = GRID_W * BLOCKSIZE
        self.height = GRID_H * BLOCKSIZE
        self.n_games = 0
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

        # Initialize Frame Iteration Count
        self.frame_iteration = 0

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
        self.food = self._srandom_point()

        # If food is placed on snake, place food again
        while self.food in self.snake.body:
            self._place_food()

    # Place snake on Screen
    def _place_snake(self) -> None:
        """
        Place snake on the screen
        """
        self.snake.head = self._random_point()

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

    def _is_collision(self) -> bool:
        """
        Check if the snake has collided with itself or the wall
        Updated this function to check for dangers for next step as well
        """
        """
        Check if the snake has collided with itself
        """
        if self.snake.head.x < 0 or self.snake.head.x > self.width - BLOCKSIZE or self.snake.head.y < 0 or self.snake.head.y > self.height - BLOCKSIZE:
            return True
        return (self.snake.head in self.snake.body[1:])

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
            for point in self.snake.body[1:-1]:
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
        filename = os.path.join(GIF_path, f'Game {n} - {GRID_W} X {GRID_H} Grid - {now}.gif')

        clip = mpy.ImageSequenceClip(imgs, fps=FPS)
        clip.write_gif(filename, fps=FPS)
        os.startfile(filename)
        print(f'GIF saved to {filename}')

        # Remove all the images
        print(f'Removing {len(imgs)} images...')
        for img in imgs:
            os.remove(img)
        print(f'Removed {len(imgs)} images.\n')

    def play_step(self, action: Direction = None) -> None:
        """
        Play a single step of the game
        """
        self.frame_iteration += 1

        # Step 1 - Get input from the User
        self._get_input()

        # Step 2 - Move the Snake
        if self.ishuman:
            self.snake.move(self.snake.direction)
        else:
            self.snake.move(action)
            pass
        pygame.image.save(self.display, os.path.join(GIF_path, f"screenshot0{self.img_cnt}.png"))
        self.img_cnt += 1

        # Step 3 - Place new food or just move
        if self.snake.head == self.food:
            self.score += 1
            self.snake.length += 1
            self._place_food()
        else:
            try:
                self.snake.tail.pop()
            except IndexError:
                pass

        # Step 4 - Check if Game Over
        game_over = False
        if self._is_collision():
            game_over = True

            if self.save_gif:
                self._save_gif()

            if self.ishuman:
                return game_over, self.score
            else:
                pass

            if not self.ishuman:
                pass
        # pygame.image.save(self.display, os.path.join(IMG_DIR, f"screenshot0{self.img_cnt}.png"))
        # self.img_cnt += 1

        # Step 5 - Update UI and clock
        self._update_ui()
        self.clock.tick(self.speed)

        # Step 6 - Return Game Over and Score
        return game_over, self.score


if __name__ == '__main__':

    game = SnakeGame(ishuman=True)
    print('Starting Game...')

    try:
        while True:
            game_over, score = game.play_step()

            if game_over:
                print(f'Game Over! Your score is {score}.\n')
                break
    except KeyboardInterrupt:
        print('\n\nExiting...')
        print(f'Your score is {score}.\n')
        pygame.quit()
        quit()
