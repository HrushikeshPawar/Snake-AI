import pygame
from collections import namedtuple, deque
from enum import Enum
from random import randint
import os
import glob
# from PIL import Image
import moviepy.editor as mpy
from time import perf_counter
from datetime import datetime


# Initialize pygame
pygame.init()
font = pygame.font.Font('Lora-Regular.ttf', 20)


# Define Constants
BLOCKSIZE   = 20
GRID_H      = 10
GRID_W      = 10
SPEED       = 20
BORDER      = 3

# Colors
BLACK       = (0, 0, 0)
GREY        = (150, 150, 150)
WHITE       = (255, 255, 255)
RED         = (255, 0, 0)
GREEN       = (0, 255, 0)
GREEN2      = (100, 255, 0)
BLUE        = (0, 0, 255)
BLUE2       = (0, 100, 255)

# DIRS
IMG_DIR     = 'Pics'


# The Point Class to store the coordinates of the snake
class Point(namedtuple('Point', 'x, y')):

    # Adding two point objects
    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    # Subtracting two point objects
    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)

    # Negation of a point object
    def __neg__(self) -> 'Point':
        return Point(-self.x, -self.y)

    # Multiplication of a point object by a scalar
    def __mul__(self, other: int) -> 'Point':
        return Point(self.x * other, self.y * other)

    # Multiplication of a scalar by a point object
    def __rmul__(self, other: int) -> 'Point':
        return Point(self.x * other, self.y * other)


# Define Direction Class
class Direction(Enum):
    UP    = Point(0, -1)
    DOWN  = Point(0, 1)
    LEFT  = Point(-1, 0)
    RIGHT = Point(1, 0)

    # Multiplication of a point object by a scalar
    def __mul__(self, other: int) -> Point:
        return Point(self.x * other, self.y * other)

    # Multiplication of a scalar by a point object
    def __rmul__(self, other: int) -> Point:
        return Point(self.x * other, self.y * other)


# The Snake Class
class Snake:

    def __init__(self, head: Point, lenght: int = 1, direction: Direction = None) -> None:

        self.head       = head
        self.length     = lenght
        self.direction  = direction

        # Initialize the tail
        if self.length == 1:
            self.tail = []
        else:
            shift = self.direction.value * BLOCKSIZE
            self.tail = list(deque([Point(self.head.x - (i * shift).x, self.head.y - (i * shift).y) for i in range(1, self.length)]))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f'''Snake(\n\thead\t  = {self.head},\n\ttail\t  = {self.tail},\n\tlength\t  = {self.length},\n\tdirection =   {self.direction}\n)'''

    def move(self, action) -> None:
        """
        Move the snake in the given direction
        """
        if isinstance(action, Direction):
            self.direction = action
            self.tail.insert(0, self.head)
            self.head += self.direction.value * BLOCKSIZE

        else:
            pass


# Define Game Class
class Game:

    def __init__(self, height: int = BLOCKSIZE * GRID_H, width: int = BLOCKSIZE * GRID_W,
                 speed: int = SPEED, n_games: int = 0, ishuman: bool = False, solver: str = None,
                 save_gif: bool = False) -> None:

        # Set required variables
        self.height   = height
        self.width    = width
        self.speed    = speed
        self.n_games  = n_games
        self.ishuman  = ishuman
        self.save_gif = save_gif

        # Initialize Pygame Window
        self.display = pygame.display.set_mode((self.width, self.height))

        if self.ishuman:
            pygame.display.set_caption("Snake Game")
        else:
            pygame.display.set_caption(f"Snake Game  ({n_games}th Game)")
            self.solver = solver
        self.clock = pygame.time.Clock()

        # Initialize the Game
        self.reset_game()
        self.start_time = perf_counter()
        pygame.image.save(self.display, os.path.join(IMG_DIR, "screenshot00.png"))
        self.img_cnt = 1

    def reset_game(self) -> None:
        """
        Reset the game
        """
        self.snake = None
        self._place_snake()

        self.food  = None
        self._place_food()

        self.score = 0
        self.game_over = False
        self.frame_iteration = 0

    def random_point(self) -> None:
        """
        Generate a random point
        """
        # x = randint(0, (self.width - BLOCKSIZE) // BLOCKSIZE) * BLOCKSIZE
        # y = randint(0, (self.height - BLOCKSIZE) // BLOCKSIZE) * BLOCKSIZE
        x = randint(0, GRID_W - 1) * BLOCKSIZE
        y = randint(0, GRID_H - 1) * BLOCKSIZE
        return Point(x, y)

    def _place_snake(self) -> None:
        """
        Place the snake on the screen
        """
        self.snake = Snake(head=self.random_point())

    def _place_food(self) -> None:
        """
        Place the food on the screen
        """
        self.food = self.random_point()
        while self.food in self.snake.tail or self.food == self.snake.head:
            self.food = self.random_point()

    def _get_input(self) -> None:

        for event in pygame.event.get():

            # CHECK IF USER QUITS
            if event.type == pygame.QUIT:
                # print('\n\nExiting...')
                # print(f'Your score is {score}.\n')
                # pygame.quit()
                # quit()
                raise KeyboardInterrupt

            # CHECK IF USER PRESSES A KEY
            if self.ishuman and event.type == pygame.KEYDOWN:
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
        Check if the snake has collided with itself
        """
        if self.snake.head.x < 0 or self.snake.head.x > self.width - BLOCKSIZE or self.snake.head.y < 0 or self.snake.head.y > self.height - BLOCKSIZE:
            return True
        return (self.snake.head in self.snake.tail)

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
            for point in self.snake.tail[:-1]:
                pygame.draw.rect(self.display, BLUE, (point.x, point.y, BLOCKSIZE - BORDER, BLOCKSIZE - BORDER))
                pygame.draw.rect(self.display, BLUE2, (point.x + 4, point.y + 4, 12 - BORDER, 12 - BORDER))
            point = self.snake.tail[-1]
            pygame.draw.rect(self.display, GREEN, (point.x, point.y, BLOCKSIZE - BORDER, BLOCKSIZE - BORDER))
            pygame.draw.rect(self.display, GREEN2, (point.x + 4, point.y + 4, 12 - BORDER, 12 - BORDER))
        except IndexError:
            pass

        # Draw the score
        text = font.render(f'Score: {self.score}', True, WHITE)
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

        # frames = []
        print('\nMaking GIF...')

        imgs = glob.glob(os.path.join(IMG_DIR, "*.png"))
        list.sort(imgs, key=lambda x: int(x.split('screenshot')[1].split('.png')[0]))
        # for i in imgs:
        #     new_frame = Image.open(i)
        #     frames.append(new_frame)

        # # Save into a GIF file that loops forever
        # frames[0].save(
        #     filename, format='GIF', append_images=frames[1:], save_all=True,
        #     duration=duration, loop=0
        # )
        txt_path = os.path.join(IMG_DIR, 'Image_List.txt')
        with open(txt_path, 'w') as file:
            for item in imgs:
                file.write(f"{item}\n")

        FPS = len(imgs) // duration
        now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        n = len(glob.glob(os.path.join(IMG_DIR, "*.gif")))
        filename = os.path.join(IMG_DIR, f'Game {n} - {self.solver} - {GRID_W} X {GRID_H} Grid - {now}.gif')

        clip = mpy.ImageSequenceClip(imgs, fps=FPS)
        clip.write_gif(filename, fps=FPS)
        os.startfile(filename)

        # Remove all the images
        for img in imgs:
            os.remove(img)
        print(f'Removed {len(imgs)} images.')

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
        pygame.image.save(self.display, os.path.join(IMG_DIR, f"screenshot0{self.img_cnt}.png"))
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

    game = Game(ishuman=True)

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
