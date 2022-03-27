from collections import namedtuple
from typing import Union
import matplotlib.pyplot as plt
from IPython import display


# Point Class
class Point(namedtuple('Point', ['x', 'y'])):

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)

    def __neg__(self) -> 'Point':
        return Point(-self.x, -self.y)

    def __mul__(self, other: int) -> 'Point':
        return Point(self.x * other, self.y * other)

    def __rmul__(self, other: int) -> 'Point':
        return Point(self.x * other, self.y * other)

    def __str__(self) -> str:
        return f"Point({self.x}, {self.y})"

    __repr__ = __str__

    def __eq__(self, other: 'Point') -> bool:
        return isinstance(other, Point) and self.x == other.x and self.y == other.y

    def __ne__(self, other: 'Point') -> bool:
        return not isinstance(other, Point) or self.x != other.x or self.y != other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def copy(self) -> 'Point':
        return Point(self.x, self.y)


class Direction():
    # Four Major Directions
    UP    = Point(0, -1)
    DOWN  = Point(0, 1)
    LEFT  = Point(-1, 0)
    RIGHT = Point(1, 0)

    # Four Diagonal Directions
    UP_LEFT    = Point(-1, -1)
    UP_RIGHT   = Point(1, -1)
    DOWN_LEFT  = Point(-1, 1)
    DOWN_RIGHT = Point(1, 1)

    def __init__(self) -> None:
        pass

    # Multiplication of a point object by a scalar
    def __mul__(self, other: int) -> Point:
        return Point(self.value.x * other, self.value.y * other)

    # Multiplication of a scalar by a point object
    def __rmul__(self, other: int) -> Point:
        return Point(self.value.x * other, self.value.y * other)


# Snake VISIONS
# 8 lines to see around the snake
VISION_8 = (
    # Basic Vision          # Diagonal Vision
    #     L0                #     L1
    Direction.UP,           Direction.UP_RIGHT,

    #     L2                #     L3
    Direction.RIGHT,        Direction.DOWN_RIGHT,

    #     L4                #     L5
    Direction.DOWN,         Direction.DOWN_LEFT,

    #     L6                #     L7
    Direction.LEFT,         Direction.UP_LEFT,
)


# 4 lines to see around the snake
VISION_4 = tuple([VISION_8[i] for i in range(0, 8, 2)])


# Vision Class for snake
class Vision:

    def __init__(
        self,
        direction : Direction,
        dist_to_wall: Union[float, int],
        is_food_visible: bool,
        is_self_visible: bool
    ) -> None:
        self.direction = direction
        self.dist_to_wall       = dist_to_wall
        self.is_food_visible   = is_food_visible
        self.is_self_visible    = is_self_visible

    def __str__(self) -> str:
        return f"Vision(direction={self.direction},\tdist_to_wall={self.dist_to_wall},\tis_food_visible={self.is_food_visible},\tis_self_visible={self.is_self_visible})"

    __repr__ = __str__


# Plotting Function
def plot(scores, mean_scores, fpath, save_plot=False):

    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Generations')
    plt.ylabel('Score')
    plt.plot(scores, color='red')
    plt.plot(mean_scores, color='green')
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

    if save_plot:
        plt.savefig(fpath)


if __name__ == '__main__':

    print(Direction.UP)
    print(Direction.DOWN)
    print(Direction.UP + Direction.DOWN)
    print(2 * Direction.UP)
