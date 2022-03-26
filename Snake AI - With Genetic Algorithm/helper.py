"""
This will contain all the requried helper functions and classes
"""


# from collections import namedtuple
# from enum import Enum
from typing import Union, Dict, Tuple


# # Point Class
# class Point(namedtuple('Point', 'x y')):

#     def __add__(self, other: 'Point') -> 'Point':
#         return Point(self.x + other.x, self.y + other.y)

#     def __sub__(self, other: 'Point') -> 'Point':
#         return Point(self.x - other.x, self.y - other.y)

#     def __neg__(self) -> 'Point':
#         return Point(-self.x, -self.y)

#     def __mul__(self, other: int) -> 'Point':
#         return Point(self.x * other, self.y * other)

#     def __rmul__(self, other: int) -> 'Point':
#         return Point(self.x * other, self.y * other)

#     def __str__(self) -> str:
#         return f"Point({self.x}, {self.y})"

#     __repr__ = __str__

#     def __eq__(self, other: 'Point') -> bool:
#         return isinstance(other, Point) and self.x == other.x and self.y == other.y

#     def __ne__(self, other: 'Point') -> bool:
#         return not isinstance(other, Point) or self.x != other.x or self.y != other.y

#     def __hash__(self) -> int:
#         return hash((self.x, self.y))


# # Direction Class
# class Direction(Enum):

#     # Four Major Directions
#     UP    = Point(0, -1)
#     DOWN  = Point(0, 1)
#     LEFT  = Point(-1, 0)
#     RIGHT = Point(1, 0)

#     # Four Diagonal Directions
#     UP_LEFT    = Point(-1, -1)
#     UP_RIGHT   = Point(1, -1)
#     DOWN_LEFT  = Point(-1, 1)
#     DOWN_RIGHT = Point(1, 1)

#     # Eight Sub Diagonal Directions
#     UP_UP_LEFT       = Point(-1, -2)
#     UP_UP_RIGHT      = Point(1, -2)
#     UP_LEFT_LEFT     = Point(-2, -1)
#     UP_RIGHT_RIGHT   = Point(2, -1)
#     DOWN_DOWN_LEFT   = Point(-1, 2)
#     DOWN_DOWN_RIGHT  = Point(1, 2)
#     DOWN_LEFT_LEFT   = Point(-2, 1)
#     DOWN_RIGHT_RIGHT = Point(2, 1)

#     # Multiplication of a point object by a scalar
#     def __mul__(self, other: int) -> Point:
#         return Point(self.value.x * other, self.value.y * other)

#     # Multiplication of a scalar by a point object
#     def __rmul__(self, other: int) -> Point:
#         return Point(self.value.x * other, self.value.y * other)


# # Snake Visions
# # 16 lines to see around the snake
# VISION16 = (
#     # Basic Vision          Sub - Diagonal                   Diagonal                 Sub - Diagonal
#     # L0                         L1                             L2                          L3
#     Direction.UP,       Direction.UP_UP_RIGHT,          Direction.UP_RIGHT,       Direction.UP_RIGHT_RIGHT,

#     # L4                         L5                             L6                          L7
#     Direction.RIGHT,    Direction.DOWN_RIGHT_RIGHT,     Direction.DOWN_RIGHT,     Direction.DOWN_DOWN_RIGHT,

#     # L8                        L9                             L10                          L11
#     Direction.DOWN,     Direction.DOWN_DOWN_LEFT,       Direction.DOWN_LEFT,      Direction.DOWN_LEFT_LEFT,

#     # L12                       L13                             L14                          L15
#     Direction.LEFT,     Direction.UP_LEFT_LEFT,         Direction.UP_LEFT,        Direction.UP_UP_LEFT,
# )


class Slope(object):
    __slots__ = ('rise', 'run')

    def __init__(self, rise: int, run: int):
        self.rise = rise
        self.run = run


class Point(object):
    __slots__ = ('x', 'y')

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def copy(self) -> 'Point':
        x = self.x
        y = self.y
        return Point(x, y)

    def to_dict(self) -> Dict[str, int]:
        d = {}
        d['x'] = self.x
        d['y'] = self.y
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> 'Point':
        return Point(d['x'], d['y'])

    def __eq__(self, other: Union['Point', Tuple[int, int]]) -> bool:
        if isinstance(other, tuple) and len(other) == 2:
            return other[0] == self.x and other[1] == self.y
        elif isinstance(other, Point) and self.x == other.x and self.y == other.y:
            return True
        return False

    def __sub__(self, other: Union['Point', Tuple[int, int]]) -> 'Point':
        if isinstance(other, tuple) and len(other) == 2:
            diff_x = self.x - other[0]
            diff_y = self.y - other[1]
            return Point(diff_x, diff_y)
        elif isinstance(other, Point):
            diff_x = self.x - other.x
            diff_y = self.y - other.y
            return Point(diff_x, diff_y)
        return None

    def __rsub__(self, other: Tuple[int, int]):
        diff_x = other[0] - self.x
        diff_y = other[1] - self.y
        return Point(diff_x, diff_y)

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __str__(self) -> str:
        return '({}, {})'.format(self.x, self.y)


# These lines are defined such that facing "up" would be L0 ###
# Create 16 lines to be able to "see" around
VISION_16 = (
    #   L0            L1             L2             L3
    Slope(-1, 0), Slope(-2, 1),  Slope(-1, 1),  Slope(-1, 2),
    #   L4            L5             L6             L7
    Slope(0, 1),  Slope(1, 2),   Slope(1, 1),   Slope(2, 1),
    #   L8            L9             L10            L11
    Slope(1, 0),  Slope(2, -1),  Slope(1, -1),  Slope(1, -2),
    #   L12           L13            L14            L15
    Slope(0, -1), Slope(-1, -2), Slope(-1, -1), Slope(-2, -1)
)


# 8 lines to see around the snake
VISION_8 = tuple([VISION_16[i] for i in range(0, 16, 2)])

# 4 lines to see around the snake
VISION_4 = tuple([VISION_16[i] for i in range(0, 16, 4)])


# Vision Class for snake
class Vision:

    def __init__(
        self,
        dist_to_wall: Union[float, int],
        is_apple_visible: bool,
        is_self_visible: bool
    ) -> None:
        self.dist_to_wall       = dist_to_wall
        self.is_apple_visible   = is_apple_visible
        self.is_self_visible    = is_self_visible
