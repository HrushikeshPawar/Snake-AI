from collections import deque
import json
import os
from genetic_algorithm.individual import Individual
from settings import settings
from typing import Any, List, Optional, Dict, Tuple, Union
import numpy as np
from helper import Point, Direction, Vision, VISION_8, VISION_4
from neural_network import FeedForwardNetwork, get_activation_by_name
from random import randint, choice


# Defining Constants
GRID_H, GRID_W = settings['grid_size']


# The Snake Class
class Snake(Individual):

    # Initializing the Snake Class, different from initializing the snake on the grid (see below)
    def __init__(
        self,
        grid_size                   : Tuple[int, int] = settings['grid_size'],
        chromosome                  : Optional[Dict[str, List[np.ndarray]]] = None,
        hidden_layer_architecture   : Optional[List[int]] = None,
        hidden_layer_activation     : Optional[str] = None,
        output_layer_activation     : Optional[str] = None,
        start_position              : Optional[Point] = None,
        start_direction             : Optional[Direction] = None,
        lifespan                    : Optional[Union[int, float]] = np.inf,
    ):
        # super().__init__()

        # Initializing snake's various values
        self.lifespan                   = lifespan  # How long the snake can live
        self.score                      = 0  # Number of apples snake gets
        self._fitness                   = 0  # Overall fitness
        self._frames                    = 0  # Number of frames that the snake has been alive
        self._frames_since_last_apple   = 0  # Number of frames since last apple
        self.possible_directions        = (Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT)  # Possible directions for snake to move
        self.grid_size                  = grid_size  # Size of the grid available for snake to move
        self._vision_type               = VISION_8 if settings['vision_type'] == 8 else VISION_4
        # self._chromosome                = chromosome

        # The constants for snake's NN Architecture
        self.hidden_layer_architecture  = hidden_layer_architecture
        self.hidden_layer_activation    = hidden_layer_activation
        self.output_layer_activation    = output_layer_activation

        # Setup the snake's NN Architecture
        num_inputs = len(self._vision_type) * 8 + len(self.possible_directions)  # Number of inputs to the NN
        self.vision_as_array: np.ndarray = np.zeros((num_inputs, 1))
        self.network_architecture = [num_inputs]                           # Inputs to the NN
        self.network_architecture.extend(self.hidden_layer_architecture)   # Hidden layers
        self.network_architecture.append(len(self.possible_directions))    # Output layer
        self.network = FeedForwardNetwork(
            layer_nodes=self.network_architecture,
            hidden_activation=get_activation_by_name(self.hidden_layer_activation),
            output_activation=get_activation_by_name(self.output_layer_activation)
        )

        # If chromosome is set, take it
        if chromosome:
            self.network.params = chromosome
        else:
            pass

        # Initialize the Snake and its food on the grid
        self.initialize_snake(start_position, start_direction)

        # Required if the snake is to be saved
        self.starting_position  = self.head
        self.starting_direction = self.direction
        self.starting_food_position = self.food

    # Initialize the snake on the grid
    def initialize_snake(self, start_position: Optional[Point] = None, start_direction: Optional[Direction] = None) -> None:
        # Initialize the snake's head and body
        self.head                   = start_position if start_position else self._random_point()
        self.snake_array            = deque([self.head.copy()])
        self._body_locations        = set(self.snake_array)
        self.direction              = start_direction if start_position else self._random_direction()
        self.tail_direction         = None
        self.length                 = 1
        self._vision_type           = VISION_8 if settings['vision_type'] == 8 else VISION_4
        self._vision: List[Vision]  = [None] * len(self._vision_type)
        self.is_alive               = True

        # Initialize the snake's food on the grid
        self._place_food()

    # Get a random point on the grid
    def _random_point(self) -> Point:
        """
        Generate a random point
        """
        x = randint(0, GRID_W - 1)
        y = randint(0, GRID_H - 1)
        return Point(x, y)

    # Place food on the screen
    def _place_food(self) -> None:
        """
        Place food randomly on the screen
        """
        possible_positions = set(Point(i, j) for i in range(self.grid_size[0]) for j in range(self.grid_size[1])) - self._body_locations
        # possible_positions = []
        # for i in range(self.grid_size[0]):
        #     for j in range(self.grid_size[1]):
        #         if (i, j) not in self._body_locations:
        #             # self.food = Point(i, j)
        #             possible_positions.append(Point(i, j))

        if possible_positions:
            self.food = choice(list(possible_positions))
        else:
            print('You WON !!!!!')
        # self.food = self._random_point()
        # if self.food in self._body_locations:
        #     if self.length >= self.grid_size[0] * self.grid_size[1]:
        #         print('You WON !!!!!')
        #     else:
        #         self._place_food()

    # Get random Direction
    def _random_direction(self) -> Direction:
        """
        Get random direction
        """
        return self.possible_directions[randint(0, 3)]

    # Defining the (abstract) fitness property of the Individual class
    @property
    def fitness(self):
        return self._fitness

    # The fitness function for the snake's performance
    def calculate_fitness(self):
        # Give positive minimum fitness for roulette wheel selection
        self._fitness = (self._frames) + ((2**self.score) + (self.score**2.1) * 500) - (((.25 * self._frames)**1.3) * (self.score**1.2))
        # self._fitness = (self._frames) + ((2**self.score) + (self.score**2.1) * 500) - (((.25 * self._frames)) * (self.score))
        self._fitness = max(self._fitness, .1)

    # Defining the (abstract) chromosome property of the Individual class
    @property
    def chromosome(self):
        # return self._chromosome
        pass

    # The encode function for the snake's chromosome
    def encode_chromosome(self):
        # # L = len(self.network.params) // 2
        # L = len(self.network.layer_nodes)
        # # Encode weights and bias
        # for layer in range(1, L):
        #     l = str(layer)
        #     self._chromosome['W' + l] = self.network.params['W' + l].flatten()
        #     self._chromosome['b' + l] = self.network.params['b' + l].flatten()
        pass

    # The decode function for the snake's chromosome
    def decode_chromosome(self):
        # # L = len(self.network.params) // 2
        # L = len(self.network.layer_nodes)
        # # Decode weights and bias
        # for layer in range(1, L):
        #     l = str(layer)
        #     w_shape = (self.network_architecture[layer], self.network_architecture[layer-1])
        #     b_shape = (self.network_architecture[layer], 1)
        #     self.network.params['W' + l] = self._chromosome['W' + l].reshape(w_shape)
        #     self.network.params['b' + l] = self._chromosome['b' + l].reshape(b_shape)
        pass

    # Check if the given point is in the grid
    def _is_inside_grid(self, point: Point) -> bool:
        """
        Check if the given point is in the grid
        """
        return 0 <= point.x < GRID_W and 0 <= point.y < GRID_H

    # Check if food is in the given direction
    def _is_food_in_direction(self, direction: Direction) -> bool:
        """
        Check if food is in the given direction
        """
        p = self.head.copy()
        p += direction
        while self._is_inside_grid(p):
            if p == self.food:
                return 1.0
            p += direction

        return 0.0

    # Check if the snake's body is in the given direction
    def _is_body_in_direction(self, direction: Direction) -> bool:
        """
        Check if body of the snake is in the given direction
        """
        p = self.head.copy()
        p += direction
        while self._is_inside_grid(p):
            if p in self._body_locations:
                return 1.0
            p += direction

        return 0.0

    # Find distance from head to wall in given direction
    def _distance_to_wall(self, direction: Direction) -> float:
        """
        Find distance from head to wall in given direction
        """
        p = self.head.copy()
        p += direction
        distance = 0.0
        while self._is_inside_grid(p):
            distance += 1.0
            p += direction
        if distance == 0:
            return 0.0
        return 1 - 1 / distance

    # Look in given direction and calculate all the vision variables
    def look_in_direction(self, direction: Direction) -> None:
        """
        Look in given direction and calculate all the vision variables
        """
        dist_to_wall    = self._distance_to_wall(direction)
        is_food_visible = self._is_food_in_direction(direction)
        is_body_visible = self._is_body_in_direction(direction)

        return Vision(direction=direction, dist_to_wall=dist_to_wall, is_food_visible=is_food_visible, is_self_visible=is_body_visible)

    # Convert the vision variables into a numpy array
    def _vision_as_input_array(self):

        # Split _vision into np array where rows [0-2] are _vision[0].dist_to_wall, _vision[0].dist_to_apple, _vision[0].dist_to_self,
        # rows [3-5] are _vision[1].dist_to_wall, _vision[1].dist_to_apple, _vision[1].dist_to_self, etc. etc. etc.
        for va_index, v_index in zip(range(0, len(self._vision), 3), range(0, len(self._vision))):
            vision: Vision = self._vision[v_index]
            self.vision_as_array[va_index, 0] = vision.dist_to_wall
            self.vision_as_array[va_index + 1, 0] = vision.is_food_visible
            self.vision_as_array[va_index + 2, 0] = vision.is_self_visible

        # Now, encode the direction of the snake's head as one-hot encoding at the end of the array
        i = len(self._vision) * 3

        direc = self.direction
        self.vision_as_array[i, 0] = 1 if direc == Direction.UP else 0
        self.vision_as_array[i + 1, 0] = 1 if direc == Direction.RIGHT else 0
        self.vision_as_array[i + 2, 0] = 1 if direc == Direction.DOWN else 0
        self.vision_as_array[i + 3, 0] = 1 if direc == Direction.LEFT else 0

    # Update the snake's vision
    def update_vision(self) -> None:

        for i, dir in enumerate(self._vision_type):
            self._vision[i] = self.look_in_direction(dir)

        # Convert it to a numpy array
        self._vision_as_input_array()

    # Update the counters
    def update(self) -> bool:
        if self.is_alive:
            self._frames += 1
            self.update_vision()
            self.network.feed_forward(self.vision_as_array)
            self.direction = self.possible_directions[np.argmax(self.network.out)]
            return True
        else:
            return False

    # Check for collision with the wall and the body
    def _check_collision(self) -> bool:
        """
        Check for collision with the wall and the body
        """
        # Check for collision with the wall
        if self.head.x < 0 or self.head.x >= GRID_W or self.head.y < 0 or self.head.y >= GRID_H:
            return True
        # Check for collision with the body
        if self.head in list(self._body_locations)[1:]:
            return True

        return False

    # Move the snake
    def move(self) -> bool:

        # Check if the snake is alive
        if not self.is_alive:
            return False

        # Check if the direction is valid
        if self.direction not in self.possible_directions:
            return False

        # Move the snake one cell ahead in the given direction
        head = self.head.copy()
        head += self.direction

        # if the snake has reached its tail
        if head == self.snake_array[-1]:
            self.snake_array.pop()
            self.snake_array.appendleft(head)

        # Check if the snake has eaten the food
        elif self.head == self.food:
            self.score += 1
            self.length += 1
            self._frames_since_last_apple = 0

            self.snake_array.appendleft(head)
            self._body_locations.update({head})
            # # Remove the tail
            # tail = self.snake_array.pop()
            # self._body_locations.symmetric_difference_update({tail})

            self._place_food()

        else:
            # Move the snake's head
            self.snake_array.appendleft(head)
            self._body_locations.update({head})

            # Remove the tail
            tail = self.snake_array.pop()
            self._body_locations.symmetric_difference_update({tail})

        self.head = head
        # Check for collisions
        if self._check_collision():
            self.is_alive = False
            return False

        self._frames_since_last_apple += 1

        if self.score >= GRID_H * GRID_W:
            print(f'Snake Scoring more than {GRID_H * GRID_W}. Snake Score: {self.score}. Snake Length: {self.length} or {len(self._body_locations)}')
            self.is_alive = False
            print('Snake WON !!!!')
            return False

        # Finally, check if the no. of frames without food is greater than the max allowed
        if self._frames_since_last_apple > GRID_H * GRID_W:
            self.is_alive = False
            return False

        return True


# The provision to save the snakes
def save_snakes(population_folder: str, individual_name: str, snake: Snake, settings: Dict[str, Any]) -> None:
    """
    Save the snakes
    """
    # Make the population folder, if it doesn't exists
    if not os.path.exists(population_folder):
        os.makedirs(population_folder)

    # First save the settings for the current run
    if 'settings.json' not in os.listdir(population_folder):
        with open(os.path.join(population_folder, 'settings.json'), 'w') as f:
            json.dump(settings, f, sort_keys=True, indent=4)

    # Make the folder for the current individual
    individual_folder = os.path.join(population_folder, individual_name)
    os.mkdir(individual_folder)

    # Save some constructor information for replay
    # NOTE: No need to save chromosome since that is saved as .npy
    # NOTE: No need to save board_size or hidden_layer_architecture
    #        since these are taken from settings
    constructor = {}
    constructor['starting_positions'] = snake.starting_position
    constructor['starting_direction'] = snake.starting_direction
    constructor['starting_food_position'] = snake.starting_food_position
    snake_constructor_file = os.path.join(individual_folder, 'constructor_params.json')

    # Save
    with open(snake_constructor_file, 'w') as f:
        json.dump(constructor, f, sort_keys=True, indent=4)

    # Save the network
    len_of_nodes = len(snake.network.layer_nodes)
    for node in range(len_of_nodes):
        weight_name = 'W' + str(node)
        bias_name   = 'b' + str(node)

        weights = snake.network.params[weight_name]
        bias    = snake.network.params[bias_name]

        np.save(os.path.join(individual_folder, weight_name), weights)
        np.save(os.path.join(individual_folder, bias_name), bias)


# The provision to load the snakes
def load_snakes(population_folder: str, individual_name: str, settings: Optional[Union[Dict[str, Any], str]] = None) -> Snake:

    if not settings:
        f = os.path.join(population_folder, 'settings.json')
        if not os.path.exists(f):
            raise Exception("Settings needs to be passed as an argument if 'settings.json' does not exist under population folder!!!")

        with open(f, 'r', encoding='utf-8') as f:
            settings = json.load(f)

    elif isinstance(settings, dict):
        settings = settings

    elif isinstance(settings, str):
        with open(settings, 'r', encoding='utf-8') as f:
            settings = json.load(f)

    params = {}
    for fname in os.listdir(os.path.join(population_folder, individual_name)):
        extension = fname.rsplit('.npy', 1)
        if len(extension) == 2:
            param = extension[0]
            params[param] = np.load(os.path.join(population_folder, individual_name, fname))
        else:
            continue

    constructor_params = {}
    snake_constructor_file = os.path.join(population_folder, individual_name, 'constructor_params.json')
    with open(snake_constructor_file, 'r', encoding='utf-8') as f:
        constructor_params = json.load(f)

    snake = Snake(
        grid_size=settings['grid_size'],
        starting_positions=constructor_params['starting_positions'],
        starting_direction=constructor_params['starting_direction'],
        starting_food_position=constructor_params['starting_food_position'],
        hidden_layer_architecture=settings['hidden_layer_architecture'],
        hidden_layer_activation=settings['hidden_layer_activation'],
        output_layer_activation=settings['output_layer_activation'],
        lifespan=settings['lifespan'],
        chromosome=params
    )

    return snake
