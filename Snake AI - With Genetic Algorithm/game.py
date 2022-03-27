from snake import Snake  # , save_snakes, load_snakes
from settings import settings
import pygame
import numpy as np
import random
from math import sqrt
from tqdm import tqdm
from typing import List, Tuple
from genetic_algorithm.individual import Individual
from genetic_algorithm.population import Population
from genetic_algorithm.selection import elitism_selection, roulette_wheel_selection
from genetic_algorithm.crossover import simulated_binary_crossover as SBX
from genetic_algorithm.crossover import single_point_binary_crossover
from genetic_algorithm.mutation import gaussian_mutation, random_uniform_mutation

# Initialize the Pygame environment
pygame.init()

# Define the CONSTANTS
GRID_H, GRID_W = settings['grid_size']
BLOCKSIZE = settings['block_size']
BORDER    = settings['border']
SPEED     = settings['speed']
FONT      = pygame.font.Font(settings['font'], 20)

# Colors
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


#  The Main Game Class
#  This class is responsible for the game loop and its components
class Main_Game:

    def __init__(self, show: bool = False) -> None:

        self.show               = show
        self.settings           = settings
        self._SBX_eta           = self.settings['SBX_eta']
        self._mutation_bins     = np.cumsum([self.settings['probability_gaussian'], self.settings['probability_random_uniform']])
        self._crossover_bins    = np.cumsum([self.settings['probability_SBX'], self.settings['probability_SPBX']])
        self._SPBX_type         = self.settings['SPBX_type'].lower()
        self._mutation_rate     = self.settings['mutation_rate']

        # Determine size of next gen based off selection type
        self._next_gen_size = None
        if self.settings['selection_type'].lower() == 'plus':
            self._next_gen_size = self.settings['num_parents'] + self.settings['num_offspring']
        elif self.settings['selection_type'].lower() == 'comma':
            self._next_gen_size = self.settings['num_offspring']
        else:
            raise Exception(f"Selection type {self.settings['selection_type']} is invalid")

        # self.width             = (GRID_W * 4) * BLOCKSIZE
        # self.height            = (GRID_H * 4) * BLOCKSIZE
        self.width             = (GRID_W) * BLOCKSIZE
        self.height            = (GRID_H) * BLOCKSIZE
        self.snake_board_width      = GRID_W * BLOCKSIZE
        self.snake_board_height    = GRID_H * BLOCKSIZE

        individuals: List[Individual] = []

        # Generate the first generation of individuals
        for _ in range(self.settings['num_parents'] + 1):
            individual = Snake(
                grid_size=(GRID_W, GRID_H),
                hidden_layer_architecture=self.settings['hidden_network_architecture'],
                hidden_layer_activation=self.settings['hidden_layer_activation'],
                output_layer_activation=self.settings['output_layer_activation']
            )
            individuals.append(individual)

        self.best_fitness = 0
        self.best_score = 0

        self._current_individual = 0
        self.population = Population(individuals)

        self.snake: Snake = self.population.individuals[self._current_individual]
        self.current_generation = 0

        # Initialize the window
        # self.init_window()

        # Initialize the game loop
        # while self.current_generation < self.settings['total_generations']:
        #     self.game_loop()
            # self.clock.tick(SPEED)
        self.play()

    # Initialize the game window
    def init_window(self):
        # Initialize Window
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Snake Game  (Generation {self.current_generation + 1})")
        self.clock = pygame.time.Clock()

        # Create Snake Window
        self.snake_window = Snake_Window(parent=self.display, board_size=(self.snake_board_width, self.snake_board_height), snake=self.snake)

    # Play the Game
    def play(self):

        # Initialize the window
        if self.show:
            self.init_window()

        # Continue playing till number of generations is reached
        
        while self.current_generation < self.settings['total_generations']:

            with tqdm(total=self._next_gen_size, desc=f"Current Generation {self.current_generation + 1}") as pbar:
                while (self.current_generation > 0 and self._current_individual < self._next_gen_size - 1) or \
                      (self.current_generation == 0 and self._current_individual < settings['num_parents']):

                    self.game_loop()
                    if not self.snake.is_alive:
                        pbar.update(1)

                    if self.show:
                        self.clock.tick(SPEED)

            print(f'======================= Gneration {self.current_generation + 1} =======================')
            print('----Max fitness:', self.population.fittest_individual.fitness)
            print('----Best Score:', self.population.fittest_individual.score)
            print('----Average fitness:', self.population.average_fitness)
            print('\n\n')

            if self.show:
                pygame.display.quit()
            self.next_generation()

    # The Game Loop
    def game_loop(self) -> None:

        if self.show:
            self.snake_window._update_ui()

        # If the current snake is alive
        if self.snake.is_alive:
            self.snake.update()
            self.snake.move()

            # Check for the score
            if self.snake.score > GRID_H * GRID_W:
                print(f'Snake Scoring more than {GRID_H * GRID_W}. Snake Score: {self.snake.score}. Snake Length: {self.snake.length} or {len(self.snake._body_locations)}')
            if self.snake.score > self.best_score:
                self.best_score = self.snake.score

        # If the snake is dead
        else:
            # Calculate fitness of current individual
            self.snake.calculate_fitness()
            fitness = self.snake.fitness

            if fitness > self.best_fitness:
                self.best_fitness = fitness

            # Change the snake
            self._current_individual += 1
            self.snake = self.population.individuals[self._current_individual]

            if self.show:
                self.snake_window.snake = self.snake

    # Create the next generation
    def next_generation(self) -> None:

        self.current_generation += 1
        self._current_individual = 0

        # Calculate fitness of individuals
        self.population.calculate_fitness()

        # Create the next generation of individuals first with elite parents
        self.population.individuals = elitism_selection(self.population, self.settings['num_parents'])

        random.shuffle(self.population.individuals)
        next_pop: List[Snake] = []

        # parents + offspring selection type ('plus')
        if self.settings['selection_type'].lower() == 'plus':

            # Decrement lifespan
            # for individual in self.population.individuals:
            #     individual.lifespan -= 1

            for individual in self.population.individuals:
                individual: Snake
                individual.lifespan -= 1

                params                      = self.snake.network.params
                board_size                  = individual.grid_size
                hidden_layer_architecture   = individual.hidden_layer_architecture
                hidden_activation           = individual.hidden_layer_activation
                output_activation           = individual.output_layer_activation
                lifespan                    = individual.lifespan
                # starting_position           = individual.starting_position
                # starting_food_position      = individual.starting_food_position
                # starting_direction          = individual.starting_direction

                # If the individual is still alive, they survive
                if lifespan > 0:
                    s = Snake(
                        grid_size=board_size,
                        chromosome=params,
                        hidden_layer_architecture=hidden_layer_architecture,
                        hidden_layer_activation=hidden_activation,
                        output_layer_activation=output_activation,
                        lifespan=lifespan,
                    )
                    next_pop.append(s)
            # print('----Elite parents:', len(self.population.individuals))
            # print('Creating childern from elite parents')

        while len(next_pop) < self._next_gen_size:

            # Select parents for crossover
            parent1, parent2 = roulette_wheel_selection(self.population, 2)
            parent1: Snake
            parent2: Snake

            L             = len(parent1.network.layer_nodes)
            child1_params = {}
            child2_params = {}

            for node in range(1, L):
                # Get the Weight and Bais of parents, which work as our chromosomes
                parent1_W_l = parent1.network.params['W' + str(node)]
                parent2_W_l = parent2.network.params['W' + str(node)]
                parent1_b_l = parent1.network.params['b' + str(node)]
                parent2_b_l = parent2.network.params['b' + str(node)]

                # Crossover
                # NOTE: I am choosing to perform the same type of crossover on the weights and the bias.
                child1_W_l, child2_W_l, child1_b_l, child2_b_l = self._crossover(parent1_W_l, parent2_W_l, parent1_b_l, parent2_b_l)

                # Mutation
                # NOTE: I am choosing to perform the same type of mutation on the weights and the bias.
                # child1_W_l, child2_W_l, child1_b_l, child2_b_l = self._mutation(child1_W_l, child2_W_l, child1_b_l, child2_b_l)
                self._mutation(child1_W_l, child2_W_l, child1_b_l, child2_b_l)

                # Assign children from crossover/mutation
                child1_params['W' + str(node)] = child1_W_l
                child2_params['W' + str(node)] = child2_W_l
                child1_params['b' + str(node)] = child1_b_l
                child2_params['b' + str(node)] = child2_b_l

                # Clip to [-1, 1]
                np.clip(child1_params['W' + str(node)], -1, 1, out=child1_params['W' + str(node)])
                np.clip(child2_params['W' + str(node)], -1, 1, out=child2_params['W' + str(node)])
                np.clip(child1_params['b' + str(node)], -1, 1, out=child1_params['b' + str(node)])
                np.clip(child2_params['b' + str(node)], -1, 1, out=child2_params['b' + str(node)])

            # Create children from chromosomes generated above
            child1 = Snake(
                parent1.grid_size,
                chromosome=child1_params,
                hidden_layer_architecture=parent1.hidden_layer_architecture,
                hidden_layer_activation=parent1.hidden_layer_activation,
                output_layer_activation=parent1.output_layer_activation,
                lifespan=self.settings['lifespan']
            )
            child2 = Snake(
                parent2.grid_size,
                chromosome=child2_params,
                hidden_layer_architecture=parent2.hidden_layer_architecture,
                hidden_layer_activation=parent2.hidden_layer_activation,
                output_layer_activation=parent2.output_layer_activation,
                lifespan=self.settings['lifespan']
            )

            # Add children to the next generation
            next_pop.extend([child1, child2])

        # Set the next generation
        random.shuffle(next_pop)
        self.population.individuals = next_pop

        # Initialize the window
        if self.show:
            self.init_window()

    # Crossover
    def _crossover(
        self,
        parent1_weights: np.ndarray,
        parent2_weights: np.ndarray,
        parent1_bias   : np.ndarray,
        parent2_bias   : np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        rand_crossover = random.random()
        crossover_bucket = np.digitize(rand_crossover, self._crossover_bins)
        child1_weights, child2_weights = None, None
        child1_bias, child2_bias = None, None

        # SBX
        if crossover_bucket == 0:
            child1_weights, child2_weights = SBX(parent1_weights, parent2_weights, self._SBX_eta)
            child1_bias, child2_bias       = SBX(parent1_bias, parent2_bias, self._SBX_eta)

        # Single point binary crossover (SPBX)
        elif crossover_bucket == 1:
            child1_weights, child2_weights = single_point_binary_crossover(parent1_weights, parent2_weights, major=self._SPBX_type)
            child1_bias, child2_bias       = single_point_binary_crossover(parent1_bias, parent2_bias, major=self._SPBX_type)

        else:
            raise Exception('Unable to determine valid crossover based off probabilities')

        return child1_weights, child2_weights, child1_bias, child2_bias

    # Mutation
    def _mutation(
        self,
        child1_weights: np.ndarray,
        child2_weights: np.ndarray,
        child1_bias   : np.ndarray,
        child2_bias   : np.ndarray
    ) -> None:

        scale = .2
        rand_mutation = random.random()
        mutation_bucket = np.digitize(rand_mutation, self._mutation_bins)

        mutation_rate = self._mutation_rate
        if self.settings['mutation_rate_type'].lower() == 'decaying':
            mutation_rate = mutation_rate / sqrt(self.current_generation + 1)

        # Gaussian
        if mutation_bucket == 0:
            # Mutate weights
            child1_weights  = gaussian_mutation(child1_weights, mutation_rate, scale=scale)
            child2_weights  = gaussian_mutation(child2_weights, mutation_rate, scale=scale)

            # Mutate bias
            child1_bias     = gaussian_mutation(child1_bias, mutation_rate, scale=scale)
            child2_bias     = gaussian_mutation(child2_bias, mutation_rate, scale=scale)

        # Uniform random
        elif mutation_bucket == 1:
            # Mutate weights
            random_uniform_mutation(child1_weights, mutation_rate, -1, 1)
            random_uniform_mutation(child2_weights, mutation_rate, -1, 1)

            # Mutate bias
            random_uniform_mutation(child1_bias, mutation_rate, -1, 1)
            random_uniform_mutation(child2_bias, mutation_rate, -1, 1)

        else:
            raise Exception('Unable to determine valid mutation based off probabilities.')

        return child1_weights, child2_weights, child1_bias, child2_bias


# The Snake Window Class
class Snake_Window:
    def __init__(self, parent: pygame.display, board_size=Tuple[int, int], snake: Snake = None) -> None:
        # super().__init__(parent)
        self.board_size = board_size
        self.display    = parent

        if snake:
            self.snake = snake

    def start_game(self) -> None:
        self.snake = Snake(self.board_size)

    def update(self) -> None:
        if self.snake.is_alive:
            self.snake.update()

        else:
            pass

    # Update the pygame window
    def _update_ui(self) -> None:
        """
        Update the UI
        """
        # Clear the screen
        self.display.fill(BLACK)
        # self.display.fill(WHITE)

        # Draw the food
        pygame.draw.rect(self.display, RED, (self.snake.food.x * BLOCKSIZE, self.snake.food.y * BLOCKSIZE, BLOCKSIZE - BORDER, BLOCKSIZE - BORDER))

        # Draw the snake
        pygame.draw.rect(self.display, GREY, (self.snake.head.x * BLOCKSIZE, self.snake.head.y * BLOCKSIZE, BLOCKSIZE - BORDER, BLOCKSIZE - BORDER))
        pygame.draw.rect(self.display, WHITE, (self.snake.head.x * BLOCKSIZE + 4, self.snake.head.y * BLOCKSIZE + 4, 12 - BORDER, 12 - BORDER))
        try:
            for point in list(self.snake._body_locations)[1:]:
                pygame.draw.rect(self.display, BLUE, (point.x * BLOCKSIZE, point.y * BLOCKSIZE, BLOCKSIZE - BORDER, BLOCKSIZE - BORDER))
                pygame.draw.rect(self.display, BLUE2, (point.x * BLOCKSIZE + 4, point.y * BLOCKSIZE + 4, 12 - BORDER, 12 - BORDER))
            point = list(self.snake._body_locations)[1:][-1]
            pygame.draw.rect(self.display, GREEN, (point.x * BLOCKSIZE, point.y * BLOCKSIZE, BLOCKSIZE - BORDER, BLOCKSIZE - BORDER))
            pygame.draw.rect(self.display, GREEN2, (point.x * BLOCKSIZE + 4, point.y * BLOCKSIZE + 4, 12 - BORDER, 12 - BORDER))
        except IndexError:
            pass

        # Draw the score
        text = FONT.render(f'Score: {self.snake.score}', True, WHITE)
        self.display.blit(text, (1, 1))

        # Update the display
        pygame.display.update()


if __name__ == '__main__':
    main_game = Main_Game()
