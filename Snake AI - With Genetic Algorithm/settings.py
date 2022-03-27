import numpy as np


settings = {
    # Regarding Game Environment
    'block_size'    : 20,
    'grid_size'     : (6, 6),
    'border'        : 3,
    'speed'         : 5,
    'font'          : 'Lora-Regular.ttf',

    #  Colors for the Game
    'black'         : (0, 0, 0),
    'white'         : (255, 255, 255),
    'grey'          : (150, 150, 150),
    'red'           : (255, 0, 0),
    'green'         : (0, 255, 0),
    'green2'        : (100, 255, 0),
    'blue'          : (0, 0, 255),
    'blue2'         : (0, 100, 255),

    # File Paths
    'GIF_path'          : 'Data/GIFs/',
    'Graph_path'        : 'Data/Graphs/',
    'Model_path'        : 'Data/Models/',
    'Checkpoint_path'   : 'Data/Checkpoints/',
    'Log_path'          : 'Data/Logs/',

    # Neural Network Settings
    'num_parents'                   : 500,        # Number of parents that will be used for reproducing
    'num_offspring'                 : 1000,       # Number of offspring that will be created. Keep num_offspring >= num_parents
    'total_generations'             : 5000,
    'hidden_layer_activation'       : 'relu',     # Options are [relu, sigmoid, tanh, linear, leaky_relu]
    'output_layer_activation'       : 'sigmoid',  # Options are [relu, sigmoid, tanh, linear, leaky_relu]
    'hidden_network_architecture'   : [20, 12],   # A list containing number of nodes in each hidden layer
    'vision_type'                   : 8,          # Options are [4, 8, 16]

    # Genetic Algorithm Settings

    # Mutation Settings #
    # Mutation rate is the probability that a given gene in a chromosome will randomly mutate
    'mutation_rate'             : 0.05,       # Value must be between [0.00, 1.00)
    # If the mutation rate type is static, then the mutation rate will always be `mutation_rate`,
    # otherwise if it is decaying it will decrease as the number of generations increase
    'mutation_rate_type'        : 'static',   # Options are [static, decaying]
    # The probability that if a mutation occurs, it is gaussian
    'probability_gaussian'      : 1.0,        # Values must be between [0.00, 1.00]
    # The probability that if a mutation occurs, it is random uniform
    'probability_random_uniform': 0.0,        # Values must be between [0.00, 1.00]

    # Cross Over Settings #
    # eta related to SBX. Larger values create a distribution closer around the parents while smaller values venture further from them.
    # Only used if probability_SBX > 0.00
    'SBX_eta'                       : 100,
    # Probability that when crossover occurs, it is simulated binary crossover
    'probability_SBX'               : 0.5,
    # The type of SPBX to consider. If it is 'r' then it flattens a 2D array in row major ordering.
    # If SPBX_type is 'c' then it flattens a 2D array in column major ordering.
    'SPBX_type'                     : 'r',        # Options are 'r' for row or 'c' for column
    # Probability that when crossover occurs, it is single point binary crossover
    'probability_SPBX'              : 0.5,
    # Crossover selection type determines the way in which we select individuals for crossover
    'crossover_selection_type'      : 'roulette_wheel',

    # Selection Settings #
    # Number of parents that will be used for reproducing
    'num_parents'                   : 500,
    # Number of offspring that will be created. Keep num_offspring >= num_parents
    'num_offspring'                 : 1000,
    # The selection type to use for the next generation.
    # If selection_type == 'plus':
    #     Then the top num_parents will be chosen from (num_offspring + num_parents)
    # If selection_type == 'comma':
    #     Then the top num_parents will be chosen from (num_offspring)
    # NOTE: if the lifespan of the individual is 1, then it cannot be selected for the next generation
    # If enough indivduals are unable to be selected for the next generation, new random ones will take their place.
    # NOTE: If selection_type == 'comma' then lifespan is ignored.
    #   This is equivalent to lifespan = 1 in this case since the parents never make it to the new generation.
    'selection_type'                : 'plus',     # Options are ['plus', 'comma']

    # Individual Settings #
    'lifespan'                      : np.inf,          # Number of generations an individual can live for
}
