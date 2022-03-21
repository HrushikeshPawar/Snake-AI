# genetic_algorithm/selection.py

import numpy as np
from typing import Tuple
from .individual import Individual


# Different Selection Methods
def simulated_binary_crossover(
    parent1: Individual,
    parent2: Individual,
    eta    : float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulated Binary Crossover
    """
    # Calculating the Gamma Value
    random_number = np.random.random(parent1.chromosome.shape)
    gamma = np.empty(parent1.chromosome.shape)

    # When gamma < 0.5
    gamma[random_number < 0.5] = (2 * random_number[random_number < 0.5]) ** (1 / (eta + 1))

    # When gamma >= 0.5
    gamma[random_number >= 0.5] = (2 * (1 - random_number[random_number >= 0.5])) ** (-1 / (eta + 1))

    # Calculating child1 chromosome
    chromosome1 = 0.5 * ((1 + gamma) * parent1.chromosome + (1 - gamma) * parent2.chromosome)
    chromosome2 = 0.5 * ((1 - gamma) * parent1.chromosome + (1 + gamma) * parent2.chromosome)

    return chromosome1, chromosome2
