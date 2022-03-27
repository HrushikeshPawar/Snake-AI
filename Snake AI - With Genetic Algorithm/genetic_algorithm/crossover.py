# genetic_algorithm/selection.py

import numpy as np
from typing import Tuple


# Different Selection Methods
def simulated_binary_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    eta    : float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulated Binary Crossover
    """
    # Calculating the Gamma Value
    random_number = np.random.random(parent1.shape)
    gamma = np.empty(parent1.shape)

    # When gamma < 0.5
    gamma[random_number < 0.5] = (2 * random_number[random_number < 0.5]) ** (1 / (eta + 1))

    # When gamma >= 0.5
    gamma[random_number >= 0.5] = (2 * (1 - random_number[random_number >= 0.5])) ** (-1 / (eta + 1))

    # Calculating child1 chromosome
    chromosome1 = 0.5 * ((1 + gamma) * parent1 + (1 - gamma) * parent2)
    chromosome2 = 0.5 * ((1 - gamma) * parent1 + (1 + gamma) * parent2)

    return chromosome1, chromosome2


def uniform_binary_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    mask = np.random.uniform(0, 1, size=offspring1.shape)
    offspring1[mask > 0.5] = parent2[mask > 0.5]
    offspring2[mask > 0.5] = parent1[mask > 0.5]

    return offspring1, offspring2


def single_point_binary_crossover(parent1: np.ndarray, parent2: np.ndarray, major='r') -> Tuple[np.ndarray, np.ndarray]:
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    rows, cols = parent2.shape
    row = np.random.randint(0, rows)
    col = np.random.randint(0, cols)

    if major.lower() == 'r':
        offspring1[:row, :] = parent2[:row, :]
        offspring2[:row, :] = parent1[:row, :]

        offspring1[row, :col + 1] = parent2[row, :col + 1]
        offspring2[row, :col + 1] = parent1[row, :col + 1]
    elif major.lower() == 'c':
        offspring1[:, :col] = parent2[:, :col]
        offspring2[:, :col] = parent1[:, :col]

        offspring1[:row + 1, col] = parent2[:row + 1, col]
        offspring2[:row + 1, col] = parent1[:row + 1, col]

    return offspring1, offspring2
