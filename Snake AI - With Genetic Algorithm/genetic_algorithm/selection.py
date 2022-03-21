# genetic_algorithm/Selection.py

import numpy as np
from typing import List
from .population import Population
from .individual import Individual

# Different Selection Methods


# The Roulette Wheel Selection Method
def roulette_wheel_selection(population: Population, number_of_individuals: int) -> List[Individual]:

    selection = []
    wheel = np.sum(individual.fitness for individual in population.individuals)

    for _ in range(number_of_individuals):
        random_number = np.random.uniform(0, wheel)
        current_number = 0
        for individual in population.individuals:
            current_number += individual.fitness
            if current_number >= random_number:
                selection.append(individual)
                break

    return selection
