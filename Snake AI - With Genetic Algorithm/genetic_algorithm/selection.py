# genetic_algorithm/Selection.py

import numpy as np
import random
from typing import List
from .population import Population
from .individual import Individual


# Different Selection Methods
# The Elite Selection Method
def elitism_selection(population: Population, num_individuals: int) -> List[Individual]:
    individuals = sorted(population.individuals, key=lambda individual: individual.fitness, reverse=True)
    return individuals[:num_individuals]


# The Roulette Wheel Selection Method
def roulette_wheel_selection(population: Population, number_of_individuals: int) -> List[Individual]:

    selection = []
    wheel = np.sum(individual.fitness for individual in population.individuals)

    for _ in range(number_of_individuals):
        pick = random.uniform(0, wheel)
        current = 0
        for individual in population.individuals:
            current += individual.fitness
            if current > pick:
                selection.append(individual)
                break

    return selection
