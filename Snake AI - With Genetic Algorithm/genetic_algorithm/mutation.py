# genetic_algorithm/mutation.py

import numpy as np
from typing import List
from .individual import Individual


# The Mutation methods
# Copied as it is from - https://chrispresso.io/Lets_Make_A_Genetic_Algorithm#mutation

def gaussian_mutation(
    individual: Individual,
    prob_mutation: float,
    mu: List[float] = None,
    sigma: List[float] = None
) -> None:
    """
    Perform a gaussian mutation for each gene in an individual with probability, prob_mutation.

    If mu and sigma are defined then the gaussian distribution will be drawn from that,
    otherwise it will be drawn from N(0, 1) for the shape of the individual.
    """
    chromosome = individual.chromosome
    # Determine which genes will be mutated
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    # If mu and sigma are defined, create gaussian distribution around each one
    if mu and sigma:
        gaussian_mutation = np.random.normal(mu, sigma)
    # Otherwise center around N(0,1)
    else:
        gaussian_mutation = np.random.normal(size=chromosome.shape)
    # Update
    chromosome[mutation_array] += gaussian_mutation[mutation_array]


def random_uniform_mutation(
    individual: Individual,
    prob_mutation: float,
    low: List[float],
    high: List[float]
) -> None:
    """
    Randomly mutate each gene in an individual with probability, prob_mutation.
    If a gene is selected for mutation it will be assigned a value with uniform probability
    between [low, high).
    """
    chromosome = individual.chromosome
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    uniform_mutation = np.random.uniform(low, high, size=chromosome.shape)
    chromosome[mutation_array] = uniform_mutation[mutation_array]


def uniform_mutation_with_respect_to_best_individual(
    individual: Individual,
    best_individual: Individual,
    prob_mutation: float
) -> None:
    """
    Ranomly mutate each gene in an individual with probability, prob_mutation.
    If a gene is selected for mutation it will nudged towards the gene from the best individual.

    """
    chromosome = individual.chromosome
    best_chromosome = best_individual.chromosome
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    uniform_mutation = np.random.uniform(size=chromosome.shape)
    delta = (best_chromosome[mutation_array] - chromosome[mutation_array])
    chromosome[mutation_array] += uniform_mutation[mutation_array] * delta
