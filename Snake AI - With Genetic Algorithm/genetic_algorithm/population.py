# genetic_algorithm\individual.py

from typing import List
from .individual import Individual
import numpy as np

# The Population Class
"""
It is a collection of individuals from the Individual class
4 Properties: Number of Individuals, Number of Genes, Average Fitness of Population, fittest individual
3 Methods   : Calculate Average Fitness, Get Fittest Individual, Calculate Fitness of Individual and update it
"""


class Population:

    def __init__(self, individuals: List[Individual]) -> None:
        self.individuals = individuals

    @property
    def number_of_individuals(self) -> int:
        return len(self.individuals)

    @number_of_individuals.setter
    def number_of_individuals(self, value: int) -> None:
        raise Exception("Cannot set number of individuals manually. Must change the Population.individuals list.")

    @property
    def number_of_genes(self) -> int:
        return self.individuals[0].chromosome.shape[1]

    @number_of_genes.setter
    def number_of_genes(self, value: int) -> None:
        raise Exception("Cannot set number of genes manually. Must change the Population.individuals list.")

    @property
    def average_fitness(self) -> float:
        return np.mean(np.array([individual.fitness for individual in self.individuals]))

    @average_fitness.setter
    def average_fitness(self, value: float) -> None:
        raise Exception("Cannot set average fitness manually. This is read-only property.")

    @property
    def fittest_individual(self) -> Individual:
        return max(self.individuals, key=lambda individual: individual.fitness)

    @fittest_individual.setter
    def fittest_individual(self, value: Individual) -> None:
        raise Exception("Cannot set fittest individual manually. This is read-only property.")

    def calculate_fitness(self) -> None:
        for individual in self.individuals:
            individual.calculate_fitness()

    def get_fitness_standard_deviation(self) -> float:
        return np.std(np.array([individual.fitness for individual in self.individuals]))
