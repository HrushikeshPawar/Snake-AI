# genetic_algorithm\individual.py

# Learning about abstractclasses
# https://docs.python.org/3/library/abc.html
from abc import ABC, abstractmethod


#  The Individual Class
"""
Two (abstract) properties: fitness and chromosome
Three (abstract) methods: calculate_fitness & encode and decode chromosome
No input while initializing

> This is a good abstract outline for an individual. It will require further implementation,
  but for now it will suffice. It lays a foundation for what all individuals in any Genetic Algorithm need:
    - A way to calculate fitness
    - Encoding and Decoding of the chromosome.

Quote Taken from : Let's Make a Genetic Algorithm! - by Chris Presso - https://chrispresso.io/Lets_Make_A_Genetic_Algorithm#implementation
"""


class Individual(ABC):

    def __init__(self):
        # As we are setting the fitness and chromosome as abstract properties,
        # we do not initialize them here
        # Instead, we initialize them in the subclasses and leave here the setter properties
        pass

    @property
    @abstractmethod
    def fitness(self):
        # Fitness is a property that need to be set in the subclasses
        # Here we just raise an error in case it is not set
        raise Exception('Fitness Property not set, must be defined.')

    @fitness.setter
    @abstractmethod
    def fitness(self, value):
        # This is the setter for our fitness property
        # We cannot let it be set manual in the subclasses, but it needs to use the calculate_fitness method
        # Here we just raise an error in case the method is not used or is tried to set manually
        raise Exception('Fitness Property cannot be set manually, must be calculated. Use calculate_fitness method.')

    @abstractmethod
    def calculate_fitness(self):
        # Must be set in subclass, just raise an error here in case it is not set
        raise Exception('calculate_functionx must be defined.')

    @property
    @abstractmethod
    def chromosome(self):
        # Chromosome is a property that need to be set in the subclasses
        # Here we just raise an error in case it is not set
        raise Exception('Chromosome Property not set, must be defined.')

    @chromosome.setter
    @abstractmethod
    def chromosome(self, value):
        # This is the setter for our chromosome property
        # We cannot let it be set manual in the subclasses, but it needs to use the decode method
        # Here we just raise an error in case the method is not used or is tried to set manually
        # For us it will be an np.array of dimension 1
        raise Exception('Chromosome Property cannot be set manually.')

    @abstractmethod
    def decode_chromosome(self):
        # Must be set in subclass, just raise an error here in case it is not set
        raise Exception('decode_chromosome must be defined.')

    @abstractmethod
    def encode_chromosome(self):
        # Must be set in subclass, just raise an error here in case it is not set
        raise Exception('encode_chromosome must be defined.')
