import numpy as np
from individual import Individual


def fitness_std(population: list) -> float:
    fitnesses = [individual.fitness for individual in population]
    return np.std(fitnesses)


def unique_fitnesses(population: list) -> int:
    fitnesses = [individual.fitness for individual in population]
    unique_fitnesses = set(fitnesses)
    return len(unique_fitnesses)


def unique_fitnesses_normed(population: list) -> float:
    num_unique_fitnesses = unique_fitnesses(population)
    return num_unique_fitnesses / len(population)
