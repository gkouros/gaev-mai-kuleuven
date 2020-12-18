import numpy as np
from individual import Individual


def calc_fitness(route: list, distance_matrix: np.array) -> float:
    """ Calculates the fitness of the individual as the total distance

    Args:
        route (list) : The route to calculate the fitness of
        distance_matrix (np.array): The cost matrix of the city edges

    Returns:
        float: The total distance of the route of the individual
    """
    dist = 0
    size = len(distance_matrix)

    for idx, from_city in enumerate(route):
        to_city = route[(idx+1) % size]
        dist += distance_matrix[from_city, to_city]

    return dist


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
