import numpy as np
from individual import Individual
from itertools import permutations, chain
from fitness_utils import calc_fitness


def swap_mutation(individual: Individual) -> Individual:
    """ Performs swap mutation

    Two genes are randomly selected and their values are swapped.

    Args:
        individual (Individual): Individual that will be mutated
        sigma (int): The mutation strength
    Returns:
        individual (Individual): The mutated individual.
    """
    route = individual.route
    sigma = individual.sigma
    gamma = individual.gamma

    for _ in range(round(individual.sigma)):
        idx1, idx2 = np.random.choice(len(route), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]

    sigma += gamma * (np.random.random() - 0.5)
    sigma = max(0, sigma)

    mutated_individual = Individual(individual.distance_matrix, route,
                                    sigma, gamma)

    return mutated_individual


def inversion_mutation(individual: Individual) -> Individual:
    """ Performs inversion mutation

    A random sequence of genes is selected and the order of the genes in that
    sequence is reversed.

    Args:
        individual (Individual): Individual that will be mutated with a mutation rate alpha.
        sigma (int): The mutation strength
    Returns:
        individual (Individual): The mutated individual.
    """
    route = individual.route
    sigma = individual.sigma
    gamma = individual.gamma

    for _ in range(round(individual.sigma)):
        idx1, idx2 = np.sort(np.random.choice(len(route), 2))
        route[idx1:idx2] = route[idx1:idx2][::-1]

    sigma += gamma * (np.random.random() - 0.5)
    sigma = max(0, sigma)

    mutated_individual = Individual(individual.distance_matrix, route,
                                    sigma, gamma)

    return mutated_individual

def greedy_mutation(individual: Individual) -> Individual:
    sigma = individual.sigma
    gamma = individual.gamma
    num_cities = len(individual.distance_matrix)
    route = individual.route
    k = 4  # np.random.choice(range(4, 8))

    for _ in range(int(individual.sigma)):
        nodes = sorted(np.random.choice(num_cities, k))
        segments = []
        start = 0

        for idx in range(k):
            segments += [route[start:nodes[idx]+1]]
            start = nodes[idx]+1

        segments += [route[start:]]

        best_route = route
        best_fitness = individual.fitness
        perms = permutations(segments)

        for perm in perms:
            new_route = list(chain.from_iterable(perm))
            new_fitness = calc_fitness(new_route, individual.distance_matrix)

            if new_fitness < best_fitness:
                best_fitness = new_fitness
                best_route = new_route
                route = new_route

    mutated_individual = Individual(individual.distance_matrix, best_route,
                                    sigma, gamma)

    return mutated_individual
