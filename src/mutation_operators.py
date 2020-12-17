import numpy as np
from individual import Individual


def swap_mutation(individual: Individual) -> Individual:
    """ Performs swap mutation

    Two genes are randomly selected and their values are swapped.

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
