import numpy as np
from individual import Individual


def swap_mutation(individual: Individual) -> Individual:
    """ Performs swap mutation

    Two genes are randomly selected and their values are swapped.

    Args:
        individual (Individual): Individual that will be mutated with a mutation rate alpha.
    Returns:
        individual (Individual): The mutated individual.
    """
    i, j = np.random.choice(len(individual.route), 2)
    individual.route[i], individual.route[j] = \
        individual.route[j], individual.route[i]
    return individual


def inversion_mutation(individual: Individual) -> Individual:
    """ Performs inversion mutation

    A random sequence of genes is selected and the order of the genes in that
    sequence is reversed.

    Args:
        individual (Individual): Individual that will be mutated with a mutation rate alpha.
    Returns:
        individual (Individual): The mutated individual.
    """
    idx1, idx2 = np.sort(np.random.choice(len(individual.route), 2))
    individual.route[idx1:idx2] = \
            individual.route[idx1:idx2][::-1]
    return individual
