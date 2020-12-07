import numpy as np
from individual import Individual


def lambda_plus_mu_elimination(offspring: list, population: list, lambda_: int
                              ) -> list:
    """ Performs the (λ + μ)-elimination step of the evolutionary algorithm.

    Args:
        offspring (list): List of the offspring.
        population (list): List of the individuals in a population.
        lambda_ (int): Number of top lambda_ candidates that will be retained.

    Returns:
        new_combined: Top lambda_ candidates that retained.

    """
    combined = population + offspring
    new_combined = sorted(combined, key=lambda k: k.fitness, reverse=False)
    return new_combined[0:lambda_]
