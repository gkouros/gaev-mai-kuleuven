import numpy as np
from individual import Individual


def k_tournament_selection(population: list, k: int = 3):
    """ Performs k-tournament selection

    Args:
        population (list): List of the individuals in a population
        k (int): Number of individuals that engage in the tournament selection

    Returns:
        Individual: The individual that won the tournament selection

    """

    # Generate a list of k random indices from the population
    indices = np.random.choice(len(population), k)

    # create a list of individuals based on the selected indices
    selected = [population[idx] for idx in indices]

    # calculate and store in a list the fitness for all k individuals
    fitnesses = [s.fitness for s in selected]

    # find the index of the individual with the best fitness value out of the
    # k selected individuals
    min_idx = np.argmin(fitnesses)

    return selected[min_idx]
