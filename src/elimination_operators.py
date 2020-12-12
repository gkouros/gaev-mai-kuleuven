""" Elimination methods for evolutionary algorithms """


def fitness_sharing_elimination(
        offspring: list, population: list, lambda_: int) -> list:
    """ Performs the (λ+μ)-elimination with fitness sharing

    Args:
        offspring (list): List of the offspring
        population (list): List of the individuals in a population
        lambda_ (int): Number of top lambda_ candidates that will be retained

    Returns:
        new_combined: Top lambda_ candidates that are retained
    """
    return lambda_plus_mu_elimination(offspring, population, lambda_, True)


def lambda_plus_mu_elimination(
        offspring: list, population: list, lambda_: int,
        fitness_sharing=False) -> list:
    """ Performs the (λ+μ)-elimination step of the evolutionary algorithm

    Args:
        offspring (list): List of the offspring
        population (list): List of the individuals in a population
        lambda_ (int): Number of top lambda_ candidates that will be retained

    Returns:
        new_combined: Top lambda_ candidates that are retained

    """
    # combine population and offspring
    combined = population + offspring

    # update fitness based on fitness sharing scheme
    if fitness_sharing:
        for idx in range(1, len(combined)):
            combined[idx].calc_shared_fitness(
                combined[:idx] + combined[idx+1:], alpha=1/4, sigma=2)

    # sort new population
    combined = sorted(combined, key=lambda k: k.fitness, reverse=False)

    # prune and return new population
    return combined[:lambda_]


def replace_worst(offspring: list, population: list, lambda_: int) -> list:
    """ Performs the (λ-μ)-elimination step of the evolutionary algorithm.

    Args:
        offspring (list): List of the offspring
        population (list): List of the individuals in a population
        lambda_ (int): Number of candidates that will be retained

    Returns:
        new_combined: The surviving candidates
    """
    combined = population[:-len(offspring)] + offspring
    new_combined = sorted(combined, key=lambda k: k.fitness, reverse=False)
    return new_combined[0:lambda_]
