""" Evolutionary Algorithm implementation for solving the
Travelling Salesman Problem
"""

__author__ = 'Georgios Kouros (r0816917)'
__email__ = 'georgios.kouros@student.kuleuven.be'
__maintainer__ = "Georgios Kouros"
__credits__ = 'Jeffrey Quickens, Konstantinos Gkentsidis'

import time
import random
from copy import deepcopy
from itertools import permutations, chain
import numpy as np

import Reporter


# Modify the class name to match your student number.
class r0816917:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):

        # Read distance matrix from file.
        file = open(filename)
        distance_matrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Your code here.
        # create an evolutionary algorithm object
        ev = TSPEvolutionaryAlgorithm(
            distance_matrix,
            lambda_=10,
            mu=5,
            k=4,
            recombination_probability=0.9,
            mutation_probability=0.9,
            local_search_probability=1,
            mutation_strength=1,
            fitness_sharing_alpha=1,
            fitness_sharing_sigma=len(distance_matrix)//5)

        #  print(ev.get_config())

        while not ev.converged(
                improvement_criterion=True,
                improvement_threshold=100):

            # Your code here.

            # run an iteration for a new generation
            ev.update()

            # extract results of current generation
            mean_objective = ev.mean_objective
            best_objective = ev.best_objective
            best_solution = np.array(ev.best_solution.route)

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best
            #  solution with city numbering starting from 0
            time_left = self.reporter.report(mean_objective,
                                             best_objective,
                                             best_solution)

            # print state of generation
            print(ev.state, f'- time_left: {int(time_left)}')

            if time_left < 0:
                break

        # Your code here.
        #  if time_left < 0:
        #      print('Timed out!')
        #  else:
        #      print('Converged!')

        return 0

###############################################################################
########################### Representation ####################################
###############################################################################


class Individual:
    """ Wrapper class for the representation of a candidate solution of TSP """

    def __init__(self, distance_matrix, route=None, sigma=1, gamma=2):
        """ Initializes a candidate solution

        Args:
            distance_matrix (np.array): The cost matrix of the cities
            route (list): The route of the candidate solution (random if None)
            sigma (int): The mutation strength of an individual
            gamma (int): The update weight of the mutation strength
        """
        self.size = len(distance_matrix)
        self.distance_matrix = distance_matrix
        self.route = None
        self.fitness = None
        self.edges = None
        self.sigma = max(1, sigma + gamma * (np.random.random() - 0.5))
        self.gamma = gamma

        if route is None:
            self.set_route(np.random.permutation(self.size))
        else:
            self.set_route(route)

    def __str__(self) -> str:
        return f"Route: {self.route}, Total distance: {self.fitness}"

    def __getitem__(self, key: int) -> int:
        """ Returns the city corresponding to the given key in the route

        Args:
            key (int): The idx of the city to return from the route

        Returns:
            int: The city corresponding to the key
        """
        if key >= self.size:
            raise ValueError('Index out of bounds')

        return self.route[key]

    def calc_fitness(self) -> float:
        """ Calculates the fitness of the individual as the total distance

        Returns:
            float: The total distance of the route of the individual
        """
        dist = 0

        for idx, from_city in enumerate(self.route):
            to_city = self.route[(idx+1) % self.size]
            dist += self.distance_matrix[from_city, to_city]

        return dist

    def set_route(self, route: list) -> None:
        """ Sets the route/chromosome of the individual

            Also updates the fitness and the edges list

            Args:
                route (list): The sequence of cities
        """
        if len(route) != len(set(route)):
            raise ValueError('Invalid udpate of route of individual')

        self.route = route
        self.fitness = self.calc_fitness()
        self.edges = [(route[idx], route[(idx + 1) % self.size])
                      for idx in range(self.size)]

    def distance_to(self, individual) -> int:
        """ Calculates the distance between the indiividual and another

        This distance is defined as the number of different edges between the
        individuals

        Args:
            individual (Individual): The individual to measure the distance to

        Returns:
            int: The distance to the given individuals
        """
        edges1 = self.edges
        edges2 = individual.edges
        intersection = list(set(edges1) & set(edges2))
        num_edges = len(self.edges)

        return num_edges - len(intersection)


###############################################################################
################################# EV Class ####################################
###############################################################################


class TSPEvolutionaryAlgorithm:

    def __init__(self,
                 distance_matrix: np.array,
                 lambda_: int = 100,
                 mu: int = 20,
                 k: int = 3,
                 recombination_probability: float = 0.9,
                 mutation_probability: float = 0.1,
                 local_search_probability: float = 0.3,
                 mutation_strength: int = 1,
                 fitness_sharing_alpha: float = 1,
                 fitness_sharing_sigma: float = 1):

        # params
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.iteration = 0

        # hyperparameters
        self.lambda_ = lambda_
        self.mu = mu
        self.k = k
        self.recombination_probability = recombination_probability
        self.mutation_probability = mutation_probability
        self.mutation_strength = mutation_strength
        self.local_search_probability = local_search_probability
        self.fs_alpha = fitness_sharing_alpha
        self.fs_sigma = fitness_sharing_sigma
        self.counts = [1, 1, 1]

        # flags
        self.fitness_sharing = True
        self.heuristic_search = True

        # define operators
        self.selection = k_tournament_selection
        self.recombination = order_crossover
        self.local_search = two_opt

        # initialize metrics
        self.mean_objective = None
        self.diversity = None

        # initialize population
        self.population = None
        self.generate_population(distance_matrix,
                                 heuristic_search=self.heuristic_search)

        # metrics history
        self.best_history = [self.best_objective]
        self.mean_history = [self.mean_objective]
        self.diversity_history = [self.diversity]

    def mutation(self, individual: Individual):
        """ Wrapper of mutation operators

        Args:
            individual (Individual): The individual to mutate

        Returns:
            Individual: The mutated individual
        """
        ''' adaptive selection of mutation operator
        #  s = sum(self.counts)
        #  f = [c / s for c in self.counts]
        #  F = [f[0], f[0]+f[1], 1]
        '''
        F = [0, 1, 1]  # only greedy mutation

        prob = np.random.random()

        if prob < F[0]:
            new_individual = inversion_mutation(individual)
            if new_individual.fitness < individual.fitness:
                self.counts[0] += 1
        elif prob < F[1]:
            new_individual = greedy_mutation(individual)
            if new_individual.fitness < individual.fitness:
                self.counts[1] += 1
        else:
            new_individual = swap_mutation(individual)
            if new_individual.fitness < individual.fitness:
                self.counts[2] += 1

        return new_individual

    def generate_population(
            self, distance_matrix, heuristic_search=False) -> None:
        """ Generates the initial population 90% random and 10% heuristics

        Args:
            distance_matrix (np.array): The distances between the cities
        """
        self.population = []
        if heuristic_search:
            num_heuristics = int(self.lambda_ * 0.2)
            self.sorted_city_map = self.calc_sorted_city_map()
            heuristic_solutions = self.find_heuristic_solutions(
                num_heuristics, steps=round(self.num_cities))

            # optimize heuristic candidate solutions
            #  for individual in heuristic_solutions:
            #      new_route = self.local_search(individual.route,
            #                                    self.distance_matrix)
            #      individual.set_route(new_route)

            self.population += heuristic_solutions

        num_randoms = self.lambda_ - len(self.population)
        random_solutions = [
            Individual(distance_matrix, sigma=self.mutation_strength)
            for _ in range(num_randoms)]
        self.population = random_solutions + heuristic_solutions

        # optimize whole initial population
        #  for individual in self.population:
        #      new_route = self.local_search(individual.route,
        #                                    self.distance_matrix)
        #      individual.set_route(new_route)

        # sort population
        self.population = sorted(self.population, key=lambda k: k.fitness)

        # calculate initial mean fitness and diversity
        self.mean_objective = self.calc_mean_objective()
        self.diversity = unique_fitnesses_normed(self.population)

    def find_heuristic_solutions(
            self, num_heuristics: int, steps: int = -1) -> list:
        """ Finds a number of greedy heuristic solutions

        Args:
            num_heuristics (int): The number of solutions to find
            step (int): The number of greedy steps to perform

        Returns:
            list: The list of greedy heuristic solutions
        """
        heuristic_solutions = []
        start_cities = np.random.choice(self.num_cities, num_heuristics)
        heuristic_solutions = [self.find_heuristic_solution(city, steps)
                               for city in start_cities]

        return heuristic_solutions

    def find_heuristic_solution(
            self, city: int, steps: int = -1) -> Individual:
        """ Finds a greedy heuristic solution starting from a given city

        Args:
            city (int): The city to start from
            step (int): The number of greedy steps to perform

        Returns:
            Individual: A greedy heuristic solution
        """
        available_cities = list(range(self.num_cities))
        new_route = [city]
        available_cities.remove(city)

        if steps == -1:
            steps = self.num_cities

        while len(new_route) != len(self.distance_matrix) and steps > 0:
            nearest_neighbours = self.sorted_city_map[new_route[-1]]
            for next_nn in nearest_neighbours:
                if next_nn not in new_route:
                    new_route.append(next_nn)
                    available_cities.remove(next_nn)
                    break

            steps -= 1

        random.shuffle(available_cities)
        new_route += available_cities

        individual = Individual(self.distance_matrix, new_route,
                                sigma=self.mutation_strength)

        return individual

    def calc_sorted_city_map(self) -> dict:
        """ Finds the list of nearest neighbours of each city """
        sorted_city_map = {}

        for city in range(self.distance_matrix.shape[0]):
            sorted_city_map[city] = np.argsort(
                self.distance_matrix[city, :])[1:]

        return sorted_city_map

    def get_config(self) -> None:
        """ Returns a string with the configuration of the algorithm """
        return 'TSP EA Config\n' + \
            '-------------\n' + \
            f'lambda = {self.lambda_}\n' + \
            f'mu = {self.mu}\n' + \
            f'k = {self.k}\n' + \
            f'p_c = {self.recombination_probability}\n' + \
            f'p_m = {self.mutation_probability}\n' + \
            f'p_l = {self.local_search_probability}\n' + \
            f'sigma_mu = {self.mutation_strength}\n' + \
            f'fs_alpha = {self.fs_alpha}\n' + \
            f'fs_sigma = {self.fs_sigma}\n' + \
            '-------------'

    @property
    def state(self) -> str:
        """ Returns the state of the optimization """
        return f'#{self.iteration} ' + \
            f'Best Objective: {self.best_objective} - ' + \
            f'Mean Objective: {self.mean_objective} - ' + \
            f'Diversity: {self.diversity}'

    def converged(self,
                  improvement_criterion: bool = False,
                  improvement_threshold: int = 50,
                  max_iterations: int = 0
                  ) -> None:
        """ Returns True if the optimization has converged """
        converged = False

        if max_iterations:
            if self.iteration == max_iterations:
                converged = True
                print('Reached maximum number of iterations')
        elif abs(self.best_objective - self.mean_objective) < 1e-8:
            converged = True
            print('Mean objective reached best objective')
        elif improvement_criterion:
            if self.iteration > improvement_threshold:
                if np.std(self.best_history[-improvement_threshold:]) < 1e-7:
                    converged = True
        else:
            converged = False

        return converged

    def calc_diversity(self) -> dict:
        """ Calculates the diversity of the population """
        return unique_fitnesses_normed(self.population)

    def calc_mean_objective(self) -> float:
        """ Returns the mean fitness of the population """
        num_individuals = len(self.population)
        sum_fitness = sum([individual.fitness
                           for individual in self.population])

        return sum_fitness / num_individuals

    @property
    def best_objective(self) -> float:
        """ Returns the best fitness of the population """
        return self.population[0].fitness

    @property
    def best_solution(self) -> float:
        """ Returns the best candidate solution of the population """
        return self.population[0]

    def update(self) -> None:
        """ Performs an iteration of the genetic algorithm """
        self.iteration += 1

        prob_c = np.random.rand()  # probability for applying recombination
        prob_m = np.random.rand()  # probability for applying mutation
        prob_l = np.random.rand()  # probability for applying local search

        all_offspring = []
        for _ in range(int(np.ceil(self.mu / 2))):
            # selection
            parents = [self.selection(self.population, k=self.k)
                       for _ in range(2)]

            # recombination
            if prob_c < self.recombination_probability:
                offspring = self.recombination(*parents)
            else:
                offspring = parents

            # mutation
            if prob_m < self.mutation_probability:
                offspring = [self.mutation(o) for o in offspring]

            for child in offspring:
                if len(all_offspring) < self.mu:
                    all_offspring += [child]

        # perform local search
        if prob_l < self.local_search_probability:
            #  print('local_search')
            for individual in all_offspring:
                new_route = self.local_search(individual.route,
                                              self.distance_matrix)
                individual.set_route(new_route)

        # elimination
        if self.fitness_sharing:
            self.population = fitness_sharing_elimination(
                all_offspring, self.population, self.lambda_,
                self.fs_alpha, self.fs_sigma)
        else:
            #  self.population = k_tournament_elimination(
            #  self.population = replace_worst
            self.population = lambda_plus_mu_elimination(
                all_offspring, self.population, self.lambda_)

        # update mean objective
        self.mean_objective = self.calc_mean_objective()

        # calculate diversity
        self.diversity = self.calc_diversity()

        # update metrics
        self.best_history.append(self.best_objective)
        self.mean_history.append(self.mean_objective)
        self.diversity_history.append(self.diversity)

        #  sigmas = [ind.sigma for ind in self.population]
        #  print(min(sigmas), max(sigmas), np.std(sigmas))


###############################################################################
############################### Operators #####################################
###############################################################################


######################## Selection Operators #################################

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

    return deepcopy(selected[min_idx])


##################### Recombination Operators #################################

def order_crossover(parent1, parent2):
    """ Performs the order crossover operator and produces two offspring

    Args:
        parent1 (Individual): First candidate solution to be recombination
        parent2 (Individual): Second candidate solution to be recombination

    Returns:
        Individual: First offspring produced by operator
        Individual: Second offspring produced by operator
    """
    size = parent1.size
    parents = (parent1, parent2)

    # randomly pick points for recombination
    start, end = sorted(np.random.choice(size, 2))

    # initialize children
    children = [[None] * size for _ in range(2)]
    children[0][start:end] = parents[0].route[start:end]
    children[1][start:end] = parents[1].route[start:end]

    # fill rest of the route of the children
    for child, parent in zip(children, parents[::-1]):
        idx1 = end % size
        idx2 = end % size

        while any(np.equal(child, None)):
            if parent[idx2] not in child:
                child[idx1] = parent[idx2]
                idx1 = (idx1 + 1) % size

            idx2 = (idx2 + 1) % size

    sigma = (parent1.sigma + parent2.sigma) / 2

    try:
        children = [Individual(parent.distance_matrix, child,
                               sigma=sigma, gamma=parent1.gamma)
                    for child, parent in zip(children, parents)]
    except ValueError as err:
        print(f'Recombination failed [{err}]. Returning parents instead.')
        children = [deepcopy(parent) for parent in parents]

    return children


##################### Mutation Operators #################################

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
        individual (Individual): Individual that will be mutated
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
    """ Mutation operator that greedily connects four segments of a route

    Args:
        individual (Individual): The candidate solution to mutate

    Returns:
        Individual: The mutated individual
    """
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
                                    individual.sigma, individual.gamma)

    return mutated_individual


####################### Elimination Operators #################################

def fitness_sharing_elimination(
        offspring: list, population: list, lambda_: int,
        alpha: float, sigma: int) -> list:
    """ Performs the (λ+μ)-elimination with fitness sharing

    Args:
        offspring (list): List of the offspring
        population (list): List of the individuals in a population
        lambda_ (int): Number of top lambda_ candidates that will be retained

    Returns:
        new_combined: Top lambda_ candidates that are retained
    """
    # combine population and offspring
    combined = population + offspring

    # sort new population
    combined = sorted(combined, key=lambda k: k.fitness, reverse=False)

    survivors = [combined[0]]

    for _ in range(lambda_):
        fitnesses = [calc_shared_fitnesses(combined, survivors, alpha, sigma)]
        idx = np.argmin(fitnesses)
        survivors.append(combined[idx])

    return survivors


def lambda_plus_mu_elimination(
        offspring: list, population: list, lambda_: int):
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

    # sort new population
    combined = sorted(combined, key=lambda k: k.fitness, reverse=False)

    # pick top lambda candidates
    combined = combined[:lambda_]

    return combined


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


def k_tournament_elimination(offspring: list, population: list, lambda_: int,
                             k: int = 3) -> list:
    """ Performs elimnination with k-tournament selection

    Args:
        offspring (list): List of the offspring
        population (list): List of the individuals in a population
        lambda_ (int): Number of candidates that will be retained
        k (int): The number of participants per tournament

    Returns:
        new_combined: The surviving candidates
    """
    combined = population + offspring
    combined = sorted(combined, key=lambda k: k.fitness, reverse=False)
    new_population = [combined[0]] + \
        [k_tournament_selection(combined, k) for _ in range(lambda_-1)]
    return new_population


####################### Local Search Operators ################################

def two_opt(route: list, distance_matrix: np.array, timeout: float = 0.2):
    """ The local search operator 2-opt

    Args:
        route (list): The route to optimize
        distance_matrix (np.array): The cost matrix of city distances
        timeout (int): The maximum allowed time limit for performing 2-opt

    Returns:
        list: The optimized route
    """
    best = deepcopy(route)
    improved = True
    start_time = time.time()

    while improved and time.time() - start_time < timeout:

        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue

                if cost_change(distance_matrix,
                               best[i - 1], best[i],
                               best[j - 1], best[j]) <= 0:
                    best[i:j] = best[j - 1:i - 1:-1]
                    improved = True

    return best

def cost_change(distance_matrix: np.array, n1: int, n2: int, n3: int, n4: int):
    """ Calculates the cost of an edge swap

    Args:
        distance_matrix (np.array): The cost matrix of city distances
        n1 (int): The idx of node1
        n2 (int): The idx of node2
        n3 (int): The idx of node3
        n4 (int): The idx of node4

    Return:
        Int: The calcluated cost change
    """
    return distance_matrix[n1][n3] + distance_matrix[n2][n4] - \
        distance_matrix[n1][n2] - distance_matrix[n3][n4]


############################## Utilities ######################################

def calc_shared_fitnesses(population: list, survivors: list,
                          alpha: float = 1, sigma: int = 1) -> None:
    """ Calculates the fitness of a population given an existing survivor list

    Args:
        population (list): The population to calculate the shared fitness of
        survivors (list): The list of survivors affecting neighbourhood
        alpha (float):
        sigma (int):

    Returns:
        float: The total distance of the route of the individual
    """
    fitnesses = np.array([individual.fitness for individual in population])
    dists = np.array(
        [[individual.distance_to(survivor)
          for survivor in survivors]
            for individual in population])

    shared = (1 - (dists / sigma) ** alpha)
    shared *= np.array(dists <= sigma)
    sum_shared = np.sum(shared, axis=1)
    shared_fitnesses = fitnesses * sum_shared
    shared_fitnesses = np.where(np.isnan(shared_fitnesses),
                                np.inf, shared_fitnesses)

    return shared_fitnesses


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
    """ Calculates the standard deviation of a population

    Args:
        population (list): The list of individual candidate solutions

    Returns:
        float: The standard deviation of the population
    """
    fitnesses = [individual.fitness for individual in population]

    return np.std(fitnesses)


def unique_fitnesses(population: list) -> int:
    """ Calculates the number of unique fitnesses in the population

    Args:
        population (list): The list of individual candidate solutions

    Returns:
        int: The number of unique fitnesses in the population
    """
    fitnesses = [individual.fitness for individual in population]
    unique = set(fitnesses)

    return len(unique)


def unique_fitnesses_normed(population: list) -> float:
    """ Calculates the normalized number of unique fitnesses in the population

    Args:
        population (list): The list of individual candidate solutions

    Returns:
        int: The normalized number of unique fitnesses in the population
    """
    num_unique_fitnesses = unique_fitnesses(population)

    return num_unique_fitnesses / len(population)
