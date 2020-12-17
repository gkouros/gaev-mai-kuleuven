import numpy as np
import random
from copy import deepcopy
import multiprocessing

from individual import Individual
from selection_operators import k_tournament_selection
from recombination_operators import order_crossover
from mutation_operators import swap_mutation, inversion_mutation
from elimination_operators import lambda_plus_mu_elimination, replace_worst, \
        fitness_sharing_elimination, k_tournament_elimination
from local_search_operators import two_opt, three_opt

from fitness_utils import fitness_std, unique_fitnesses_normed
from adaptivity_utils import adapt_param


class TSPEvolutionaryAlgorithm:

    def __init__(self,
                 distance_matrix: np.array,
                 lambda_: int = 100,
                 mu: int = 20,
                 k: int = 4,
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
        self.local_search_probability = local_search_probability
        self.fs_alpha = fitness_sharing_alpha
        self.fs_sigma = fitness_sharing_sigma

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

        print('Population initialized!')

    def mutation(self, individual):
        prob = random.random()

        if prob < 1:
            return inversion_mutation(individual)
        else:
            return swap_mutation(individual)

    def generate_population(self, distance_matrix, heuristic_search=False
                           ) -> None:
        """ Generates the initial population

        90% random and 10% heuristics

        Args:
            distance_matrix (np.array): The distances between the cities
        """
        self.population = []
        if heuristic_search:
            num_heuristics = int(self.lambda_ * 0.1)
            self.sorted_city_map = self.calc_sorted_city_map()
            heuristic_solutions = self.find_heuristic_solutions(
                num_heuristics, steps=round(self.num_cities))

            #  #  find local optimums of heuristic solutions
            for individual in heuristic_solutions:
                new_route = self.local_search(individual.route,
                                              self.distance_matrix)
                individual.set_route(new_route)

            self.population += heuristic_solutions

        num_randoms = self.lambda_ - len(self.population)

        self.population += [
            Individual(distance_matrix) for _ in range(num_randoms)]

        self.population += heuristic_solutions
        #
        #  for individual in self.population:
        #      new_route = self.local_search(individual.route,
        #                                    self.distance_matrix)
        #      individual.set_route(new_route)

        # calculate mean fitness of initial population
        self.mean_objective = self.calc_mean_objective()
        self.diversity = unique_fitnesses_normed(self.population)

        # sort population
        self.population = sorted(self.population, key=lambda k: k.fitness)

    def find_heuristic_solutions(self, num_heuristics: int, steps: int = -1
                                 ) -> list:
        heuristic_solutions = []
        start_cities = np.random.choice(self.num_cities, num_heuristics)
        heuristic_solutions = [self.find_heuristic_solution(city, steps)
                               for city in start_cities]

        return heuristic_solutions

    def find_heuristic_solution(self, city: int, steps: int = -1) -> Individual:
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

        individual = Individual(self.distance_matrix, new_route)

        return individual

    def calc_sorted_city_map(self) -> dict:
        """ Finds the list of nearest neighbours of each city """
        sorted_city_map = {}

        for city in range(self.distance_matrix.shape[0]):
            sorted_city_map[city] = np.argsort(
                self.distance_matrix[city, :])[1:]

        return sorted_city_map

    @property
    def state(self) -> str:
        """ Returns the state of the optimization """
        return \
                f'#{self.iteration} ' + \
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
            if  self.iteration > improvement_threshold:
                if np.std(self.best_history[-improvement_threshold:]) < 1e-7:
                    converged = True
        else:
            converged = False

        return converged

    def calc_diversity(self) -> dict:
        return unique_fitnesses_normed(self.population)

    def calc_mean_objective(self) -> float:
        """ Returns the mean fitness of the population """
        mean = 0
        num_individuals = len(self.population)
        for individual in self.population:
            mean += individual.actual_fitness / num_individuals

        return mean

    @property
    def best_objective(self) -> float:
        """ Returns the best fitness of the population """
        return self.population[0].fitness

    @property
    def best_solution(self) -> float:
        """ returns the best candidate solution of the population """
        return self.population[0]


    def update(self) -> None:
        """ Performs an iteration of the genetic algorithm """
        self.iteration += 1

        prob_c = np.random.rand()  # probability of recombination
        prob_m = np.random.rand()  # probability of mutation
        prob_l = np.random.rand()  # probability of local search

        all_offspring = []
        for _ in range(self.mu//2):
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

            all_offspring += offspring

        # perform local search
        if prob_l < self.local_search_probability:
            for individual in all_offspring:
                new_route = self.local_search(individual.route,
                                              self.distance_matrix)
                individual.set_route(new_route)

        # elimination
        if self.fitness_sharing: #and self.diversity < 0.3:
            self.population = fitness_sharing_elimination(
                all_offspring, self.population, self.lambda_,
                self.fs_alpha, self.fs_sigma)
        else:
            self.population = lambda_plus_mu_elimination(
            #  self.population = k_tournament_elimination(
            #  self.population = replace_worst
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
