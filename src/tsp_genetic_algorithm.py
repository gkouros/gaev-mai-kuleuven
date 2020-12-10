import numpy as np
import random

from individual import Individual
from selection_operators import k_tournament_selection
from recombination_operators import order_crossover
from mutation_operators import swap_mutation, inversion_mutation
from elimination_operators import lambda_plus_mu_elimination


class TSPGeneticAlgorithm:

    def __init__(self,
                 distance_matrix: np.array,
                 lambda_: int,
                 mu: int,
                 k: int,
                 recombination_probability: float,
                 mutation_probability: float):

        self.lambda_ = lambda_
        self.mu = mu
        self.k = k
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.recombination_probability = recombination_probability
        self.mutation_probability = mutation_probability
        self.iteration = 0
        self.mean_objective = None

        # initialize population
        self.population = None
        self.generate_population(distance_matrix)

        # define operators
        self.select = k_tournament_selection
        self.recombine = order_crossover
        self.mutate = inversion_mutation
        self.eliminate = lambda_plus_mu_elimination

    def generate_population(self, distance_matrix) -> None:
        """ Generates the initial population

        90% random and 10% heuristics

        Args:
            distance_matrix (np.array): The distances between the cities
        """
        num_randoms = round(0.9 * self.lambda_)
        self.population = [
            Individual(distance_matrix) for _ in range(num_randoms)]

        # compute heuristic solutions
        num_heuristics = self.lambda_ - num_randoms
        self.sorted_city_map = self.calc_sorted_city_map()
        heuristic_solutions = self.find_heuristic_solutions(num_heuristics)
        self.population += heuristic_solutions

    def find_heuristic_solutions(self, num_heuristics: int) -> list:
        heuristic_solutions = []
        start_cities = np.random.choice(self.num_cities, num_heuristics)
        heuristic_solutions = [
            self.find_heuristic_solution(city) for city in start_cities]

        return heuristic_solutions

    def find_heuristic_solution(self, city: int) -> Individual:
        new_route = [city]

        while len(new_route) != len(self.distance_matrix):
            nearest_neighbours = self.sorted_city_map[new_route[-1]]
            for next_nn in nearest_neighbours:
                if next_nn not in new_route:
                    new_route.append(next_nn)
                    break

        individual = Individual(self.distance_matrix)
        individual.set_route(new_route)

        return individual

    def calc_sorted_city_map(self) -> dict:
        sorted_city_map = {}

        for city in range(self.distance_matrix.shape[0]):
            sorted_city_map[city] = np.argsort(
                self.distance_matrix[city, :])[1:]

        return sorted_city_map

    @property
    def state(self) -> str:
        """ Returns the state of the optimization """
        return f'#{self.iteration} ' +\
               f'Best Objective: {self.best_objective} - ' + \
               f'Mean Objective: {self.mean_objective}'

    def converged(self) -> None:
        """ Returns True if the optimization has converged """
        return self.mean_objective is not None and \
            abs(self.best_objective - self.mean_objective) < 1e-8

    def calc_mean_objective(self) -> float:
        """ Returns the mean fitness of the population """
        mean = 0
        num_individuals = len(self.population)
        for individual in self.population:
            mean += individual.fitness / num_individuals

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

        all_offspring = []
        for _ in range(self.mu//2):
            # selection
            parents = [self.select(self.population, k=self.k)
                       for _ in range(2)]

            # recombination
            if prob_c < self.recombination_probability:
                offspring = self.recombine(*parents)
            else:
                offspring = parents

            # mutation
            if prob_m < self.mutation_probability:
                offspring = [self.mutate(o) for o in offspring]

            all_offspring += offspring

        # elimination
        self.population = self.eliminate(all_offspring, self.population,
                                         self.lambda_)

        # update mean objective
        self.mean_objective = self.calc_mean_objective()
