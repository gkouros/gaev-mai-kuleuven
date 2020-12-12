#!/usr/bin/env python3

import numpy as np
import Reporter
from tsp_genetic_algorithm import TSPGeneticAlgorithm


class TSPDemo:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def optimize(self, filename):
        """ The evolutionary algorithm's main loop

        Args:
            filename (str): The name of the file containing the TSP problem
        """

        # Read distance matrix from file.
        file = open(filename)
        distance_matrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Your code here.

        # Initialize a genetic algorithm instance using the given TSP problem
        ga = TSPGeneticAlgorithm(distance_matrix, lambda_=100, mu=20, k=4,
                                 recombination_probability=0.9,
                                 mutation_probability=0.1)

        while not ga.converged():

            # Your code here.

            # perform an optimization step
            ga.update()

            # extract results of current generation
            mean_objective = ga.mean_objective
            best_objective = ga.best_objective
            best_solution = ga.best_solution

            # print state of generation
            print(ga.state)

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing
            #    the best solution with city numbering starting from 0
            time_left = self.reporter.report(mean_objective,
                                             best_objective,
                                             best_solution)

            if time_left < 0:
                break

        # Your code here.

        # print final state of optimization
        if time_left < 0:
            print('Timed out')
        else:
            print('Converged!')

        return 0


if __name__ == '__main__':
    demo = TSPDemo()
    #  demo.optimize('datasets/tour29.csv')
    #  demo.optimize('datasets/tour100.csv')
    demo.optimize('datasets/tour194.csv')
    #  demo.optimize('datasets/tour929.csv')
