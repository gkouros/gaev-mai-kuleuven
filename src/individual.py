import numpy as np


class Individual:

    def __init__(self, distance_matrix):
        self.size = len(distance_matrix)
        self.distance_matrix = distance_matrix
        self.route = np.random.permutation(self.size)
        self.fitness = self.calc_fitness()

    def __getitem__(self, key):
        if key >= self.size:
            raise ValueError('Index out of bounds')

        return self.route[key]

    def __str__(self):
        return "Route: {self.route}, Total distance: {self.distance}"

    def calc_fitness(self):
        """ Calculates the fitness of the individual as the total distance

        Returns:
            float: The total distance of the route of the individual
        """
        dist = 0

        for idx, from_city in enumerate(self.route):
            to_city = self.route[(idx+1) % self.size]
            dist += self.distance_matrix[from_city, to_city]

        return dist

    def set_route(self, route):
        self.route = route
        self.fitness = self.calc_fitness()
