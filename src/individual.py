import numpy as np


class Individual:
    """ A candidate solution class for the TSP problem """

    def __init__(self, distance_matrix, route=None, sigma=10, gamma=2):
        self.size = len(distance_matrix)
        self.distance_matrix = distance_matrix
        self.route = None
        self.fitness = None
        self.actual_fitness = None
        self.edges = None
        self.sigma = sigma
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
        self.actual_fitness = self.fitness = self.calc_fitness()
        self.edges = [(route[idx], route[(idx + 1) % self.size])
                      for idx in range(self.size)]

    def distance_to(self, individual) -> int:
        """ Calculates the distance between two individuals

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


    def calc_shared_fitness(self, population=None, alpha=1, sigma=1) -> None:
        if population is None:
            return

        # calculate distances of individual to population
        dists = np.array([self.distance_to(ind) for ind in population])

        # calculate fitness weight based on similar candidates
        sh = (1 - (dists[dists <= sigma] / sigma) ** alpha)
        sum_sh = max(1, np.sum(sh))
        self.fitness *= sum_sh
