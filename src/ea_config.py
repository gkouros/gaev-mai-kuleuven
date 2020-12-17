class EAConfig:
    def __init__(
        selection,
        recombination,
        mutation,
        elimination,
        local_search,
        heuristic_search,
        mutation_adaptivity,
        fitness_sharing,
        mutation_probability,
        recombination_probability,
        num_local_searches,
        num_heuristic_searches,

    ):
    self.selction = selection
    self.recombination = recombination
    self.mutation = mutation
    self.local_search = local_search
    self.heuristic_search = heuristic_search
    self.mutation_adaptivity = mutation_adaptivity
    self.fitness_sharing = fitness_sharing
    self.mutation_probability = mutation_probability,
    self.recombination_probability = recombination_probability,
    self.num_local_searches = num_local_searches,
    self.num_heuristic_searches = num_heuristic_searches,

