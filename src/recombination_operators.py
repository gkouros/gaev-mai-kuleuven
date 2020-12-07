import numpy as np
from individual import Individual


def order_crossover(mother, father):
    size = mother.size

    start, end = sorted(np.random.randint(size) for i in range(2))
    child1 = [-1] * size
    child2 = [-1] * size
    child1_inherited = []
    child2_inherited = []
    for i in range(start, end + 1):
        child1[i] = mother.route[i]
        child2[i] = father.route[i]
        child1_inherited.append(mother.route[i])
        child2_inherited.append(father.route[i])

    current_father_position, current_mother_position = 0, 0

    inherited_pos = list(range(start, end + 1))
    i = 0
    while i < size:
        if i in inherited_pos:
            i += 1
            continue

        test_child1 = child1[i]
        if test_child1==-1:
            father_city = father.route[current_father_position]
            while father_city in child1_inherited:
                current_father_position += 1
                father_city = father.route[current_father_position]
            child1[i] = father_city
            child1_inherited.append(father_city)

        test_child2 = child2[i]
        if test_child2==-1: #to be filled
            mother_city = mother.route[current_mother_position]
            while mother_city in child2_inherited:
                current_mother_position += 1
                mother_city = mother.route[current_mother_position]
            child2[i] = mother_city
            child2_inherited.append(mother_city)

        i +=1

    c1 = Individual(mother.distance_matrix)
    c1.route = child1
    c1.fitness =  c1.calc_fitness()

    c2 = Individual(mother.distance_matrix)
    c2.route = child2
    c2.fitness = c2.calc_fitness()

    return c1, c2
