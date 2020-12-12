import numpy as np
from individual import Individual


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
    start, end = sorted(np.random.choice(size, 2))

    child1 = [None] * size
    child2 = [None] * size

    child1[start:end] = parent1.route[start:end]
    child2[start:end] = parent2.route[start:end]

    for child, parent in zip([child1, child2], [parent2, parent1]):
        idx1 = end % size
        idx2 = end % size

        while any(np.equal(child, None)):
            if parent[idx2] not in child:
                child[idx1] = parent[idx2]
                idx1 = (idx1 + 1) % size

            idx2 = (idx2 + 1) % size

    try:
        individual1 = Individual(parent1.distance_matrix, child1)
        individual2 = Individual(parent2.distance_matrix, child2)
    except ValueError as err:
        print(f'Recombination failed [{err}]. Returning parents instead.')
        return parent1, parent2

    return individual1, individual2


#  """ old implementation """
#  def order_crossover(mother, father):
#      size = mother.size
#
#      start, end = sorted(np.random.randint(size) for i in range(2))
#      child1 = [-1] * size
#      child2 = [-1] * size
#      child1_inherited = []
#      child2_inherited = []
#      for i in range(start, end + 1):
#          child1[i] = mother.route[i]
#          child2[i] = father.route[i]
#          child1_inherited.append(mother.route[i])
#          child2_inherited.append(father.route[i])
#
#      current_father_position, current_mother_position = 0, 0
#
#      inherited_pos = list(range(start, end + 1))
#      i = 0
#      while i < size:
#          if i in inherited_pos:
#              i += 1
#              continue
#
#          test_child1 = child1[i]
#          if test_child1==-1:
#              father_city = father.route[current_father_position]
#              while father_city in child1_inherited:
#                  current_father_position += 1
#                  father_city = father.route[current_father_position]
#              child1[i] = father_city
#              child1_inherited.append(father_city)
#
#          test_child2 = child2[i]
#          if test_child2==-1: #to be filled
#              mother_city = mother.route[current_mother_position]
#              while mother_city in child2_inherited:
#                  current_mother_position += 1
#                  mother_city = mother.route[current_mother_position]
#              child2[i] = mother_city
#              child2_inherited.append(mother_city)
#
#          i +=1
#
#      c1 = Individual(mother.distance_matrix, child1)
#      c2 = Individual(mother.distance_matrix, child2)
#
#      return c1, c1
