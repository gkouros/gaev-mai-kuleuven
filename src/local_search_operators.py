import time
import random
import numpy as np
import random
from numba import jit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

@jit
def two_opt(route: list, distance_matrix: np.array):
    """ The local search operator 2-opt

    Args:
        route (list): The route to optimize
        distance_matrix (np.array): The cost matrix of city distances

    Returns:
        list: The optimized route
    """
    best = route

    improved = True
    count = 0
    while improved and count < 100:
        count += 1
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 2, len(route)):
                if cost_change(distance_matrix,
                               best[i - 1], best[i],
                               best[j - 1], best[j]) < 0:
                    best[i:j] = best[j - 1:i - 1:-1]
                    improved = True

    return best

@jit
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

def three_opt(tour, distances, timeout=1):
    """Iterative improvement based on 3 exchange."""
    #  ts = time.time()
    while True:# and time.time() - ts < timeout:
        delta = 0
        segments = all_segments(len(tour))
        random.shuffle(segments)
        for (a, b, c) in segments:
            delta += reverse_segment_if_better(tour, a, b, c, distances)

            if time.time() - ts < timeout:
                break

            if delta >= 0:
                break

        if delta >= 0:
            break

    return tour

def all_segments(n: int, timeout=1):
    """Generate all segments combinations"""
    return [(i, j, k)
        for i in range(n)
        for j in range(i + 2, n)
        for k in range(j + 2, n + (i > 0))]

def reverse_segment_if_better(tour, i, j, k, distances):
    """If reversing tour[i:j] would make the tour shorter, then do it."""
    # Given tour [...A-B...C-D...E-F...]
    A, B, C, D, E, F = tour[i-1], tour[i], tour[j-1], tour[j], tour[k-1], tour[k % len(tour)]
    d0 = distances[A, B] + distances[C, D] + distances[E, F]
    d1 = distances[A, C] + distances[B, D] + distances[E, F]
    d2 = distances[A, B] + distances[C, E] + distances[D, F]
    d3 = distances[A, D] + distances[E, B] + distances[C, F]
    d4 = distances[F, B] + distances[C, D] + distances[E, A]

    if d0 > d1:
        tour[i:j] = tour[i:j][::-1]
        return -d0 + d1
    elif d0 > d2:
        tour[j:k] = tour[j:k][::-1]
        return -d0 + d2
    elif d0 > d4:
        tour[i:k] = tour[i:k][::-1]
        return -d0 + d4
    elif d0 > d3:
        tmp = tour[j:k] + tour[i:j]
        tour[i:k] = tmp
        return -d0 + d3
    return 0
