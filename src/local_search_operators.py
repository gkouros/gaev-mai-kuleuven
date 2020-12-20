import time
import random
import numpy as np
import random


def two_opt(route, distance_matrix, timeout=1):
    best = route
    improved = True
    ts = time.time()
    size = len(distance_matrix)

    while improved and time.time() - ts < timeout:

        improved = False
        for i in range(1, len(route) - 2):
            j_list = list(range(i + 1, len(route)))
            random.shuffle(j_list)
            for j in j_list:
                if j - i == 1:
                    continue

                if cost_change(distance_matrix,
                               best[i - 1], best[i],
                               best[j - 1], best[j]) <= 0:
                    best[i:j] = best[j - 1:i - 1:-1]
                    improved = True

    #  print(time.time() - ts)
    return best

def cost_change(distance_matrix, n1, n2, n3, n4):
    return distance_matrix[n1][n3] + distance_matrix[n2][n4] - \
            distance_matrix[n1][n2] - distance_matrix[n3][n4]

def three_opt(tour, distances, timeout=1):
    """Iterative improvement based on 3 exchange."""
    ts = time.time()
    while True and time.time() - ts < timeout:
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
