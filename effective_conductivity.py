import numpy as np
from scipy import linalg

def ec(tri, constants, r, cell):
    n = int(len(tri.points) / 27)
    summ, value = 0, 0
    indptr, indices = tri.vertex_neighbor_vertices
    non_periodic_points = tri.points[0:n]
    for point_index in range(n):
        neighbours_indices = indices[indptr[point_index]:indptr[point_index+1]]
        for neighbour_index in neighbours_indices:
            p1 = constants[point_index]
            p2 = constants[neighbour_index % n]
            value = _g_mk(non_periodic_points[point_index], tri.points[neighbour_index], r, cell) * (p1 - p2)**2
            summ += value
    return summ/2

def min_ec(tri, r, cell):
    gs, B = _construct_coefficients(tri, r, cell)
    cons = _find_minima(gs, B)
    return ec(tri, cons, r, cell)

def _g_mk(dm, dk, r, cell):
    delta_mk = cell.distance(dm, dk) - 2*r
    return -np.pi * r * np.log(delta_mk)

def _construct_coefficients(tri, r, cell):
    n = int(len(tri.points) / 27)
    g_s = np.zeros((n, n))
    g_s[n-1][n-1] = 1
    vec_b = np.zeros(n)
    indptr, indices = tri.vertex_neighbor_vertices
    _periodic_points = tri.points
    non_periodic_points = tri.points[0:n]

    for point_index in range(n-1):
        neighbours_indices = indices[indptr[point_index]:indptr[point_index+1]]
        bsum = 0
        for neighbour_index in neighbours_indices:
            g = _g_mk(non_periodic_points[point_index], tri.points[neighbour_index], r, cell)
            g_s[point_index][point_index] += g
            g_s[point_index][neighbour_index % n] = -g
            factor = _periodic_points[neighbour_index] - non_periodic_points[neighbour_index % n]
            bsum += g * factor[0]
        vec_b[point_index] = bsum
    
    return g_s, vec_b

def _find_minima(M, b):
    return linalg.solve(M, b, check_finite=False, overwrite_a=True, overwrite_b=True)