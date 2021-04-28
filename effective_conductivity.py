import numpy as np
from scipy import linalg

def ec(tri, constants, r, cell):
    base_n = 3**(len(cell.shape))
    n = int(len(tri.points) / base_n)
    summ, value = 0, 0
    indptr, indices = tri.vertex_neighbor_vertices
    _periodic_points = tri.points
    non_periodic_points = tri.points[0:n]

    for point_index in range(n):
        neighbours_indices = indices[indptr[point_index]:indptr[point_index+1]]
        for neighbour_index in neighbours_indices:
            periodic_shift = _periodic_points[neighbour_index] - non_periodic_points[neighbour_index % n]
            p1 = constants[point_index]
            p2 = constants[neighbour_index % n] + periodic_shift[0]
            value = g_mk(non_periodic_points[point_index], tri.points[neighbour_index], r, cell) * (p1 - p2)**2
            summ += value
    return summ/2

def min_ec(tri, r, cell):
    gs, B = construct_coefficients(tri, r, cell)
    cons = find_minima(gs, B)
    return ec(tri, cons, r, cell)

def g_mk(dm, dk, r, cell):
    delta_mk = cell.distance(dm, dk) - 2*r
    if len(cell.shape) == 2:
        return np.pi * np.sqrt(r/delta_mk)
    if len(cell.shape) == 3:
        return -np.pi * r * np.log(delta_mk)
    return None

def construct_coefficients(tri, r, cell):
    base_n = 3**(len(cell.shape))
    n = int(len(tri.points) / base_n)
    print('N of base points: %d, n of periodic points: %d' % (n, len(tri.points)))
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
            g = g_mk(non_periodic_points[point_index], tri.points[neighbour_index], r, cell)
            g_s[point_index][point_index] += g
            g_s[point_index][neighbour_index % n] = -g
            factor = _periodic_points[neighbour_index] - non_periodic_points[neighbour_index % n]
            bsum += g * factor[0]
        vec_b[point_index] = bsum
    
    return g_s, vec_b

def find_minima(M, b):
    return linalg.solve(M, b, check_finite=False, overwrite_a=True, overwrite_b=True)