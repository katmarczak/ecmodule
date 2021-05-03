import numpy as np
from scipy import linalg

def gmk_factory(r, cell):
    g_mk = None
    dm = cell.shape.size
    if dm == 2:
        if np.all(np.isclose(r, r[0])):
            def g_mk(dm, dk, r, cell):
                r = r[0]
                delta_mk = cell.distance(dm, dk) - 2*r
                return np.pi * np.sqrt(r/delta_mk)
        else:
            def g_mk(dm, dk, r, cell):
                delta_mk = cell.distance(dm, dk) - (r[0]+r[1])
                return (np.pi / np.sqrt(delta_mk)) * np.sqrt( (2*r[0]*r[1]) / (r[0]+r[1]) )
    if dm == 3:
        def g_mk(dm, dk, r, cell):
            delta_mk = cell.distance(dm, dk) - 2*r
            return np.pi * np.sqrt(r/delta_mk)
    
    return g_mk

def ec(tri, constants, r, cell):
    base_n = 3**(cell.shape.size)
    n = int(len(tri.points) / base_n)
    summ, value = 0, 0
    indptr, indices = tri.vertex_neighbor_vertices
    _periodic_points = tri.points
    non_periodic_points = tri.points[0:n]

    g_mk = gmk_factory(r, cell)

    for point_index in range(n):
        neighbours_indices = indices[indptr[point_index]:indptr[point_index+1]]
        for neighbour_index in neighbours_indices:
            periodic_shift = _periodic_points[neighbour_index] - non_periodic_points[neighbour_index % n]
            p1, p2 = constants[point_index], constants[neighbour_index % n] + periodic_shift[0]
            radii = r[point_index], r[neighbour_index % n]
            value = g_mk(non_periodic_points[point_index], tri.points[neighbour_index], radii, cell) * (p1 - p2)**2
            summ += value
    return summ/2

def min_ec(tri, r, cell):
    gs, B = construct_coefficients(tri, r, cell)
    cons = find_minima(gs, B)
    return ec(tri, cons, r, cell)

def construct_coefficients(tri, r, cell):
    base_n = 3**(cell.shape.size)
    n = int(len(tri.points) / base_n)
    g_s = np.zeros((n, n))
    g_s[n-1][n-1] = 1
    vec_b = np.zeros(n)
    indptr, indices = tri.vertex_neighbor_vertices
    _periodic_points = tri.points
    non_periodic_points = tri.points[0:n]

    g_mk = gmk_factory(r, cell)

    for point_index in range(n-1):
        neighbours_indices = indices[indptr[point_index]:indptr[point_index+1]]
        bsum = 0
        for neighbour_index in neighbours_indices:
            radii = r[point_index], r[neighbour_index % n]
            g = g_mk(non_periodic_points[point_index], tri.points[neighbour_index], radii, cell)
            g_s[point_index][point_index] += g
            g_s[point_index][neighbour_index % n] = -g
            factor = _periodic_points[neighbour_index] - non_periodic_points[neighbour_index % n]
            bsum += g * factor[0]
        vec_b[point_index] = bsum
    
    return g_s, vec_b

def find_minima(M, b):
    return linalg.solve(M, b, check_finite=False, overwrite_a=True, overwrite_b=True)