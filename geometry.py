from typing import List
from scipy.spatial import Delaunay
import numpy as np

class _PeriodicCell:
    def __init__(self):
        pass
    
    def distance(self, p1, p2):
        pass
    
    def random_point(self):
        pass

class PeriodicCell2D(_PeriodicCell):    
    def __init__(self, vecs: List[int]):
        super()
        self.x = vecs[0]
        self.y = vecs[1]
        self.shape = np.array([self.x, self.y])
    
    def random_point(self):
        return np.array([np.random.random() * self.x, np.random.random() * self.y])
    
    def distance(self, p1, p2):
        d = p1 - p2
        periodic_d = np.minimum(np.abs(d), self.shape - np.abs(d))
        return np.linalg.norm(periodic_d)

class PeriodicCell3D(_PeriodicCell):    
    def __init__(self, vecs: List[int]):
        super()
        self.x = vecs[0]
        self.y = vecs[1]
        self.z = vecs[2]
        self.shape = np.array([self.x, self.y, self.z])
    
    def random_point(self):
        return np.array([np.random.random() * self.x, np.random.random() * self.y, np.random.random() * self.z])
    
    def distance(self, p1, p2):
        d = p1 - p2
        periodic_d = np.minimum(np.abs(d), self.shape - np.abs(d))
        return np.linalg.norm(periodic_d)

def _rsa3d(radius, n, cell):
    A = []
    while len(A) < n:
        sphere_new = cell.random_point()
        collision = False
        for sphere in A:
            if cell.distance(sphere, sphere_new) < 2*radius:
                collision = True
                break
        if not collision:
            A.append(sphere_new)
    return np.array(A)

def rsa3d(cell, n, conc = None):
    if not conc:
        conc = 0.3
    radius = np.cbrt( 3*conc / (4*np.pi*n) )
    
    A = []
    while len(A) < n:
        sphere_new = cell.random_point()
        collision = False
        for sphere in A:
            if cell.distance(sphere, sphere_new) < 2*radius:
                collision = True 
                break
        if not collision:
            A.append(sphere_new)
    return np.array(A), radius

def periodize3d(points, cell):
    periodic_points = []
    lx, ly, lz = cell.shape
    
    xs = [0, -lx, lx]
    ys = [0, -ly, ly]
    zs = [0, -lz, lz]
    
    for x in xs:
        for y in ys:
            for z in zs:
                shift = [x, y, z]
                periodic_points.append(np.add(points, shift))
                                
    return np.array(periodic_points) # shape: 27 x n_of_points x 3

def periodize2d(points, cell):
    periodic_points = []
    lx, ly = cell.shape
    
    xs = [0, -lx, lx]
    ys = [0, -ly, ly]
    
    for x in xs:
        for y in ys:
            shift = [x, y]
            periodic_points.append(np.add(points, shift))
                                
    return np.array(periodic_points) # shape: 9 x n_of_points x 2 

def make_delaunay(points, cell):
    '''make periodic delaunay from non-periodic points''' 
    per_points = None
    n = len(points)
    if len(cell.shape) == 2:
        per_points = periodize2d(points, cell).reshape(9*n, 2)
        per_points = np.column_stack((per_points[:, 0], per_points[:, 1]))
    if len(cell.shape) == 3:
        per_points = periodize3d(points, cell).reshape(27*n, 3)
        per_points = np.column_stack((per_points[:, 0], per_points[:, 1], per_points[:, 2]))
    
    tri = Delaunay(per_points, qhull_options='Qc')
    return tri