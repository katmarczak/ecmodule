import numpy as np
import matplotlib.pyplot as plt

def _points_on_sphere(sx, sy, sz, r):
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    
    x = r*x + sx
    y = r*y + sy
    z = r*z + sz
    return (x, y, z)

def plot_points(disks_centers, radius=None, cell=None, style='solid'):
    '''plot points with scatter if no radius, otherwise plot spheres with given radius'''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    xs = disks_centers[:, 0:1]
    ys = disks_centers[:, 1:2]  
    zs = disks_centers[:, 2:3]
    
    if radius is None:
        ax.scatter(xs, ys, zs)
    else:
        for (xi, yi, zi) in zip(xs, ys, zs):
            (xs, ys, zs) = _points_on_sphere(xi, yi, zi, radius)
            if style == 'wireframe':
                ax.plot_wireframe(xs, ys, zs)
            else:
                ax.plot_surface(xs, ys, zs)
    
    if cell is not None:
        xlim, ylim, zlim = cell.shape

        ax.set_xlim3d(0, xlim)
        ax.set_xlabel('X')
        ax.set_ylim3d(0, ylim)
        ax.set_ylabel('Y')
        ax.set_zlim3d(0, zlim)
        ax.set_zlabel('Z')
    
    plt.show()

def _collect_edges(tri):
    edges = set()

    def sorted_tuple(a,b):
        return (a, b) if a < b else (b, a)
    
    for (i0, i1, i2, i3) in tri.simplices:
        edges.add(sorted_tuple(i0,i1))
        edges.add(sorted_tuple(i0,i2))
        edges.add(sorted_tuple(i0,i3))
        edges.add(sorted_tuple(i1,i2))
        edges.add(sorted_tuple(i1,i3))
        edges.add(sorted_tuple(i2,i3))
    return edges

def plot_tri(ax, points, tri):
    # plotting edges
    edges = _collect_edges(tri)
    x = np.array([])
    y = np.array([])
    z = np.array([])
    for (i,j) in edges:
        x = np.append(x, [points[i, 0], points[j, 0], np.nan])      
        y = np.append(y, [points[i, 1], points[j, 1], np.nan])      
        z = np.append(z, [points[i, 2], points[j, 2], np.nan])
    ax.plot3D(x, y, z, color='g', lw='0.1')
    
    # plotting initial points
    n = int(len(tri.points) / 27)
    ax.scatter(points[:n,0], points[:n,1], points[:n,2], color='b')
