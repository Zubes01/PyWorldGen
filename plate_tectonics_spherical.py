import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import SphericalVoronoi, geometric_slerp

num_regions = 25
plot_3d_sphere = True

# select num_regions random points on the unit sphere
phi = np.random.uniform(0, 2 * np.pi, num_regions)
cos_theta = np.random.uniform(-1, 1, num_regions) # necessary to ensure uniform distribution on sphere
theta = np.arccos(cos_theta) # if we were to use uniform distribution for theta, points would cluster at the poles
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)
points = np.vstack((x, y, z)).T

# compute spherical Voronoi tesselation
radius = 1
center = np.array([0, 0, 0])
sv = SphericalVoronoi(points, radius, center)

# sort vertices (optional, helpful for plotting)
sv.sort_vertices_of_regions()

if plot_3d_sphere:
    # plot Voronoi regions
    t_vals = np.linspace(0, 1, 2000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot generator points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b')

    # plot Voronoi vertices
    ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2],
                    c='g')

    # indicate Voronoi regions (as Euclidean polygons)
    for region in sv.regions:
        n = len(region)
        for i in range(n):
            # for each edge of the region, plot a great circle arc
            start = sv.vertices[region][i]
            end = sv.vertices[region][(i + 1) % n]
            result = geometric_slerp(start, end, t_vals)
            ax.plot(result[..., 0],
                    result[..., 1],
                    result[..., 2],
                    c='k')    
        

    # plot the surface of the sphere (only for positive x, y, and z)
    u = np.linspace(0, 2 * np.pi, 10)
    v = np.linspace(0, np.pi, 10)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='c', alpha=0.1)

    ax.azim = 10
    ax.elev = 40

    ax.set_box_aspect([1,1,1])  # equal aspect ratio

    fig.set_size_inches(4, 4)
    plt.show()

def unit_sphere_to_mercator(x, y, z):
    """Convert 3D unit sphere coordinates to 2D Mercator projection."""
    lon = np.arctan2(y, x)
    lat = np.arcsin(z)
    x_merc = lon
    y_merc = np.log(np.tan(np.pi / 4 + lat / 2))
    return x_merc, y_merc

def plot_mercator_borders(show_axes=False):
    # plot the borders of the mercator projection, i.e. the edges of the square
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)

    # make the aspect ratio equal
    plt.gca().set_aspect('equal', adjustable='box')
    
    if show_axes:
        plt.xlabel('Longitude (radians)')
        plt.ylabel('Latitude (radians)')
    else:
        plt.xticks([])
        plt.yticks([])

# plot generator points in Mercator projection
points_merc = np.array([unit_sphere_to_mercator(*p) for p in points])
plt.figure()
plt.scatter(points_merc[:, 0], points_merc[:, 1], c='b')

# plot Voronoi vertices in Mercator projection
for region in sv.regions:
    n = len(region)

    # select a random color for this region
    color = np.random.rand(3,)

    line_list = []
    for i in range(n):
        # for each edge of the region, plot a great circle arc
        start = sv.vertices[region][i]
        end = sv.vertices[region][(i + 1) % n]

        # convert start and end points to Mercator
        start_merc = unit_sphere_to_mercator(*start)
        end_merc = unit_sphere_to_mercator(*end)

        # interpolate in 3D and convert to Mercator
        t_vals = np.linspace(0, 1, 100)
        result = geometric_slerp(start, end, t_vals)
        result_merc = np.array([unit_sphere_to_mercator(*p) for p in result])

        # determine if the arc crosses the discontinuity at lon = ±π
        if np.any(np.abs(np.diff(result_merc[:, 0])) > np.pi):
            # split the arc into two segments
            mid_index = np.argmax(np.abs(np.diff(result_merc[:, 0])))
            line_list.append(result_merc[:mid_index+1])
            line_list.append(result_merc[mid_index+1:])
        else:
            line_list.append(result_merc)

    for line in line_list:
        plt.plot(line[:, 0], line[:, 1], c=color)

    # color in this region (approximate by filling the polygon formed by the vertices)
    #region_vertices_merc = np.array([unit_sphere_to_mercator(*sv.vertices[v]) for v in region])
    #plt.fill(region_vertices_merc[:, 0], region_vertices_merc[:, 1], color=color, alpha=0.3)

plot_mercator_borders()

plt.show()
