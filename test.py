import pyvista as pv
import numpy as np

# Example: one hexagon and one pentagon
points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1.5, 0.87, 0],
    [1, 1.73, 0],
    [0, 1.73, 0],
    [-0.5, 0.87, 0],
])
faces = np.hstack([[6, 0, 1, 2, 3, 4, 5]])  # '6' = number of vertices in the face

mesh = pv.PolyData(points, faces)
mesh.cell_data["color"] = [[255, 0, 0]]  # RGB for face

plotter = pv.Plotter()
plotter.add_mesh(mesh, color=True, show_edges=True)
plotter.show()