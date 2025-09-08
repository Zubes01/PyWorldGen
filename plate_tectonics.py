import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Parameters
map_size = 500
num_plates = 10

# Step 1: Plate seeds
np.random.seed(42)  # reproducibility
seeds = np.random.rand(num_plates, 2) * map_size

# Step 2: Voronoi diagram
vor = Voronoi(seeds)

# Step 3: Assign properties
plate_types = np.random.choice(["continental", "oceanic"], size=num_plates, p=[0.7, 0.3])
velocities = np.random.randn(num_plates, 2)  # random directions

# Step 4: Plot
fig, ax = plt.subplots(figsize=(8, 8))
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors="black")

# Color regions
for i, region in enumerate(vor.regions):
    if not region or -1 in region:
        continue
    polygon = [vor.vertices[j] for j in region]
    color = "tan" if plate_types[i % len(plate_types)] == "continental" else "lightblue"
    ax.fill(*zip(*polygon), color=color, alpha=0.5)

# Velocity arrows
for i, seed in enumerate(seeds):
    ax.arrow(seed[0], seed[1], velocities[i,0]*10, velocities[i,1]*10,
             head_width=5, color="red")

ax.set_xlim(0, map_size)
ax.set_ylim(0, map_size)
plt.show()
