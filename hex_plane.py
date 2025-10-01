import numpy as np
import matplotlib.pyplot as plt
import random
import opensimplex

class HexPlane:
    def __init__(self, length, height):
        self.length = length
        self.height = height
        self.world_hexagons = []
        self.noise_gen = opensimplex.OpenSimplex(seed=random.randint(0, 10000))
        self.noise_z_val = random.uniform(0, 1000)  # Random z value for 3D noise slice
        self.create_hexagonal_plane()

    def create_hexagonal_plane(self):
        """
        Create a hexagonal grid of WorldHexagons in a 2D plane.
        """
        hex_radius = 1.0
        hex_height = np.sqrt(3) * hex_radius

        for row in range(self.height):
            for col in range(self.length):
                x_offset = col * 1.5 * hex_radius
                y_offset = row * hex_height + (col % 2) * (hex_height / 2)
                center = np.array([x_offset, y_offset])

                # Calculate vertices of the hexagon
                vertices = []
                for i in range(6):
                    angle = np.pi / 3 * i
                    vertex = center + np.array([hex_radius * np.cos(angle), hex_radius * np.sin(angle)])
                    vertices.append(vertex)

                self.world_hexagons.append(WorldHexagon(center, vertices))

    def map_perlin_noise(self, scale=0.065):
        """
        Map Perlin noise values to each hexagon for terrain generation.
        """
        for hexagon in self.world_hexagons:
            x, y = hexagon.center
            # Scale coordinates for noise
            # Note: Using simplex noise instead of Perlin for better performance and quality
            # Note: using 3D noise with a fixed z value to get a 2D slice for randomness
            value = opensimplex.noise3(x * scale, y * scale, self.noise_z_val)
            hexagon.perlin_value = value

    def assign_perlin_terrain_colors(self, water_threshold=0.5, beach_threshold=0.52, land_threshold=0.75, mountain_threshold=0.83, water_low_color=(0, 0, 0.2), water_high_color=(0, 0, 1), beach_color=(0.86, 0.80, 0.20), land_low_color=(0, 0.4, 0), land_high_color=(0, 1, 0), mountain_low_color=(0.7, 0.7, 0.7), mountain_high_color=(0.8, 0.8, 0.8), snow_color=(1, 1, 1)):
        """
        Assign terrain colors to each hexagon based on Perlin noise values.
        """
        for hexagon in self.world_hexagons:
            if hexagon.perlin_value is None:
                raise ValueError("Perlin noise values not mapped. Call map_perlin_noise() first.")
            # noise value ranges from -1 to 1
            # Normalize to 0-1
            normalized_value = (hexagon.perlin_value + 1) / 2
            if normalized_value < water_threshold:
                # Blue value should move from water_low_color to water_high_color based on normalized_value
                t = normalized_value / water_threshold
                hexagon.color = tuple(np.array(water_low_color) * (1 - t) + np.array(water_high_color) * t)
            elif normalized_value < beach_threshold:
                hexagon.color = beach_color
            elif normalized_value < land_threshold:
                # Green value should move from land_low_color to land_high_color based on normalized_value
                t = (normalized_value - water_threshold) / (land_threshold - water_threshold)
                hexagon.color = tuple(np.array(land_low_color) * (1 - t) + np.array(land_high_color) * t)
            elif normalized_value < mountain_threshold:
                # Gray value should move from mountain_low_color to mountain_high_color based on normalized_value
                t = (normalized_value - land_threshold) / (1 - land_threshold)
                hexagon.color = tuple(np.array(mountain_low_color) * (1 - t) + np.array(mountain_high_color) * t)
            else:
                # White color for snow
                hexagon.color = snow_color

    def plot(self):
        """
        Plot the hexagonal grid with assigned colors.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        for hexagon in self.world_hexagons:
            polygon = plt.Polygon(hexagon.face, edgecolor='black', alpha=0.5, facecolor=hexagon.color)
            ax.add_patch(polygon)

        ax.set_aspect('equal')
        ax.autoscale_view()
        # remove all axes
        ax.axis('off')
        plt.show()

class WorldHexagon:
    def __init__(self, hex_center, hex_face):
        self.center = hex_center
        self.face = hex_face
        self.perlin_value = None  # Placeholder for Perlin noise value
        self.color = None  # Placeholder for color based on terrain type
        self.neighbors = []  # List of neighboring WorldHexagons

def main():
    length = 80  # Number of hexagons along the length
    height = 50  # Number of hexagons along the height
    hex_plane = HexPlane(length, height)
    hex_plane.map_perlin_noise()
    hex_plane.assign_perlin_terrain_colors()
    hex_plane.plot()

if __name__ == '__main__':
    main()