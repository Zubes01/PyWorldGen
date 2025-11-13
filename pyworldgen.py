"""
Imports
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from math import sqrt, log10
from collections import defaultdict, deque
import opensimplex
import random
import time
import pickle
import os
from tqdm import tqdm
from scipy.spatial import SphericalVoronoi
import heapq
from vispy import app, scene
from vispy.visuals.filters import ShadingFilter, WireframeFilter

"""
Helper functions
"""
def convert_to_sea_level_height(height_zero_to_one):
    """
    Convert a normalized height value (0 to 1) to a height relative to sea level (0 km).
    height_zero_to_one: normalized height value (0 to 1)
    0 corresponds to LOWEST_POINT_ELEVATION_KM
    1 corresponds to HIGHEST_POINT_ELEVATION_KM

    returns height relative to sea level in km
    """
    absolute_height_km = LOWEST_POINT_ELEVATION_KM + (height_zero_to_one * (HIGHEST_POINT_ELEVATION_KM - LOWEST_POINT_ELEVATION_KM))
    sea_level_height_km = absolute_height_km - SEA_LEVEL_HEIGHT
    return sea_level_height_km

def convert_from_sea_level_height_km(sea_level_height_km):
    """
    Convert a height relative to sea level (0 km) to a normalized height value (0 to 1).
    highest_possible_height_km: maximum height in km (e.g., 9 km for Mount Everest)
    lowest_possible_height_km: minimum height in km (e.g., -11 km for Mariana Trench)
    """
    absolute_height_km = sea_level_height_km + SEA_LEVEL_HEIGHT
    height_zero_to_one = (absolute_height_km - LOWEST_POINT_ELEVATION_KM) / (HIGHEST_POINT_ELEVATION_KM - LOWEST_POINT_ELEVATION_KM)
    return height_zero_to_one

def convert_difference_in_km_to_normalized(diff_km):
    """
    Convert a difference in height in km to a normalized difference (0 to 1).
    """
    normalized_diff = diff_km / (HIGHEST_POINT_ELEVATION_KM - LOWEST_POINT_ELEVATION_KM)
    return normalized_diff

def convert_difference_in_normalized_to_km(diff_normalized):
    """
    Convert a difference in normalized height (0 to 1) to a difference in km.
    """
    diff_km = diff_normalized * (HIGHEST_POINT_ELEVATION_KM - LOWEST_POINT_ELEVATION_KM)
    return diff_km

"""
CONSTANTS
"""
# Calculation tolerances
FLOAT_TOLERANCE = 1e-6 # Tolerance for floating point comparisons

# Geographic constants
SEA_LEVEL_HEIGHT = 0.0  # Sea level at 0 km
HIGHEST_POINT_ELEVATION_KM = 9.0  # Height of Mount Everest rounded to nearest km
LOWEST_POINT_ELEVATION_KM = -11.0  # Depth of Mariana Trench rounded to nearest km
OCEANIC_PLATE_AVG_HEIGHT_LOW_KM = -5.0  # Average height of old oceanic crust is around -5 km below sea level
OCEANIC_PLATE_AVG_HEIGHT_HIGH_KM = -2.0  # Average height of young oceanic crust is as little as -2 km below sea level
CONTINENTAL_PLATE_AVG_HEIGHT_LOW_KM = 0.0  # Average height of continental plates can be as low as around sea level
CONTINENTAL_PLATE_AVG_HEIGHT_HIGH_KM = 1.0  # Average height of continental plates can be up to about 1 km above sea level

# Globe generation parameters
GLOBE_RECURSION_LEVEL = 6  # Level of recursion for icosphere generation
GLOBAL_NOISE_SCALE = 2  # Scale for noise mapping
GLOBAL_NOISE_NUM_OCTAVES = 3  # Number of octaves for noise generation
GLOBAL_NOISE_AMPLITUDE = 0.15  # Amplitude for noise contribution to height when using plate-based generation

# Tectonic plate parameters
NUM_TECTONIC_PLATES = 15 # Number of tectonic plates to create
OCEANIC_PLATE_RATIO = 0.7  # Ratio of oceanic plates to total plates
PLATE_INTERNAL_NOISE_SCALE = 2  # Scale for noise mapping in plate assignment
PLATE_INTERNAL_NOISE_NUM_OCTAVES = 3  # Number of octaves for noise generation in plate assignment
PLATE_INTERNAL_NOISE_AMPLITUDE = 0.0  # Amplitude for noise contribution to plate internal height variation

# Tectonic simulation parameters
MILLION_YEARS_PER_SIMULATION_STEP = 10 # 10 million years per simulation step
GAUSSIAN_DECAY_RATE = 0.1  # Rate for Gaussian decay function in deformation simulation
DEFORMATION_THRESHOLD = 0.001  # Minimum deformation magnitude to apply to a tile

# Deformation rates (in km per million years)
OCEANIC_OCEANIC_DIVERGENT_UPLIFT_RATE_KM_PER_MILLION_YEAR = 0.1  # Uplift rate for oceanic-oceanic divergent boundaries (for both plates)
OCEANIC_OCEANIC_CONVERGENT_UPLIFT_RATE_KM_PER_MILLION_YEAR = 0.2  # Uplift rate for oceanic-oceanic convergence (for the non-subducting plate)
OCEANIC_OCEANIC_CONVERGENT_SUBDUCTION_RATE_KM_PER_MILLION_YEAR = -0.3  # Subduction rate for oceanic-oceanic convergence (for the subducting plate)
# note that generally, transform boundaries do not change elevation 
CONTINENTAL_CONTINENTAL_DIVERGENT_SUBDUCTION_RATE_KM_PER_MILLION_YEAR = -0.2  # Subduction rate for continental-continental divergence (for both plates)
CONTINENTAL_CONTINENTAL_CONVERGENT_UPLIFT_RATE_KM_PER_MILLION_YEAR = 0.5  # Uplift rate for continental-continental convergence (for both plates)
# note that generally, transform boundaries do not change elevation
OCEANIC_CONTINENTAL_DIVERGENT_UPLIFT_RATE_KM_PER_MILLION_YEAR = 0.05  # Uplift rate for oceanic-continental divergent boundaries (for oceanic plate)
OCEANIC_CONTINENTAL_DIVERGENT_SUBDUCTION_RATE_KM_PER_MILLION_YEAR = -0.15  # Subduction rate for oceanic-continental divergent boundaries (for continental plate)
OCEANIC_CONTINENTAL_CONVERGENT_UPLIFT_RATE_KM_PER_MILLION_YEAR = 0.2  # Uplift rate for oceanic-continental convergence (for continental plate)
OCEANIC_CONTINENTAL_CONVERGENT_SUBDUCTION_RATE_KM_PER_MILLION_YEAR = -0.3  # Subduction rate for oceanic-continental convergence (for oceanic plate)
# note that generally, transform boundaries do not change elevation

# Noise parameters for tectonic deformation
NOISE_SCALE_FOR_TECTONIC_DEFORMATION = 8  # Scale for noise mapping in tectonic deformation
NOISE_OCTAVES_FOR_TECTONIC_DEFORMATION = 2  # Number of octaves for noise generation in tectonic deformation
NOISE_AMPLITUDE_FOR_TECTONIC_DEFORMATION = 0.2  # Amplitude for noise contribution to tectonic deformation

# Terrain coloring thresholds
WATER_THRESHOLD = convert_from_sea_level_height_km(0.0) # everything below this will be water
BEACH_THRESHOLD = convert_from_sea_level_height_km(0.5) # everything between WATER_THRESHOLD and BEACH_THRESHOLD will be beach
LAND_THRESHOLD = convert_from_sea_level_height_km(4.0) # everything between BEACH_THRESHOLD and LAND_THRESHOLD will be land
MOUNTAIN_THRESHOLD = convert_from_sea_level_height_km(5.0) # everything between LAND_THRESHOLD and MOUNTAIN_THRESHOLD will be mountain
# above MOUNTAIN_THRESHOLD will be snow

# Terrain colors
WATER_LOW_COLOR = (0, 0, 0.2)
WATER_HIGH_COLOR = (0, 0, 1)
BEACH_COLOR = (0.86, 0.80, 0.20)
LAND_LOW_COLOR = (0, 0.4, 0)
LAND_HIGH_COLOR = (0, 1, 0)
MOUNTAIN_LOW_COLOR = (0.7, 0.7, 0.7)
MOUNTAIN_HIGH_COLOR = (0.8, 0.8, 0.8)
SNOW_COLOR = (1, 1, 1)


"""
Classes
"""
class SimulatedGlobe:
    """
    Creates a sphere tiled with tiles (hexagons and 12 pentagons) for world generation
    The globe is made up of WorldTiles, each with a center and vertices in 3D space
    The tile locations are generated by creating an icosphere and then taking the dual
    of the triangular mesh to create tiles (hexagons and pentagons)
    """

    """
    Standard functions
    """
    def __init__(self, recursion_level, use_cache=True):
        self.recursion_level = recursion_level
        self.use_cache = use_cache
        self.world_tiles = [] # list of all WorldTile objects on the sphere
        self.tectonic_plates = [] # list of TectonicPlate objects
        self.plate_boundaries = [] # list of PlateBoundary objects
        self.noise_4d_val = np.random.uniform(0, 1000)  # Random value for 4D noise slice
        self.create_world_tiles(recursion_level)

        self.world_edges, self.world_vertices = self.get_world_edges_and_world_vertices()

    """
    Generating the tiles
    """
    def create_world_tiles(self, rec_lvl):
        """
        Create WorldTile objects for each tile (hex/pentagon) on the sphere.
        """

        def create_icosphere(recursion_level):
            """
            Creates a geodesic sphere from an icosahedron.
            
            Args:
            recursion_level: The number of times to subdivide the faces.
            
            Returns:
            A tuple containing two numpy arrays: vertices and faces.
            """
            phi = (1.0 + sqrt(5.0)) / 2.0
            
            vertices = np.array([
                [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
                [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
                [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
            ])
            vertices /= np.linalg.norm(vertices, axis=1)[:, np.newaxis]

            faces = np.array([
                [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
            ])

            # Memoization dictionary for middle points
            middle_point_cache = {}

            def get_middle_point(p1, p2):
                nonlocal vertices
                
                # Sort points to ensure a consistent key
                smaller_index, larger_index = min(p1, p2), max(p1, p2)
                key = (smaller_index, larger_index)

                if key in middle_point_cache:
                    return middle_point_cache[key]
                else:
                    # Calculate midpoint and normalize
                    midpoint = (vertices[p1] + vertices[p2]) / 2.0
                    midpoint /= np.linalg.norm(midpoint)
                    
                    # Add to vertices and cache index
                    vertices = np.vstack([vertices, midpoint])
                    new_index = len(vertices) - 1
                    middle_point_cache[key] = new_index
                    return new_index

            for _ in tqdm(range(recursion_level), position=0, desc="Subdividing faces"):
                new_faces = []
                for tri in tqdm(faces, position=1, leave=False):
                    v1, v2, v3 = tri
                    a = get_middle_point(v1, v2)
                    b = get_middle_point(v2, v3)
                    c = get_middle_point(v3, v1)
                    
                    new_faces.append([v1, a, c])
                    new_faces.append([v2, b, a])
                    new_faces.append([v3, c, b])
                    new_faces.append([a, b, c])
                faces = np.array(new_faces)
                
            return vertices, faces

        def icosphere_to_hexsphere(vertices, faces):
            """
            Convert an icosphere into a hexagon-tiled sphere (with 12 pentagons).
            
            The dual of the icosphere subdivision creates hexagons and pentagons.
            Each vertex in the original mesh becomes the center of a hex/pentagon,
            and we create faces by connecting the centroids of adjacent triangles.
            
            Args:
            vertices: Vertex array from icosphere
            faces: Face array from icosphere
            
            Returns:
            hex_faces: List of vertex indices forming each hex/pentagon
            """
            # Find which faces are adjacent to each vertex
            vertex_to_faces = defaultdict(list)
            for face_idx, face in tqdm(enumerate(faces), position=0, desc="Finding adjacent faces"):
                for vertex in face:
                    vertex_to_faces[vertex].append(face_idx)
            
            # Calculate face centroids (these become the vertices of the hex tiles)
            face_centroids = []
            for face in tqdm(faces, position=0, desc="Calculating face centroids"):
                centroid = np.mean(vertices[face], axis=0)
                centroid /= np.linalg.norm(centroid)  # Project back to sphere
                face_centroids.append(centroid)
            face_centroids = np.array(face_centroids)
            
            # For each vertex in the original mesh, create a hex/pentagon
            # by ordering the centroids of adjacent faces
            hex_faces = []

            for vertex_idx in tqdm(range(len(vertices)), position=0, desc="Creating hex faces"):
                adjacent_faces = vertex_to_faces[vertex_idx]
                
                # Order the adjacent face centroids in circular order
                # We do this by sorting by angle around the vertex
                center = vertices[vertex_idx]
                
                # Create a local coordinate system
                # Use the first adjacent face centroid to define a reference direction
                ref_vec = face_centroids[adjacent_faces[0]] - center
                ref_vec /= np.linalg.norm(ref_vec)
                
                # Create perpendicular vector for angle calculation
                normal = center / np.linalg.norm(center)
                tangent = ref_vec - np.dot(ref_vec, normal) * normal
                tangent /= np.linalg.norm(tangent)
                bitangent = np.cross(normal, tangent)
                
                # Calculate angles for each adjacent face centroid
                angles = []
                for face_idx in adjacent_faces:
                    vec = face_centroids[face_idx] - center
                    vec /= np.linalg.norm(vec)
                    # Project onto tangent plane
                    vec_proj = vec - np.dot(vec, normal) * normal
                    vec_proj /= np.linalg.norm(vec_proj)
                    
                    # Calculate angle
                    x = np.dot(vec_proj, tangent)
                    y = np.dot(vec_proj, bitangent)
                    angle = np.arctan2(y, x)
                    angles.append((angle, face_idx))
                
                # Sort by angle
                angles.sort()
                ordered_faces = [face_idx for _, face_idx in angles]
                
                hex_faces.append(ordered_faces)
            
            return hex_faces, face_centroids

        # Create the cache directory if it doesn't exist
        os.makedirs('./.cache', exist_ok=True)

        # Check if cached tiles exist
        if self.use_cache:
            try:
                with open(f'./.cache/world_tiles_{rec_lvl}.pkl', 'rb') as f:
                    self.world_tiles = pickle.load(f)
                    print("\tLoaded cached world tiles.")
                    return
            except FileNotFoundError:
                print("\tNo cached world tiles found, generating new ones.")

        # Generate new tiles
        print("\tGenerating new world tiles...")
        start_time = time.time()
        vertices, faces = create_icosphere(rec_lvl)
        hex_faces, face_centroids = icosphere_to_hexsphere(vertices, faces)
        for hf in hex_faces:
            poly_verts = face_centroids[hf]
            self.world_tiles.append(WorldTile(len(self.world_tiles), np.mean(poly_verts, axis=0), poly_verts))
        print(f"\tTile generation took {time.time() - start_time:.2f} seconds.")

        # Convert 3D vertices to 2D (longitude, latitude)
        print("\tCalculating equirectangular coordinates for tiles...")
        start_time = time.time()
        for tile in tqdm(self.world_tiles, position=0, desc="Calculating equirectangular coordinates"):
            lon_lat = []
            for v in tile.vertices:
                x, y, z = v
                lon = np.arctan2(y, x)
                lat = np.arcsin(z / np.linalg.norm(v))
                lon_lat.append((lon, lat))

            # Unwrap longitudes to avoid stretching across the map edge
            lons = np.array([ll[0] for ll in lon_lat])
            lats = np.array([ll[1] for ll in lon_lat])
            # Find jumps in longitude
            lon_diffs = np.diff(lons)
            if np.any(np.abs(lon_diffs) > np.pi):
                # If a jump > pi is found, unwrap longitudes
                lons_unwrapped = lons.copy()
                for i in range(1, len(lons_unwrapped)):
                    diff = lons_unwrapped[i] - lons_unwrapped[i-1]
                    if diff > np.pi:
                        lons_unwrapped[i:] -= 2 * np.pi
                    elif diff < -np.pi:
                        lons_unwrapped[i:] += 2 * np.pi
                lon_lat = list(zip(lons_unwrapped, lats))
            tile.equirectangular_coords = lon_lat
        print(f"\tEquirectangular coordinate calculation took {time.time() - start_time:.2f} seconds.")


        # Save the world tiles to a file for faster loading next time
        with open(f'./.cache/world_tiles_{self.recursion_level}.pkl', 'wb') as f:
            pickle.dump(self.world_tiles, f)
        print("\tGenerated and cached new world tiles.")

    def populate_neighbor_lists(self):
        """
        Populate the neighbor lists for each tile.
        """
        # Create a mapping from vertex (as a tuple) to tiles that share it
        vertex_to_tiles = defaultdict(list)
        for tile in tqdm(self.world_tiles, position=0, desc="Mapping vertices to tiles"):
            for vertex in tile.vertices:
                vertex_key = tuple(np.round(vertex, decimals=6))  # Round to avoid floating point issues
                vertex_to_tiles[vertex_key].append(tile)

        # For each tile, find neighbors by looking up shared vertices
        for tile in tqdm(self.world_tiles, position=0, desc="Finding neighbors"):
            neighbor_set = set()
            for vertex in tile.vertices:
                vertex_key = tuple(np.round(vertex, decimals=6))
                for neighbor in vertex_to_tiles[vertex_key]:
                    if neighbor is not tile:
                        neighbor_set.add(neighbor)
            tile.neighbors = list(neighbor_set)

    def get_world_edges_and_world_vertices(self):
        """
        Gets the edges and vertices of the globe's tiles.
        """
        num_decimal_places = log10(1 / FLOAT_TOLERANCE)

        all_world_vertices = defaultdict(lambda: None) # key: (position tuple), value: WorldVertex
        all_world_edges_dict = defaultdict(lambda: None)    # key: (v1_key, v2_key), value: index in all_world_edges
        all_world_edges = []
        num_vertices = 0
        for tile in tqdm(self.world_tiles, position=0, desc="Processing tiles to get vertices"):
            for v in tile.vertices:
                v_key = tuple(np.array(v * 10 ** num_decimal_places).astype(int)) # Round to avoid floating point issues

                if all_world_vertices[v_key] is None:
                    new_vertex = WorldVertex(num_vertices, v)
                    all_world_vertices[v_key] = new_vertex
                    num_vertices += 1
                all_world_vertices[v_key].adjacent_tiles.append(tile)
                tile.world_vertices.add(all_world_vertices[v_key])

        for tile in tqdm(self.world_tiles, position=0, desc="Processing tiles to get edges"):
            vertex_keys = np.array(tile.vertices * 10 ** num_decimal_places).astype(int)
            tile_vertex_keys = [tuple(vk) for vk in vertex_keys]
            num_tile_vertices = len(tile_vertex_keys)
            for i in range(num_tile_vertices):
                v1_key = tile_vertex_keys[i]
                v2_key = tile_vertex_keys[(i + 1) % num_tile_vertices] # wrap around to first vertex

                edge_key = tuple(sorted((v1_key, v2_key)))
                if all_world_edges_dict[edge_key] is None:
                    v1 = all_world_vertices[v1_key]
                    v2 = all_world_vertices[v2_key]
                    v1.neighboring_vertices.append(v2)
                    v2.neighboring_vertices.append(v1)
                    new_edge = WorldEdge(v1, v2)
                    all_world_edges.append(new_edge)
                    all_world_edges_dict[edge_key] = len(all_world_edges) - 1
                edge_idx = all_world_edges_dict[edge_key]
                all_world_edges[edge_idx].adjacent_tiles.append(tile)
                tile.world_edges.add(all_world_edges[edge_idx])

        return all_world_edges, list(all_world_vertices.values())
            
    """
    Basic terrain generation (using OpenSimplex noise)
    """
    def map_noise(self, scale, num_octaves, importance=None):
        """
        Adds 4D noise values to each tile for terrain generation.
        scale: Scale factor for noise frequency
        num_octaves: Number of noise octaves to sum
        importance: Importance factor for noise contribution to height (0 to 1).
                    If None, the noise will completely overwrite existing height values.
        """
        noise_gen = opensimplex.OpenSimplex(seed=random.randint(0, 10000))
        for tile in tqdm(self.world_tiles, position=0, desc="Mapping noise"):
            x, y, z = tile.center
            # Scale coordinates for noise
            # Note: using 4D noise with a fixed w value to get a 3D slice for randomness
            value = 0

            for octave in range(num_octaves):
                frequency = 2 ** octave
                amplitude = 0.5 ** octave
                value += amplitude * noise_gen.noise4(x * scale * frequency, y * scale * frequency, z * scale * frequency, self.noise_4d_val)

            # noise value ranges from -1 to 1
            # Normalize to 0-1
            normalized_value = (value + 1) / 2

            if importance is None:
                tile.height = normalized_value
            else:
                tile.height = (1 - importance) * tile.height + importance * normalized_value

    def add_noise(self, scale, num_octaves, amplitude):
        """
        Adds 4D noise values to existing height values of each tile.
        scale: Scale factor for noise frequency
        num_octaves: Number of noise octaves to sum
        amplitude: Amplitude factor for noise contribution to height
        """
        noise_gen = opensimplex.OpenSimplex(seed=random.randint(0, 10000))
        for tile in tqdm(self.world_tiles, position=0, desc="Adding noise"):
            x, y, z = tile.center
            # Scale coordinates for noise
            # Note: using 4D noise with a fixed w value to get a 3D slice for randomness
            value = 0

            for octave in range(num_octaves):
                frequency = 2 ** octave
                amp = 0.5 ** octave
                value += amp * noise_gen.noise4(x * scale * frequency, y * scale * frequency, z * scale * frequency, self.noise_4d_val)

            # noise value ranges from -1 to 1
            tile.height += amplitude * value

    """
    Tectonic plate generation and simulation
    """
    def create_tectonic_plates_with_voronoi(self, num_plates):
        """
        Create tectonic plates using spherical Voronoi tessellation.
        """
        # Randomly select seed points on the sphere
        seed_points = []
        for _ in range(num_plates):
            phi = np.arccos(1 - 2 * random.random())  # polar angle
            theta = 2 * np.pi * random.random()       # azimuthal angle
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            seed_points.append([x, y, z])
        seed_points = np.array(seed_points)

        # Create Spherical Voronoi tessellation
        sv = SphericalVoronoi(seed_points)
        sv.sort_vertices_of_regions()

        # Create a mapping from a tectonic plate ids to its tiles
        plate_ids_to_tiles = {}
        for plate_id in range(num_plates):
            plate_ids_to_tiles[plate_id] = []

        # Assign tiles to plates based on closest seed point
        for tile in tqdm(self.world_tiles, position=0, desc="Assigning tiles to tectonic plates (Voronoi)"):
            min_dist = float('inf')
            assigned_plate_id = None
            for plate_id, center in enumerate(seed_points):
                dist = np.arccos(np.dot(tile.center, center))
                if dist < min_dist:
                    min_dist = dist
                    assigned_plate_id = plate_id
            tile.plate_id = assigned_plate_id
            plate_ids_to_tiles[assigned_plate_id].append(tile)

        # Finally, create the plate objects from the tile assignments
        self.tectonic_plates = []
        for plate_id in range(num_plates):
            plate = TectonicPlate(plate_id, plate_ids_to_tiles[plate_id])
            self.tectonic_plates.append(plate)

    def create_tectonic_plates_using_random_growth(self, num_plates):
        """
        Create tectonic plates using a random plate growth algorithm.

        Idea: start with no tiles assigned to plates.

        Select num_plates random seed tiles on the globe, and assign each to a different plate.

        Then, choose a random unassigned tile that is adjacent to an assigned tile, 
        and assign it to the same plate as that adjacent tile.
        """
        plate_lists = [[] for _ in range(num_plates)]
        unassigned_tiles = set(self.world_tiles)

        # Select random seed tiles for each plate
        for plate_id in range(num_plates):
            seed_tile = random.choice(list(unassigned_tiles))
            seed_tile.plate_id = plate_id
            plate_lists[plate_id].append(seed_tile)
            unassigned_tiles.remove(seed_tile)

        frontier_tiles = set()
        for plate_id in range(num_plates):
            for tile in plate_lists[plate_id]:
                for neighbor in tile.neighbors:
                    if neighbor in unassigned_tiles:
                        frontier_tiles.add(neighbor)

        while len(unassigned_tiles) > 0:
            current_tile = random.choice(list(frontier_tiles))
            # Find assigned neighbor(s)
            assigned_neighbors = [n for n in current_tile.neighbors if n not in unassigned_tiles]
            
            # Randomly select one assigned neighbor to determine plate assignment
            chosen_neighbor = random.choice(assigned_neighbors)
            current_tile.plate_id = chosen_neighbor.plate_id
            plate_lists[chosen_neighbor.plate_id].append(current_tile)
            unassigned_tiles.remove(current_tile)
            frontier_tiles.remove(current_tile)

            # Add new frontier tiles
            for neighbor in current_tile.neighbors:
                if neighbor in unassigned_tiles:
                    frontier_tiles.add(neighbor)

        # Finally, create the plate objects from the tile assignments
        self.tectonic_plates = []
        for plate_id in range(num_plates):
            plate = TectonicPlate(plate_id, plate_lists[plate_id])
            self.tectonic_plates.append(plate)

    def create_tectonic_plates_using_noise_guided_growth(self, num_plates):
        """
        Create tectonic plates using a noise-guided plate growth algorithm.

        Idea: start with no tiles assigned to plates.

        Select num_plates random seed tiles on the globe, and assign each to a different plate.

        Then, choose a random unassigned tile that is adjacent to an assigned tile, 
        and assign it to the same plate as that adjacent tile.

        TODO: This version has an issue where plates can end up inside other plates!
        """
        noise_gen = opensimplex.OpenSimplex(seed=random.randint(0, 10000))
        random_noise_offset = np.random.uniform(0, 1000)
        noise_values = {}
        for tile in self.world_tiles:
            x, y, z = tile.center

            num_octaves = 3
            scale = 10
            amp = 1

            noise_value = 0
            for octave in range(num_octaves):
                frequency = 2 ** octave
                amp = 0.5 ** octave
                noise_value += amp * noise_gen.noise4(x * scale * frequency, y * scale * frequency, z * scale * frequency, random_noise_offset)
            noise_values[tile] = noise_value

        plate_lists = [[] for _ in range(num_plates)]
        unassigned_tiles = set(self.world_tiles)

        # Select random seed tiles for each plate
        for plate_id in range(num_plates):
            seed_tile = random.choice(list(unassigned_tiles))
            seed_tile.plate_id = plate_id
            plate_lists[plate_id].append(seed_tile)
            unassigned_tiles.remove(seed_tile)

        # Create a min-heap based on noise values for frontier tiles
        in_heap = set()
        frontier_heap = []
        for plate_id in range(num_plates):
            for tile in plate_lists[plate_id]:
                for neighbor in tile.neighbors:
                    if neighbor in unassigned_tiles:
                        heapq.heappush(frontier_heap, (noise_values[neighbor], neighbor))
                        in_heap.add(neighbor)

        while len(unassigned_tiles) > 0:
            current_tile = heapq.heappop(frontier_heap)[1]
            # Find assigned neighbor(s)
            assigned_neighbors = [n for n in current_tile.neighbors if n not in unassigned_tiles]
            
            # Randomly select one assigned neighbor to determine plate assignment
            chosen_neighbor = random.choice(assigned_neighbors)
            current_tile.plate_id = chosen_neighbor.plate_id
            plate_lists[chosen_neighbor.plate_id].append(current_tile)
            unassigned_tiles.remove(current_tile)

            # Add new frontier tiles
            for neighbor in current_tile.neighbors:
                if neighbor in unassigned_tiles and neighbor not in in_heap:
                    heapq.heappush(frontier_heap, (noise_values[neighbor], neighbor))
                    in_heap.add(neighbor)

        # Finally, create the plate objects from the tile assignments
        self.tectonic_plates = []
        for plate_id in range(num_plates):
            plate = TectonicPlate(plate_id, plate_lists[plate_id])
            self.tectonic_plates.append(plate)

    def create_plate_boundaries(self):
        """
        Identify plate boundaries and create PlateBoundary objects.
        """
        boundary_dict = {}  # key: (plate1_id, plate2_id), value: list of boundary tiles

        for tile in tqdm(self.world_tiles, position=0, desc="Identifying plate boundaries"):
            for neighbor in tile.neighbors:
                if neighbor.plate_id != tile.plate_id:
                    plate_pair = tuple(sorted((tile.plate_id, neighbor.plate_id)))
                    if plate_pair not in boundary_dict:
                        boundary_dict[plate_pair] = set()
                    boundary_dict[plate_pair].add(tile)

        # Create PlateBoundary objects
        for (plate1_id, plate2_id), boundary_tiles in boundary_dict.items():
            plate1 = self.tectonic_plates[plate1_id]
            plate2 = self.tectonic_plates[plate2_id]

            boundary = PlateBoundary(plate1, plate2, list(boundary_tiles))
            self.plate_boundaries.append(boundary)
            plate1.boundaries.append(boundary)
            plate2.boundaries.append(boundary)

    def assign_heights_using_plates(self):
        """
        Simulate tectonic activity!
        """
        for boundary in self.plate_boundaries:
            boundary.simulate_tectonic_activity()

    """
    Tile coloring
    """
    def assign_terrain_colors_by_height(self):
        """
        Assign terrain colors to each tile based on height values.
        """
        for tile in self.world_tiles:
            if tile.height is None:
                raise ValueError("Height values not mapped. Call map_noise() or assign height values using another function first.")
            if tile.height < WATER_THRESHOLD:
                # Blue value should move from water_low_color to water_high_color based on hexagon.height
                t = tile.height / WATER_THRESHOLD
                tile.color = tuple(np.array(WATER_LOW_COLOR) * (1 - t) + np.array(WATER_HIGH_COLOR) * t)
            elif tile.height < BEACH_THRESHOLD:
                tile.color = BEACH_COLOR
            elif tile.height < LAND_THRESHOLD:
                # Green value should move from land_low_color to land_high_color based on hexagon.height
                t = (tile.height - WATER_THRESHOLD) / (LAND_THRESHOLD - WATER_THRESHOLD)
                tile.color = tuple(np.array(LAND_LOW_COLOR) * (1 - t) + np.array(LAND_HIGH_COLOR) * t)
            elif tile.height < MOUNTAIN_THRESHOLD:
                # Gray value should move from mountain_low_color to mountain_high_color based on hexagon.height
                t = (tile.height - LAND_THRESHOLD) / (MOUNTAIN_THRESHOLD - LAND_THRESHOLD)
                tile.color = tuple(np.array(MOUNTAIN_LOW_COLOR) * (1 - t) + np.array(MOUNTAIN_HIGH_COLOR) * t)
            else:
                # White color for snow
                tile.color = SNOW_COLOR

    def assign_tectonic_plate_colors(self):
        """
        Assign colors to tiles based on their tectonic plate.
        """
        plate_colors = {}
        for plate in self.tectonic_plates:
            # Assign a random color to each plate
            plate_colors[plate.plate_id] = (random.random(), random.random(), random.random())

        for tile in self.world_tiles:
            if tile.plate_id is not None:
                tile.color = plate_colors[tile.plate_id]
            else:
                tile.color = (0.5, 0.5, 0.5)  # Gray for unassigned tiles

    def assign_plate_boundary_colors(self):
        """
        Assign colors to tiles based on plate boundaries.
        """
        for tile in self.world_tiles:
            tile.color = np.array([0.8, 0.8, 0.8])  # Light gray base color

        for boundary in self.plate_boundaries:
            for tile in boundary.direct_boundary_tiles:
                if boundary.boundary_type == 'convergent':
                    tile.color -= np.array([0, 0.33, 0.33]) * boundary.activity_level  # Red for convergent
                elif boundary.boundary_type == 'divergent':
                    tile.color -= np.array([0.33, 0.33, 0]) * boundary.activity_level  # Blue for divergent
                elif boundary.boundary_type == 'transform':
                    tile.color -= np.array([0.33, 0, 0.33]) * boundary.activity_level  # Green for transform

        for tile in self.world_tiles:
            # Clamp color values to [0, 1]
            tile.color = tuple(np.clip(tile.color, 0, 1))

            # convert back to tuple
            tile.color = tuple(tile.color)

    def assign_colors_based_on_boundary_proximity(self, max_distance=0.1):
        """
        Assign colors to tiles based on proximity to plate boundaries.
        """
        for tile in self.world_tiles:
            tile.color = tuple(np.array([1.0, 1.0, 1.0]))  # Start with white color

        for boundary in self.plate_boundaries:
            visited_tiles = set()
            min_heap = []

            starting_tile = boundary.direct_boundary_tiles[0]
            starting_dist = boundary.calculate_distance_to_boundary_line(starting_tile.center)

            heapq.heappush(min_heap, (starting_dist, starting_tile))
            visited_tiles.add(starting_tile)

            while min_heap:
                current_distance, current_tile = heapq.heappop(min_heap)

                # Assign color based on distance
                t = current_distance / max_distance
                t = min(max(t, 0), 1)  # Clamp t to [0, 1]
                boundary_color = np.array([1.0, 0.0, 0.0])  # Red color for boundary
                base_color = np.array([1.0, 1.0, 1.0])  # White base color
                blended_color = boundary_color * (1 - t) + base_color * t # blend of boundary and base colors
                final_color = np.min(np.stack((current_tile.color, blended_color), axis=0), axis=0)  # Darken existing color towards boundary color
                current_tile.color = tuple(final_color)

                # Explore neighbors
                for neighbor in current_tile.neighbors:
                    if neighbor in visited_tiles:
                        continue

                    # Calculate distance to boundary
                    distance = boundary.calculate_distance_to_boundary_line(neighbor.center)
                    if distance < max_distance:
                        heapq.heappush(min_heap, (distance, neighbor))
                        visited_tiles.add(neighbor)

            print(f"Num tiles visited for boundary between plate {boundary.plate1.plate_id} and plate {boundary.plate2.plate_id}: {len(visited_tiles)}")

    """
    Visualization
    """
    def plot_equirectangular(self):
        # Simple equirectangular projection for testing
        fig, ax = plt.subplots(figsize=(10, 5))
        for tile in self.world_tiles:
            polygon = plt.Polygon(tile.equirectangular_coords, edgecolor=None, alpha=0.5, facecolor=tile.color)
            ax.add_patch(polygon)

        ax.set_xlim(-np.pi - 0.3, np.pi + 0.3) # Add some padding for longitudes that were unwrapped
        ax.set_ylim(-np.pi/2, np.pi/2)
        ax.set_aspect('equal')
        plt.show()

    def plot_sphere(self):
        """
        Visualize the hexagon and pentagon-tiled sphere.
        """
        fig = plt.figure(figsize=(5, 5))
        
        # Hex-tiled sphere
        ax = fig.add_subplot(projection='3d')
        hex_polys = []
        colors = []
        for tile in self.world_tiles:
            poly_verts = tile.vertices
            hex_polys.append(poly_verts)
            colors.append(tile.color)

        poly = Poly3DCollection(hex_polys, facecolors=colors, linewidths=1, edgecolors='black', alpha=0.9)
        ax.add_collection3d(poly)
        
        max_range = 1.2
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        ax.set_box_aspect([1, 1, 1])
        
        plt.tight_layout()
        plt.show()

    def plot_plate_movement_vectors_vispy(self, view):
        for plate in self.tectonic_plates:
            # Draw simple plate movement vectors from plate centroid
            if plate.total_movement_vector is not None and plate.centroid is not None:
                normal_vec = plate.centroid / np.linalg.norm(plate.centroid)

                # triangle arrow parameters
                base_width = 0.003
                base_center = plate.centroid + normal_vec * 0.05  # offset from surface slightly

                # compute tangent vector for arrow direction
                tangent_vec = np.cross(plate.total_movement_vector, (plate.centroid / np.linalg.norm(plate.centroid)))
                arrow_length = np.linalg.norm(tangent_vec) * 0.0001 + 0.01  # scale arrow length based on movement vector magnitude

                # define arrow vertices
                tip_point = base_center + tangent_vec * arrow_length
                base_point1 = base_center - tangent_vec * (arrow_length * 0.3) + np.cross(normal_vec, tangent_vec) * (base_width / 2)
                base_point2 = base_center - tangent_vec * (arrow_length * 0.3) - np.cross(normal_vec, tangent_vec) * (base_width / 2)

                # create arrow mesh
                arrow_verts = np.array([tip_point, base_point1, base_point2])
                arrow_faces = np.array([[0, 1, 2]])
                arrow = scene.visuals.Mesh(vertices=arrow_verts.astype(np.float32), faces=arrow_faces.astype(np.int32), color='blue', shading=None)
                view.add(arrow)
            else:
                print(f"Plate {plate.plate_id} either has no movement vector defined or no centroid.")
                continue

    def plot_boundary_vectors_vispy(self, view):
        for boundary in self.plate_boundaries:
            # draw two arrows at the boundary midpoint indicating the movement of each plate at that boundary
            if boundary.boundary_centroid is not None and boundary.plate1.total_movement_vector is not None and boundary.plate2.total_movement_vector is not None:
                tangent_vec_1 = np.cross(boundary.plate1.total_movement_vector, (boundary.boundary_centroid / np.linalg.norm(boundary.boundary_centroid)))
                tangent_vec_2 = np.cross(boundary.plate2.total_movement_vector, (boundary.boundary_centroid / np.linalg.norm(boundary.boundary_centroid)))
                arrow_1_length = np.linalg.norm(tangent_vec_1) * 0.0001 + 0.01
                arrow_2_length = np.linalg.norm(tangent_vec_2) * 0.0001 + 0.01
                arrows_base_width = 0.003
                arrow_1_color = 'green'
                arrow_2_color = 'orange'

                # define arrow 1 vertices
                tip_point_1 = boundary.boundary_centroid + tangent_vec_1 * arrow_1_length
                base_point1_1 = boundary.boundary_centroid - tangent_vec_1 * (arrow_1_length * 0.3) + np.cross(boundary.boundary_centroid, tangent_vec_1) * (arrows_base_width / 2)
                base_point2_1 = boundary.boundary_centroid - tangent_vec_1 * (arrow_1_length * 0.3) - np.cross(boundary.boundary_centroid, tangent_vec_1) * (arrows_base_width / 2)
                arrow_1_verts = np.array([tip_point_1, base_point1_1, base_point2_1])
                arrow_1_faces = np.array([[0, 1, 2]])
                arrow_1 = scene.visuals.Mesh(vertices=arrow_1_verts.astype(np.float32), faces=arrow_1_faces.astype(np.int32), color=arrow_1_color, shading=None)
                view.add(arrow_1)

                # define arrow 2 vertices
                tip_point_2 = boundary.boundary_centroid + tangent_vec_2 * arrow_2_length
                base_point1_2 = boundary.boundary_centroid - tangent_vec_2 * (arrow_2_length * 0.3) + np.cross(boundary.boundary_centroid, tangent_vec_2) * (arrows_base_width / 2)
                base_point2_2 = boundary.boundary_centroid - tangent_vec_2 * (arrow_2_length * 0.3) - np.cross(boundary.boundary_centroid, tangent_vec_2) * (arrows_base_width / 2)
                arrow_2_verts = np.array([tip_point_2, base_point1_2, base_point2_2])
                arrow_2_faces = np.array([[0, 1, 2]])
                arrow_2 = scene.visuals.Mesh(vertices=arrow_2_verts.astype(np.float32), faces=arrow_2_faces.astype(np.int32), color=arrow_2_color, shading=None)
                view.add(arrow_2)
            else:
                print(f"Boundary between plate {boundary.plate1.plate_id} and plate {boundary.plate2.plate_id} missing data for drawing boundary vectors.")
                continue

    def plot_randomly_sampled_movement_vectors_vispy(self, view, num_vectors=200):
        # draw movement vectors at randomly sampled tile centers
        sampled_tiles = random.sample(self.world_tiles, k=num_vectors)
        for tile in sampled_tiles:
            plate = self.tectonic_plates[tile.plate_id]
            if plate.total_movement_vector is not None:
                normal_vec = tile.center / np.linalg.norm(tile.center)

                # triangle arrow parameters
                base_width = 0.003
                base_center = tile.center + normal_vec * 0.05  # offset from surface slightly

                tangent_vec = np.cross(plate.total_movement_vector, (tile.center / np.linalg.norm(tile.center)))
                arrow_length = np.linalg.norm(tangent_vec) * 0.0001 + 0.01  # scale arrow length based on movement vector magnitude

                # define arrow vertices
                tip_point = base_center + tangent_vec * arrow_length
                base_point1 = base_center - tangent_vec * (arrow_length * 0.3) + np.cross(normal_vec, tangent_vec) * (base_width / 2)
                base_point2 = base_center - tangent_vec * (arrow_length * 0.3) - np.cross(normal_vec, tangent_vec) * (base_width / 2)

                # create arrow mesh
                arrow_verts = np.array([tip_point, base_point1, base_point2])
                arrow_faces = np.array([[0, 1, 2]])
                arrow = scene.visuals.Mesh(vertices=arrow_verts.astype(np.float32), faces=arrow_faces.astype(np.int32), color='purple', shading=None)
                view.add(arrow)
            else:
                print(f"Tile's plate {plate.plate_id} has no movement vector defined.")
                continue

    def plot_sphere_vispy(self, visualize_heights=False, height_scale=0.1, shading=True, draw_plate_vectors=True, draw_north_pole=False, draw_boundary_component_vectors=True, draw_randomly_sampled_vectors=True):
        print(app.use_app())

        # Create VisPy canvas with 3D camera
        canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
        view = canvas.central_widget.add_view()
        view.camera = scene.cameras.TurntableCamera(fov=45, distance=10)

        colorset = set()
        for tile in self.world_tiles:
            colorset.add(tile.color)
        print(f"Unique colors used: {len(colorset)}")

        all_faces = []
        all_verts = []
        all_face_colors = []
        vert_offset = 0

        if visualize_heights:
            extra_verts = []
            extra_faces = []
            extra_colors = []
            extra_vert_offset = 0

        for tile in self.world_tiles:
            if visualize_heights:
                # set the center vertex to be offset by height
                normal_vec = tile.center / np.linalg.norm(tile.center)
                center_vertex = tile.center + normal_vec * (tile.height * height_scale)

                # numpy has a bug where after being pickled, numpy.ndarray's isbuiltin is set to 0
                # this fixes that issue
                equi_type = getattr(np, str(tile.vertices.dtype))
                tile.vertices = tile.vertices.astype(equi_type)

                hex_verts = tile.vertices.copy()
                for v_idx in range(len(hex_verts)):
                    normal_vec = hex_verts[v_idx] / np.linalg.norm(hex_verts[v_idx])
                    hex_verts[v_idx] = hex_verts[v_idx] + normal_vec * (tile.height * height_scale)
                these_verts = np.vstack([center_vertex, hex_verts])

                # Add 2 new faces for every edge: connecting the original cordinates to the heighted coordinates
                num_vertices = len(tile.vertices)
                for i in range(num_vertices):
                    v1_base = tile.vertices[i]
                    v2_base = tile.vertices[(i + 1) % num_vertices]
                    v1_height = hex_verts[i]
                    v2_height = hex_verts[(i + 1) % num_vertices]

                    extra_verts.extend([v1_base, v2_base, v2_height, v1_height])
                    extra_faces.append([
                        extra_vert_offset,
                        extra_vert_offset + 1,
                        extra_vert_offset + 2,
                    ])
                    extra_faces.append([
                        extra_vert_offset,
                        extra_vert_offset + 2,
                        extra_vert_offset + 3,
                    ])
                    extra_vert_offset += 4
                
                extra_colors.append([tile.color] * (num_vertices * 2))
            else:
                these_verts = np.vstack([tile.center, tile.vertices])

            if len(tile.vertices) == 5:
                # Pentagons need to be triangulated
                faces = np.array([
                    [0, 1, 2],
                    [0, 2, 3],
                    [0, 3, 4],
                    [0, 4, 5],
                    [0, 5, 1],
                ])
            else:
                # Hexagons can be triangulated similarly
                faces = np.array([
                    [0, 1, 2],
                    [0, 2, 3],
                    [0, 3, 4],
                    [0, 4, 5],
                    [0, 5, 6],
                    [0, 6, 1],
                ])

            all_verts.append(these_verts)
            all_faces.append(faces + vert_offset)
            all_face_colors.append([tile.color] * faces.shape[0])
            vert_offset += these_verts.shape[0]

        mesh = scene.visuals.Mesh(vertices=np.concatenate(all_verts), faces=np.concatenate(all_faces), face_colors=np.concatenate(all_face_colors), shading=None)
        view.add(mesh)
        if visualize_heights:
            extra_mesh = scene.visuals.Mesh(vertices=np.array(extra_verts), faces=np.array(extra_faces), face_colors=np.concatenate(extra_colors), shading=None)
            view.add(extra_mesh)

        if draw_plate_vectors:
            self.plot_plate_movement_vectors_vispy(view)

        if draw_boundary_component_vectors:
            self.plot_boundary_vectors_vispy(view)

        if draw_randomly_sampled_vectors:
            self.plot_randomly_sampled_movement_vectors_vispy(view, num_vectors=200)

        if draw_north_pole:
            # draw a pyramid at the north pole for reference
            north_pole_verts = np.array([
                [0, 0, 1.1],
                [0.1, 0, 1],
                [0, 0.1, 1],
                [-0.1, 0, 1],
                [0, -0.1, 1],
            ])
            north_pole_faces = np.array([
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 1],
            ])
            north_pole_mesh = scene.visuals.Mesh(vertices=north_pole_verts, faces=north_pole_faces, color='red', shading=None)
            view.add(north_pole_mesh)

        if shading:
            global shading_state_index
            global wireframe_state_index
            global shininess_state_index

            # Use filters to affect the rendering of the mesh.
            wireframe_filter = WireframeFilter(enabled=False)
            # Note: For convenience, this `ShadingFilter` would be created automatically by
            # the `MeshVisual with, e.g. `mesh = MeshVisual(..., shading='smooth')`. It is
            # created manually here for demonstration purposes.
            shading_filter = ShadingFilter(shading='smooth', shininess=1000)
            # The wireframe filter is attached before the shading filter otherwise the
            # wireframe is not shaded.
            mesh.attach(wireframe_filter)
            mesh.attach(shading_filter)

            if visualize_heights:
                extra_wireframe_filter = WireframeFilter(enabled=False)
                extra_shading_filter = ShadingFilter(shading='smooth', shininess=1000)
                extra_mesh.attach(extra_wireframe_filter)
                extra_mesh.attach(extra_shading_filter)

            def attach_headlight(view):
                light_dir = (0, 1, 0, 0)
                shading_filter.light_dir = light_dir[:3]
                initial_light_dir = view.camera.transform.imap(light_dir)

                @view.scene.transform.changed.connect
                def on_transform_change(event):
                    transform = view.camera.transform
                    shading_filter.light_dir = transform.map(initial_light_dir)[:3]


            attach_headlight(view)

            shading_states = (
                dict(shading=None),
                dict(shading='flat'),
                dict(shading='smooth'),
            )
            shading_state_index = shading_states.index(
                dict(shading=shading_filter.shading))

            wireframe_states = (
                dict(wireframe_only=False, faces_only=False,),
                dict(wireframe_only=True, faces_only=False,),
                dict(wireframe_only=False, faces_only=True,),
            )
            wireframe_state_index = wireframe_states.index(dict(
                wireframe_only=wireframe_filter.wireframe_only,
                faces_only=wireframe_filter.faces_only,
            ))

            shininess_states = (1, 10, 100, 1000)
            shininess_state_index = 3

            def cycle_state(states, index):
                new_index = (index + 1) % len(states)
                return states[new_index], new_index


            @canvas.events.key_press.connect
            def on_key_press(event):
                global shading_state_index
                global wireframe_state_index
                global shininess_state_index
                if event.key == 's':
                    state, shading_state_index = cycle_state(shading_states,
                                                            shading_state_index)
                    for attr, value in state.items():
                        setattr(shading_filter, attr, value)
                        if visualize_heights:
                            setattr(extra_shading_filter, attr, value)
                    mesh.update()
                    if visualize_heights:
                        extra_mesh.update()
                elif event.key == 'w':
                    wireframe_filter.enabled = not wireframe_filter.enabled
                    if visualize_heights:
                        extra_wireframe_filter.enabled = not extra_wireframe_filter.enabled
                    mesh.update()
                    if visualize_heights:
                        extra_mesh.update()
                elif event.key == 'f':
                    state, wireframe_state_index = cycle_state(wireframe_states,
                                                            wireframe_state_index)
                    for attr, value in state.items():
                        setattr(wireframe_filter, attr, value)
                        if visualize_heights:
                            setattr(extra_wireframe_filter, attr, value)
                    mesh.update()
                    if visualize_heights:
                        extra_mesh.update()
                elif event.key == 'h':
                    state, shininess_state_index = cycle_state(
                        shininess_states, shininess_state_index)
                    shading_filter.shininess = state
                    if visualize_heights:
                        extra_shading_filter.shininess = state
                    mesh.update()
                    if visualize_heights:
                        extra_mesh.update()
            
        app.run()
        
    def fill_in_sea(self):
        """
        Fills the oceans by adjusting the heights of tiles below sea level. Does not affect tile color.
        """
        for tile in self.world_tiles:
            if tile.height < WATER_THRESHOLD:
                tile.height = WATER_THRESHOLD

    """Testing and debugging"""
    def print_geometry_statistics(self):
        """
        Print statistics about the hexagonal grid.
        """
        num_tiles = len(self.world_tiles)
        num_pentagons = sum(1 for t in self.world_tiles if len(t.vertices) == 5)
        num_hexagons = num_tiles - num_pentagons

        max_x_coord_for_center_of_tile = max(t.center[0] for t in self.world_tiles)
        min_x_coord_for_center_of_tile = min(t.center[0] for t in self.world_tiles)
        max_y_coord_for_center_of_tile = max(t.center[1] for t in self.world_tiles)
        min_y_coord_for_center_of_tile = min(t.center[1] for t in self.world_tiles)
        max_z_coord_for_center_of_tile = max(t.center[2] for t in self.world_tiles)
        min_z_coord_for_center_of_tile = min(t.center[2] for t in self.world_tiles)

        print(f"Total tiles (including pentagons): {num_tiles}")
        print(f"Number of pentagons: {num_pentagons}")
        print(f"Number of hexagons: {num_hexagons}")
        print(f"Max X coord for center of tile: {max_x_coord_for_center_of_tile}")
        print(f"Min X coord for center of tile: {min_x_coord_for_center_of_tile}")
        print(f"Max Y coord for center of tile: {max_y_coord_for_center_of_tile}")
        print(f"Min Y coord for center of tile: {min_y_coord_for_center_of_tile}")
        print(f"Max Z coord for center of tile: {max_z_coord_for_center_of_tile}")
        print(f"Min Z coord for center of tile: {min_z_coord_for_center_of_tile}")

    def paint_random_and_neighbors_red(self):
        """
        For testing: Paint a random tile and its neighbors red.
        """
        random_hex = random.choice(self.world_tiles)
        random_hex.color = (1.0, 0, 0)  # Red
        for neighbor in random_hex.neighbors:
            neighbor.color = (1.0, 0, 0)  # Red

    def make_random_tower(self, height_increase=0.5):
        """
        For testing: Increase the height of a random tile to create a tower.
        """
        random_hex = random.choice(self.world_tiles)
        random_hex.height += height_increase

    def print_world_edge_and_vertex_statistics(self):
        """
        Print statistics about the edges and vertices of the world tiles.
        """
        print(f"Total world_edges: {len(self.world_edges)}")
        print(f"Total unique vertices in the world: {len(self.world_vertices)}")
        for vertex in self.world_vertices:
            assert len(vertex.adjacent_tiles) == 3, "Each vertex should be adjacent to exactly 3 tiles."
            assert len(vertex.neighboring_vertices) == 3, "Each vertex should be adjacent to exactly 3 vertices."

class TectonicPlate:
    """
    Represents a tectonic plate consisting of multiple WorldTiles.
    """
    def __init__(self, plate_id, tiles):
        self.plate_id = plate_id
        self.tiles = tiles  # List of WorldTile objects belonging to this plate
        self.boundaries = []  # List of PlateBoundary objects
        self.noise_4d_val = np.random.uniform(0, 1000)  # Random value for 4D noise slice. Used for noise within the plate
        self.noise_gen = opensimplex.OpenSimplex(seed=random.randint(0, 10000))

        # Identify centroid
        self.centroid = self.get_plate_centroid()

        # Identify external vertices
        self.external_vertices = self.get_plate_external_vertices()

        # Assign oceanic or continental plate type
        if random.random() < OCEANIC_PLATE_RATIO:
            self.plate_type = 'oceanic'
        else:
            self.plate_type = 'continental'

        # Randomly generate an average height for the plate based on its type
        if self.plate_type == 'oceanic':
            self.avg_height = np.random.uniform(convert_from_sea_level_height_km(OCEANIC_PLATE_AVG_HEIGHT_LOW_KM), convert_from_sea_level_height_km(OCEANIC_PLATE_AVG_HEIGHT_HIGH_KM))
        else:
            self.avg_height = np.random.uniform(convert_from_sea_level_height_km(CONTINENTAL_PLATE_AVG_HEIGHT_LOW_KM), convert_from_sea_level_height_km(CONTINENTAL_PLATE_AVG_HEIGHT_HIGH_KM))

        # Assign the heights of tiles within the plate using noise
        self.set_internal_heights()

        # Assign random movement vectors and rotation speeds
        movement_vector = np.random.uniform(-2 * np.pi, 2 * np.pi, size=3) # angular rotation vector on a sphere (rotation about X, rotation about Y, rotation about Z)
        rotation_scalar = np.random.uniform(-2 * np.pi, 2 * np.pi)
        rotation_vector = rotation_scalar * ( self.centroid / np.linalg.norm(self.centroid) ) # angular rotation speed about the center of the plate
        """
        These movement vectors represent angular velocities (radians per unit time) around the X, Y, and Z axes.
        To convert these angular velocities into linear velocities at a point on the sphere's surface, we can use the cross product:
        v = ω x r
        where:
        v is the linear velocity vector at the point on the surface.
        ω is the angular velocity vector ( the movement vector + the rotation vector).
        r is the position vector from the center of the sphere to the point on the surface (tile.center).
        This will give us the linear velocity vector at that point on the sphere's surface due to the plate's rotation.
        """
        self.total_movement_vector = movement_vector + rotation_vector # the combined movement vector

    def __repr__(self):
        return f"TectonicPlate(plate_id={self.plate_id}, num_tiles={len(self.tiles)}, plate_type={self.plate_type})"
    
    def __eq__(self, value):
        if not isinstance(value, TectonicPlate):
            return False
        return self.plate_id == value.plate_id
    
    def get_plate_centroid(self):
        """
        Returns the centroid of the plate based on the centers of its tiles.
        """
        if not self.tiles:
            return None
        sum_coords = np.sum([tile.center for tile in self.tiles], axis=0)
        centroid = sum_coords / len(self.tiles)
        centroid /= np.linalg.norm(centroid)  # Normalize to lie on sphere surface
        return centroid
    
    def get_plate_external_vertices(self):
        """
        Identify and return the set of vertices on the plate's exterior.
        """
        vertex_count = {}  # key: vertex (as tuple), value: count of how many times it appears in plate tiles
        for tile in self.tiles:
            for vertex in tile.vertices:
                vertex_key = tuple(np.round(vertex, decimals=8))
                if vertex_key in vertex_count:
                    vertex_count[vertex_key] += 1
                else:
                    vertex_count[vertex_key] = 1

        external_vertices = set()
        for vertex_key, count in vertex_count.items():
            if count < 3:  # Appears less than 3 times means it's on the exterior
                external_vertices.add(vertex_key)

        self.external_vertices = external_vertices

        return external_vertices

    def set_internal_heights(self):
        """
        Set the heights of all tiles in the plate to the plate's average height.
        """
        def get_local_noise(tile, scale, num_octaves, noise_gen):
            x, y, z = tile.center
            # Scale coordinates for noise
            # Note: using 4D noise with a fixed w value to get a 3D slice for randomness
            value = 0

            for octave in range(num_octaves):
                frequency = 2 ** octave
                amplitude = 0.5 ** octave
                value += amplitude * noise_gen.noise4(x * scale * frequency, y * scale * frequency, z * scale * frequency, self.noise_4d_val)

            # noise value ranges from -1 to 1
            return value

        for tile in self.tiles:
            noise_factor = get_local_noise(tile, PLATE_INTERNAL_NOISE_SCALE, PLATE_INTERNAL_NOISE_NUM_OCTAVES, self.noise_gen) * PLATE_INTERNAL_NOISE_AMPLITUDE
            tile.height = self.avg_height + noise_factor

class PlateBoundary:
    """
    Represents a boundary between two tectonic plates.
    """
    def __init__(self, plate1, plate2, boundary_tiles):
        # Validate inputs
        if plate1 == plate2:
            raise ValueError("PlateBoundary cannot be created between the same plate.")
        if not boundary_tiles:
            raise ValueError("PlateBoundary must have at least one boundary tile.")

        self.plate1 = plate1  # TectonicPlate object
        self.plate2 = plate2  # TectonicPlate object
        self.direct_boundary_tiles = boundary_tiles  # List of WorldTile objects along the boundary
        self.boundary_line = self.get_boundary_line()  # List of vertices forming the boundary line
        self.boundary_centroid = self.get_boundary_centroid()  # Centroid of the boundary line
        self.boundary_type, self.activity_level = self.classify_boundary_type_and_calculate_activity_level()
        self.noise_4d_val = np.random.uniform(0, 1000)  # Random value for 4D noise slice. Used for noise along the boundary
        self.noise_gen = opensimplex.OpenSimplex(seed=random.randint(0, 10000))
        
    def __repr__(self):
        return f"PlateBoundary(plate1={self.plate1}, plate2={self.plate2}, num_boundary_tiles={len(self.direct_boundary_tiles)}, boundary_type={self.boundary_type}, activity_level={self.activity_level})"
    
    def __eq__(self, value):
        if not isinstance(value, PlateBoundary):
            return False
        return (self.plate1 == value.plate1 and self.plate2 == value.plate2) or (self.plate1 == value.plate2 and self.plate2 == value.plate1)

    def get_boundary_line(self):
        """
        Returns a list of vertices that together form the plate boundary.
        """
        return self.plate1.external_vertices.intersection(self.plate2.external_vertices)
    
    def calculate_distance_to_boundary_line(self, point):
        """
        Calculate the minimum distance from a point to the boundary line.
        """
        min_distance = float('inf')
        for vertex in self.boundary_line:
            dist = np.arccos(np.dot(point, vertex))
            if dist < min_distance:
                min_distance = dist
        return min_distance
    
    def get_boundary_centroid(self):
        """
        Returns the centroid of the boundary line vertices.
        """
        if not self.boundary_line:
            return None
        sum_coords = np.sum([np.array(vertex) for vertex in self.boundary_line], axis=0)
        centroid = sum_coords / len(self.boundary_line)
        centroid /= np.linalg.norm(centroid)  # Normalize to lie on sphere surface
        return centroid

    def classify_boundary_type_and_calculate_activity_level(self):
        """
        Classify the boundary type (convergent, divergent, transform) based on plate movement vectors.
        Also calculates an activity level based on the relative movement of the plates at the boundary.
        """
        # Determine boundary type and activity level based on plate movement vectors
        if self.plate1.total_movement_vector is not None and self.plate2.total_movement_vector is not None:
            plate1_movement_at_boundary = np.cross(self.plate1.total_movement_vector, self.boundary_centroid)
            plate2_movement_at_boundary = np.cross(self.plate2.total_movement_vector, self.boundary_centroid)

            relative_movement_vector = plate2_movement_at_boundary - plate1_movement_at_boundary
            relative_movement_vector /= np.linalg.norm(relative_movement_vector)

            vec_plate_1_to_plate_2 = ((self.plate2.centroid / np.linalg.norm(self.plate2.centroid)) - (self.plate1.centroid / np.linalg.norm(self.plate1.centroid)))
            vec_plate_1_to_plate_2 /= np.linalg.norm(vec_plate_1_to_plate_2)

            dot_product = np.dot(relative_movement_vector, -vec_plate_1_to_plate_2)

            if dot_product > 0.5: # plates are moving toward each other: convergent
                boundary_type = 'convergent'
                activity_level = np.linalg.norm(relative_movement_vector) * dot_product
            elif dot_product < -0.5: # plates are moving away from each other: divergent
                boundary_type = 'divergent'
                activity_level = np.linalg.norm(relative_movement_vector) * -dot_product
            else: # plates are sliding past each other: transform
                boundary_type = 'transform'
                activity_level = np.linalg.norm(relative_movement_vector) * (1 - abs(dot_product))
        else:
            print(f"One or both plates ({self.plate1.plate_id}, {self.plate2.plate_id}) have no movement vector defined!")
        
        return boundary_type, activity_level
    
    def simulate_tectonic_activity(self):
        """
        Simulate the tectonic activity at this boundary, modifying the heights of nearby tiles.
        """
        if self.boundary_type == 'convergent':
            if self.plate1.plate_type == 'continental' and self.plate2.plate_type == 'continental':
                # Continental-Continental Convergent Boundary: Mountain Building
                max_uplift_both_sides = CONTINENTAL_CONTINENTAL_CONVERGENT_UPLIFT_RATE_KM_PER_MILLION_YEAR * MILLION_YEARS_PER_SIMULATION_STEP
                actual_uplift_both_sides = convert_difference_in_km_to_normalized(max_uplift_both_sides) * self.activity_level
                self.simulate_deformation(
                    starting_tiles=self.direct_boundary_tiles,
                    deformation_magnitude=actual_uplift_both_sides,
                    deformation_decay_function=self.gaussian_decay_function,
                    plate_id=self.plate1.plate_id
                    )
                self.simulate_deformation(
                    starting_tiles=self.direct_boundary_tiles,
                    deformation_magnitude=actual_uplift_both_sides,
                    deformation_decay_function=self.gaussian_decay_function,
                    plate_id=self.plate2.plate_id
                    )
            elif self.plate1.plate_type == 'oceanic' and self.plate2.plate_type == 'oceanic':
                # Oceanic-Oceanic Convergent Boundary: Trench Formation
                max_uplift_non_subducting = OCEANIC_OCEANIC_CONVERGENT_UPLIFT_RATE_KM_PER_MILLION_YEAR * MILLION_YEARS_PER_SIMULATION_STEP
                actual_uplift_non_subducting = convert_difference_in_km_to_normalized(max_uplift_non_subducting) * self.activity_level

                max_subsidence_subducting = OCEANIC_OCEANIC_CONVERGENT_SUBDUCTION_RATE_KM_PER_MILLION_YEAR * MILLION_YEARS_PER_SIMULATION_STEP
                actual_subsidence_subducting = convert_difference_in_km_to_normalized(max_subsidence_subducting) * self.activity_level

                subducting_plate_id = self.plate1.plate_id if self.plate1.avg_height < self.plate2.avg_height else self.plate2.plate_id
                non_subducting_plate_id = self.plate2.plate_id if subducting_plate_id == self.plate1.plate_id else self.plate1.plate_id

                self.simulate_deformation(
                    starting_tiles=self.direct_boundary_tiles,
                    deformation_magnitude=actual_uplift_non_subducting,
                    deformation_decay_function=self.gaussian_decay_function,
                    plate_id=non_subducting_plate_id
                    )
                self.simulate_deformation(
                    starting_tiles=self.direct_boundary_tiles,
                    deformation_magnitude=actual_subsidence_subducting,
                    deformation_decay_function=self.gaussian_decay_function,
                    plate_id=subducting_plate_id
                    )
            elif (self.plate1.plate_type == 'oceanic' and self.plate2.plate_type == 'continental') or (self.plate1.plate_type == 'continental' and self.plate2.plate_type == 'oceanic'):
                # Oceanic-Continental Convergent Boundary: Subduction Zone
                max_uplift_non_subducting = OCEANIC_CONTINENTAL_CONVERGENT_UPLIFT_RATE_KM_PER_MILLION_YEAR * MILLION_YEARS_PER_SIMULATION_STEP
                actual_uplift_non_subducting = convert_difference_in_km_to_normalized(max_uplift_non_subducting) * self.activity_level

                max_subsidence_subducting = OCEANIC_CONTINENTAL_CONVERGENT_SUBDUCTION_RATE_KM_PER_MILLION_YEAR * MILLION_YEARS_PER_SIMULATION_STEP
                actual_subsidence_subducting = convert_difference_in_km_to_normalized(max_subsidence_subducting) * self.activity_level

                subducting_plate_id = self.plate1.plate_id if self.plate1.plate_type == 'oceanic' else self.plate2.plate_id
                non_subducting_plate_id = self.plate2.plate_id if subducting_plate_id == self.plate1.plate_id else self.plate1.plate_id

                self.simulate_deformation(
                    starting_tiles=self.direct_boundary_tiles,
                    deformation_magnitude=actual_uplift_non_subducting,
                    deformation_decay_function=self.gaussian_decay_function,
                    plate_id=non_subducting_plate_id
                    )
                self.simulate_deformation(
                    starting_tiles=self.direct_boundary_tiles,
                    deformation_magnitude=actual_subsidence_subducting,
                    deformation_decay_function=self.gaussian_decay_function,
                    plate_id=subducting_plate_id
                    )
            else:
                print(f"Unknown plate types for plates {self.plate1.plate_id} and {self.plate2.plate_id}")
        elif self.boundary_type == 'divergent':
            if self.plate1.plate_type == 'continental' and self.plate2.plate_type == 'continental':
                # Continental-Continental Divergent Boundary: Rift Valley Formation
                max_subsidence_both_sides = CONTINENTAL_CONTINENTAL_DIVERGENT_SUBDUCTION_RATE_KM_PER_MILLION_YEAR * MILLION_YEARS_PER_SIMULATION_STEP
                actual_subsidence_both_sides = convert_difference_in_km_to_normalized(max_subsidence_both_sides) * self.activity_level
                self.simulate_deformation(
                    starting_tiles=self.direct_boundary_tiles,
                    deformation_magnitude=-actual_subsidence_both_sides,
                    deformation_decay_function=self.gaussian_decay_function,
                    plate_id=self.plate1.plate_id
                    )
                self.simulate_deformation(
                    starting_tiles=self.direct_boundary_tiles,
                    deformation_magnitude=-actual_subsidence_both_sides,
                    deformation_decay_function=self.gaussian_decay_function,
                    plate_id=self.plate2.plate_id
                    )
            elif self.plate1.plate_type == 'oceanic' and self.plate2.plate_type == 'oceanic':
                # Oceanic-Oceanic Divergent Boundary: Mid-Ocean Ridge Formation
                max_uplift_both_sides = OCEANIC_OCEANIC_DIVERGENT_UPLIFT_RATE_KM_PER_MILLION_YEAR * MILLION_YEARS_PER_SIMULATION_STEP
                actual_uplift_both_sides = convert_difference_in_km_to_normalized(max_uplift_both_sides) * self.activity_level
                self.simulate_deformation(
                    starting_tiles=self.direct_boundary_tiles,
                    deformation_magnitude=actual_uplift_both_sides,
                    deformation_decay_function=self.gaussian_decay_function,
                    plate_id=self.plate1.plate_id
                    )
                self.simulate_deformation(
                    starting_tiles=self.direct_boundary_tiles,
                    deformation_magnitude=actual_uplift_both_sides,
                    deformation_decay_function=self.gaussian_decay_function,
                    plate_id=self.plate2.plate_id
                    )
            elif (self.plate1.plate_type == 'oceanic' and self.plate2.plate_type == 'continental') or (self.plate1.plate_type == 'continental' and self.plate2.plate_type == 'oceanic'):
                # Oceanic-Continental Divergent Boundary: Mixed Ridge and Rift Formation
                max_uplift_oceanic_side = OCEANIC_CONTINENTAL_DIVERGENT_UPLIFT_RATE_KM_PER_MILLION_YEAR * MILLION_YEARS_PER_SIMULATION_STEP
                actual_uplift_oceanic_side = convert_difference_in_km_to_normalized(max_uplift_oceanic_side) * self.activity_level

                max_subsidence_continental_side = OCEANIC_CONTINENTAL_DIVERGENT_SUBDUCTION_RATE_KM_PER_MILLION_YEAR * MILLION_YEARS_PER_SIMULATION_STEP
                actual_subsidence_continental_side = convert_difference_in_km_to_normalized(max_subsidence_continental_side) * self.activity_level

                oceanic_plate_id = self.plate1.plate_id if self.plate1.plate_type == 'oceanic' else self.plate2.plate_id
                continental_plate_id = self.plate2.plate_id if oceanic_plate_id == self.plate1.plate_id else self.plate1.plate_id

                self.simulate_deformation(
                    starting_tiles=self.direct_boundary_tiles,
                    deformation_magnitude=actual_uplift_oceanic_side,
                    deformation_decay_function=self.gaussian_decay_function,
                    plate_id=oceanic_plate_id
                    )
                self.simulate_deformation(
                    starting_tiles=self.direct_boundary_tiles,
                    deformation_magnitude=-actual_subsidence_continental_side,
                    deformation_decay_function=self.gaussian_decay_function,
                    plate_id=continental_plate_id
                    )
            else:
                print(f"Unknown plate types for plates {self.plate1.plate_id} and {self.plate2.plate_id}")
        elif self.boundary_type == 'transform':
            if self.plate1.plate_type == 'continental' and self.plate2.plate_type == 'continental':
                # Continental-Continental Transform Boundary: There is typically little vertical movement
                pass
            elif self.plate1.plate_type == 'oceanic' and self.plate2.plate_type == 'oceanic':
                # Oceanic-Oceanic Transform Boundary: There is typically little vertical movement
                pass
            elif (self.plate1.plate_type == 'oceanic' and self.plate2.plate_type == 'continental') or (self.plate1.plate_type == 'continental' and self.plate2.plate_type == 'oceanic'):
                # Oceanic-Continental Transform Boundary: There is typically little vertical movement
                pass
            else:
                print(f"Unknown plate types for plates {self.plate1.plate_id} and {self.plate2.plate_id}")
        else:
            print(f"Unknown boundary type: {self.boundary_type}")

    def gaussian_decay_function(self, distance):
        """
        Example decay function: Gaussian decay based on distance.
        """
        return np.exp(-GAUSSIAN_DECAY_RATE * (distance ** 2))

    def simulate_deformation(self, starting_tiles, deformation_magnitude, deformation_decay_function, plate_id):
        """
        Simulate deformation by using multi-source BFS from the starting tiles. Stop when deformation magnitude falls below a threshold.

        starting_tiles: list of WorldTile objects where deformation starts (i.e. the tiles that are directly on the boundary of a tectonic plate)
        deformation_magnitude: initial magnitude of deformation at the starting tiles
        deformation_decay_function: function that takes in distance from starting tile and returns decay factor (0-1)
        plate_id: the plate id to which the deformation is being applied (only tiles belonging to this plate will be deformed)
        """
        def get_local_noise(tile, scale, num_octaves, noise_gen):
            x, y, z = tile.center
            # Scale coordinates for noise
            # Note: using 4D noise with a fixed w value to get a 3D slice for randomness
            value = 0

            for octave in range(num_octaves):
                frequency = 2 ** octave
                amplitude = 0.5 ** octave
                value += amplitude * noise_gen.noise4(x * scale * frequency, y * scale * frequency, z * scale * frequency, self.noise_4d_val)

            # noise value ranges from -1 to 1
            return value

        visited_tiles = set()
        queue = deque()

        for tile in starting_tiles:
            queue.append((tile, 0))  # (tile, distance from starting tile)
            visited_tiles.add(tile)

        while queue:
            current_tile, distance = queue.popleft()

            decay_factor = deformation_decay_function(distance)
            current_deformation = deformation_magnitude * decay_factor
            
            noise_factor = get_local_noise(current_tile, NOISE_SCALE_FOR_TECTONIC_DEFORMATION, NOISE_OCTAVES_FOR_TECTONIC_DEFORMATION, self.noise_gen) * NOISE_AMPLITUDE_FOR_TECTONIC_DEFORMATION

            noise_scaled_deformation = current_deformation + noise_factor

            current_tile.height += noise_scaled_deformation

            if current_deformation < DEFORMATION_THRESHOLD:
                continue

            for neighbor in current_tile.neighbors:
                if neighbor not in visited_tiles and neighbor.plate_id == plate_id:
                    queue.append((neighbor, distance + 1)) # distance here does not have to be the true distance, we use the number of hops for simplicity
                    visited_tiles.add(neighbor)

class WorldTile:
    """
    Represents a tile on the sphere (hexagon or pentagon).
    """

    """
    Standard functions
    """
    def __init__(self, id, center, vertices):
        self.id = id  # Unique identifier
        self.center = center  # 3D coordinates on the sphere [x, y, z]
        self.vertices = vertices  # List of 3D vertex coordinates [[x, y, z], ...]
        self.height = None  # Value assigned later
        self.color = None  # RGB tuple with values 0-1, assigned later
        self.equirectangular_coords = None  # List of (lon, lat) tuples for 2D mapping, assigned later
        self.neighbors = []  # List of neighboring WorldTiles
        self.plate_id = None  # Assigned later for tectonic plates

        self.world_vertices = set()  # Set of WorldVertex objects corresponding to the vertices of this tile
        self.world_edges = set()  # Set of WorldEdge objects corresponding to the edges of this tile

    def __repr__(self):
        return f"WorldTile(center={self.center}, num_vertices={len(self.vertices)}, height={self.height}, plate_id={self.plate_id})"
    
    def __eq__(self, value):
        # for the purposes of our simulation, two WorldTiles are equal if they have the same id
        if not isinstance(value, WorldTile):
            return False
        return self.id == value.id
    
    def __lt__(self, value):
        return self.id < value.id
    
    def __hash__(self):
        return hash(self.id)

class WorldVertex:
    """
    Represents a vertex in 3D space.
    """
    def __init__(self, id, position):
        self.id = id # Unique identifier (to avoid issues with floating point precision)
        self.position = position  # 3D coordinates [x, y, z]
        self.adjacent_tiles = []  # List of WorldTiles that share this vertex (should only ever be 3)
        self.neighboring_vertices = []  # List of WorldVertex objects that are directly connected via edges

    def __repr__(self):
        return f"WorldVertex(id={self.id}, position={self.position})"
    
    def __eq__(self, value):
        if not isinstance(value, WorldVertex):
            return False
        return self.id == value.id
    
    def __hash__(self):
        return hash(self.id)
    
class WorldEdge:
    """
    Represents an edge between two vertices.
    """
    def __init__(self, vertex1, vertex2):
        self.vertex1 = vertex1  # WorldVertex object
        self.vertex2 = vertex2  # WorldVertex object
        self.adjacent_tiles = []  # List of WorldTiles that share this edge (should only ever be 2)

    def __repr__(self):
        return f"WorldEdge(vertex1={self.vertex1}, vertex2={self.vertex2})"
    
    def __eq__(self, value):
        if not isinstance(value, WorldEdge):
            return False
        return (self.vertex1 == value.vertex1 and self.vertex2 == value.vertex2) or (self.vertex1 == value.vertex2 and self.vertex2 == value.vertex1)
    
    def __hash__(self):
        return hash((min(hash(self.vertex1), hash(self.vertex2)), max(hash(self.vertex1), hash(self.vertex2))))

"""
Main execution
"""
def main():
    print("Creating simulated globe...")
    start_time = time.time()
    globe = SimulatedGlobe(recursion_level=GLOBE_RECURSION_LEVEL, use_cache=True)
    end_time = time.time()
    print(f"Globe creation took {end_time - start_time:.2f} seconds.")

    print("Printing geometry statistics...")
    start_time = time.time()
    globe.print_geometry_statistics()
    end_time = time.time()
    print(f"Geometry statistics printing took {end_time - start_time:.2f} seconds.")

    print("Printing world edge and vertex statistics...")
    start_time = time.time()
    globe.print_world_edge_and_vertex_statistics()
    end_time = time.time()
    print(f"World edge and vertex statistics printing took {end_time - start_time:.2f} seconds.")

    print("Populating neighbor lists...")
    start_time = time.time()
    globe.populate_neighbor_lists()
    end_time = time.time()
    print(f"Neighbor list population took {end_time - start_time:.2f} seconds.")

    print("Assigning tectonic plates...")
    start_time = time.time()
    globe.create_tectonic_plates_using_random_growth(num_plates=NUM_TECTONIC_PLATES)
    end_time = time.time()
    print(f"Tectonic plate assignment took {end_time - start_time:.2f} seconds.")

    print("Creating plate boundaries...")
    start_time = time.time()
    globe.create_plate_boundaries()
    end_time = time.time()
    print(f"Plate boundary creation took {end_time - start_time:.2f} seconds.")

    print("Assigning heights based on tectonic plates...")
    start_time = time.time()
    globe.assign_heights_using_plates()
    end_time = time.time()
    print(f"Height assignment took {end_time - start_time:.2f} seconds.")

    # print("Mapping noise...")
    # start_time = time.time()
    # globe.map_noise(scale=GLOBAL_NOISE_SCALE, num_octaves=GLOBAL_NOISE_NUM_OCTAVES, importance=GLOBAL_NOISE_IMPORTANCE)
    # end_time = time.time()
    # print(f"Noise mapping took {end_time - start_time:.2f} seconds.")

    print("Adding noise to terrain globally...")
    start_time = time.time()
    globe.add_noise(scale=GLOBAL_NOISE_SCALE, num_octaves=GLOBAL_NOISE_NUM_OCTAVES, amplitude=GLOBAL_NOISE_AMPLITUDE)
    end_time = time.time()
    print(f"Global noise addition took {end_time - start_time:.2f} seconds.")

    # print("Assigning tectonic plate colors...")
    # start_time = time.time()
    # globe.assign_tectonic_plate_colors()
    # end_time = time.time()
    # print(f"Tectonic plate color assignment took {end_time - start_time:.2f} seconds.")

    # print("Assigning plate boundary colors...")
    # start_time = time.time()
    # globe.assign_plate_boundary_colors()
    # end_time = time.time()
    # print(f"Plate boundary color assignment took {end_time - start_time:.2f} seconds.")

    # print("Assigning colors based on boundary proximity...")
    # start_time = time.time()
    # globe.assign_colors_based_on_boundary_proximity()
    # end_time = time.time()
    # print(f"Boundary proximity color assignment took {end_time - start_time:.2f} seconds.")

    print("Assigning terrain colors based on height...")
    start_time = time.time()
    globe.assign_terrain_colors_by_height()
    end_time = time.time()
    print(f"Height-based color assignment took {end_time - start_time:.2f} seconds.")

    # print("Filling in seas...")
    # start_time = time.time()
    # globe.fill_in_sea()
    # end_time = time.time()
    # print(f"Sea filling took {end_time - start_time:.2f} seconds.")

    print("Plotting with VisPy...")
    start_time = time.time()
    globe.plot_sphere_vispy(visualize_heights=True, height_scale=0.1, draw_north_pole=False, draw_plate_vectors=False, draw_boundary_component_vectors=False, draw_randomly_sampled_vectors=False)
    end_time = time.time()
    print(f"VisPy plotting took {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()