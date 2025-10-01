import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from math import sqrt
from collections import defaultdict

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

    for _ in range(recursion_level):
        new_faces = []
        for tri in faces:
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
      hex_centers: Centers of the hexagons/pentagons
      hex_faces: List of vertex indices forming each hex/pentagon
    """
    # Find which faces are adjacent to each vertex
    vertex_to_faces = defaultdict(list)
    for face_idx, face in enumerate(faces):
        for vertex in face:
            vertex_to_faces[vertex].append(face_idx)
    
    # Calculate face centroids (these become the vertices of the hex tiles)
    face_centroids = []
    for face in faces:
        centroid = np.mean(vertices[face], axis=0)
        centroid /= np.linalg.norm(centroid)  # Project back to sphere
        face_centroids.append(centroid)
    face_centroids = np.array(face_centroids)
    
    # For each vertex in the original mesh, create a hex/pentagon
    # by ordering the centroids of adjacent faces
    hex_centers = []
    hex_faces = []
    
    for vertex_idx in range(len(vertices)):
        adjacent_faces = vertex_to_faces[vertex_idx]
        
        # The center of this hex/pentagon is the original vertex
        hex_centers.append(vertices[vertex_idx])
        
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
    
    hex_centers = np.array(hex_centers)
    
    return hex_centers, hex_faces, face_centroids

if __name__ == '__main__':
    # Generate a geodesic sphere with a recursion level
    recursion = 3  # Try different values: 1, 2, 3, etc.
    vertices, faces = create_icosphere(recursion_level=recursion)
    
    print(f"Icosphere - Vertices: {vertices.shape[0]}, Faces: {faces.shape[0]}")
    
    # Convert to hexagon sphere
    hex_centers, hex_faces, face_centroids = icosphere_to_hexsphere(vertices, faces)
    
    print(f"Hex sphere - Tiles: {len(hex_faces)}")
    
    # Count pentagons and hexagons
    pentagon_count = sum(1 for hf in hex_faces if len(hf) == 5)
    hexagon_count = sum(1 for hf in hex_faces if len(hf) == 6)
    print(f"Pentagons: {pentagon_count}, Hexagons: {hexagon_count}")
    
    # Visualize the hexagon-tiled sphere
    fig = plt.figure(figsize=(5, 5))
    
    # Hex-tiled sphere
    ax = fig.add_subplot(projection='3d')
    hex_polys = []
    colors = []
    for hf in hex_faces:
        poly_verts = face_centroids[hf]
        hex_polys.append(poly_verts)
        # Color pentagons red, hexagons cyan
        colors.append('red' if len(hf) == 5 else 'cyan')
    
    poly2 = Poly3DCollection(hex_polys, facecolors=colors, 
                             linewidths=1, edgecolors='black', alpha=0.6)
    ax.add_collection3d(poly2)
    ax.set_title('Hex-Tiled Sphere (Dual)')
    
    max_range = 1.2
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.show()