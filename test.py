import numpy as np
from math import sqrt
from sympy import false
from vispy import app, scene, visuals
import torch 
import time
import os
import pickle
from collections import defaultdict
from tqdm import tqdm

def create_icosahedron():
    """
    Create an icosahedron centered at the origin with vertices on the unit sphere.
    Returns:
        vertices: np.ndarray of shape (12, 3)
        faces: np.ndarray of shape (20, 3)
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
    
    return vertices, faces

def subdivide_icosahedron(verts, faces, recursion_level):
    """
    Subdivide each triangular face of the icosahedron to create a finer mesh.
    Args:
        verts: np.ndarray of shape (N, 3) - vertices of the icosahedron
        faces: np.ndarray of shape (M, 3) - faces of the icosahedron
        recursion_level: int - number of times to subdivide each face
    Returns:
        new_verts: np.ndarray - vertices after subdivision
        new_faces: np.ndarray - faces after subdivision
    """
    def midpoint(v1, v2):
        mid = (v1 + v2) / 2.0
        return mid / np.linalg.norm(mid)

    for _ in range(recursion_level):
        vert_dict = {}
        new_faces = []
        for tri in faces:
            v1, v2, v3 = tri
            a = tuple(sorted((v1, v2)))
            b = tuple(sorted((v2, v3)))
            c = tuple(sorted((v3, v1)))

            if a not in vert_dict:
                vert_dict[a] = len(verts)
                verts = np.vstack([verts, midpoint(verts[v1], verts[v2])])
            if b not in vert_dict:
                vert_dict[b] = len(verts)
                verts = np.vstack([verts, midpoint(verts[v2], verts[v3])])
            if c not in vert_dict:
                vert_dict[c] = len(verts)
                verts = np.vstack([verts, midpoint(verts[v3], verts[v1])])

            a_idx = vert_dict[a]
            b_idx = vert_dict[b]
            c_idx = vert_dict[c]

            new_faces.append([v1, a_idx, c_idx])
            new_faces.append([v2, b_idx, a_idx])
            new_faces.append([v3, c_idx, b_idx])
            new_faces.append([a_idx, b_idx, c_idx])

        faces = np.array(new_faces)

    return verts, faces

def subdivide_icosahedron_torch(verts, faces, recursion_level):
    """
    Subdivide each triangular face of the icosahedron to create a finer mesh using PyTorch.
    Args:
        verts: torch.Tensor of shape (N, 3) - vertices of the icosahedron
        faces: torch.Tensor of shape (M, 3) - faces of the icosahedron
        recursion_level: int - number of times to subdivide each face
    Returns:
        new_verts: torch.Tensor - vertices after subdivision
        new_faces: torch.Tensor - faces after subdivision
    """
    def deduplicate_vertices(verts, faces):
        mps = verts.device.type == 'mps'

        if mps:
            print("Warning: deduplication must be done on CPU on MPS device due to unimplemented operations.")
            original_device = verts.device
            verts = verts.cpu()
            faces = faces.cpu()

        # Get unique vertices and inverse indices
        unique_verts, inverse_indices = torch.unique(verts, dim=0, sorted=False, return_inverse=True)

        # Update faces directly using inverse mapping
        faces = inverse_indices[faces]

        if mps:
            unique_verts = unique_verts.to(original_device)
            faces = faces.to(original_device)

        return unique_verts, faces

    # use mps
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device: ", device)
    verts = torch.tensor(verts, dtype=torch.float32, device=device)
    faces = torch.tensor(faces, dtype=torch.long, device=device)

    for _ in range(recursion_level):
        all_triangles = verts[faces]

        v1 = all_triangles[:, 0, :]
        v2 = all_triangles[:, 1, :]
        v3 = all_triangles[:, 2, :]

        a = (v1 + v2) / 2.0
        b = (v2 + v3) / 2.0
        c = (v3 + v1) / 2.0
        a = a / a.norm(dim=1, keepdim=True)
        b = b / b.norm(dim=1, keepdim=True)
        c = c / c.norm(dim=1, keepdim=True)

        new_verts = torch.cat([verts, a, b, c], dim=0) # same as np.vstack
        a_idx = torch.arange(verts.shape[0], verts.shape[0] + a.shape[0], device=device)
        b_idx = torch.arange(verts.shape[0] + a.shape[0], verts.shape[0] + a.shape[0] + b.shape[0], device=device)
        c_idx = torch.arange(verts.shape[0] + a.shape[0] + b.shape[0], new_verts.shape[0], device=device)
        new_faces = torch.cat([
            torch.stack([faces[:, 0], a_idx, c_idx], dim=1),
            torch.stack([faces[:, 1], b_idx, a_idx], dim=1),
            torch.stack([faces[:, 2], c_idx, b_idx], dim=1),
            torch.stack([a_idx, b_idx, c_idx], dim=1)
        ], dim=0)

        verts = new_verts
        faces = new_faces

    verts, faces = deduplicate_vertices(verts, faces)

    return verts, faces

def geodesic_to_hexsphere(vertices, faces):
    """
    Convert an icosphere into a hexagon-tiled sphere (with 12 pentagons).
    
    The dual of the icosphere subdivision creates hexagons and pentagons.
    Each vertex in the original mesh becomes the center of a hex/pentagon,
    and we create faces by connecting the centroids of adjacent triangles.
    
    Args:
    vertices: Vertex array from icosphere
    faces: Face array from icosphere
    
    Returns:
    all_vertices: List of all vertices (original + new centroids)
        eg. [ [x, y, z], ... ]
    hex_faces: List of vertex indices forming each hex/pentagon
        eg. [ [v1, v2, v3, v4, v5, v6], ... ]
    hex_centers: List of centers of each hex/pentagon
        eg. [ [x, y, z], ... ]

    Note that avg(hex_faces[0]) == hex_centers[0]

    Procedure:
    1. Each vertex in the original mesh becomes the center of a hex/pentagon.
    2. For each vertex, find all adjacent faces (triangles).
    3. The centroids of these faces form the vertices of the hex/pentagon.
    """
    # get the neighbor faces for each vertex
    neighbor_faces = defaultdict(list)
    for face_idx, face in enumerate(faces):
        for vertex in face:
            neighbor_faces[vertex].append(face_idx)

    hex_faces = []
    hex_centers = []
    all_vertices = vertices.tolist()  # start with original vertices
    vertex_map = {}  # map from vertex index to hex_faces index
    for vertex_idx, face_indices in neighbor_faces.items():
        # Calculate centroid of each adjacent face
        centroids = []
        for face_idx in face_indices:
            face_verts = vertices[faces[face_idx]]
            centroid = np.mean(face_verts, axis=0)
            centroid /= np.linalg.norm(centroid)  # project onto unit sphere
            centroids.append(centroid)
        
        # Add centroids to all_vertices and get their indices
        centroid_indices = []
        for centroid in centroids:
            all_vertices.append(centroid.tolist())
            centroid_indices.append(len(all_vertices) - 1)
        if len(centroid_indices) == 5:
            all_vertices.append(centroid.tolist())
            centroid_indices.append(len(all_vertices) - 1)  # duplicate last centroid to make pentagon into hexagon
        
        hex_faces.append(centroid_indices)
        hex_centers.append(vertices[vertex_idx].tolist())
        vertex_map[vertex_idx] = len(hex_faces) - 1  # map original vertex to hex_faces index

    all_vertices = np.array(all_vertices)
    hex_centers = np.array(hex_centers)
    hex_faces = np.array(hex_faces)

    return hex_faces, hex_centers, all_vertices

def geodesic_to_hexsphere_torch(vertices, faces):
    """
    Convert an icosphere into a hexagon-tiled sphere (with 12 pentagons).
    
    The dual of the icosphere subdivision creates hexagons and pentagons.
    Each vertex in the original mesh becomes the center of a hex/pentagon,
    and we create faces by connecting the centroids of adjacent triangles.
    
    Args:
    vertices: Vertex array from icosphere
    faces: Face array from icosphere
    
    Returns:
    all_vertices: List of all vertices (true vertices + centroids at the end)
        eg. [ [x, y, z], ... ]
    hex_faces: List of vertex indices forming each hex/pentagon
        eg. [ [v1, v2, v3, v4, v5, v6], ... ]
    hex_centers: List of vertex indices that are the centers of each hex/pentagon
        eg. [ c1, c2, c3, ... ]

    Note that avg(hex_faces[0]) == hex_centers[0]

    Procedure:
    1. Each vertex in the original mesh becomes the center of a hex/pentagon.
    2. For each vertex, find all adjacent faces (triangles).
    3. The centroids of these faces form the vertices of the hex/pentagon.
    """
    # use mps
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device: ", device)

    vertices = torch.tensor(vertices, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.long)

    expanded_faces = vertices[faces]

    # compute centroids of each face
    centroids = torch.mean(expanded_faces, dim=1)
    centroids = centroids / centroids.norm(dim=1, keepdim=True)
    # now, centroids will be the vertices of the hex/pentagons!

    # each vertex becomes a face center
    hex_centers = vertices * vertices.norm(dim=1, keepdim=True) # moves the point inward to be flat agains the center of the hex face
    # now we have the center points of each hex/pentagon!

    #TODO

def subdivide_icosahedron_with_neighbors_torch( recursion_level ):
    """
    Subdivide each triangular face of a icosahedron to create a finer mesh using PyTorch.
    Also returns the neighboring face indices for each vertex.
    Args:
        recursion_level: int - number of times to subdivide each face
    Returns:
        new_verts: torch.Tensor - vertices after subdivision
        new_faces: torch.Tensor - faces after subdivision
        vertex_neighbors: List of sets - each set contains indices of neighboring vertices
    """
    def deduplicate_vertices(verts, faces):
        mps = verts.device.type == 'mps'

        if mps:
            print("Warning: deduplication must be done on CPU on MPS device due to unimplemented operations.")
            original_device = verts.device
            verts = verts.cpu()
            faces = faces.cpu()

        # Get unique vertices and inverse indices
        unique_verts, inverse_indices = torch.unique(verts, dim=0, sorted=False, return_inverse=True)

        # Update faces directly using inverse mapping
        faces = inverse_indices[faces]

        if mps:
            unique_verts = unique_verts.to(original_device)
            faces = faces.to(original_device)

        return unique_verts, faces
    
    # use mps
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device: ", device)

    phi = torch.tensor((1.0 + sqrt(5.0)) / 2.0, dtype=torch.float32, device=device)

    icosahedron_verts = torch.tensor([
        [-1, phi, 0], # vertex 0
        [1, phi, 0], # vertex 1
        [-1, -phi, 0], # vertex 2
        [1, -phi, 0], # vertex 3
        [0, -1, phi], # vertex 4
        [0, 1, phi], # vertex 5
        [0, -1, -phi], # vertex 6
        [0, 1, -phi], # vertex 7
        [phi, 0, -1], # vertex 8
        [phi, 0, 1], # vertex 9
        [-phi, 0, -1], # vertex 10
        [-phi, 0, 1] # vertex 11
    ], dtype=torch.float32, device=device)

    icosahedron_faces = torch.tensor([
        [0, 11, 5], # face 0
        [0, 5, 1], # face 1
        [0, 1, 7], # face 2
        [0, 7, 10], # face 3
        [0, 10, 11], # face 4
        [1, 5, 9], # face 5
        [5, 11, 4], # face 6
        [11, 10, 2], # face 7
        [10, 7, 6], # face 8
        [7, 1, 8], # face 9
        [3, 9, 4], # face 10
        [3, 4, 2], # face 11
        [3, 2, 6], # face 12
        [3, 6, 8], # face 13
        [3, 8, 9], # face 14
        [4, 9, 5], # face 15
        [2, 4, 11], # face 16
        [6, 2, 10], # face 17
        [8, 6, 7], # face 18
        [9, 8, 1]  # face 19
    ], dtype=torch.long, device=device)

    vert_neighboring_faces = torch.tensor([
        [0, 1, 2, 3, 4], # vertex 0
        [1, 2, 9, 19, 5], # vertex 1
        [7, 16, 11, 12, 17], # vertex 2
        [10, 11, 12, 13, 14], # vertex 3
        [6, 15, 10, 11, 16], # vertex 4
        [0, 1, 5, 15, 6], # vertex 5
        [8, 17, 12, 13, 18], # vertex 6
        [2, 3, 8, 18, 9], # vertex 7
        [9, 18, 13, 14, 19], # vertex 8
        [5, 15, 10, 14, 19], # vertex 9
        [3, 4, 7, 17, 8], # vertex 10
        [0, 4, 7, 16, 6], # vertex 11
        ], dtype=torch.long, device=device
    )
    # why the funky order? Because the neighboring faces are given in a clockwise order around each vertex (using the tangent plane to the vertex)
    # this bookeeping will help later when constructing the hex faces

    # TODO: track neighbors during subdivision
    verts = icosahedron_verts
    faces = icosahedron_faces
    for _ in range(recursion_level):
        all_triangles = verts[faces]

        v1 = all_triangles[:, 0, :]
        v2 = all_triangles[:, 1, :]
        v3 = all_triangles[:, 2, :]

        a = (v1 + v2) / 2.0
        b = (v2 + v3) / 2.0
        c = (v3 + v1) / 2.0
        a = a / a.norm(dim=1, keepdim=True)
        b = b / b.norm(dim=1, keepdim=True)
        c = c / c.norm(dim=1, keepdim=True)

        new_verts = torch.cat([verts, a, b, c], dim=0) # same as np.vstack
        a_idx = torch.arange(verts.shape[0], verts.shape[0] + a.shape[0], device=device)
        b_idx = torch.arange(verts.shape[0] + a.shape[0], verts.shape[0] + a.shape[0] + b.shape[0], device=device)
        c_idx = torch.arange(verts.shape[0] + a.shape[0] + b.shape[0], new_verts.shape[0], device=device)
        new_faces = torch.cat([
            torch.stack([faces[:, 0], a_idx, c_idx], dim=1),
            torch.stack([faces[:, 1], b_idx, a_idx], dim=1),
            torch.stack([faces[:, 2], c_idx, b_idx], dim=1),
            torch.stack([a_idx, b_idx, c_idx], dim=1)
        ], dim=0)

        verts = new_verts
        faces = new_faces

    verts, faces = deduplicate_vertices(verts, faces)

    return icosahedron_verts, icosahedron_faces, vert_neighboring_faces

def geodesic_to_hexsphere_with_neighbors_torch( vertices, faces, vertex_neighbors ):
    """
    Convert an icosphere into a hexagon-tiled sphere (with 12 pentagons) using PyTorch.
    Uses neighbor information for each vertex to speed up processing.
    
    The dual of the icosphere creates hexagons and pentagons.
    Each vertex in the original mesh becomes the center of a hex/pentagon,
    and we create faces by connecting the centroids of adjacent triangles.
    
    Args:
    vertices: Vertex array from icosphere
    faces: Face array from icosphere
    vertex_neighbors: list of sets - each set contains the indices of neighboring faces to that vertex
    
    Returns:
    all_vertices: List of all vertices (true vertices + centroids at the end)
        eg. [ [x, y, z], ... ]
    hex_faces: List of vertex indices forming each hex/pentagon
        eg. [ [v1, v2, v3, v4, v5, v6], ... ]
    hex_centers: List of vertex indices that are the centers of each hex/pentagon
        eg. [ c1, c2, c3, ... ]
    vertex_neighbors_hex: List of sets - each set contains indices of neighboring vertices in hexsphere
    """

    # use mps
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device: ", device)

    vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(faces, dtype=torch.long, device=device)

    expanded_faces = vertices[faces]

    # compute centroids of each face
    centroids = torch.mean(expanded_faces, dim=1) # dim 0 is face index, dim 1 is vertex coordinates (x,y,z) [we want to average over the 3 vertices]
    centroids = centroids / centroids.norm(dim=1, keepdim=True) # project onto unit sphere
    # now, centroids will be the vertices of the hex/pentagons!

    # each vertex becomes a face center
    hex_centers = vertices * vertices.norm(dim=1, keepdim=True) # moves the point inward to be flat agains the center of the hex face
    # now we have the center points of each hex/pentagon!

    hexagon_faces = vertex_neighbors

    return hexagon_faces, hex_centers, centroids

def plot_points_and_faces_as_mesh(verts, faces):
    # Create VisPy canvas with 3D camera
    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(fov=45, distance=10)

    mesh = scene.visuals.Mesh(vertices=verts, faces=faces, color='blue', shading=None)

    filter = visuals.filters.mesh.WireframeFilter(color='black', width=1)
    mesh.attach(filter)

    view.add(mesh)

    app.run()

def plot_sphere_vispy(hexagon_faces, hexagon_colors):
    from vispy import app, scene
    print(app.use_app())

    # calculate the hexagon centers from the faces
    hexagon_centers = np.mean(hexagon_faces, axis=1)

    # Create VisPy canvas with 3D camera
    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(fov=45, distance=10)

    #colorset = set(hexagon_colors)
    #print(f"Unique colors used: {len(colorset)}")

    all_faces = []
    all_verts = []
    all_face_colors = []
    vert_offset = 0

    for hexagon_face, hexagon_center, hexagon_color in zip(hexagon_faces, hexagon_centers, hexagon_colors):
        these_verts = np.vstack([hexagon_center, hexagon_face])
        if len(hexagon_face) == 5:
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
        all_face_colors.append([hexagon_color] * faces.shape[0])
        vert_offset += these_verts.shape[0]

    mesh = scene.visuals.Mesh(vertices=np.concatenate(all_verts), faces=np.concatenate(all_faces), face_colors=np.concatenate(all_face_colors), shading='smooth')
    view.add(mesh)

    app.run()

def order_hex_face_vertices(hex_face):
    centroid = np.mean(hex_face, axis=0)

    # Compute the local face normal
    v1, v2 = hex_face[1] - hex_face[0], hex_face[2] - hex_face[0]
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)

    # Create a local coordinate system (u, v) on the face
    u = v1 / np.linalg.norm(v1)
    v = np.cross(normal, u)

    # Project vertices into this local 2D basis
    rel = hex_face - centroid
    x_local = np.dot(rel, u)
    y_local = np.dot(rel, v)

    # Compute angles in this local plane
    angles = np.arctan2(y_local, x_local)
    ordered_face = hex_face[np.argsort(angles)]

    return ordered_face

def sort_hex_faces(hex_faces):
    new_hex_faces = []
    for hex_face in hex_faces:
        # order each hex face
        ordered_face = order_hex_face_vertices(hex_face)
        new_hex_faces.append(ordered_face)
    return new_hex_faces

def order_hex_face_vertices_given_verts_and_indices(hex_face, all_verts):
    centroid = np.mean(all_verts[hex_face], axis=0)

    # Compute the local face normal
    v1, v2 = all_verts[hex_face][1] - all_verts[hex_face][0], all_verts[hex_face][2] - all_verts[hex_face][0]
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)

    # Create a local coordinate system (u, v) on the face
    u = v1 / np.linalg.norm(v1)
    v = np.cross(normal, u)

    # Project vertices into this local 2D basis
    rel = all_verts[hex_face] - centroid
    x_local = np.dot(rel, u)
    y_local = np.dot(rel, v)

    # Compute angles in this local plane
    angles = np.arctan2(y_local, x_local)
    ordered_face = hex_face[np.argsort(angles)]

    return ordered_face

def sort_hex_faces_given_verts_and_indices(hex_faces, all_verts):
    new_hex_faces = []
    for hex_face in hex_faces:
        # order each hex face
        ordered_face = order_hex_face_vertices_given_verts_and_indices(hex_face, all_verts)
        new_hex_faces.append(ordered_face)
    return new_hex_faces

if __name__ == "__main__":
    recursion_level = 1

    # verts, faces = create_icosahedron()

    # start_time = time.time()
    # verts1, faces1 = subdivide_icosahedron(verts, faces, recursion_level=recursion_level)
    # end_time = time.time()
    # print(f"Subdivision with numpy took {end_time - start_time:.2f} seconds")

    # start_time = time.time()
    # verts2, faces2 = subdivide_icosahedron_torch(verts, faces, recursion_level=recursion_level)
    # verts2 = verts2.cpu().numpy()
    # faces2 = faces2.cpu().numpy()
    # end_time = time.time()
    # print(f"Subdivision with torch took {end_time - start_time:.2f} seconds")

    # print(f"Vertex count post-subdivision: {len(verts2)}")

    # start_time = time.time()
    # hex_faces, hex_centers, all_vertices = geodesic_to_hexsphere(verts2, faces2)
    # end_time = time.time()
    # print(f"Geodesic to hexsphere conversion took {end_time - start_time:.2f} seconds")

    start_time = time.time()
    verts, faces, vertex_neighbors = subdivide_icosahedron_with_neighbors_torch( recursion_level=recursion_level )
    verts = verts.cpu().numpy()
    faces = faces.cpu().numpy()
    vertex_neighbors = vertex_neighbors.cpu().numpy()
    end_time = time.time()
    print(f"Subdivision with torch + neighbors took {end_time - start_time:.2f} seconds")

    start_time = time.time()
    sorted_hex_faces, hex_centers, all_vertices = geodesic_to_hexsphere_with_neighbors_torch( verts, faces, vertex_neighbors )
    hex_centers = hex_centers.cpu().numpy()
    all_vertices = all_vertices.cpu().numpy()
    end_time = time.time()
    print(f"Geodesic to hexsphere with neighbors took {end_time - start_time:.2f} seconds")

    # start_time = time.time()
    # sorted_hex_faces = sort_hex_faces_given_verts_and_indices(sorted_hex_faces, all_vertices)
    # end_time = time.time()
    # print(f"Sorting hex faces took {end_time - start_time:.2f} seconds")

    for i, face in enumerate(sorted_hex_faces):
        print(f"Hex face {i}: {face}")

    print(f"hex center example: {hex_centers[0]}")
    hexagon_colors = np.array(
        [
            [1, 0, 0],  # Red for face 0
            [0, 1, 0],  # Green for face 1
            [0, 0, 1],  # Blue for face 2
            [1, 1, 0],  # Yellow for face 3
            [1, 0, 1],  # Magenta for face 4
            [0, 1, 1],  # Cyan for face 5
            [0.5, 0.5, 0],  # Olive for face
            [0.5, 0, 0.5],  # Purple for face 6
            [0, 0.5, 0.5],  # Teal for face 7
            [0.25, 0.75, 0.25],  # Light Green for face 8
            [0.75, 0.25, 0.75],  # Light Purple for face 9
            [0.75, 0.75, 0.25],  # Light Yellow for face 10
            [0.1, 0.6, 0.8],  # Sky Blue for face 11
            [0.9, 0.4, 0.2],  # Orange for face 12
            [0.3, 0.3, 0.3],  # Gray for face 13
            [0.6, 0.2, 0.7],  # Violet for face 14
            [0.2, 0.7, 0.3],  # Lime Green for face 15
            [0.4, 0.4, 0.1],  # Brown for face 16
            [0.8, 0.1, 0.5],  # Pink for face 17
            [0.1, 0.8, 0.6],  # Aquamarine for face 18
            [0.5, 0.5, 0.5],  # Silver for face 19
            
        ]
    )
    plot_sphere_vispy(all_vertices[sorted_hex_faces], hexagon_colors=hexagon_colors)
