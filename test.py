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
        original_device = verts.device
        verts = verts.cpu()
        faces = faces.cpu()

        # Get unique vertices and inverse indices TODO: this isn't working on MPS due to the OP not being implemented!
        unique_verts, inverse_indices = torch.unique(verts, dim=0, sorted=False, return_inverse=True)

        # Update faces directly using inverse mapping
        faces = inverse_indices[faces]

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

    # vert_dict = {}
    # unique_verts = []
    # unique_vert_idx = 0
    # for i in tqdm(range(verts.shape[0]), desc="Removing duplicate vertices"):
    #     if tuple(verts[i].cpu().numpy()) not in vert_dict:
    #         vert_dict[tuple(verts[i].cpu().numpy())] = unique_vert_idx
    #         unique_verts.append(verts[i].cpu().numpy())
    #         unique_vert_idx += 1

    # for i in tqdm(range(faces.shape[0]), desc="Updating face indices"):
    #     for j in range(3):
    #         faces[i, j] = vert_dict[tuple(verts[faces[i, j]].cpu().numpy())]

    # verts = torch.tensor(unique_verts, dtype=torch.float32)

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

    for k, v in neighbor_faces.items():
        # key is vertex index, value is list of face indices
        assert len(v) in [5, 6], f"Vertex {k} has {len(v)} adjacent faces, expected 5 or 6."
        v.sort()

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

def plot_points_and_faces_as_mesh(verts, faces):
    # Create VisPy canvas with 3D camera
    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(fov=45, distance=10)

    mesh = scene.visuals.Mesh(vertices=verts, faces=faces, color='blue', shading=None)

    filter = visuals.filters.mesh.WireframeFilter(color='black', width=5)
    mesh.attach(filter)

    view.add(mesh)

    app.run()

if __name__ == "__main__":
    recursion_level = 4

    verts, faces = create_icosahedron()

    start_time = time.time()
    verts1, faces1 = subdivide_icosahedron(verts, faces, recursion_level=recursion_level)
    end_time = time.time()
    print(f"Subdivision with numpy took {end_time - start_time:.2f} seconds")

    start_time = time.time()
    verts2, faces2 = subdivide_icosahedron_torch(verts, faces, recursion_level=recursion_level)
    verts2 = verts2.cpu().numpy()
    faces2 = faces2.cpu().numpy()
    end_time = time.time()
    print(f"Subdivision with torch took {end_time - start_time:.2f} seconds")

    start_time = time.time()
    hex_faces, hex_centers, all_vertices = geodesic_to_hexsphere(verts2, faces2)
    end_time = time.time()
    print(f"Geodesic to hexsphere conversion took {end_time - start_time:.2f} seconds")

    plot_points_and_faces_as_mesh(all_vertices, hex_faces)
