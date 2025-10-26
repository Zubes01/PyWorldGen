import torch as th
import numpy as np
from vispy import app, scene, visuals
import time
from collections import defaultdict
    
def create_icosahedron_torch(device='cpu'):
    phi = (th.tensor([1.0], device=device) + th.sqrt(th.tensor(5.0, device=device))) / 2.0

    vertices = th.tensor([
        [-1,  phi, 0],
        [ 1,  phi, 0],
        [-1, -phi, 0],
        [ 1, -phi, 0],
        [0, -1,  phi],
        [0,  1,  phi],
        [0, -1, -phi],
        [0,  1, -phi],
        [ phi, 0, -1],
        [ phi, 0,  1],
        [-phi, 0, -1],
        [-phi, 0,  1],
    ], dtype=th.float32, device=device)

    # Normalize vertices to lie on the unit sphere
    vertices = vertices / vertices.norm(dim=1, keepdim=True)

    faces = th.tensor([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ], dtype=th.long, device=device)

    edges = th.tensor([
        [0, 11], [11, 5], [5, 0],
        [0, 5], [5, 1], [1, 0],
        [0, 1], [1, 7], [7, 0],
        [0, 7], [7, 10], [10, 0],
        [0, 10], [10, 11], [11, 0],
        [1, 5], [5, 9], [9, 1],
        [5, 11], [11, 4], [4, 5],
        [11, 10], [10, 2], [2, 11],
        [10, 7], [7, 6], [6, 10],
        [7, 1], [1, 8], [8, 7],
        [3, 9], [9, 4], [4, 3],
        [3, 4], [4, 2], [2, 3],
        [3, 2], [2, 6], [6, 3],
        [3, 6], [6, 8], [8, 3],
        [3, 8], [8, 9], [9, 3],
        [4, 9], [9, 5], [5, 4],
        [2, 4], [4, 11], [11, 2],
        [6, 2], [2, 10], [10, 6],
        [8, 6], [6, 7], [7, 8],
        [9, 8], [8, 1], [1, 9],
    ], dtype=th.long, device=device)

    return vertices, faces, edges

def subdivide_icosahedron_torch(verts, faces, recursion_level, device='cpu'):
    def deduplicate_vertices(verts, faces):
        mps = verts.device.type == 'mps'

        if mps:
            print("Warning: deduplication must be done on CPU on MPS device due to unimplemented operations.")
            original_device = verts.device
            verts = verts.cpu()
            faces = faces.cpu()

        # Get unique vertices and inverse indices
        unique_verts, inverse_indices = th.unique(verts, dim=0, sorted=False, return_inverse=True)

        # Update faces directly using inverse mapping
        faces = inverse_indices[faces]

        if mps:
            unique_verts = unique_verts.to(original_device)
            faces = faces.to(original_device)

        return unique_verts, faces

    verts = verts.to(device)
    faces = faces.to(device)

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

        new_verts = th.cat([verts, a, b, c], dim=0) # same as np.vstack
        a_idx = th.arange(verts.shape[0], verts.shape[0] + a.shape[0], device=device)
        b_idx = th.arange(verts.shape[0] + a.shape[0], verts.shape[0] + a.shape[0] + b.shape[0], device=device)
        c_idx = th.arange(verts.shape[0] + a.shape[0] + b.shape[0], new_verts.shape[0], device=device)
        new_faces = th.cat([
            th.stack([faces[:, 0], a_idx, c_idx], dim=1),
            th.stack([faces[:, 1], b_idx, a_idx], dim=1),
            th.stack([faces[:, 2], c_idx, b_idx], dim=1),
            th.stack([a_idx, b_idx, c_idx], dim=1)
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

def dual_icosahedron_torch(verts, faces, device='cpu'):
    verts = verts.to(device)
    faces = faces.to(device)

    triangle_face_centers = verts[faces].mean(dim=1)
    triangle_face_centers = triangle_face_centers / triangle_face_centers.norm(dim=1, keepdim=True)

    values = th.arange(verts.shape[0], device=device).unsqueeze(1).unsqueeze(2)  # (max_val+1, 1, 1)
    mask = (faces.unsqueeze(0) == values)  # (max_val+1, num_rows, num_cols)

    row_contains_value = mask.any(dim=2)  # (max_val+1, num_rows)

    k = 5  # we know that we will have hexagons/pentagons

    # Use topk or argsort to get indices
    row_indices = th.arange(faces.shape[0], device=device).unsqueeze(0).expand(verts.shape[0], -1)
    # Mask out non-matching rows by setting them to -1
    row_indices_masked = th.where(row_contains_value, row_indices, th.tensor(-1, device=device))

    result = th.topk(row_indices_masked, k=k, dim=1).values  # (max_val+1, k)

    return triangle_face_centers, result

def separated_dual_icosahedron_torch(verts, faces, device='cpu'):
    verts = verts.to(device)
    faces = faces.to(device)

    triangle_face_centers = verts[faces].mean(dim=1)
    triangle_face_centers = triangle_face_centers / triangle_face_centers.norm(dim=1, keepdim=True)

    indices = th.arange(0, verts.shape[0]-1, device=device).unsqueeze(1).unsqueeze(2)  # (max_val+1, 1, 1)
    Z = (faces.unsqueeze(0) == indices).sum(dim=(1, 2))  # Z[i] = number of faces adjacent to vertex i
    # print(Z)
    pentagonal_verts = th.nonzero(Z == 5).squeeze(1) # indices of vertices that will become pentagons
    hexagonal_verts = th.nonzero(Z == 6).squeeze(1) # indices of vertices that will become hexagons

    # print(f"Pentagonal verts: {pentagonal_verts.shape[0]}, Hexagonal verts: {hexagonal_verts.shape[0]}")
    # print(f"example: pentagonal verts: {pentagonal_verts[:5]}")
    # print(f"example: hexagonal verts: {hexagonal_verts[:5]}")

    # first we deal with the pentagonal verts
    flattened_faces = faces.flatten()
    pent_mask = pentagonal_verts.unsqueeze(1) == flattened_faces.unsqueeze(0)
    pent_indices = th.arange(flattened_faces.size(0), device=device).unsqueeze(0) + 1
    pent_matches = th.where(pent_mask, pent_indices, th.tensor(0, device=device))
    pent_mask2 = pent_matches != 0
    pentagonal_faces = pent_matches[pent_mask2].reshape(pent_matches.size(0), -1) - 1  # convert back to original indices
    pentagonal_faces = pentagonal_faces // 3 # because we flattened the faces

    # print(f"Example pentagonal face indices: {pentagonal_faces[0]}")
    # print(f"Max pentagonal face index: {pentagonal_faces.max()}")
    # print(f"Min pentagonal face index: {pentagonal_faces.min()}")

    # now we deal with the hexagonal verts
    if hexagonal_verts.numel() != 0:
        hex_mask = hexagonal_verts.unsqueeze(1) == flattened_faces.unsqueeze(0)
        hex_indices = th.arange(flattened_faces.size(0), device=device).unsqueeze(0) + 1
        hex_matches = th.where(hex_mask, hex_indices, th.tensor(0, device=device))
        hex_mask2 = hex_matches != 0
        hexagonal_faces = hex_matches[hex_mask2].reshape(hex_matches.size(0), -1) - 1  # convert back to original indices
        hexagonal_faces = hexagonal_faces // 3 # because we flattened the faces
    else:
        hexagonal_faces = th.empty((0, 6), dtype=th.long, device=device)

    # print(f"Example hexagonal face indices: {hexagonal_faces[0]}")
    # print(f"Max hexagonal face index: {hexagonal_faces.max()}")
    # print(f"Min hexagonal face index: {hexagonal_faces.min()}")

    return triangle_face_centers, pentagonal_faces, hexagonal_faces

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

def plot_sphere_vispy(hexagon_faces, hexagon_colors):
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

if __name__ == "__main__":
    recursion_level = 6
    device = 'mps' if th.backends.mps.is_available() else 'cpu'
    print("Using device: ", device)

    start_time = time.time()
    ico_verts, ico_faces, ico_edges = create_icosahedron_torch(device=device)
    end_time = time.time()
    print(f"Icosahedron creation took {end_time - start_time:.2f} seconds")

    start_time = time.time()
    triangular_vertices, triangular_faces = subdivide_icosahedron_torch(
        ico_verts, ico_faces, recursion_level, device=device
    )
    end_time = time.time()
    print(f"Subdivision to level {recursion_level} took {end_time - start_time:.2f} seconds")

    print(f"Total triangular vertices: {triangular_vertices.shape[0]}")
    print(f"Total triangular faces: {triangular_faces.shape[0]}")

    start_time = time.time()
    test_faces, test_centers, test_verts = geodesic_to_hexsphere(
        triangular_vertices.cpu().numpy(), triangular_faces.cpu().numpy()
    )
    end_time = time.time()
    print(f"Dual conversion (numpy) took {end_time - start_time:.2f} seconds")

    start_time = time.time()
    final_vertices, pentagonal_faces, hexagonal_faces = separated_dual_icosahedron_torch(
        triangular_vertices, triangular_faces, device=device
    )
    end_time = time.time()
    print(f"Dual conversion took {end_time - start_time:.2f} seconds")
    final_vertices = final_vertices.cpu().numpy()
    pentagonal_faces_with_duped_final_vertex = th.cat([pentagonal_faces, pentagonal_faces[:, 4:5]], dim=1)
    all_faces = np.vstack([pentagonal_faces_with_duped_final_vertex.cpu().numpy(), hexagonal_faces.cpu().numpy()])

    start_time = time.time()
    sorted_hex_faces = sort_hex_faces(final_vertices[all_faces])
    end_time = time.time()
    print(f"Sorting hex faces took {end_time - start_time:.2f} seconds")

    print(f"Total pentagonal/hexagonal vertices: {final_vertices.shape[0]}")
    print(f"Total pentagonal/hexagonal faces: {all_faces.shape[0]}")

    plot_sphere_vispy(sorted_hex_faces, hexagon_colors=np.random.rand(all_faces.shape[0], 3))

    #plot_points_and_faces_as_mesh(triangular_vertices.cpu().numpy(), triangular_faces.cpu().numpy())