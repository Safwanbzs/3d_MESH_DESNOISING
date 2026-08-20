import open3d as o3d
import numpy as np


# -----------------------------
# 1. Create a mesh
# -----------------------------

mesh = o3d.geometry.TriangleMesh.create_sphere(
    radius=1.0,
    resolution=20
)

mesh.compute_vertex_normals()
vertices = np.asarray(mesh.vertices)
noise = np.random.normal(
    0,
    0.05,
    vertices.shape
)


vertices += noise
mesh.compute_vertex_normals()



# -----------------------------
# 2. Convert mesh data to arrays
# -----------------------------


normals = np.asarray(mesh.vertex_normals)
triangles = np.asarray(mesh.triangles)


# -----------------------------
# 3. Build vertex neighborhoods
# -----------------------------

neighbors = [set() for _ in range(len(vertices))]

for triangle in triangles:

    a, b, c = triangle

    neighbors[a].add(b)
    neighbors[a].add(c)

    neighbors[b].add(a)
    neighbors[b].add(c)

    neighbors[c].add(a)
    neighbors[c].add(b)


# -----------------------------
# 4. Bilateral filter parameters
# -----------------------------

sigma_s = 0.4
sigma_n = 0.1


# -----------------------------
# 5. Create new vertex array
# -----------------------------

new_vertices = vertices.copy()


# -----------------------------
# 6. Filter every vertex
# -----------------------------

for i in range(len(vertices)):

    weighted_position = np.zeros(3)
    total_weight = 0.0

    for j in neighbors[i]:

        # Distance between vertices
        distance = np.linalg.norm(
            vertices[i] - vertices[j]
        )

        # Spatial weight
        spatial_weight = np.exp(
            -(distance ** 2) /
            (2 * sigma_s ** 2)
        )

        # Normal similarity
        normal_similarity = np.dot(
            normals[i],
            normals[j]
        )

        # Normal weight
        normal_weight = np.exp(
            -((1 - normal_similarity) ** 2) /
            (2 * sigma_n ** 2)
        )

        # Bilateral weight
        weight = spatial_weight * normal_weight

        # Weighted position
        weighted_position += weight * vertices[j]

        # Sum weights
        total_weight += weight

    # Normalize
    if total_weight > 0:

        new_vertices[i] = (
            weighted_position / total_weight
        )


# -----------------------------
# 7. Put filtered vertices
#    back into the mesh
# -----------------------------

mesh.vertices = o3d.utility.Vector3dVector(
    new_vertices
)

mesh.compute_vertex_normals()


# -----------------------------
# 8. Display result
# -----------------------------

o3d.visualization.draw_geometries(
    [mesh],mesh_show_wireframe=True,
    window_name="Bilateral Filtered Mesh"
)