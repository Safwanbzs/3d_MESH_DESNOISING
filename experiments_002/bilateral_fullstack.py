import open3d as o3d
import numpy as np


# -----------------------------
# 1. Create clean mesh
# -----------------------------

mesh = o3d.geometry.TriangleMesh.create_sphere(
    radius=1.0,
    resolution=20
)

mesh.compute_vertex_normals()


# -----------------------------
# 2. Store clean vertices
# -----------------------------

clean_vertices = np.asarray(mesh.vertices).copy()


# -----------------------------
# 3. Add noise
# -----------------------------

noise_strength = 0.05

noise = np.random.normal(
    0,
    noise_strength,
    clean_vertices.shape
)

noisy_vertices = clean_vertices + noise


# -----------------------------
# 4. Create noisy mesh
# -----------------------------

noisy_mesh = o3d.geometry.TriangleMesh()

noisy_mesh.vertices = o3d.utility.Vector3dVector(
    noisy_vertices
)

noisy_mesh.triangles = mesh.triangles

noisy_mesh.compute_vertex_normals()


# -----------------------------
# 5. Get noisy mesh data
# -----------------------------

vertices = np.asarray(
    noisy_mesh.vertices
)

normals = np.asarray(
    noisy_mesh.vertex_normals
)

triangles = np.asarray(
    noisy_mesh.triangles
)


# -----------------------------
# 6. Build neighborhoods
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
# 7. Bilateral parameters
# -----------------------------

sigma_s = 0.5
sigma_n = 0.5


# -----------------------------
# 8. Create filtered vertices
# -----------------------------

new_vertices = vertices.copy()


# -----------------------------
# 9. Bilateral filtering
# -----------------------------

for i in range(len(vertices)):

    weighted_position = np.zeros(3)
    total_weight = 0.0

    for j in neighbors[i]:

        # Distance
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
        weight = (
            spatial_weight *
            normal_weight
        )

        # Accumulate weighted positions
        weighted_position += (
            weight * vertices[j]
        )

        total_weight += weight

    # Normalize
    if total_weight > 0:

        new_vertices[i] = (
            weighted_position /
            total_weight
        )


# -----------------------------
# 10. Create filtered mesh
# -----------------------------

filtered_mesh = o3d.geometry.TriangleMesh()

filtered_mesh.vertices = o3d.utility.Vector3dVector(
    new_vertices
)

filtered_mesh.triangles = noisy_mesh.triangles

filtered_mesh.compute_vertex_normals()

#
# # -----------------------------
# # 11. Display all three
# # -----------------------------
#
# o3d.visualization.draw_geometries(
#     [mesh],
#     window_name="Clean Mesh"
# )
#
# o3d.visualization.draw_geometries(
#     [noisy_mesh],
#     window_name="Noisy Mesh"
# )
#
# o3d.visualization.draw_geometries(
#     [filtered_mesh],
#     window_name="Bilateral Filtered Mesh"
# )

# -----------------------------
# Display side by side
# -----------------------------
#
# clean_display = mesh.translate(
#     [-2.5, 0, 0],
#     relative=True
# )
#
# noisy_display = noisy_mesh.translate(
#     [0, 0, 0],
#     relative=True
# )
#
# filtered_display = filtered_mesh.translate(
#     [2.5, 0, 0],
#     relative=True
# )
#
# o3d.visualization.draw_geometries(
#     [
#         clean_display,
#         noisy_display,
#         filtered_display
#     ],
#     window_name="Clean | Noisy | Bilateral"
# )
# -----------------------------
# 12. Calculate errors
# -----------------------------

filtered_vertices = np.asarray(
    filtered_mesh.vertices
)


# Distance from noisy mesh to clean mesh
noisy_error = np.linalg.norm(
    noisy_vertices - clean_vertices,
    axis=1
)

# Distance from filtered mesh to clean mesh
filtered_error = np.linalg.norm(
    filtered_vertices - clean_vertices,
    axis=1
)


# Average error
mean_noisy_error = np.mean(noisy_error)
mean_filtered_error = np.mean(filtered_error)


print("\n===== DENOISING RESULTS =====")

print("Mean error before filtering:",
      mean_noisy_error)

print("Mean error after filtering:",
      mean_filtered_error)

print("Error reduction:",
      mean_noisy_error - mean_filtered_error)

print(
    "Percentage improvement:",
    (1 - mean_filtered_error / mean_noisy_error) * 100,
    "%"
)