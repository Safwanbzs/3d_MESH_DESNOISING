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
# 3. Add random noise
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
# 5. Display noisy mesh
# -----------------------------

o3d.visualization.draw_geometries(
    [noisy_mesh],
    window_name="Noisy Sphere"
)