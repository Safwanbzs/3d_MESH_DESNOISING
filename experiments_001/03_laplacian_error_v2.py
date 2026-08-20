import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

mesh = o3d.geometry.TriangleMesh.create_sphere(
    radius=1.0,
    resolution=20
)
clean_vertices = np.asarray(mesh.vertices).copy()
vertices = np.asarray(mesh.vertices)

np.random.seed(42)
noise = np.random.normal(
    0,
    0.05,
    vertices.shape
)

vertices += noise

mesh.vertices = o3d.utility.Vector3dVector(vertices)
noisy_error = np.linalg.norm(
    vertices - clean_vertices,
    axis=1
)
print("Noisy mean error:", noisy_error.mean())
neighbors = [set() for _ in range(len(vertices))]

for triangle in np.asarray(mesh.triangles):
    a, b, c = triangle

    neighbors[a].update([b, c])
    neighbors[b].update([a, c])
    neighbors[c].update([a, b])

lambda_ = 0.5
iterations = 10

for _ in range(iterations):

    new_vertices = vertices.copy()

    for i in range(len(vertices)):

        neighbor_indices = list(neighbors[i])

        if len(neighbor_indices) == 0:
            continue

        average = vertices[neighbor_indices].mean(axis=0)

        new_vertices[i] = (
            vertices[i]
            + lambda_ * (average - vertices[i])
        )

    vertices = new_vertices
denoised_error = np.linalg.norm(
    vertices - clean_vertices,
    axis=1
)
print("Denoised mean error:", denoised_error.mean())
print("Noisy mean error:", noisy_error.mean())
print("Noisy max error:", noisy_error.max())

print("Denoised mean error:", denoised_error.mean())
print("Denoised max error:", denoised_error.max())

plt.hist(noisy_error, bins=30, alpha=0.5, label="Noisy")
plt.hist(denoised_error, bins=30, alpha=0.5, label="Laplacian")

plt.xlabel("Vertex error")
plt.ylabel("Number of vertices")
plt.legend()
plt.show()