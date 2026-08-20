import open3d as o3d
import numpy as np

mesh = o3d.geometry.TriangleMesh.create_sphere(
    radius=1.0,
    resolution=20
)

vertices = np.asarray(mesh.vertices)

noise = np.random.normal(
    0,
    0.05,
    vertices.shape
)

vertices += noise

mesh.vertices = o3d.utility.Vector3dVector(vertices)

# neighbors = [set() for _ in range(len(vertices))]
#
# for triangle in np.asarray(mesh.triangles):
#     a, b, c = triangle
#
#     neighbors[a].update([b, c])
#     neighbors[b].update([a, c])
#     neighbors[c].update([a, b])

# Tell Open3D to compute the neighbors
mesh.compute_adjacency_list()
# Access the result
neighbors = mesh.adjacency_list

lambda_ = 0.31415

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

mesh.vertices = o3d.utility.Vector3dVector(new_vertices)

mesh.compute_vertex_normals()

o3d.visualization.draw_geometries(
    [mesh],
    mesh_show_wireframe=True
)