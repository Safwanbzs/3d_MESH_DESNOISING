import open3d as o3d
import numpy as np

# Create a sphere
mesh = o3d.geometry.TriangleMesh.create_sphere(
    radius=1.0,
    resolution=10
)

# Calculate vertex normals
mesh.compute_vertex_normals()

vertices = np.asarray(mesh.vertices)
normals = np.asarray(mesh.vertex_normals)
triangles = np.asarray(mesh.triangles)

# Pick one vertex
vertex_id = 100

print("Selected vertex:")
print("ID:", vertex_id)
print("Position:", vertices[vertex_id])
print("Normal:", normals[vertex_id])


# Find triangles containing this vertex
neighbor_triangles = []

for i, triangle in enumerate(triangles):
    if vertex_id in triangle:
        neighbor_triangles.append(i)

# Find neighboring vertices
neighbors = set()

for triangle_id in neighbor_triangles:
    triangle = triangles[triangle_id]

    for vertex in triangle:
        if vertex != vertex_id:
            neighbors.add(vertex)

print("\nNeighbor vertices:")

for neighbor_id in neighbors:

    position = vertices[neighbor_id]
    normal = normals[neighbor_id]

    distance = np.linalg.norm(
        vertices[vertex_id] - position
    )

    normal_similarity = np.dot(
        normals[vertex_id],
        normal
    )

    print("\nNeighbor ID:", neighbor_id)
    print("Position:", position)
    print("Normal:", normal)
    print("Distance:", distance)
    print("Normal similarity:", normal_similarity)