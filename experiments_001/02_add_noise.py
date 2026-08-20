import open3d as o3d
import numpy as np

mesh = o3d.geometry.TriangleMesh.create_sphere(
    radius=1.0,
    resolution=20
)

mesh.compute_vertex_normals()

vertices = np.asarray(mesh.vertices)

print("Vertices:", len(vertices))
print("First vertex:", vertices[0])

noise = np.random.normal(
    0,
    0.05,
    vertices.shape
)
# noise = np.clip(noise, -0.5, 0.5)

vertices += noise

# mesh.vertices = o3d.utility.Vector3dVector(vertices)

mesh.compute_vertex_normals()

o3d.visualization.draw_geometries(
    [mesh],
    mesh_show_wireframe=True
)