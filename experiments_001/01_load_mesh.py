import open3d as o3d

mesh = o3d.geometry.TriangleMesh.create_sphere(
    radius=1.0,
    resolution=20
)

mesh.compute_vertex_normals()

print(mesh)
print("Vertices:", len(mesh.vertices))
print("Normals:", len(mesh.vertex_normals))
print("Triangles:", len(mesh.triangles))

o3d.visualization.draw_geometries(
    [mesh],
    mesh_show_wireframe=True
)