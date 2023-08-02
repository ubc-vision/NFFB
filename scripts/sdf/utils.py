import torch
import trimesh
import mcubes


def create_mesh(model, mesh_out_path, grid_res):
    # Prepare directory
    num_samples = grid_res ** 3

    sdf_values = torch.zeros(num_samples, 1)

    bound_min = torch.FloatTensor([-1.0, -1.0, -1.0])
    bound_max = torch.FloatTensor([1.0, 1.0, 1.0])

    X = torch.linspace(bound_min[0], bound_max[0], grid_res)
    Y = torch.linspace(bound_min[1], bound_max[1], grid_res)
    Z = torch.linspace(bound_min[2], bound_max[2], grid_res)

    xx, yy, zz = torch.meshgrid(X, Y, Z, indexing='ij')
    inputs = torch.concat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda() # [N, 3]

    head = 0
    max_batch = int(2 ** 18)

    while head < num_samples:
        sample_subset = inputs[head : min(head + max_batch, num_samples), :]

        sdf_values[head : min(head + max_batch, num_samples), 0] = (
            model(sample_subset).squeeze(1).detach().cpu()
        )
        head += max_batch

    sdf_values = sdf_values.reshape(grid_res, grid_res, grid_res)

    numpy_3d_sdf_tensor = sdf_values.data.cpu().numpy()

    verts, faces = mcubes.marching_cubes(numpy_3d_sdf_tensor, 0.0)

    vertices = verts / (grid_res - 1.0) * 2.0 - 1.0

    print(f'\nSaving mesh to {mesh_out_path}...', end="")

    mesh = trimesh.Trimesh(vertices, faces, process=False) 		# important, process=True leads to seg fault...
    mesh.export(mesh_out_path)

    print(f"==> Finished saving mesh.")
