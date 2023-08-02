"""
These codes are adapted from torch-ngp (https://github.com/ashawkey/torch-ngp/tree/main)
"""

from torch.utils.data import Dataset

import numpy as np
import trimesh
import pysdf


class SDFDataset(Dataset):
    def __init__(self, path, size=100, num_samples=2**18, clip_sdf=None):
        super().__init__()
        self.path = path

        # load obj
        self.mesh = trimesh.load(path, force='mesh')

        # normalize to [-1, 1] (different from instant-sdf where is [0, 1])
        vs = self.mesh.vertices
        vmin = vs.min(0)
        vmax = vs.max(0)
        v_center = (vmin + vmax) / 2
        v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
        vs = (vs - v_center[None, :]) * v_scale
        self.mesh.vertices = vs

        print(f"[INFO] mesh: {self.mesh.vertices.shape} {self.mesh.faces.shape}")

        if not self.mesh.is_watertight:
            print(f"[WARN] mesh is not watertight! SDF maybe incorrect.")

        self.sdf_fn = pysdf.SDF(self.mesh.vertices, self.mesh.faces)

        self.num_samples = num_samples
        assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."
        self.clip_sdf = clip_sdf

        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, _):
        # online sampling
        sdfs = np.zeros((self.num_samples, 1))
        # surface
        points_surface = self.mesh.sample(self.num_samples * 2 // 3)
        # perturb surface
        points_surface[self.num_samples // 3:] += 0.01 * np.random.randn(self.num_samples // 3, 3)
        # random
        points_uniform = np.random.rand(self.num_samples // 3, 3) * 2 - 1
        points = np.concatenate([points_surface, points_uniform], axis=0).astype(np.float32)

        sdfs[self.num_samples // 3:] = -self.sdf_fn(points[self.num_samples // 3:])[:,None].astype(np.float32)

        # clip sdf
        if self.clip_sdf is not None:
            sdfs = sdfs.clip(-self.clip_sdf, self.clip_sdf)

        results = {
            'sdfs': sdfs,
            'points': points,
        }

        return results