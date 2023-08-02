"""
These codes are adapted from tiny-cuda-nn (https://github.com/NVlabs/tiny-cuda-nn)
"""

import torch
from torch.utils.data import Dataset

import math


class ImageDataset(Dataset):
    def __init__(self, data, size=100, num_samples=2**18, split='train'):
        super().__init__()

        # assign image
        self.data = data

        self.img_wh = (self.data.shape[0], self.data.shape[1])
        self.img_shape = torch.tensor([self.img_wh[0], self.img_wh[1]]).float()

        print(f"[INFO] image: {self.data.shape}")

        self.num_samples = num_samples

        self.split = split
        self.size = size

        if self.split.startswith("test"):
            half_dx =  0.5 / self.img_wh[0]
            half_dy =  0.5 / self.img_wh[1]
            xs = torch.linspace(half_dx, 1-half_dx, self.img_wh[0])
            ys = torch.linspace(half_dy, 1-half_dy, self.img_wh[1])
            xv, yv = torch.meshgrid([xs, ys], indexing="ij")

            xy = torch.stack((xv.flatten(), yv.flatten())).t()

            xy_max_num = math.ceil(xy.shape[0] / 1024.0)
            padding_delta = xy_max_num * 1024 - xy.shape[0]
            zeros_padding = torch.zeros((padding_delta, 2))
            self.xs = torch.cat([xy, zeros_padding], dim=0)


    def __len__(self):
        return self.size


    def __getitem__(self, _):
        if self.split.startswith('train'):
            xs = torch.rand([self.num_samples, 2], dtype=torch.float32)

            assert torch.sum(xs < 0) == 0, "The coordinates for input image should be non-negative."

            with torch.no_grad():
                scaled_xs = xs * self.img_shape
                indices = scaled_xs.long()
                lerp_weights = scaled_xs - indices.float()

                x0 = indices[:, 0].clamp(min=0, max=self.img_wh[0]-1).long()
                y0 = indices[:, 1].clamp(min=0, max=self.img_wh[1]-1).long()
                x1 = (x0 + 1).clamp(min=0, max=self.img_wh[0]-1).long()
                y1 = (y0 + 1).clamp(min=0, max=self.img_wh[1]-1).long()

                rgbs = self.data[x0, y0] * (1.0 - lerp_weights[:, 0:1]) * (1.0 - lerp_weights[:, 1:2]) + \
                       self.data[x0, y1] * (1.0 - lerp_weights[:, 0:1]) * lerp_weights[:, 1:2] + \
                       self.data[x1, y0] * lerp_weights[:, 0:1] * (1.0 - lerp_weights[:, 1:2]) + \
                       self.data[x1, y1] * lerp_weights[:, 0:1] * lerp_weights[:, 1:2]
        else:
            xs = self.xs
            rgbs = self.data

        results = {
            'points': xs,
            'rgbs': rgbs,
        }

        return results