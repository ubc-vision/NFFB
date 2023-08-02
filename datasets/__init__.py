from datasets.nerf.nerf import NeRFDataset
from datasets.nerf.nsvf import NSVFDataset
from datasets.nerf.colmap import ColmapDataset
from datasets.nerf.nerfpp import NeRFPPDataset
from datasets.nerf.rtmv import RTMVDataset


dataset_dict = {'nerf': NeRFDataset,
                'nsvf': NSVFDataset,
                'colmap': ColmapDataset,
                'nerfpp': NeRFPPDataset,
                'rtmv': RTMVDataset}