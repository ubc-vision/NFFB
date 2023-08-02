import os
import numpy as np

import random

import torch
import pytorch_lightning


def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                break
        else:
            checkpoint_[k] = v
    return checkpoint_


def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    if not ckpt_path: return
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)


def slim_ckpt(ckpt_path, save_poses=False):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # pop unused parameters
    keys_to_pop = ['directions', 'model.density_grid', 'model.grid_coords']
    if not save_poses: keys_to_pop += ['poses']
    for k in ckpt['state_dict']:
        if k.startswith('val_lpips'):
            keys_to_pop += [k]
    for k in keys_to_pop:
        ckpt['state_dict'].pop(k, None)
    return ckpt['state_dict']



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    pytorch_lightning.seed_everything(seed, workers=True)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True



def process_batch_in_chunks(in_ccords, model, max_chunk_size=1024):
    chunk_outs = []

    coord_chunks = torch.split(in_ccords, max_chunk_size)
    for chunk_batched_in in coord_chunks:
        tmp_img = model(chunk_batched_in)
        chunk_outs.append(tmp_img.detach())

    batched_out = torch.cat(chunk_outs, dim=0)

    return batched_out