#!/bin/bash

export ROOT_DIR=NeRF/Synthetic_NeRF

python train_nerf.py \
    --root_dir $ROOT_DIR/Chair \
    --exp_name Chair --no_save_test \
    --num_epochs 30 --batch_size 4096 --lr 2e-2 --eval_lpips

python train_nerf.py \
    --root_dir $ROOT_DIR/Drums \
    --exp_name Drums --no_save_test \
    --num_epochs 30 --batch_size 4096 --lr 2e-2 --eval_lpips

python train_nerf.py \
    --root_dir $ROOT_DIR/Ficus \
    --exp_name Ficus --no_save_test \
    --num_epochs 30 --batch_size 4096 --lr 2e-2 --eval_lpips

python train_nerf.py \
    --root_dir $ROOT_DIR/Hotdog \
    --exp_name Hotdog --no_save_test \
    --num_epochs 30 --batch_size 4096 --lr 2e-2 --eval_lpips

python train_nerf.py \
    --root_dir $ROOT_DIR/Lego \
    --exp_name Lego --no_save_test \
    --num_epochs 30 --batch_size 4096 --lr 2e-2 --eval_lpips

python train_nerf.py \
    --root_dir $ROOT_DIR/Materials \
    --exp_name Materials --no_save_test \
    --num_epochs 30 --batch_size 4096 --lr 2e-2 --eval_lpips

python train_nerf.py \
    --root_dir $ROOT_DIR/Mic \
    --exp_name Mic --no_save_test \
    --num_epochs 30 --batch_size 4096 --lr 2e-2 --eval_lpips

python train_nerf.py \
    --root_dir $ROOT_DIR/Ship \
    --exp_name Ship --no_save_test \
    --num_epochs 30 --batch_size 4096 --lr 2e-2 --eval_lpips
