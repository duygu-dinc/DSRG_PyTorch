#!/bin/bash
arch=deeplab_large_fov
image_list='./train_noval.txt'
image_path='./benchmark_RELEASE/dataset/img'
cls_labels_path='./DSRG_PyTorch/datalist/PascalVOC/cls_labels.npy'
log_path='./train_log/1'
pred_path='./DSRG_PyTorch/SBDresult/1'
#trained=${log_path}/last_checkpoint.pth.tar
trained=/content/drive/MyDrive/Training/last_checkpoint.pth.tar
smooth=True
color_mask=1
gpu=0

python3 ./DSRG_PyTorch/test_multiprocess.py \
  --arch ${arch} \
  --trained ${trained} \
  --image-list ${image_list} \
  --image-path ${image_path} \
  --pred-path ${pred_path} \
  --cls-labels-path ${cls_labels_path} \
  --gpu ${gpu} \
  --num-gpu 1 \
  --split-size 1 \
  --smooth \
  --color-mask ${color_mask} \
