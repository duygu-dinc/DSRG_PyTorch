#!/bin/bash
image_list='./DSRG_PyTorch/datalist/PascalVOC/val_id.txt'
gt_path="./DSRG_PyTorch/dataset/VOC2012/SegmentationClassAug/"
result_path='./DSRG_PyTorch/result/1'
#pred_path=${result_path}/label_mask
pred_path=${result_path}/pred
save_name=${result_path}/evaluation_result_direct.txt
#color_mask=0
color_mask=1

python ./DSRG_PyTorch/evaluation.py \
  --image-list ${image_list} \
  --pred-path ${pred_path} \
  --gt-path ${gt_path} \
  --save-name ${save_name} \
  --class-num 21 \
  --color-mask ${color_mask} 
