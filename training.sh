#! /usr/bin/bash

config_path=/mnt/c/Datasets/LettucePG/config.yaml
bbox_ul_path=/mnt/c/Datasets/LettucePG/bbox_gt_ul/bbox_gt_ul.yaml
clip_prompts=( \
    "a lettuce plant" \
    "a photo of lettuce" \
    "a head of lettuce" \
    "a lettuce seedling" \
    "a young lettuce plant" \
    "a small green plant" \
    "lettuce growing in soil" \
    "a photo of lettuce from above" \
    "a rosette of green leaves" \
    "a leafy green vegetable" \
    "a crop of lettuce" \
)

python3 train.py yolo_world $config_path --yaml-path $bbox_ul_path --model YOLO-S-WORLD --prompts "${clip_prompts[@]}"
