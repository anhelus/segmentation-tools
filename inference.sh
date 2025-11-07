#! /usr/bin/bash

config_path=/mnt/c/Datasets/LettucePG/config.yaml
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

python3 main.py dino $config_path --save-metrics-only --model GDINO-TINY
python3 main.py dino $config_path --save-metrics-only --model GDINO-BASE

python3 main.py owl $config_path --save-metrics-only --model OWLVIT-BASE-16 --prompts "${clip_prompts[@]}"
python3 main.py owl $config_path --save-metrics-only --model OWLVIT-BASE-32 --prompts "${clip_prompts[@]}"
python3 main.py owl $config_path --save-metrics-only --model OWLVIT-LARGE --prompts "${clip_prompts[@]}"
python3 main.py owl $config_path --save-metrics-only --model OWL2-BASE --prompts "${clip_prompts[@]}"
python3 main.py owl $config_path --save-metrics-only --model OWL2-BASE-ENSEMBLE --prompts "${clip_prompts[@]}"
python3 main.py owl $config_path --save-metrics-only --model OWL2-BASE-FINETUNED --prompts "${clip_prompts[@]}"
python3 main.py owl $config_path --save-metrics-only --model OWL2-LARGE --prompts "${clip_prompts[@]}"
python3 main.py owl $config_path --save-metrics-only --model OWL2-LARGE-ENSEMBLE --prompts "${clip_prompts[@]}"
python3 main.py owl $config_path --save-metrics-only --model OWL2-LARGE-FINETUNED --prompts "${clip_prompts[@]}"

python3 main.py yolo_world $config_path --save-metrics-only --model YOLO-S-WORLD --prompts "${clip_prompts[@]}"
python3 main.py yolo_world $config_path --save-metrics-only --model YOLO-M-WORLD --prompts "${clip_prompts[@]}"
python3 main.py yolo_world $config_path --save-metrics-only --model YOLO-L-WORLD --prompts "${clip_prompts[@]}"
python3 main.py yolo_world $config_path --save-metrics-only --model YOLO-X-WORLD --prompts "${clip_prompts[@]}"
python3 main.py yolo_world $config_path --save-metrics-only --model YOLO-S-WORLD-v2 --prompts "${clip_prompts[@]}"
python3 main.py yolo_world $config_path --save-metrics-only --model YOLO-M-WORLD-v2 --prompts "${clip_prompts[@]}"
python3 main.py yolo_world $config_path --save-metrics-only --model YOLO-L-WORLD-v2 --prompts "${clip_prompts[@]}"
python3 main.py yolo_world $config_path --save-metrics-only --model YOLO-X-WORLD-v2 --prompts "${clip_prompts[@]}"

python3 main.py grounded_sam $config_path --save-metrics-only --dino-model GDINO-TINY --sam-model SAM-T
python3 main.py grounded_sam $config_path --save-metrics-only --dino-model GDINO-TINY --sam-model SAM-S
python3 main.py grounded_sam $config_path --save-metrics-only --dino-model GDINO-TINY --sam-model SAM-B
python3 main.py grounded_sam $config_path --save-metrics-only --dino-model GDINO-TINY --sam-model SAM-L
python3 main.py grounded_sam $config_path --save-metrics-only --dino-model GDINO-TINY --sam-model SAM-2.1-T
python3 main.py grounded_sam $config_path --save-metrics-only --dino-model GDINO-TINY --sam-model SAM-2.1-S
python3 main.py grounded_sam $config_path --save-metrics-only --dino-model GDINO-TINY --sam-model SAM-2.1-B
python3 main.py grounded_sam $config_path --save-metrics-only --dino-model GDINO-TINY --sam-model SAM-2.1-L
python3 main.py grounded_sam $config_path --save-metrics-only --dino-model GDINO-BASE --sam-model SAM-T
python3 main.py grounded_sam $config_path --save-metrics-only --dino-model GDINO-BASE --sam-model SAM-S
python3 main.py grounded_sam $config_path --save-metrics-only --dino-model GDINO-BASE --sam-model SAM-B
python3 main.py grounded_sam $config_path --save-metrics-only --dino-model GDINO-BASE --sam-model SAM-L
python3 main.py grounded_sam $config_path --save-metrics-only --dino-model GDINO-BASE --sam-model SAM-2.1-T
python3 main.py grounded_sam $config_path --save-metrics-only --dino-model GDINO-BASE --sam-model SAM-2.1-S
python3 main.py grounded_sam $config_path --save-metrics-only --dino-model GDINO-BASE --sam-model SAM-2.1-B
python3 main.py grounded_sam $config_path --save-metrics-only --dino-model GDINO-BASE --sam-model SAM-2.1-L
