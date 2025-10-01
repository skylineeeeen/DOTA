#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python dota_gda_em_aug.py     --config configs/rn50 \
                                                --log-path ./log \
                                                --datasets  caltech101/dtd/eurosat/fgvc/food101/oxford_flowers/oxford_pets/stanford_cars/sun397/ucf101 \
                                                --backbone RN50

