#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python dota_gda_em_aug.py     --config configs/vit \
                                                --log-path ./log \
                                                --datasets  I/A/V/R/S \
                                                --backbone ViT-B/16

