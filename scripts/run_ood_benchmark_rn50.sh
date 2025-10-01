#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python dota_gda_em_aug.py     --config configs/rn50 \
                                                --log-path ./log \
                                                --datasets  I/A/V/R/S \
                                                --backbone RN50

