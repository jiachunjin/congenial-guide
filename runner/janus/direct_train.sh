#!/bin/bash
CUDA_VISIBLE_DEVICES=2,3,4,5 accelerate launch \
    --num_processes 4 \
    --config_file config/accelerate_config/deepspeed.yaml \
    runner/janus/direct_train.py \
    --config config/janus/direct_train.yaml