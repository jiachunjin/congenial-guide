#!/bin/bash
accelerate launch \
    --config_file config/accelerate_config/deepspeed.yaml \
    runner/llamagen/direct_train.py \
    --config config/llamagen/qz_direct_train.yaml