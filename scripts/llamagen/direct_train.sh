#!/bin/bash
accelerate launch \
    --config_file config/accelerate_config/single_node.yaml \
    runner/llamagen/direct_train.py \
    --config config/llamagen/qz_direct_train.yaml