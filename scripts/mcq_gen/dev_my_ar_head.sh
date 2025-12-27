#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
--num_processes 8 \
--config_file config/accelerate_config/single_node.yaml \
runner/mcq_gen/dev_my_ar_head.py \
--config config/mcq_gen/dev_my_ar_head.yaml