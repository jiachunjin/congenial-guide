#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
--num_processes 8 \
--config_file config/accelerate_config/deepspeed.yaml \
runner/vq_distill/distill.py \
--config config/vq_distill/distill_mcq_4code.yaml