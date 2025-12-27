#!/bin/bash
accelerate launch \
--num_processes 8 \
--config_file config/accelerate_config/single_node.yaml \
runner/mixture_modality/moe.py \
--config config/mixture_modality/moe.yaml