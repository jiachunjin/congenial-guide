#!/bin/bash
accelerate launch \
--config_file config/accelerate_config/single_node.yaml \
runner/sft/rec_align.py \
--config config/sft/rec_align.yaml