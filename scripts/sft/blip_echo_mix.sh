#!/bin/bash
accelerate launch \
--config_file config/accelerate_config/single_node.yaml \
runner/sft/blip3o.py \
--config config/sft/echo4o_blip3o.yaml