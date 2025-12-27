#!/bin/bash
accelerate launch \
--config_file config/accelerate_config/single_node.yaml \
runner/sft/t2i_with_recA.py \
--config config/sft/T2I_with_recA.yaml