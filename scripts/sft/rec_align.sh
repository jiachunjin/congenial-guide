#!/bin/bash
accelerate launch \
--config_file config/accelerate_config/deepspeed.yaml \
runner/sft/rec_align.py \
--config config/sft/rec_align.yaml