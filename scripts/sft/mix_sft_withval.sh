#!/bin/bash
accelerate launch \
--config_file config/accelerate_config/deepspeed.yaml \
runner/sft/blip3o.py \
--config config/sft/mix_data_with_val.yaml