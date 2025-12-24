#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1

python runner/mixture_modality/moe_gen.py \
--exp_dir /inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/experiment/hdd_exp/1224_new_save \
--step 60000 \
--cfg_scale 3.0 \
--batch_size 4

# /inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/experiment/hdd_exp/1224_new_save
# /inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/experiment/mcq_gen/1221_new
# /inspire/hdd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/jjc/experiment/mcq_gen/1222_newlr
# /inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/experiment/mcq_gen/1212_moe
# /inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/experiment/mcq_gen/1213_mot