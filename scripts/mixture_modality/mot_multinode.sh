#!/bin/bash
export NCCL_GDR_LEVEL=4
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch \
  --num_processes ${PET_NPROC_PER_NODE} \
  --num_machines ${PET_NNODES} \
  --distributed_type DEEPSPEED \
  --mixed_precision bf16 \
  --config_file config/accelerate_config/multi_node.yaml \
  runner/mixture_modality/moe.py \
  --config config/mixture_modality/mot.yaml
