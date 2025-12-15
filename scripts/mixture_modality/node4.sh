#!/bin/bash
export NCCL_GDR_LEVEL=4
export NCCL_IB_DISABLE=1

# 多节点分布式训练环境变量
export MASTER_ADDR=252.1.82.59
export MASTER_PORT=19534
export WORLD_SIZE=32
export RANK=24
export LOCAL_WORLD_SIZE=8

accelerate launch \
  --num_processes 32 \
  --num_machines 4 \
  --machine_rank 3 \
  --main_process_ip 252.1.82.59 \
  --main_process_port 19534 \
  --config_file config/accelerate_config/multi_node.yaml \
  runner/mixture_modality/moe.py \
  --config config/mixture_modality/mot_multi_node.yaml
 