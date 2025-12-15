#!/bin/bash
export NCCL_GDR_LEVEL=4
export NCCL_IB_DISABLE=1

# 设置每个节点的进程数，默认为8（如果未设置环境变量）
PET_NPROC_PER_NODE=${PET_NPROC_PER_NODE:-8}
PET_NNODES=${PET_NNODES:-1}
NUM_PROCESSES=$((PET_NPROC_PER_NODE * PET_NNODES))

echo "Using ${NUM_PROCESSES} processes total"

accelerate launch \
  --num_processes ${NUM_PROCESSES} \
  --num_machines ${PET_NNODES} \
  --config_file config/accelerate_config/multi_node.yaml \
  runner/mixture_modality/moe.py \
  --config config/mixture_modality/mot_multi_node_h100.yaml
