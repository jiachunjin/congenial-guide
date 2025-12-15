#!/bin/bash

# ============ NCCL 通信优化 ============
# 如果有 InfiniBand/RoCE，启用 IB（注释掉 NCCL_IB_DISABLE）
# export NCCL_IB_DISABLE=1  # 仅在没有 IB 时启用

# IB/RoCE 优化（如果有高速网络）
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

# GPU Direct RDMA
export NCCL_GDR_LEVEL=5
export NCCL_NET_GDR_LEVEL=5

# 通信算法优化
export NCCL_ALGO=Tree
export NCCL_PROTO=Simple

# 缓冲区和异步优化
export NCCL_BUFFSIZE=8388608
export NCCL_NTHREADS=512
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=4

# 调试信息（调试时启用，生产时注释掉）
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

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
 