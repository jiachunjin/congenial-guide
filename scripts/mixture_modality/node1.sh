#!/bin/bash

# ============ NCCL 通信优化 ============
# 指定网络接口（重要！根据你的集群修改）
# 运行 `ip link show` 查看可用网卡，选择高速网卡
export NCCL_SOCKET_IFNAME=eth0  # 改成你的高速网卡名，如 ib0, bond0, eth0

# 如果没有 InfiniBand，使用 TCP
export NCCL_IB_DISABLE=1
# 如果有 InfiniBand，注释上一行，取消下面的注释
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_TIMEOUT=23
# export NCCL_IB_RETRY_CNT=7

# TCP Socket 优化（无 IB 时重要）
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=8

# 缓冲区优化
export NCCL_BUFFSIZE=16777216  # 16MB，增大缓冲区
export NCCL_NTHREADS=512

# P2P 和 SHM 优化
export NCCL_P2P_LEVEL=NVL      # 节点内使用 NVLink（如果有）
export NCCL_SHM_DISABLE=0      # 启用共享内存

# 调试信息（先启用看看通信情况，确认无问题后注释掉）
export NCCL_DEBUG=WARN
# export NCCL_DEBUG=INFO  # 更详细的信息

# 多节点分布式训练环境变量
export MASTER_ADDR=252.1.82.59
export MASTER_PORT=19534
export WORLD_SIZE=32
export RANK=0
export LOCAL_WORLD_SIZE=8

accelerate launch \
  --num_processes 32 \
  --num_machines 4 \
  --machine_rank 0 \
  --main_process_ip 252.1.82.59 \
  --main_process_port 19534 \
  --config_file config/accelerate_config/multi_node.yaml \
  runner/mixture_modality/moe.py \
  --config config/mixture_modality/mot_multi_node.yaml
