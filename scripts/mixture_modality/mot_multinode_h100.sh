#!/bin/bash
export NCCL_GDR_LEVEL=4
export NCCL_IB_DISABLE=1

# 设置每个节点的进程数，默认为8（如果未设置环境变量）
PET_NPROC_PER_NODE=${PET_NPROC_PER_NODE:-8}
# 总节点数 (默认为2，请根据实际修改)
PET_NNODES=${PET_NNODES:-2}
# 当前节点的 Rank (0 为主节点，1 为从节点...，必须在不同节点上设置不同值)
PET_NODE_RANK=${PET_NODE_RANK:-0}
# 计算总进程数 (Total World Size)
TOTAL_PROCESSES=$((PET_NPROC_PER_NODE * PET_NNODES))

echo "=================================================="
echo "Distributed Training Config:"
echo "  Nodes: ${PET_NNODES}"
echo "  GPU per Node: ${PET_NPROC_PER_NODE}"
echo "  Total Processes (World Size): ${TOTAL_PROCESSES}"
echo "  Current Node Rank: ${PET_NODE_RANK}"
echo "  Master Addr: ${MASTER_ADDR}:${MASTER_PORT}"
echo "=================================================="

accelerate launch \
  --num_processes ${TOTAL_PROCESSES} \
  --num_machines ${PET_NNODES} \
  --machine_rank ${PET_NODE_RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --config_file config/accelerate_config/multi_node.yaml \
  runner/mixture_modality/moe.py \
  --config config/mixture_modality/mot_multi_node_h100.yaml


# NCCL_IB_QPS_PER_CONNECTION
# NCCL_GDR_LEVEL
# NCCL_IB_PCI_RELAXED_ORDERING
# NCCL_IB_TC
# NCCL_NVLS_ENABLE
# NCCL_IB_GID_INDEX
# GLOO_SOCKET_IFNAME
# NCCL_SOCKET_IFNAME
# NCCL_DEBUG
# NCCL_IB_TIMEOUT
# NCCL_IB_RETRY_CNT
# NCCL_IB_HCA
# MASTER_PORT
# MASTER_ADDR
# PET_MASTER_PORT
# PET_MASTER_ADDR
# WORLD_SIZE
# RANK
# PET_NPROC_PER_NODE
# PET_NNODES
# PET_NODE_RANK