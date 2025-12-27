#!/bin/bash
export NCCL_IB_DISABLE=0

PET_NPROC_PER_NODE=${PET_NPROC_PER_NODE:-8}
PET_NNODES=${PET_NNODES:-2}
PET_NODE_RANK=${PET_NODE_RANK:-0}
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
  runner/sft/t2i_with_recA.py \
  --config config/sft/T2I_with_recA.yaml
