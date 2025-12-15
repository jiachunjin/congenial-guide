#!/bin/bash

# 如果没有设置 CUDA_VISIBLE_DEVICES，自动检测所有可用GPU
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # 检测GPU数量
    num_gpus=$(nvidia-smi -L | wc -l)
    # 生成GPU列表: 0,1,2,3,4,5,6,7
    gpu_list=$(seq -s, 0 $((num_gpus-1)))
    echo "自动检测到 $num_gpus 张GPU，使用: $gpu_list"
else
    gpu_list="$CUDA_VISIBLE_DEVICES"
    echo "使用指定的GPU: $gpu_list"
fi

IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python runner/vq_distill/eval_vqav2.py \
        --exp-dir /inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/experiment/vq_llava_distill/1203_multivq_mlp_4B_256_8x2048 \
        --step 85000 \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait

