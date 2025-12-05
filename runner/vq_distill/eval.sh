# EXPDIR=1202_mlp_16_8B
# EXPDIR=1202_vq_mlp_65536_4B
# EXPDIR=1203_multivq_mlp_4B_128_8x8192
# EXPDIR=1203_multivq_mlp_4B_256_8x2048
EXPDIR=1204_mcq_attn_4B_256_8x2048

for STEP in 100000; do
    echo "Running evaluation for step ${STEP}..."
    python runner/vq_distill/eval.py \
    --exp_dir /inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/experiment/vq_llava_distill/${EXPDIR} \
    --step ${STEP} \
    --device cuda:0

    python evaluation/understanding/mme/calculation.py --results_dir /inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/codebase/congenial-guide/evaluation/understanding/mme/${EXPDIR}_${STEP} > evaluation/understanding/mme/${EXPDIR}_${STEP}/_results.txt
done