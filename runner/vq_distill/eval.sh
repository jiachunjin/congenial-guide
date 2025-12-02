EXPDIR=1202_mlp_16_8B
STEP=14000
python runner/vq_distill/eval.py \
--exp_dir /inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/experiment/vq_llava_distill/${EXPDIR} \
--step ${STEP} \
--device cuda:0
python evaluation/understanding/mme/calculation.py --results_dir /inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/codebase/congenial-guide/evaluation/understanding/mme/${EXPDIR}_${STEP}