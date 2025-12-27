accelerate launch \
runner/geneval/geneval_generation.py \
--exp_dir /inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/experiment/hdd_exp/1227_t2i_with_recA \
--step 5000 \
--metadata_file evaluation/generation/geneval/evaluation_metadata.jsonl \
--outdir asset/geneval