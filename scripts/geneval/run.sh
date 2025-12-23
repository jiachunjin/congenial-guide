accelerate launch \
runner/geneval/geneval_generation.py \
--exp_dir /inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/ssd/jjc/experiment/mcq_gen/1219_newdataloader \
--step 110000 \
--metadata_file evaluation/generation/evaluation_metadata.jsonl \
--outdir asset/geneval