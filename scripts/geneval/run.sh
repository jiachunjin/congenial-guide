accelerate launch \
runner/geneval/geneval_generation.py \
--exp_dir /inspire/hdd/project/advanced-machine-learning-and-deep-learning-applications/yangyi-253108120173/home_jjc/experiment/mcq_gen/1225_sft_blip3o \
--step 2000 \
--metadata_file evaluation/generation/geneval/evaluation_metadata.jsonl \
--outdir asset/geneval