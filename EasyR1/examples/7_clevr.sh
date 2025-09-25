#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/KL_scienceqa_textvqa_vizwiz_gqa_geo_pathvqa/global_step_38/actor/huggingface  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/clevr/train/train_dataset.parquet@train \
    data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/clevr/test/test_dataset.parquet@test \
    data.prompt_key=problem \
    data.image_key=image \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=NEW_PPO_clip_scienceqa_textvqa_vizwiz_gqa_geo_pathvqa_clevr \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=3 


# python3 scripts/model_merger.py --local_dir checkpoints/mllm_cl/NEW_scienceqa_textvqa_vizwiz_gqa_geo_pathvqa_clevr/global_step_45/actor
# nohup bash /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/7_clevr.sh > /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/kl_logs/7_clver.log 2>&1 &

# python3 scripts/model_merger.py --local_dir checkpoints/mllm_cl/NEW_scienceqa_textvqa_vizwiz_gqa_geo_pathvqa_clevr/global_step_45/actor