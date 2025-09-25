#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/KL_scienceqa_textvqa_vizwiz_gqa_geo/global_step_40/actor/huggingface  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=flaviagiammarino/path-vqa@train \
    data.val_files=flaviagiammarino/path-vqa@test \
    data.prompt_key=question \
    data.image_key=image \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=KL_scienceqa_textvqa_vizwiz_gqa_geo_pathvqa \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=1

# python3 scripts/model_merger.py --local_dir checkpoints/mllm_cl/KL_scienceqa_textvqa_vizwiz_gqa_geo/global_step_40/actor
# nohup bash /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/5_pathvqa.sh > /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/kl_logs/5_pathvqa.log 2>&1 &


