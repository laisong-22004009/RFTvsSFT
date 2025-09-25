#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/KL_scienceqa_textvqa_vizwiz_gqa_geo_pathvqa/global_step_38/actor/huggingface  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=flaviagiammarino/vqa-rad@train \
    data.val_files=flaviagiammarino/vqa-rad@test \
    data.prompt_key=question \
    data.image_key=image \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=KL_scienceqa_textvqa_vizwiz_gqa_geo_pathvqa_vqarad \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=10


# python3 scripts/model_merger.py --local_dir checkpoints/mllm_cl/KL_scienceqa_textvqa_vizwiz_gqa_geo_pathvqa/global_step_38/actor
# nohup bash /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/6_vqarad.sh > /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/kl_logs/6_vqarad.log 2>&1 &


sftp ZHAO_Haohan@