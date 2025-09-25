#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/home/esg8sdce/esg8sdceuser01/project/EasyR1/checkpoints/easy_r1/qwen2_5_vl_7b_scienceqa_textvqa_vizwiz_GQA_grpo/global_step_312/actor/huggingface  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=NEW_scienceqa_textvqa_vizwiz_gqa_geo \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=10


# nohup bash /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/qwen2_5_vl_7b_geo3k_grpo.sh > /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/kl_logs/qwen2_5_vl_7b_geo3k_grpo.log 2>&1 &
