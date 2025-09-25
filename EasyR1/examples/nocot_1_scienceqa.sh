#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/ScienceQA/train/train-00000-of-00001.parquet@train \
    data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/ScienceQA/test/test-00000-of-00001.parquet@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    trainer.experiment_name=NO-COT_scienceqa \
    trainer.n_gpus_per_node=8 \
    data.format_prompt=./examples/format_prompt/no_cot.jinja \
    worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
    trainer.total_epochs=1 \
    trainer.val_before_train=false


# nohup bash /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/nocot_1_scienceqa.sh > /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/kl_logs/nocot_1_scienceqa.log 2>&1 &
