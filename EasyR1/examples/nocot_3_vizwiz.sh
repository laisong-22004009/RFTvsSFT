#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/NO-COT_scienceqa_textvqa/global_step_67/actor/huggingface  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/VizWiz/train/train-00000-of-00001.parquet@train \
    data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/VizWiz/test/test-00000-of-00001.parquet@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    trainer.experiment_name=NO-COT_scienceqa_textvqa_vizwiz \
    trainer.n_gpus_per_node=8 \
    data.format_prompt=./examples/format_prompt/no_cot.jinja \
    worker.reward.reward_function=./examples/reward_function/math.py:no_cot \
    worker.actor.fsdp.torch_dtype=bf16 \
    trainer.total_epochs=1


# nohup bash /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/nocot_3_vizwiz.sh > /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/kl_logs/nocot_3_vizwiz.log 2>&1 &


# python3 scripts/model_merger.py --local_dir checkpoints/mllm_cl/NO-COT_scienceqa_textvqa/global_step_67/actor