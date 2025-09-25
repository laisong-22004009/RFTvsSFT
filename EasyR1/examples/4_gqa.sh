#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/CL_oneRef_scienceqa_textvqa/global_step_135/actor/huggingface  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/GQA/train/train-00000-of-00001.parquet@train \
    data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/GQA/test/test-00000-of-00001.parquet@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    trainer.experiment_name=CL_oneRef_scienceqa_textvqa_vizwiz_gqa \
    trainer.n_gpus_per_node=8 \
    worker.actor.fsdp.torch_dtype=bf16 \
    trainer.total_epochs=1


# nohup bash /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/nocot_3_vizwiz.sh > /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/kl_logs/nocot_3_vizwiz.log 2>&1 &


# python3 scripts/model_merger.py --local_dir checkpoints/mllm_cl/CL_oneRef_scienceqa_textvqa/global_step_135/actor
