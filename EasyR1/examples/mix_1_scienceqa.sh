#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/scratch/esg8sdce/pre_trained_models/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/home/esg8sdce/esg8sdceuser01/project/rollout_sft/pass_dataset/textvqa/train-00000-of-00001.parquet@train \
    data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/TextVQA/test/test-00000-of-00001.parquet@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    worker.ref.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=Pass_scienceqa_textvqa \
    trainer.n_gpus_per_node=8 \
    worker.actor.fsdp.torch_dtype=bf16 \
    trainer.total_epochs=1

