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
    worker.ref.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=NOKL_scienceqa \
    trainer.n_gpus_per_node=8 \
    worker.actor.fsdp.torch_dtype=bf16 \
    algorithm.disable_kl=true \
    algorithm.use_kl_loss=false \
    trainer.total_epochs=5



# nohup bash /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/nokl_1_scienceqa.sh > /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/kl_logs/nokl_1_textvqa.log 2>&1 &

# python3 scripts/model_merger.py --local_dir checkpoints/mllm_cl/Mix_scienceqa_textvqa/global_step_33/actor

# /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/Mix_scienceqa_textvqa/global_step_33/actor/huggingface
