#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/NOKL_scienceqa_textvqa_vizwiz/global_step_80/actor/huggingface  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/GQA/train/train-00000-of-00001.parquet@train \
    data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/GQA/test/test-00000-of-00001.parquet@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    worker.ref.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=NOKL_scienceqa_textvqa_vizwiz_gqa \
    trainer.n_gpus_per_node=8 \
    worker.actor.fsdp.torch_dtype=bf16 \
    algorithm.disable_kl=true \
    algorithm.use_kl_loss=false \
    trainer.total_epochs=1


# nohup bash /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/nokl_4_gqa.sh > /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/kl_logs/nokl_4_gqa.log 2>&1 &


# python3 scripts/model_merger.py --local_dir checkpoints/mllm_cl/NOKL_scienceqa_textvqa_vizwiz_gqa/global_step_120/actor
