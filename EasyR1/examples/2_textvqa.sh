#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/home/esg8sdce/esg8sdceuser01/project/EasyR1/checkpoints/easy_r1/qwen2_5_vl_7b_scienceqa_grpo/global_step_90/actor/huggingface  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/TextVQA/train/train-00000-of-00001.parquet@train \
    data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/TextVQA/test/test-00000-of-00001.parquet@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    trainer.experiment_name=CL_oneRef_scienceqa_textvqa \
    trainer.n_gpus_per_node=8 \
    worker.actor.fsdp.torch_dtype=bf16 \
    trainer.total_epochs=1



# nohup bash /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/2_textvqa.sh > /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/kl_logs/oneRef_2_textvqa.log 2>&1 &

# python3 scripts/model_merger.py --local_dir checkpoints/mllm_cl/Mix_scienceqa_textvqa/global_step_33/actor

# /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/Mix_scienceqa_textvqa/global_step_33/actor/huggingface


# python3 scripts/model_merger.py --local_dir checkpoints/model_merge/AD_grpo/global_step_117/actor
# python3 scripts/model_merger.py --local_dir checkpoints/model_merge/SCI_grpo/global_step_60/actor