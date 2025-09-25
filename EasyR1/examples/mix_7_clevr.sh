#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/checkpoints/mllm_cl/Pass_scienceqa_textvqa_vizwiz_gqa_geo_pathvqa/global_step_32/actor/huggingface # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/home/esg8sdce/esg8sdceuser01/project/rollout_sft/pass_dataset/clevr/train-00000-of-00001.parquet@train \
    data.val_files=/home/esg8sdce/esg8sdceuser01/project/mllm_cl/cl_datasets/clevr/test/test_dataset.parquet@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    trainer.experiment_name=Pass_scienceqa_textvqa_vizwiz_gqa_geo_pathvqa_clevr \
    trainer.n_gpus_per_node=8 \
    worker.actor.fsdp.torch_dtype=bf16 \
    data.prompt_key=problem \
    data.image_key=image \
    trainer.total_epochs=3


# nohup bash /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/mix_7_clevr.sh > /home/esg8sdce/esg8sdceuser01/mllm_grpo/EasyR1/examples/kl_logs/pass_7_clevr.log 2>&1 &

# python3 scripts/model_merger.py --local_dir checkpoints/mllm_cl/Pass_scienceqa_textvqa_vizwiz_gqa_geo_pathvqa/global_step_32/actor



