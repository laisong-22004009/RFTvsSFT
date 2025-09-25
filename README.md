
# Reinforcement Fine-Tuning Naturally Mitigates Forgetting in Continual Post-Training

This repository contains the official implementation for our submitted paper, "**Reinforcement Fine-Tuning Naturally Mitigates Forgetting in Continual Post-Training**".

In this work, we conduct a comprehensive comparison between Supervised Fine-tuning (SFT) and Reinforcement Fine-tuning (RFT) in a continual post-training setting for Multimodal Large Language Models (MLLMs). Our key finding is that **RFT inherently resists catastrophic forgetting and preserves general model capabilities**, outperforming SFT without requiring explicit continual learning strategies.

Our implementations are built upon two fantastic open-source frameworks:
- **[EasyR1](https://github.com/hiyouga/EasyR1)** for our Reinforcement Fine-Tuning (RFT) experiments.
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** for our Supervised Fine-Tuning (SFT) baselines.

We are deeply grateful to their authors for their invaluable contributions to the community.

## Overview of Our Findings

- **RFT Resists Forgetting:** RFT-based methods maintain performance on old tasks and achieve results comparable to multi-task learning.
- **RFT Preserves General Abilities:** Unlike SFT, which degrades performance on benchmarks like MMMU and MMLU-Pro, RFT preserves and even enhances the base model's general knowledge.
- **RIF-RFT:** We propose a **R**ollout-based **I**nstance **F**iltering method for RFT to improve training stability and efficiency.

## Setup and Installation

###  Environment Setup
We recommend using Conda to manage the environment.
```bash
conda create -n cpt_rft python=3.10
conda activate cpt_rft
```

###  Install Required Frameworks
Our experiments require both `LLaMA-Factory` and `EasyR1`. We have included them as submodules for convenience.

```bash
# Initialize and clone the submodules
git submodule update --init --recursive

# Install LLaMA-Factory
cd LLaMA-Factory
pip install -e .
cd ..

# Install EasyR1
cd EasyR1
pip install -e .
cd ..
```

> **CUDA Version Compatibility:**
> Our experiments were conducted with specific CUDA versions for each framework. Please ensure your environment is compatible. While other versions might work, using these is recommended for exact replication.
> - **LLaMA-Factory**: Tested with `CUDA 12.1`.
> - **EasyR1**: Tested with `CUDA 12.4`.

## Data Preparation

We provide a Jupyter notebook, `dataproc.ipynb`, to automate the downloading and formatting of datasets used in our continual learning benchmark.

## Reproducing Our Results

All scripts and configurations are provided to replicate the key findings of our paper.

### Supervised Fine-Tuning (SFT)

SFT experiments are managed using `LLaMA-Factory`. The configuration files for our continual learning sequence are located in `LLaMA-Factory/examples/mllm_cl/`.


### Reinforcement Fine-Tuning (RFT)

RFT experiments  are implemented with `EasyR1`. The main training scripts are located in `EasyR1/examples/`. 
