# 🚁 AIR-VLA: Vision-Language-Action Systems for Aerial Manipulation

Official repository for [**AIR-VLA**](https://arxiv.org/abs/2601.21602), the first Vision-Language-Action (VLA) benchmark and dataset specifically designed for Aerial Manipulation Systems (AMS).

While existing VLA models excel in ground-based tasks, AIR-VLA bridges the gap to 3D aerial environments by addressing unique challenges: floating-base dynamics, strong UAV-manipulator coupling, and multi-step long-horizon operational tasks.

## 🌟 Key Features

* **Pioneering Aerial Benchmark:** The first standardized testbed for evaluating VLA and VLM models on aerial manipulation.

* **Comprehensive Task Suites:** Evaluates policies across 4 dimensions: *Base Manipulation, Object & Spatial Understanding, Semantic Understanding, and Long-Horizon Planning*.

* **High-Quality Multimodal Dataset:** 3,000 manually teleoperated demonstrations featuring diverse environments (home, factory, outdoor), multi-view RGB-D, and high-frequency proprioception.

## 🗂️ Dataset & Assets

**Current Status:** 🚧 **Phase 1 Release**
We are currently releasing the first subset of our HDF5 dataset (near 1000 episodes). The complete 3,000-episode dataset and further asset updates will be uploaded continuously.

Due to GitHub's file size limits, the massive HDF5 datasets are hosted on Hugging Face:

* **HDF5 Dataset:** \[https://huggingface.co/datasets/SpencerSon2001/AIR-VLA_hdf5_datasets\]


## 🛠️ Installation & Setup

### 1. Prerequisites

This project requires **NVIDIA Omniverse Isaac Sim 5.0.0**. Please install it via the Omniverse Launcher.

### 2. Python Dependencies

Due to physics engine dependencies, we use the embedded Python environment provided natively by Isaac Sim. Install the required packages using Isaac Sim's python executable:

```bash
# Locate your Isaac Sim installation path (typically ~/.local/share/ov/pkg/isaac-sim-5.0.0)
export ISAAC_PYTHON=~/.local/share/ov/pkg/isaac-sim-5.0.0/python.sh

# Install requirements
$ISAAC_PYTHON -m pip install -r requirements.txt
```

## 🚀 Usage Guide

All scripts must be executed using the Isaac Sim Python executable (`$ISAAC_PYTHON`).

### 1. Data Teleoperation & Recordinghttps://huggingface.co/datasets/SpencerSon2001/AIR-VLA_hdf5_datasets

Record new expert demonstrations using keyboard teleoperation in the Isaac Sim environment.

```bash
$ISAAC_PYTHON scripts/record.py \
    --usd_path "environments/manipulation.usd" \
    --dataset_root "data/dataset_raw/object/task_name"
```

### 2. Data Conversion (Raw to HDF5)

Convert raw recorded episodes (CSV actions + MP4 videos) into the standard HDF5 format used for VLA model training.

```bash
$ISAAC_PYTHON scripts/convert_raw2hdf5.py
```

### 3. Policy Evaluation

Evaluate trained VLA models in the physics-based simulation. The script communicates with an external policy inference server via WebSockets.

```bash
$ISAAC_PYTHON scripts/evaluation.py \
    --server_url "ws://127.0.0.1:8000" \
    --usd_root "./environments" \
    --result_root "./evaluation_results"
```

## 📝 Citation

If you find our code, environment, or dataset useful for your research, please consider citing our paper:

```bibtex
@misc{sun2026airvlavisionlanguageactionsystemsaerial,
      title={AIR-VLA: Vision-Language-Action Systems for Aerial Manipulation}, 
      author={Jianli Sun and Bin Tian and Qiyao Zhang and Chengxiang Li and Zihan Song and Zhiyong Cui and Yisheng Lv and Yonglin Tian},
      year={2026},
      eprint={2601.21602},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={[https://arxiv.org/abs/2601.21602](https://arxiv.org/abs/2601.21602)}, 
}
```
