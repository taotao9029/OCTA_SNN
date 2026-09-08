# Voxel-SNN: A Spatiotemporal Spiking Neural Network for OCTA Stroke Video Classification

[中文说明 / Chinese documentation](#中文说明)

Voxel-SNN is a reproducible spatiotemporal spiking neural network for binary stroke classification from mouse fundus OCTA videos. Asynchronous events are converted into voxel sequences, processed by convolutional LIF layers, spike-residual connections, and temporal-attention readout. The repository provides nested five-fold cross-validation, leave-date-out external validation, and auditable experiment manifests.

## Method

`video → event stream → vessel-region filtering → voxel sequence → convolutional SNN → spike residual + temporal attention → video-level probability`.

The V1 configuration uses 24 temporal bins, a 48×80 voxel grid, a 160,000-event cap per segment, light training augmentation, and trimmed logit-mean video aggregation. Outer-test data are never used to select epochs or thresholds; thresholds are determined from inner OOF predictions.

## Installation

```bash
conda env create -f yaml/environment.yaml
conda activate snn
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

CPU-only PyTorch is supported but substantially slower.

## Data format

Each event CSV must contain `timestamp(s)`, `x`, `y`, and `polarity`. Expected class layout:

```text
output_filter_event_rename/
├── 0/   # control / healthy
└── 1/   # stroke
```

Use the fixed `feature_out/split_manifest.csv`; do not regenerate folds during evaluation.

## Run

```bash
python Video2Events.py
python seg.py
python event_filter.py
python train_snnV1.py
python train_group_outV1.py
python audit_outputs.py
```

Check data paths, output paths, and the split-manifest path before training.

## Outputs and reproducibility

`output_v1/` contains model bundles, standardized predictions, original-only/TTA interfaces, `sample_view_manifest.csv`, `run_manifest.json`, checkpoint SHA-256 records, `runtime_report.json`, and `requirements-lock.txt`. Report ROC-AUC, PR-AUC, accuracy, balanced accuracy, F1, sensitivity, and specificity.

## 中文说明

Voxel-SNN 面向小鼠眼底 OCTA 视频脑卒中二分类，将异步事件编码为时空体素序列，通过卷积脉冲神经元、脉冲残差连接和时序注意力完成视频级分类。项目提供固定五折嵌套交叉验证、leave-date-out 外部验证及可审计运行清单。安装、数据格式和运行命令请参阅上方英文说明。

## Citation

Please cite the associated study and report the model version, split manifest, and Git commit used for each experiment.

## Copyright

Copyright (c) 2025 Taotao9029. All rights reserved.
