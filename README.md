# Voxel-SNN: 时空脉冲神经网络用于 OCTA 脑卒中视频分类

本仓库实现 Voxel-SNN（Spatiotemporal Voxelized Spiking Neural Network），面向小鼠眼底 OCTA 视频的脑卒中二分类。模型将异步事件流编码为时空体素序列，利用卷积脉冲神经元提取微循环空间模式，并结合脉冲残差连接与时序注意力完成视频级读出。

## 方法概览

`视频 → 事件流 → 血管区域过滤 → 时空体素 → 卷积 SNN → 脉冲残差/时序注意力 → 视频级概率`。

- 24 个时间 bin，空间尺寸 48×80，单段最多 160,000 个事件。
- 训练增强：水平翻转、时间片随机遮挡、强度扰动。
- 视频级聚合采用 `trimmed_logit_mean`。
- 固定 outer 5-fold，阈值仅由 inner OOF 确定；outer test 只用于最终评估。

## 环境安装

推荐 Linux、NVIDIA GPU、CUDA 12.8：

```bash
conda env create -f yaml/environment.yaml
conda activate snn
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

也可执行 `pip install -r requirements.txt`。无 GPU 时可使用 CPU 版 PyTorch，但训练会明显变慢。

## 数据要求

事件 CSV 必须包含 `timestamp(s)`, `x`, `y`, `polarity` 字段，目录结构为：

```text
output_filter_event/
├── 0/   # 健康
└── 1/   # 脑卒
```

五折划分使用 `split_manifest.csv`；不要在测试阶段重新随机划分。增强视图应保持统一的 `video_key` 命名。

## 运行步骤

```bash
python Video2Events.py       # 视频转事件流
python seg.py                # 生成血管分割区域
python event_filter.py       # 过滤血管区域事件
python train_snnV1.py        # 固定五折训练
python train_group_outV1.py  # leave-date-out 外部验证
python audit_outputs.py      # 生成审计清单与哈希
```

运行前请检查脚本中的数据根目录、输出目录和 `split_manifest.csv` 路径。

## 输出文件

结果位于 `output_v1/`：`snn_model_bundle.pt` 为模型；`standardized_predictions.csv` 为视频级预测；`predictions_original_only.csv` 和 `predictions_tta.csv` 为统一结果接口；`sample_view_manifest.csv` 记录样本、视图、fold 和增强信息；`run_manifest.json` 记录配置 SHA-256、训练清单 SHA-256 与 Git commit；`checkpoint_sha256.csv` 记录模型哈希；`runtime_report.json` 记录运行设备；`requirements-lock.txt` 锁定实际依赖。

## 可复现性规范

报告 ROC-AUC、PR-AUC、ACC、Balanced Accuracy、F1、敏感度和特异度。禁止使用 outer test 选择 epoch 或阈值；日期外部验证使用独立 `leave_date.csv` 并单独报告。

## 目录结构

`train_snn*.py`：五折训练；`train_group_out*.py`：日期外部验证；`Video2Events.py`、`event_filter.py`：预处理；`feature_out/`：特征与划分；`yaml/environment.yaml`：环境；`output_v1/`：模型、预测和审计产物。

## Citation

使用本代码时，请注明模型版本、数据划分清单和 Git commit。

## Copyright

Copyright (c) 2025 Taotao9029. All rights reserved.
