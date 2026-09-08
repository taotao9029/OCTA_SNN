"""SNN V1：在固定嵌套五折上增强时空建模和视频级预测稳定性。

该入口复用 train_snn.py 的数据校验、固定 split manifest、训练循环和正式
清单逻辑，仅替换 V1 明确需要改进的配置、数据增强、网络及视频聚合。outer
test 仍然只用于一次最终评估，不参与模型、轮数或阈值选择。
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import train_snn as base


MODEL_VARIANT = "snn_v1"
SAVE_ROOT = os.path.join(base.SCRIPT_DIR, "output_v1")
TIME_BINS = 24
VOXEL_H = 48
VOXEL_W = 80
MAX_EVENTS = 160_000
CACHE_VERSION = (
    f"v2_t{TIME_BINS}_h{VOXEL_H}_w{VOXEL_W}_e{MAX_EVENTS}"
)
CACHE_ROOT = os.path.join(
    base.SCRIPT_DIR,
    "output",
    "voxel_cache",
    CACHE_VERSION,
)

BATCH_SIZE = 8
MAX_SEGMENTS_PER_VIDEO = 6
EPOCHS = 100
EARLY_STOP_PATIENCE = 18
LEARNING_RATE = 1.5e-4
WEIGHT_DECAY = 7.5e-4
DROPOUT = 0.20
READOUT_HIDDEN = 160
LABEL_SMOOTHING = 0.03
FINAL_SEEDS = 5
VIDEO_AGGREGATION = "trimmed_logit_mean"

HORIZONTAL_FLIP_PROBABILITY = 0.50
TEMPORAL_MASK_PROBABILITY = 0.20
INTENSITY_JITTER = 0.10
RESIDUAL_SCALE = 2.0 ** -0.5

_BaseEventVoxelDataset = base.EventVoxelDataset
_BaseConvSNN = base.ConvSNN
_ORIGINAL_BUILD_CONFIG_SNAPSHOT = base.build_config_snapshot


class EventVoxelDatasetV1(_BaseEventVoxelDataset):
    """读取共享体素缓存，并且只对训练样本执行轻量增强。"""

    def __init__(self, samples, train=False, cache_root=None):
        super().__init__(
            samples,
            cache_root=CACHE_ROOT if cache_root is None else cache_root,
        )
        self.train = bool(train)

    def __getitem__(self, index):
        voxel, label, video_key = super().__getitem__(index)
        if not self.train:
            return voxel, label, video_key

        voxel = voxel.clone()
        if torch.rand(()) < HORIZONTAL_FLIP_PROBABILITY:
            voxel = torch.flip(voxel, dims=(-1,))

        gain = 1.0 + (
            torch.rand((), dtype=voxel.dtype) * 2.0 - 1.0
        ) * INTENSITY_JITTER
        voxel = torch.clamp(voxel * gain, min=0.0, max=1.0)

        if (
            voxel.shape[0] > 1
            and torch.rand(()) < TEMPORAL_MASK_PROBABILITY
        ):
            time_index = int(torch.randint(voxel.shape[0], size=(1,)).item())
            voxel[time_index] = 0.0
        return voxel, label, video_key


def make_loader_v1(samples, seed, train):
    dataset = EventVoxelDatasetV1(samples, train=train)
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs = {
        "dataset": dataset,
        "batch_size": base.BATCH_SIZE,
        "num_workers": base.NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": base.seed_worker,
        "generator": generator,
    }
    if base.NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
    if train:
        kwargs["sampler"] = base.VideoBalancedSampler(
            samples,
            base.MAX_SEGMENTS_PER_VIDEO,
            seed,
        )
    else:
        kwargs["shuffle"] = False
    return DataLoader(**kwargs)


class ConvSNNV1(_BaseConvSNN):
    """带脉冲残差和时序注意力读出的卷积 SNN。"""

    def __init__(self, beta=None, threshold=None):
        super().__init__(
            beta=base.LIF_BETA if beta is None else beta,
            threshold=base.LIF_THRESHOLD if threshold is None else threshold,
        )
        channel1, channel2, channel3 = base.SNN_CHANNELS
        self.skip2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channel1, channel2, 1, bias=False),
            nn.GroupNorm(8, channel2),
        )
        self.skip3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channel2, channel3, 1, bias=False),
            nn.GroupNorm(12, channel3),
        )

        temporal_features = channel3 * 2
        attention_hidden = max(channel3 // 2, 32)
        self.temporal_norm = nn.LayerNorm(temporal_features)
        self.temporal_attention = nn.Sequential(
            nn.Linear(temporal_features, attention_hidden),
            nn.GELU(),
            nn.Dropout(base.DROPOUT * 0.5),
            nn.Linear(attention_hidden, 1),
        )

        readout_features = channel3 * 6 + 6
        self.classifier = nn.Sequential(
            nn.LayerNorm(readout_features),
            nn.Linear(readout_features, base.READOUT_HIDDEN),
            nn.GELU(),
            nn.Dropout(base.DROPOUT),
            nn.Linear(base.READOUT_HIDDEN, 2),
        )

    def forward(self, voxel_sequence):
        mem1 = None
        mem2 = None
        mem3 = None
        spike_features = []
        membrane_features = []
        for time_index in range(voxel_sequence.shape[1]):
            frame = voxel_sequence[:, time_index]
            current1 = self.norm1(self.conv1(frame))
            spike1, mem1 = base.lif_step(
                current1,
                mem1,
                self.beta,
                self.threshold,
            )

            current2 = self.norm2(self.conv2(spike1))
            current2 = (
                current2 + self.skip2(spike1)
            ) * RESIDUAL_SCALE
            spike2, mem2 = base.lif_step(
                current2,
                mem2,
                self.beta,
                self.threshold,
            )

            current3 = self.norm3(self.conv3(spike2))
            current3 = (
                current3 + self.skip3(spike2)
            ) * RESIDUAL_SCALE
            spike3, mem3 = base.lif_step(
                current3,
                mem3,
                self.beta,
                self.threshold,
            )
            spike_features.append(self.pool(spike3).flatten(1))
            membrane_features.append(self.pool(mem3).flatten(1))

        spike_sequence = torch.stack(spike_features, dim=1)
        membrane_sequence = torch.stack(membrane_features, dim=1)
        temporal_sequence = torch.cat(
            [spike_sequence, membrane_sequence],
            dim=2,
        )
        temporal_normalized = self.temporal_norm(temporal_sequence)
        attention = torch.softmax(
            self.temporal_attention(temporal_normalized),
            dim=1,
        )
        attended_features = torch.sum(
            attention * temporal_normalized,
            dim=1,
        )

        input_activity = voxel_sequence.mean(dim=(-1, -2))
        activity_features = torch.cat(
            [
                input_activity.mean(dim=1),
                input_activity.amax(dim=1),
                input_activity.std(dim=1, unbiased=False),
            ],
            dim=1,
        )
        readout = torch.cat(
            [
                spike_sequence.mean(dim=1),
                spike_sequence.amax(dim=1),
                membrane_sequence.mean(dim=1),
                membrane_sequence.std(dim=1, unbiased=False),
                attended_features,
                activity_features,
            ],
            dim=1,
        )
        return self.classifier(readout)


def aggregate_video_predictions_v1(records):
    """用去极值 logit 均值降低单个增强片段的过度置信影响。"""
    segment_frame = pd.DataFrame(records)
    if segment_frame.empty:
        raise ValueError("预测记录为空")
    label_counts = segment_frame.groupby("video_key")["label"].nunique()
    if label_counts.max() > 1:
        bad_keys = label_counts[label_counts.gt(1)].index.tolist()
        raise ValueError(
            f"预测记录中同一视频存在多个标签：{bad_keys[:10]}"
        )

    probabilities = np.clip(
        segment_frame["probability"].to_numpy(np.float64),
        1e-6,
        1.0 - 1e-6,
    )
    segment_frame = segment_frame.copy()
    segment_frame["logit"] = np.log(
        probabilities / (1.0 - probabilities)
    )

    rows = []
    for video_key, group in segment_frame.groupby(
        "video_key",
        sort=True,
    ):
        logits = np.sort(group["logit"].to_numpy(np.float64))
        if len(logits) >= 5:
            logits = logits[1:-1]
        mean_logit = float(np.mean(logits))
        probability = 1.0 / (1.0 + np.exp(-mean_logit))
        rows.append(
            {
                "video_key": str(video_key),
                "label": int(group["label"].iloc[0]),
                "probability": float(probability),
            }
        )
    return pd.DataFrame(rows)


def build_config_snapshot_v1():
    config = _ORIGINAL_BUILD_CONFIG_SNAPSHOT()
    config.update(
        {
            "source_file": os.path.abspath(__file__),
            "base_source_file": os.path.abspath(base.__file__),
            "model_variant": MODEL_VARIANT,
            "video_aggregation": VIDEO_AGGREGATION,
            "horizontal_flip_probability": (
                HORIZONTAL_FLIP_PROBABILITY
            ),
            "temporal_mask_probability": TEMPORAL_MASK_PROBABILITY,
            "intensity_jitter": INTENSITY_JITTER,
            "residual_scale": RESIDUAL_SCALE,
            "temporal_attention": True,
            "spike_residual": True,
        }
    )
    return config


def save_model_bundle_v1(
    bundle_path,
    state_dicts,
    outer_fold,
    selected_epoch,
    threshold,
    test_video_keys,
):
    bundle = {
        "format_version": 2,
        "model_class": "ConvSNNV1",
        "model_variant": MODEL_VARIANT,
        "model_state_dicts": state_dicts,
        "model_config": {
            "time_bins": base.TIME_BINS,
            "voxel_h": base.VOXEL_H,
            "voxel_w": base.VOXEL_W,
            "orig_h": base.ORIG_H,
            "orig_w": base.ORIG_W,
            "lif_beta": base.LIF_BETA,
            "lif_threshold": base.LIF_THRESHOLD,
            "dropout": base.DROPOUT,
            "snn_channels": list(base.SNN_CHANNELS),
            "readout_hidden": base.READOUT_HIDDEN,
            "label_smoothing": base.LABEL_SMOOTHING,
            "max_events": base.MAX_EVENTS,
            "cache_version": base.CACHE_VERSION,
            "video_aggregation": base.VIDEO_AGGREGATION,
            "inner_seeds": base.INNER_SEEDS,
            "final_seeds": base.FINAL_SEEDS,
            "checkpoint_objective": base.CHECKPOINT_OBJECTIVE,
            "horizontal_flip_probability": (
                HORIZONTAL_FLIP_PROBABILITY
            ),
            "temporal_mask_probability": TEMPORAL_MASK_PROBABILITY,
            "intensity_jitter": INTENSITY_JITTER,
            "residual_scale": RESIDUAL_SCALE,
            "temporal_attention": True,
            "spike_residual": True,
        },
        "outer_fold": int(outer_fold),
        "selected_epoch": int(selected_epoch),
        "threshold": float(threshold),
        "threshold_objective": base.THRESHOLD_OBJECTIVE,
        "seeds": [
            int(base.SEED + outer_fold * 10000 + index)
            for index in range(base.FINAL_SEEDS)
        ],
        "test_video_keys": sorted(
            str(key) for key in test_video_keys
        ),
    }
    torch.save(bundle, bundle_path)


def configure_v1():
    """将复用入口所依赖的全局配置切换为 V1。"""
    base.SAVE_ROOT = SAVE_ROOT
    base.TIME_BINS = TIME_BINS
    base.VOXEL_H = VOXEL_H
    base.VOXEL_W = VOXEL_W
    base.MAX_EVENTS = MAX_EVENTS
    base.CACHE_VERSION = CACHE_VERSION
    base.CACHE_ROOT = CACHE_ROOT
    base.BATCH_SIZE = BATCH_SIZE
    base.MAX_SEGMENTS_PER_VIDEO = MAX_SEGMENTS_PER_VIDEO
    base.EPOCHS = EPOCHS
    base.EARLY_STOP_PATIENCE = EARLY_STOP_PATIENCE
    base.LEARNING_RATE = LEARNING_RATE
    base.WEIGHT_DECAY = WEIGHT_DECAY
    base.DROPOUT = DROPOUT
    base.READOUT_HIDDEN = READOUT_HIDDEN
    base.LABEL_SMOOTHING = LABEL_SMOOTHING
    base.FINAL_SEEDS = FINAL_SEEDS
    base.VIDEO_AGGREGATION = VIDEO_AGGREGATION

    base.EventVoxelDataset = EventVoxelDatasetV1
    base.ConvSNN = ConvSNNV1
    base.make_loader = make_loader_v1
    base.aggregate_video_predictions = aggregate_video_predictions_v1
    base.build_config_snapshot = build_config_snapshot_v1
    base.save_model_bundle = save_model_bundle_v1


def main():
    configure_v1()
    print(
        f"启动 {MODEL_VARIANT}: output={base.SAVE_ROOT}, "
        f"voxel={base.TIME_BINS}x{base.VOXEL_H}x{base.VOXEL_W}, "
        f"segments_per_video={base.MAX_SEGMENTS_PER_VIDEO}, "
        f"final_seeds={base.FINAL_SEEDS}"
    )
    base.main()


if __name__ == "__main__":
    main()
