"""基于事件 CSV 的视频级 SNN 嵌套五折训练。

数据划分完全由 OCTA_RF/feature_out/split_manifest.csv 决定。所有增强片段
按原始 video_key 进入同一个 outer/inner fold，inner OOF 用于选择阈值和最终
训练轮数，outer test 不参与模型或阈值选择。
"""

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, Sampler


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = (
    "/home/wxy/Classification/syy/syy/strock_test/OCTA_pytorch/"
    "cpu/output_filter_event_rename"
)
SPLIT_MANIFEST_PATH = (
    "/home/wxy/Classification/syy/syy/strock_test/OCTA_pytorch/"
    "OCTA_SNN/feature_out/split_manifest.csv"
)
SAVE_ROOT = os.path.join(SCRIPT_DIR, "output")

SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 5
FINAL_SEEDS = 3

ORIG_H = 1216
ORIG_W = 1936
VOXEL_H = 48
VOXEL_W = 64
TIME_BINS = 20
MAX_EVENTS = 120_000
CSV_CHUNK_SIZE = 50_000
CACHE_VERSION = (
    f"v1_t{TIME_BINS}_h{VOXEL_H}_w{VOXEL_W}_e{MAX_EVENTS}"
)
CACHE_ROOT = os.path.join(SAVE_ROOT, "voxel_cache", CACHE_VERSION)

BATCH_SIZE = 8
NUM_WORKERS = 4
MAX_SEGMENTS_PER_VIDEO = 4
EPOCHS = 80
EARLY_STOP_PATIENCE = 15
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 5.0
LIF_BETA = 0.85
LIF_THRESHOLD = 1.0
DROPOUT = 0.25
THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)
THRESHOLD_OBJECTIVE = "balanced_acc"
VIDEO_AGGREGATION = "median"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EVENT_COLUMNS = ["timestamp(s)", "x", "y", "polarity"]
RUN_START_TIME = ""
CONFIG_SNAPSHOT_PATH = ""
CONFIG_SHA256 = ""
CODE_COMMIT = ""

EPOCH_METRIC_COLUMNS = [
    "model", "ablation", "outer_fold", "inner_fold", "seed", "epoch",
    "train_loss", "val_loss", "val_auc", "val_ap",
    "val_balanced_accuracy", "learning_rate", "checkpoint_saved",
]
RUN_MANIFEST_COLUMNS = [
    "run_id", "model", "ablation", "outer_fold", "seed", "config_file",
    "config_sha256", "training_manifest_sha256", "code_commit",
    "start_time", "end_time", "selected_epoch", "selected_threshold",
    "threshold_objective", "checkpoint_path", "relative_path", "bytes",
    "sha256",
]
PREDICTION_COLUMNS = [
    "model", "ablation", "animal_id", "label", "outer_fold",
    "probability", "fold_threshold", "prediction", "threshold_objective",
    "checkpoint_id",
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def sha256_file(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=SCRIPT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )
        commit = result.stdout.strip()
        return commit if commit else "not_available"
    except (OSError, subprocess.SubprocessError):
        return "not_available"


def build_config_snapshot():
    config_names = [
        "DATA_ROOT", "SPLIT_MANIFEST_PATH", "SAVE_ROOT", "SEED",
        "OUTER_FOLDS", "INNER_FOLDS", "FINAL_SEEDS", "ORIG_H", "ORIG_W",
        "VOXEL_H", "VOXEL_W", "TIME_BINS", "MAX_EVENTS",
        "CSV_CHUNK_SIZE", "CACHE_VERSION", "BATCH_SIZE", "NUM_WORKERS",
        "MAX_SEGMENTS_PER_VIDEO", "EPOCHS", "EARLY_STOP_PATIENCE",
        "LEARNING_RATE", "WEIGHT_DECAY", "GRAD_CLIP", "LIF_BETA",
        "LIF_THRESHOLD", "DROPOUT", "THRESHOLD_OBJECTIVE",
        "VIDEO_AGGREGATION",
    ]
    config = {"source_file": os.path.abspath(__file__)}
    for name in config_names:
        value = globals()[name]
        if isinstance(value, torch.device):
            value = str(value)
        config[name] = value
    config["threshold_grid"] = {
        "min": float(THRESHOLD_GRID.min()),
        "max": float(THRESHOLD_GRID.max()),
        "step": 0.01,
    }
    return config


def prepare_run_metadata():
    global CONFIG_SNAPSHOT_PATH, CONFIG_SHA256, CODE_COMMIT

    os.makedirs(SAVE_ROOT, exist_ok=True)
    config_path = os.path.abspath(
        os.path.join(SAVE_ROOT, "config_snapshot.json")
    )
    with open(config_path, "w", encoding="utf-8") as file_obj:
        json.dump(
            build_config_snapshot(),
            file_obj,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        file_obj.write("\n")
    CONFIG_SNAPSHOT_PATH = config_path
    CONFIG_SHA256 = sha256_file(config_path)
    CODE_COMMIT = get_git_commit()


def get_original_video_key(filename):
    key = os.path.splitext(os.path.basename(str(filename)))[0]
    return re.sub(
        r"(?:_aug|_augmentation)[-_]?\d+$",
        "",
        key,
        flags=re.IGNORECASE,
    )


def read_split_manifest(manifest_path):
    manifest_path = os.path.abspath(manifest_path)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"训练划分清单不存在：{manifest_path}")
    manifest = pd.read_csv(
        manifest_path,
        dtype={"animal_id": str, "video_key": str},
    )
    manifest.columns = manifest.columns.astype(str).str.strip()
    required = {"video_key", "label", "outer_fold", "role", "inner_fold"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"训练划分清单缺少字段：{sorted(missing)}")

    manifest["video_key"] = manifest["video_key"].astype(str).str.strip()
    manifest["role"] = manifest["role"].astype(str).str.strip()
    manifest["label"] = pd.to_numeric(
        manifest["label"], errors="raise"
    ).astype(int)
    manifest["outer_fold"] = pd.to_numeric(
        manifest["outer_fold"], errors="raise"
    ).astype(int)
    manifest["inner_fold"] = pd.to_numeric(
        manifest["inner_fold"], errors="coerce"
    )
    allowed_roles = {"outer_test", "inner_train", "inner_val"}
    unknown_roles = sorted(set(manifest["role"]) - allowed_roles)
    if unknown_roles:
        raise ValueError(f"训练划分清单包含未知 role：{unknown_roles}")
    if manifest.loc[
        manifest["role"].isin({"inner_train", "inner_val"}),
        "inner_fold",
    ].isna().any():
        raise ValueError("inner_train/inner_val 记录缺少 inner_fold")
    if manifest.duplicated(
        ["video_key", "outer_fold", "role", "inner_fold"]
    ).any():
        raise ValueError("训练划分清单包含重复记录")
    return manifest


def manifest_label_map(manifest):
    label_counts = manifest.groupby("video_key")["label"].nunique()
    bad_keys = label_counts[label_counts.ne(1)].index.tolist()
    if bad_keys:
        raise ValueError(f"划分清单中同一视频存在多个标签：{bad_keys[:10]}")
    return (
        manifest[["video_key", "label"]]
        .drop_duplicates()
        .set_index("video_key")["label"]
        .astype(int)
        .to_dict()
    )


def collect_samples(root_dir, label_map):
    samples = []
    ignored_files = []
    for label in (0, 1):
        folder = os.path.join(root_dir, str(label))
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"事件类别目录不存在：{folder}")
        for filename in sorted(os.listdir(folder)):
            if not filename.lower().endswith(".csv"):
                continue
            video_key = get_original_video_key(filename)
            if video_key not in label_map:
                ignored_files.append(os.path.join(folder, filename))
                continue
            if int(label_map[video_key]) != label:
                raise ValueError(
                    f"事件目录标签与划分清单不一致：{filename}, "
                    f"folder={label}, manifest={label_map[video_key]}"
                )
            path = os.path.join(folder, filename)
            header = pd.read_csv(path, nrows=0)
            missing = set(EVENT_COLUMNS) - set(header.columns)
            if missing:
                raise ValueError(f"事件 CSV 缺少字段 {sorted(missing)}：{path}")
            samples.append({
                "path": path,
                "filename": filename,
                "video_key": video_key,
                "label": label,
            })

    found_keys = {sample["video_key"] for sample in samples}
    missing_keys = sorted(set(label_map) - found_keys)
    if missing_keys:
        raise ValueError(f"事件目录缺少清单视频：{missing_keys[:10]}")
    if ignored_files:
        print(
            f"[INFO] 忽略 {len(ignored_files)} 个不在 split manifest 中的 CSV，"
            f"例如：{ignored_files[0]}"
        )
    return samples


def validate_split_manifest(manifest, video_meta):
    video_keys = video_meta["video_key"].astype(str)
    dataset_keys = set(video_keys)
    manifest_keys = set(manifest["video_key"])
    if dataset_keys != manifest_keys:
        missing_keys = sorted(dataset_keys - manifest_keys)
        extra_keys = sorted(manifest_keys - dataset_keys)
        raise ValueError(
            "事件数据与划分清单的视频不一致："
            f"missing={missing_keys[:10]}, extra={extra_keys[:10]}"
        )

    labels = dict(zip(video_keys, video_meta["label"].astype(int)))
    expected_labels = manifest["video_key"].map(labels)
    if not manifest["label"].eq(expected_labels).all():
        bad_keys = manifest.loc[
            ~manifest["label"].eq(expected_labels), "video_key"
        ].drop_duplicates().tolist()
        raise ValueError(f"事件数据与划分清单标签不一致：{bad_keys[:10]}")

    expected_outer = list(range(1, OUTER_FOLDS + 1))
    actual_outer = sorted(manifest["outer_fold"].unique().tolist())
    if actual_outer != expected_outer:
        raise ValueError(
            f"outer fold 应为 {expected_outer}，实际为 {actual_outer}"
        )
    outer_test = manifest[manifest["role"].eq("outer_test")]
    test_counts = outer_test["video_key"].value_counts()
    if set(test_counts.index) != dataset_keys or not test_counts.eq(1).all():
        raise ValueError("每个视频必须且只能出现于一个 outer_test fold")

    for outer_fold in expected_outer:
        fold_rows = manifest[manifest["outer_fold"].eq(outer_fold)]
        test_keys = set(
            fold_rows.loc[fold_rows["role"].eq("outer_test"), "video_key"]
        )
        train_keys = dataset_keys - test_keys
        inner_rows = fold_rows[
            fold_rows["role"].isin({"inner_train", "inner_val"})
        ]
        if set(inner_rows["video_key"]) != train_keys:
            raise ValueError(f"outer fold {outer_fold} 的训练视频集合不完整")
        expected_inner = list(range(1, INNER_FOLDS + 1))
        actual_inner = sorted(
            inner_rows["inner_fold"].astype(int).unique().tolist()
        )
        if actual_inner != expected_inner:
            raise ValueError(
                f"outer fold {outer_fold} 的 inner fold 应为 "
                f"{expected_inner}，实际为 {actual_inner}"
            )
        for inner_fold in expected_inner:
            current = inner_rows[inner_rows["inner_fold"].eq(inner_fold)]
            inner_train = set(
                current.loc[current["role"].eq("inner_train"), "video_key"]
            )
            inner_val = set(
                current.loc[current["role"].eq("inner_val"), "video_key"]
            )
            if inner_train & inner_val:
                raise ValueError(
                    f"outer fold {outer_fold} / inner fold {inner_fold} 存在重叠"
                )
            if inner_train | inner_val != train_keys:
                raise ValueError(
                    f"outer fold {outer_fold} / inner fold {inner_fold} "
                    "未完整覆盖 outer train"
                )
        val_counts = inner_rows.loc[
            inner_rows["role"].eq("inner_val"), "video_key"
        ].value_counts()
        if set(val_counts.index) != train_keys or not val_counts.eq(1).all():
            raise ValueError(
                f"outer fold {outer_fold} 的每个训练视频必须且只能验证一次"
            )


def manifest_keys(manifest, outer_fold, role, inner_fold=None):
    rows = manifest[
        manifest["outer_fold"].eq(outer_fold)
        & manifest["role"].eq(role)
    ]
    if inner_fold is not None:
        rows = rows[rows["inner_fold"].eq(inner_fold)]
    return set(rows["video_key"].astype(str))


def samples_by_keys(samples, video_keys):
    video_keys = set(video_keys)
    return [
        sample for sample in samples
        if sample["video_key"] in video_keys
    ]


def _sample_event_chunks(file_path):
    selected = np.empty((0, 4), dtype=np.float32)
    selected_priority = np.empty(0, dtype=np.uint64)
    row_offset = 0
    hash_multiplier = np.uint64(11400714819323198485)
    hash_offset = np.uint64(0x9E3779B97F4A7C15)

    for chunk in pd.read_csv(
        file_path,
        usecols=EVENT_COLUMNS,
        chunksize=CSV_CHUNK_SIZE,
    ):
        values = chunk[EVENT_COLUMNS].to_numpy(dtype=np.float32)
        indices = np.arange(
            row_offset,
            row_offset + len(values),
            dtype=np.uint64,
        )
        row_offset += len(values)
        finite = np.isfinite(values).all(axis=1)
        values = values[finite]
        indices = indices[finite]
        if not len(values):
            continue
        priorities = (indices * hash_multiplier) ^ hash_offset
        selected = np.concatenate([selected, values], axis=0)
        selected_priority = np.concatenate(
            [selected_priority, priorities], axis=0
        )
        if len(selected) > MAX_EVENTS:
            keep = np.argpartition(
                selected_priority, MAX_EVENTS - 1
            )[:MAX_EVENTS]
            selected = selected[keep]
            selected_priority = selected_priority[keep]

    if len(selected):
        selected = selected[np.argsort(selected[:, 0], kind="stable")]
    return selected


def events_to_voxel(events):
    voxel = np.zeros(
        (TIME_BINS, 2, VOXEL_H, VOXEL_W),
        dtype=np.float32,
    )
    if len(events) == 0:
        return voxel

    timestamp = events[:, 0]
    x_coord = events[:, 1]
    y_coord = events[:, 2]
    polarity = events[:, 3]
    valid = (
        np.isfinite(events).all(axis=1)
        & (x_coord >= 0)
        & (x_coord < ORIG_W)
        & (y_coord >= 0)
        & (y_coord < ORIG_H)
    )
    if not valid.any():
        return voxel
    timestamp = timestamp[valid]
    x_coord = x_coord[valid]
    y_coord = y_coord[valid]
    polarity = polarity[valid]

    time_min = float(timestamp.min())
    time_span = max(float(timestamp.max()) - time_min, 1e-6)
    time_idx = np.floor(
        (timestamp - time_min) / time_span * TIME_BINS
    ).astype(np.int64)
    x_idx = np.floor(x_coord / ORIG_W * VOXEL_W).astype(np.int64)
    y_idx = np.floor(y_coord / ORIG_H * VOXEL_H).astype(np.int64)
    channel_idx = np.where(polarity > 0, 0, 1).astype(np.int64)
    time_idx = np.clip(time_idx, 0, TIME_BINS - 1)
    x_idx = np.clip(x_idx, 0, VOXEL_W - 1)
    y_idx = np.clip(y_idx, 0, VOXEL_H - 1)
    np.add.at(voxel, (time_idx, channel_idx, y_idx, x_idx), 1.0)

    voxel = np.log1p(voxel)
    nonzero = voxel[voxel > 0]
    if len(nonzero):
        scale = max(float(np.percentile(nonzero, 99.0)), 1.0)
        voxel = np.clip(voxel / scale, 0.0, 1.0)
    return voxel.astype(np.float32)


class EventVoxelDataset(Dataset):
    def __init__(self, samples, cache_root=CACHE_ROOT):
        self.samples = list(samples)
        self.cache_root = cache_root

    def __len__(self):
        return len(self.samples)

    def _cache_path(self, sample):
        label_dir = os.path.join(self.cache_root, str(sample["label"]))
        filename = os.path.splitext(sample["filename"])[0] + ".npy"
        return os.path.join(label_dir, filename)

    def _load_voxel(self, sample):
        cache_path = self._cache_path(sample)
        if os.path.isfile(cache_path):
            return np.load(cache_path, allow_pickle=False).astype(np.float32)

        voxel = events_to_voxel(_sample_event_chunks(sample["path"]))
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        temp_path = f"{cache_path}.{os.getpid()}.tmp.npy"
        np.save(temp_path, voxel.astype(np.float16), allow_pickle=False)
        try:
            os.replace(temp_path, cache_path)
        except OSError:
            if os.path.isfile(temp_path):
                os.unlink(temp_path)
        return voxel

    def __getitem__(self, index):
        sample = self.samples[index]
        voxel = torch.from_numpy(self._load_voxel(sample))
        label = torch.tensor(int(sample["label"]), dtype=torch.long)
        return voxel, label, sample["video_key"]


class VideoBalancedSampler(Sampler):
    def __init__(self, samples, max_segments_per_video, seed):
        self.video_indices = {}
        for index, sample in enumerate(samples):
            self.video_indices.setdefault(sample["video_key"], []).append(index)
        self.max_segments_per_video = int(max_segments_per_video)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        selected = []
        for indices in self.video_indices.values():
            count = min(len(indices), self.max_segments_per_video)
            chosen = rng.choice(indices, size=count, replace=False)
            selected.extend(int(index) for index in chosen)
        rng.shuffle(selected)
        return iter(selected)

    def __len__(self):
        return sum(
            min(len(indices), self.max_segments_per_video)
            for indices in self.video_indices.values()
        )


def make_loader(samples, seed, train):
    dataset = EventVoxelDataset(samples)
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs = {
        "dataset": dataset,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if train:
        kwargs["sampler"] = VideoBalancedSampler(
            samples,
            MAX_SEGMENTS_PER_VIDEO,
            seed,
        )
    else:
        kwargs["shuffle"] = False
    return DataLoader(**kwargs)


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane_over_threshold):
        ctx.save_for_backward(membrane_over_threshold)
        return (membrane_over_threshold > 0).to(
            membrane_over_threshold.dtype
        )

    @staticmethod
    def backward(ctx, grad_output):
        (membrane_over_threshold,) = ctx.saved_tensors
        surrogate = 1.0 / (
            1.0 + 25.0 * membrane_over_threshold.abs()
        ) ** 2
        return grad_output * surrogate


def lif_step(current, membrane, beta, threshold):
    if membrane is None:
        membrane = torch.zeros_like(current)
    membrane = beta * membrane + current
    spike = SurrogateSpike.apply(membrane - threshold)
    membrane = membrane - spike.detach() * threshold
    return spike, membrane


class ConvSNN(nn.Module):
    def __init__(self, beta=LIF_BETA, threshold=LIF_THRESHOLD):
        super().__init__()
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.conv1 = nn.Conv2d(2, 16, 5, stride=2, padding=2, bias=False)
        self.norm1 = nn.GroupNorm(4, 16)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, 32)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)
        self.norm3 = nn.GroupNorm(8, 64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(64, 2)

    def forward(self, voxel_sequence):
        mem1 = None
        mem2 = None
        mem3 = None
        readout_membrane = None
        readouts = []
        for time_index in range(voxel_sequence.shape[1]):
            current1 = self.norm1(self.conv1(voxel_sequence[:, time_index]))
            spike1, mem1 = lif_step(
                current1, mem1, self.beta, self.threshold
            )
            current2 = self.norm2(self.conv2(spike1))
            spike2, mem2 = lif_step(
                current2, mem2, self.beta, self.threshold
            )
            current3 = self.norm3(self.conv3(spike2))
            spike3, mem3 = lif_step(
                current3, mem3, self.beta, self.threshold
            )
            features = self.pool(spike3).flatten(1)
            current_out = self.classifier(self.dropout(features))
            if readout_membrane is None:
                readout_membrane = torch.zeros_like(current_out)
            readout_membrane = self.beta * readout_membrane + current_out
            readouts.append(readout_membrane)
        return torch.stack(readouts, dim=0).mean(dim=0)


def class_weights_from_samples(samples):
    video_labels = (
        pd.DataFrame(samples)[["video_key", "label"]]
        .drop_duplicates("video_key")
    )
    counts = video_labels["label"].value_counts().reindex([0, 1], fill_value=0)
    if (counts == 0).any():
        raise ValueError(f"训练集缺少类别：{counts.to_dict()}")
    weights = len(video_labels) / (2.0 * counts.to_numpy(np.float32))
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def calculate_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    return {
        "ACC": float(accuracy_score(y_true, y_pred)),
        "AUC": (
            float(roc_auc_score(y_true, y_prob))
            if len(np.unique(y_true)) > 1 else float("nan")
        ),
        "PR-AUC": float(average_precision_score(y_true, y_prob)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Balanced_ACC": float(balanced_accuracy_score(y_true, y_pred)),
        "Sensitivity": float(tp / max(tp + fn, 1)),
        "Specificity": float(tn / max(tn + fp, 1)),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def aggregate_video_predictions(records):
    segment_frame = pd.DataFrame(records)
    if segment_frame.empty:
        raise ValueError("预测记录为空")
    label_counts = segment_frame.groupby("video_key")["label"].nunique()
    if label_counts.max() > 1:
        bad_keys = label_counts[label_counts.gt(1)].index.tolist()
        raise ValueError(f"预测记录中同一视频存在多个标签：{bad_keys[:10]}")
    aggregation = "median" if VIDEO_AGGREGATION == "median" else "mean"
    return (
        segment_frame.groupby("video_key", as_index=False)
        .agg(label=("label", "first"), probability=("probability", aggregation))
        .sort_values("video_key")
        .reset_index(drop=True)
    )


def search_best_threshold(video_frame):
    y_true = video_frame["label"].to_numpy(np.int64)
    y_prob = video_frame["probability"].to_numpy(np.float64)
    best_score = None
    best_threshold = 0.5
    for threshold in THRESHOLD_GRID:
        prediction = (y_prob >= threshold).astype(np.int64)
        metrics = calculate_metrics(y_true, prediction, y_prob)
        score = (
            metrics["Balanced_ACC"],
            metrics["F1"],
            metrics["ACC"],
            -abs(float(threshold) - 0.5),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for voxel, label, _ in loader:
        voxel = voxel.to(DEVICE, non_blocking=True)
        label = label.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(voxel)
        loss = loss_fn(logits, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        loss_sum += float(loss.item()) * len(label)
        sample_count += len(label)
    return loss_sum / max(sample_count, 1)


@torch.no_grad()
def evaluate_model(model, loader, loss_fn=None):
    model.eval()
    records = []
    loss_sum = 0.0
    sample_count = 0
    for voxel, label, video_keys in loader:
        voxel = voxel.to(DEVICE, non_blocking=True)
        label_device = label.to(DEVICE, non_blocking=True)
        logits = model(voxel)
        if loss_fn is not None:
            loss = loss_fn(logits, label_device)
            loss_sum += float(loss.item()) * len(label)
            sample_count += len(label)
        probability = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        for key, target, prob in zip(
            video_keys,
            label.numpy(),
            probability,
        ):
            records.append({
                "video_key": str(key),
                "label": int(target),
                "probability": float(prob),
            })
    video_frame = aggregate_video_predictions(records)
    mean_loss = loss_sum / max(sample_count, 1) if loss_fn is not None else np.nan
    return mean_loss, video_frame


def clone_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def train_with_validation(
    train_samples,
    val_samples,
    outer_fold,
    inner_fold,
    save_dir,
):
    seed = SEED + outer_fold * 1000 + inner_fold
    seed_everything(seed)
    train_loader = make_loader(train_samples, seed, train=True)
    val_loader = make_loader(val_samples, seed + 1, train=False)
    model = ConvSNN().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights_from_samples(train_samples)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(EPOCHS, 1),
        eta_min=LEARNING_RATE * 0.05,
    )

    best_score = None
    best_state = None
    best_epoch = 1
    best_video_frame = None
    wait = 0
    epoch_rows = []
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_video = evaluate_model(model, val_loader, loss_fn)
        val_prob = val_video["probability"].to_numpy(np.float64)
        val_label = val_video["label"].to_numpy(np.int64)
        val_auc = (
            float(roc_auc_score(val_label, val_prob))
            if len(np.unique(val_label)) > 1 else float("nan")
        )
        val_ap = float(average_precision_score(val_label, val_prob))
        val_pred = (val_prob >= 0.5).astype(np.int64)
        val_balanced = float(
            balanced_accuracy_score(val_label, val_pred)
        )
        score = (
            val_ap if np.isfinite(val_ap) else -float("inf"),
            val_auc if np.isfinite(val_auc) else -float("inf"),
            -float(val_loss),
        )
        checkpoint_saved = int(best_score is None or score > best_score)
        if checkpoint_saved:
            best_score = score
            best_state = clone_state_dict(model)
            best_epoch = epoch
            best_video_frame = val_video.copy()
            wait = 0
        else:
            wait += 1

        epoch_rows.append({
            "model": "snn",
            "ablation": "full",
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_ap": val_ap,
            "val_balanced_accuracy": val_balanced,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "checkpoint_saved": checkpoint_saved,
        })
        scheduler.step()
        print(
            f"outer={outer_fold} inner={inner_fold} epoch={epoch} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_auc={val_auc:.4f} val_ap={val_ap:.4f}"
        )
        if wait >= EARLY_STOP_PATIENCE:
            break

    if best_state is None or best_video_frame is None:
        raise RuntimeError("inner fold 未产生有效模型")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "best_model.pt")
    torch.save(best_state, checkpoint_path)
    pd.DataFrame(epoch_rows)[EPOCH_METRIC_COLUMNS].to_csv(
        os.path.join(save_dir, "epoch_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return best_state, best_epoch, best_video_frame, epoch_rows


def train_fixed(train_samples, epochs, seed):
    seed_everything(seed)
    loader = make_loader(train_samples, seed, train=True)
    model = ConvSNN().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights_from_samples(train_samples)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(epochs), 1),
        eta_min=LEARNING_RATE * 0.05,
    )
    for _ in range(max(int(epochs), 1)):
        train_one_epoch(model, loader, loss_fn, optimizer)
        scheduler.step()
    return clone_state_dict(model)


def predict_state_dict(state_dict, samples, seed):
    model = ConvSNN().to(DEVICE)
    model.load_state_dict(state_dict, strict=True)
    loader = make_loader(samples, seed, train=False)
    _, prediction = evaluate_model(model, loader)
    return prediction


def predict_state_ensemble(state_dicts, samples, seed):
    frames = [
        predict_state_dict(state_dict, samples, seed + index)
        for index, state_dict in enumerate(state_dicts)
    ]
    reference = frames[0][["video_key", "label"]].copy()
    probabilities = []
    reference_keys = reference["video_key"].tolist()
    for frame in frames:
        if frame["video_key"].tolist() != reference_keys:
            raise AssertionError("SNN ensemble 的视频顺序不一致")
        if not frame["label"].eq(reference["label"]).all():
            raise AssertionError("SNN ensemble 的视频标签不一致")
        probabilities.append(frame["probability"].to_numpy(np.float64))
    reference["probability"] = np.mean(np.stack(probabilities), axis=0)
    return reference


def save_model_bundle(
    bundle_path,
    state_dicts,
    outer_fold,
    selected_epoch,
    threshold,
    test_video_keys,
):
    bundle = {
        "format_version": 1,
        "model_class": "ConvSNN",
        "model_state_dicts": state_dicts,
        "model_config": {
            "time_bins": TIME_BINS,
            "voxel_h": VOXEL_H,
            "voxel_w": VOXEL_W,
            "orig_h": ORIG_H,
            "orig_w": ORIG_W,
            "lif_beta": LIF_BETA,
            "lif_threshold": LIF_THRESHOLD,
            "dropout": DROPOUT,
            "max_events": MAX_EVENTS,
            "cache_version": CACHE_VERSION,
            "video_aggregation": VIDEO_AGGREGATION,
        },
        "outer_fold": int(outer_fold),
        "selected_epoch": int(selected_epoch),
        "threshold": float(threshold),
        "threshold_objective": THRESHOLD_OBJECTIVE,
        "seeds": [
            int(SEED + outer_fold * 10000 + index)
            for index in range(FINAL_SEEDS)
        ],
        "test_video_keys": sorted(str(key) for key in test_video_keys),
    }
    torch.save(bundle, bundle_path)


def save_run_manifest(summary):
    end_time = datetime.now().isoformat(timespec="seconds")
    model_paths = []
    for relative_path in summary["model_bundle"].astype(str):
        model_path = os.path.abspath(os.path.join(SAVE_ROOT, relative_path))
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"模型文件不存在：{model_path}")
        model_paths.append(model_path)
    manifest = pd.DataFrame({
        "run_id": [
            f"snn_full_outer_fold_{int(fold)}"
            for fold in summary["outer_fold"]
        ],
        "model": "snn",
        "ablation": "full",
        "outer_fold": summary["outer_fold"].astype(int),
        "seed": SEED,
        "config_file": CONFIG_SNAPSHOT_PATH,
        "config_sha256": CONFIG_SHA256,
        "training_manifest_sha256": sha256_file(SPLIT_MANIFEST_PATH),
        "code_commit": CODE_COMMIT,
        "start_time": RUN_START_TIME,
        "end_time": end_time,
        "selected_epoch": summary["selected_epoch"].astype(int),
        "selected_threshold": summary["threshold"].astype(float),
        "threshold_objective": THRESHOLD_OBJECTIVE,
        "checkpoint_path": model_paths,
        "relative_path": [
            os.path.relpath(path, SAVE_ROOT).replace(os.sep, "/")
            for path in model_paths
        ],
        "bytes": [os.path.getsize(path) for path in model_paths],
        "sha256": [sha256_file(path) for path in model_paths],
    })
    logs_dir = os.path.join(SAVE_ROOT, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    manifest[RUN_MANIFEST_COLUMNS].to_csv(
        os.path.join(logs_dir, "run_manifest.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def main():
    global RUN_START_TIME
    RUN_START_TIME = datetime.now().isoformat(timespec="seconds")
    seed_everything(SEED)
    os.makedirs(SAVE_ROOT, exist_ok=True)
    os.makedirs(os.path.join(SAVE_ROOT, "logs"), exist_ok=True)
    prepare_run_metadata()

    split_manifest = read_split_manifest(SPLIT_MANIFEST_PATH)
    label_map = manifest_label_map(split_manifest)
    samples = collect_samples(DATA_ROOT, label_map)
    sample_frame = pd.DataFrame(samples)
    video_meta = (
        sample_frame[["video_key", "label"]]
        .drop_duplicates()
        .sort_values("video_key")
        .reset_index(drop=True)
    )
    validate_split_manifest(split_manifest, video_meta)
    shutil.copyfile(
        SPLIT_MANIFEST_PATH,
        os.path.join(SAVE_ROOT, "split_manifest.csv"),
    )
    print(
        f"device={DEVICE}, segments={len(samples)}, videos={len(video_meta)}, "
        f"labels={video_meta['label'].value_counts().sort_index().to_dict()}"
    )

    all_video_keys = set(video_meta["video_key"].astype(str))
    summary_rows = []
    outer_predictions = []
    all_epoch_rows = []
    for outer_fold in range(1, OUTER_FOLDS + 1):
        print(
            f"\n================ SNN Outer Fold "
            f"{outer_fold}/{OUTER_FOLDS} ================"
        )
        fold_root = os.path.join(SAVE_ROOT, f"outer_fold_{outer_fold}")
        os.makedirs(fold_root, exist_ok=True)
        test_keys = manifest_keys(
            split_manifest, outer_fold, "outer_test"
        )
        train_keys = all_video_keys - test_keys
        outer_train_samples = samples_by_keys(samples, train_keys)
        outer_test_samples = samples_by_keys(samples, test_keys)

        oof_parts = []
        best_epochs = []
        for inner_fold in range(1, INNER_FOLDS + 1):
            inner_train_keys = manifest_keys(
                split_manifest,
                outer_fold,
                "inner_train",
                inner_fold,
            )
            inner_val_keys = manifest_keys(
                split_manifest,
                outer_fold,
                "inner_val",
                inner_fold,
            )
            inner_train_samples = samples_by_keys(
                samples, inner_train_keys
            )
            inner_val_samples = samples_by_keys(samples, inner_val_keys)
            inner_dir = os.path.join(
                fold_root, f"inner_fold_{inner_fold}"
            )
            _, best_epoch, val_video, epoch_rows = train_with_validation(
                inner_train_samples,
                inner_val_samples,
                outer_fold,
                inner_fold,
                inner_dir,
            )
            best_epochs.append(best_epoch)
            val_video["inner_fold"] = inner_fold
            oof_parts.append(val_video)
            all_epoch_rows.extend(epoch_rows)

        inner_oof = pd.concat(oof_parts, ignore_index=True)
        if inner_oof["video_key"].duplicated().any():
            raise AssertionError("inner OOF 包含重复视频")
        if set(inner_oof["video_key"]) != train_keys:
            raise AssertionError("inner OOF 未完整覆盖 outer train")
        threshold = search_best_threshold(inner_oof)
        selected_epoch = max(1, int(round(float(np.median(best_epochs)))))
        inner_oof["model"] = "snn"
        inner_oof["ablation"] = "full"
        inner_oof["animal_id"] = inner_oof["video_key"].astype(str)
        inner_oof["outer_fold"] = outer_fold
        inner_oof["fold_threshold"] = threshold
        inner_oof["prediction"] = (
            inner_oof["probability"] >= threshold
        ).astype(np.int64)
        inner_oof["threshold_objective"] = THRESHOLD_OBJECTIVE
        inner_oof.to_csv(
            os.path.join(fold_root, "inner_oof_video_pred.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        state_dicts = []
        for seed_index in range(FINAL_SEEDS):
            final_seed = SEED + outer_fold * 10000 + seed_index
            state_dicts.append(
                train_fixed(
                    outer_train_samples,
                    selected_epoch,
                    final_seed,
                )
            )
        test_video = predict_state_ensemble(
            state_dicts,
            outer_test_samples,
            SEED + outer_fold * 500,
        )
        test_video["prediction"] = (
            test_video["probability"] >= threshold
        ).astype(np.int64)
        test_video["model"] = "snn"
        test_video["ablation"] = "full"
        test_video["animal_id"] = test_video["video_key"].astype(str)
        test_video["outer_fold"] = outer_fold
        test_video["fold_threshold"] = threshold
        test_video["threshold_objective"] = THRESHOLD_OBJECTIVE

        bundle_path = os.path.join(fold_root, "snn_model_bundle.pt")
        save_model_bundle(
            bundle_path,
            state_dicts,
            outer_fold,
            selected_epoch,
            threshold,
            test_keys,
        )
        test_video["checkpoint_id"] = os.path.relpath(
            bundle_path, SAVE_ROOT
        ).replace(os.sep, "/")
        test_video.to_csv(
            os.path.join(fold_root, "outer_test_vid_pred.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        outer_predictions.append(test_video)

        metrics = calculate_metrics(
            test_video["label"],
            test_video["prediction"],
            test_video["probability"],
        )
        summary_rows.append({
            "model": "snn",
            "ablation": "full",
            "outer_fold": outer_fold,
            "selected_epoch": selected_epoch,
            "threshold": threshold,
            "test_acc": metrics["ACC"],
            "test_auc": metrics["AUC"],
            "test_prauc": metrics["PR-AUC"],
            "test_f1": metrics["F1"],
            "test_bal_acc": metrics["Balanced_ACC"],
            "test_sensitivity": metrics["Sensitivity"],
            "test_specificity": metrics["Specificity"],
            "n_test_videos": len(test_video),
            "model_bundle": os.path.relpath(
                bundle_path, SAVE_ROOT
            ).replace(os.sep, "/"),
        })
        print(
            f"outer={outer_fold} epoch={selected_epoch} "
            f"threshold={threshold:.2f} ACC={metrics['ACC']:.4f} "
            f"AUC={metrics['AUC']:.4f} PR-AUC={metrics['PR-AUC']:.4f}"
        )

    summary = pd.DataFrame(summary_rows)
    predictions = pd.concat(outer_predictions, ignore_index=True)
    if predictions["video_key"].duplicated().any():
        raise AssertionError("outer test 预测包含重复视频")
    if set(predictions["video_key"]) != all_video_keys:
        raise AssertionError("outer test 预测未完整覆盖全部视频")

    summary.to_csv(
        os.path.join(SAVE_ROOT, "outer_5fold_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        os.path.join(SAVE_ROOT, "all_outer_test_vid_pred.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    predictions[PREDICTION_COLUMNS].to_csv(
        os.path.join(SAVE_ROOT, "standardized_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    pooled_metrics = calculate_metrics(
        predictions["label"],
        predictions["prediction"],
        predictions["probability"],
    )
    pd.DataFrame([pooled_metrics]).to_csv(
        os.path.join(SAVE_ROOT, "pooled_outer_test_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(all_epoch_rows)[EPOCH_METRIC_COLUMNS].to_csv(
        os.path.join(SAVE_ROOT, "logs", "epoch_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    save_run_manifest(summary)
    print("\n================ SNN 5-fold Summary ================")
    print(summary.to_string(index=False))
    print("\n================ SNN Pooled Outer Test ================")
    print(pd.DataFrame([pooled_metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
