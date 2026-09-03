# -*- coding: utf-8 -*-
"""SNN 按采集日期进行 leave-one-date-out 训练与预测。"""

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import train_snn as model


SCRIPT_DIR = Path(__file__).resolve().parent
FUSION_DIR = SCRIPT_DIR.parent / "OCTA_Fusion"
if str(FUSION_DIR) not in sys.path:
    sys.path.insert(0, str(FUSION_DIR))

from group_out_common import save_group_predictions, validate_group_table


LEAVE_DATE_MANIFEST_PATH = (
    "/home/wxy/Classification/syy/syy/strock_test/OCTA_pytorch/"
    "OCTA_Attention/feature_out/leave_date.csv"
)
OUTPUT_ROOT = SCRIPT_DIR / "output_improved" / "snn_group_out_results"
PREDICTION_PATH = OUTPUT_ROOT / "leave_date_out_predictions.csv"
EXPECTED_DATES = {"2025-10-28", "2025-11-01", "2025-11-03"}
GROUP_TYPE = "date"
GROUP_COLUMN = "date"


def load_leave_date_manifest(manifest_path):
    manifest = pd.read_csv(
        manifest_path,
        dtype={"animal_id": str, "video_key": str, "date": str},
    )
    manifest.columns = manifest.columns.astype(str).str.strip()
    required = {"animal_id", "video_key", "label", "group", "date"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"leave_date.csv 缺少字段：{sorted(missing)}")

    for column in ("animal_id", "video_key", "date"):
        manifest[column] = manifest[column].astype(str).str.strip()
    manifest["label"] = pd.to_numeric(
        manifest["label"], errors="raise"
    ).astype(int)
    manifest["group"] = pd.to_numeric(
        manifest["group"], errors="raise"
    ).astype(int)

    if manifest["video_key"].duplicated().any():
        duplicated = manifest.loc[
            manifest["video_key"].duplicated(keep=False), "video_key"
        ].tolist()
        raise ValueError(f"leave_date.csv 存在重复视频：{duplicated[:10]}")
    if set(manifest["group"]) != {1, 2, 3}:
        raise ValueError("leave_date.csv 的 group 必须恰好为 1、2、3")
    if set(manifest["date"]) != EXPECTED_DATES:
        raise ValueError(
            "leave_date.csv 的日期必须恰好为："
            f"{sorted(EXPECTED_DATES)}"
        )
    if not manifest.groupby("group")["date"].nunique().eq(1).all():
        raise ValueError("leave_date.csv 中同一 group 对应了多个日期")
    if not manifest.groupby("date")["group"].nunique().eq(1).all():
        raise ValueError("leave_date.csv 中同一日期对应了多个 group")
    return manifest


def load_samples():
    manifest = load_leave_date_manifest(LEAVE_DATE_MANIFEST_PATH)
    label_map = (
        manifest[["video_key", "label"]]
        .set_index("video_key")["label"]
        .astype(int)
        .to_dict()
    )
    samples = model.collect_samples(model.DATA_ROOT, label_map)
    if not samples:
        raise FileNotFoundError(f"没有找到 SNN 事件样本：{model.DATA_ROOT}")

    sample_frame = pd.DataFrame(samples)
    sample_keys = set(sample_frame["video_key"].astype(str))
    manifest_keys = set(manifest["video_key"].astype(str))
    if sample_keys != manifest_keys:
        missing = sorted(sample_keys - manifest_keys)
        extra = sorted(manifest_keys - sample_keys)
        raise ValueError(
            "SNN 数据与 leave_date.csv 视频集合不一致："
            f"清单缺少={missing[:10]}，清单多出={extra[:10]}"
        )

    metadata = manifest[["video_key", "label", "group", "date"]].rename(
        columns={"label": "manifest_label"}
    )
    sample_frame = sample_frame.merge(
        metadata,
        on="video_key",
        how="left",
        validate="many_to_one",
    )
    label_mismatch = sample_frame["label"].astype(int).ne(
        sample_frame["manifest_label"].astype(int)
    )
    if label_mismatch.any():
        bad = sample_frame.loc[
            label_mismatch, "video_key"
        ].drop_duplicates().tolist()
        raise ValueError(f"SNN 数据与 leave_date.csv 标签不一致：{bad[:10]}")
    return sample_frame.drop(columns="manifest_label"), manifest


def build_video_meta(sample_frame):
    metadata = sample_frame[
        ["video_key", "label", "group", "date"]
    ].drop_duplicates()
    if metadata["video_key"].duplicated().any():
        bad = metadata.loc[
            metadata["video_key"].duplicated(keep=False), "video_key"
        ].tolist()
        raise ValueError(f"同一视频对应多个标签或日期：{bad[:10]}")
    return metadata.sort_values("video_key").reset_index(drop=True)


def select_training_configuration(
    train_samples,
    train_video_meta,
    group_index,
    output_dir,
):
    class_counts = train_video_meta["label"].value_counts()
    n_splits = min(model.INNER_FOLDS, int(class_counts.min()))
    if n_splits < 2:
        raise ValueError("Group-out 训练集内部至少需要每类 2 个视频")

    inner_cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=model.SEED + group_index * 100,
    )
    inner_parts = []
    selected_epochs = []
    train_keys = set(train_video_meta["video_key"].astype(str))

    split_iterator = inner_cv.split(
        train_video_meta,
        y=train_video_meta["label"],
        groups=train_video_meta["video_key"],
    )
    for inner_index, (fit_index, val_index) in enumerate(
        split_iterator,
        start=1,
    ):
        fit_keys = set(
            train_video_meta.iloc[fit_index]["video_key"].astype(str)
        )
        val_keys = set(
            train_video_meta.iloc[val_index]["video_key"].astype(str)
        )
        fit_samples = model.samples_by_keys(train_samples, fit_keys)
        val_samples = model.samples_by_keys(train_samples, val_keys)
        inner_dir = Path(output_dir) / "inner" / f"inner_fold_{inner_index}"
        state_dicts = []
        inner_epochs = []

        for seed_index in range(model.INNER_SEEDS):
            seed_dir = inner_dir / f"seed_{seed_index + 1}"
            state_dict, best_epoch, _, _ = model.train_with_validation(
                fit_samples,
                val_samples,
                group_index,
                inner_index,
                str(seed_dir),
                seed_index,
            )
            state_dicts.append(state_dict)
            inner_epochs.append(best_epoch)

        val_video = model.predict_state_ensemble(
            state_dicts,
            val_samples,
            model.SEED + group_index * 100 + inner_index * 10,
        )
        val_video["inner_fold"] = inner_index
        inner_parts.append(val_video)
        selected_epochs.extend(inner_epochs)
        print(
            f"group={group_index} inner={inner_index} "
            f"ensemble_epochs={inner_epochs}"
        )

    inner_oof = pd.concat(inner_parts, ignore_index=True)
    if inner_oof["video_key"].duplicated().any():
        raise AssertionError("Group-out inner OOF 出现重复视频")
    if set(inner_oof["video_key"].astype(str)) != train_keys:
        raise AssertionError("Group-out inner OOF 未完整覆盖训练视频")

    threshold = model.search_best_threshold(inner_oof)
    selected_epoch = max(
        1,
        int(round(float(np.median(selected_epochs)))),
    )
    inner_oof["prediction"] = (
        inner_oof["probability"] >= threshold
    ).astype(np.int64)
    inner_oof["fold_threshold"] = threshold
    inner_oof["threshold_objective"] = model.THRESHOLD_OBJECTIVE
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    inner_oof.to_csv(
        Path(output_dir) / "inner_oof_video_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return threshold, selected_epoch


def build_config_snapshot():
    return {
        "source_file": str(Path(__file__).resolve()),
        "model_source_file": str(Path(model.__file__).resolve()),
        "data_root": model.DATA_ROOT,
        "leave_date_manifest": LEAVE_DATE_MANIFEST_PATH,
        "output_root": str(OUTPUT_ROOT.resolve()),
        "expected_dates": sorted(EXPECTED_DATES),
        "group_type": GROUP_TYPE,
        "inner_folds": model.INNER_FOLDS,
        "inner_seeds": model.INNER_SEEDS,
        "final_seeds": model.FINAL_SEEDS,
        "threshold_objective": model.THRESHOLD_OBJECTIVE,
        "video_aggregation": model.VIDEO_AGGREGATION,
        "epochs": model.EPOCHS,
        "early_stop_patience": model.EARLY_STOP_PATIENCE,
        "snn_channels": list(model.SNN_CHANNELS),
        "readout_hidden": model.READOUT_HIDDEN,
        "label_smoothing": model.LABEL_SMOOTHING,
    }


def save_run_manifest(summary, config_path, start_time):
    config_sha256 = model.sha256_file(config_path)
    training_manifest_sha256 = model.sha256_file(
        LEAVE_DATE_MANIFEST_PATH
    )
    code_commit = model.get_git_commit()
    end_time = datetime.now().isoformat(timespec="seconds")
    rows = []
    for row in summary.itertuples(index=False):
        bundle_path = OUTPUT_ROOT / row.model_bundle
        rows.append(
            {
                "run_id": f"snn_leave_date_out_{row.outer_fold}",
                "model": "snn",
                "ablation": "full",
                "outer_fold": int(row.outer_fold),
                "held_out_group": str(row.held_out_group),
                "seed": model.SEED,
                "config_file": str(config_path.resolve()),
                "config_sha256": config_sha256,
                "training_manifest_sha256": training_manifest_sha256,
                "code_commit": code_commit,
                "start_time": start_time,
                "end_time": end_time,
                "selected_epoch": int(row.selected_epoch),
                "selected_threshold": float(row.threshold),
                "threshold_objective": model.THRESHOLD_OBJECTIVE,
                "checkpoint_path": str(bundle_path.resolve()),
                "relative_path": bundle_path.relative_to(
                    OUTPUT_ROOT
                ).as_posix(),
                "bytes": bundle_path.stat().st_size,
                "sha256": model.sha256_file(bundle_path),
            }
        )
    logs_dir = OUTPUT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        logs_dir / "run_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main():
    start_time = datetime.now().isoformat(timespec="seconds")
    model.seed_everything(model.SEED)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sample_frame, manifest = load_samples()
    video_meta = build_video_meta(sample_frame)
    held_out_dates = validate_group_table(sample_frame, GROUP_COLUMN)

    config_path = OUTPUT_ROOT / "config_snapshot.json"
    with open(config_path, "w", encoding="utf-8") as file_obj:
        json.dump(
            build_config_snapshot(),
            file_obj,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        file_obj.write("\n")
    shutil.copyfile(
        LEAVE_DATE_MANIFEST_PATH,
        OUTPUT_ROOT / "leave_date.csv",
    )

    prediction_rows = []
    summary_rows = []
    for group_index, held_out_date in enumerate(held_out_dates, start=1):
        print(
            f"\n========== SNN leave-date-out: {held_out_date} "
            f"({group_index}/{len(held_out_dates)}) =========="
        )
        train_keys = set(
            video_meta.loc[
                video_meta[GROUP_COLUMN].astype(str) != str(held_out_date),
                "video_key",
            ].astype(str)
        )
        test_keys = set(
            video_meta.loc[
                video_meta[GROUP_COLUMN].astype(str) == str(held_out_date),
                "video_key",
            ].astype(str)
        )
        train_video_meta = video_meta[
            video_meta["video_key"].isin(train_keys)
        ].copy()
        if train_video_meta["label"].nunique() < 2:
            raise ValueError(
                f"留出 {held_out_date} 后训练集只有一个类别，无法训练"
            )

        train_samples = model.samples_by_keys(
            sample_frame.to_dict("records"), train_keys
        )
        test_samples = model.samples_by_keys(
            sample_frame.to_dict("records"), test_keys
        )
        held_out_dir = (
            OUTPUT_ROOT
            / "checkpoints"
            / f"held_out_{group_index:03d}"
        )
        threshold, selected_epoch = select_training_configuration(
            train_samples,
            train_video_meta,
            group_index,
            held_out_dir,
        )

        state_dicts = []
        for seed_index in range(model.FINAL_SEEDS):
            final_seed = model.SEED + group_index * 10000 + seed_index
            state_dicts.append(
                model.train_fixed(
                    train_samples,
                    selected_epoch,
                    final_seed,
                )
            )
        test_video = model.predict_state_ensemble(
            state_dicts,
            test_samples,
            model.SEED + group_index * 500,
        )
        test_video["prediction"] = (
            test_video["probability"] >= threshold
        ).astype(np.int64)

        bundle_path = held_out_dir / "snn_model_bundle.pt"
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model_bundle(
            str(bundle_path),
            state_dicts,
            group_index,
            selected_epoch,
            threshold,
            test_keys,
        )

        for row in test_video.itertuples(index=False):
            prediction_rows.append(
                {
                    "group_type": GROUP_TYPE,
                    "held_out_group": str(held_out_date),
                    "animal_id": str(row.video_key),
                    "label": int(row.label),
                    "probability": float(row.probability),
                    "threshold": float(threshold),
                    "prediction": int(row.prediction),
                }
            )

        metrics = model.calculate_metrics(
            test_video["label"],
            test_video["prediction"],
            test_video["probability"],
        )
        summary_rows.append(
            {
                "model": "snn",
                "ablation": "full",
                "outer_fold": group_index,
                "held_out_group": str(held_out_date),
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
                "model_bundle": bundle_path.relative_to(
                    OUTPUT_ROOT
                ).as_posix(),
            }
        )
        print(
            f"date={held_out_date} epoch={selected_epoch} "
            f"threshold={threshold:.2f} ACC={metrics['ACC']:.4f} "
            f"AUC={metrics['AUC']:.4f} PR-AUC={metrics['PR-AUC']:.4f}"
        )

    predictions = save_group_predictions(
        prediction_rows,
        PREDICTION_PATH,
    )
    if predictions["animal_id"].duplicated().any():
        raise AssertionError("leave-date-out 预测包含重复视频")
    if set(predictions["animal_id"].astype(str)) != set(
        manifest["video_key"].astype(str)
    ):
        raise AssertionError("leave-date-out 预测未完整覆盖全部视频")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUTPUT_ROOT / "leave_date_out_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_run_manifest(summary, config_path, start_time)
    print(f"SNN leave-date-out 预测已保存：{PREDICTION_PATH}")


if __name__ == "__main__":
    main()
