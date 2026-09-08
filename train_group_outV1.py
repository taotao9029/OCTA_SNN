# -*- coding: utf-8 -*-
"""SNN V1 按采集日期进行 leave-one-date-out 训练与预测。

复用 ``train_group_out.py`` 的分组、内部验证、阈值选择和正式清单逻辑，
仅注入 ``train_snnV1.py`` 定义的体素配置、网络、训练增强与视频聚合。
"""

from pathlib import Path

import train_group_out as group_runner
import train_snnV1 as snn_v1


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "output_v1" / "snn_group_out_results"
PREDICTION_PATH = OUTPUT_ROOT / "leave_date_out_predictions.csv"

_BASE_GROUP_CONFIG_SNAPSHOT = group_runner.build_config_snapshot


def build_config_snapshot_v1():
    """补充V1入口和结构配置，供group-out运行清单计算SHA-256。"""
    config = _BASE_GROUP_CONFIG_SNAPSHOT()
    config.update(
        {
            "source_file": str(Path(__file__).resolve()),
            "group_runner_source_file": str(
                Path(group_runner.__file__).resolve()
            ),
            "model_source_file": str(Path(snn_v1.__file__).resolve()),
            "base_model_source_file": str(
                Path(snn_v1.base.__file__).resolve()
            ),
            "model_variant": snn_v1.MODEL_VARIANT,
            "time_bins": snn_v1.TIME_BINS,
            "voxel_h": snn_v1.VOXEL_H,
            "voxel_w": snn_v1.VOXEL_W,
            "max_events": snn_v1.MAX_EVENTS,
            "cache_version": snn_v1.CACHE_VERSION,
            "horizontal_flip_probability": (
                snn_v1.HORIZONTAL_FLIP_PROBABILITY
            ),
            "temporal_mask_probability": (
                snn_v1.TEMPORAL_MASK_PROBABILITY
            ),
            "intensity_jitter": snn_v1.INTENSITY_JITTER,
            "residual_scale": snn_v1.RESIDUAL_SCALE,
            "temporal_attention": True,
            "spike_residual": True,
        }
    )
    return config


def configure_group_out_v1():
    """让共享group-out入口完整使用SNN V1实现和独立输出目录。"""
    snn_v1.configure_v1()
    group_runner.model = snn_v1.base
    group_runner.OUTPUT_ROOT = OUTPUT_ROOT
    group_runner.PREDICTION_PATH = PREDICTION_PATH
    group_runner.build_config_snapshot = build_config_snapshot_v1


def main():
    configure_group_out_v1()
    print(
        f"启动 {snn_v1.MODEL_VARIANT} leave-date-out: "
        f"manifest={group_runner.LEAVE_DATE_MANIFEST_PATH}, "
        f"output={PREDICTION_PATH}"
    )
    group_runner.main()
    if not PREDICTION_PATH.is_file():
        raise FileNotFoundError(f"预测文件未生成：{PREDICTION_PATH}")


if __name__ == "__main__":
    main()
