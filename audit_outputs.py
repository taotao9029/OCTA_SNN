#!/usr/bin/env python3
"""为 SNN V1 现有五折/group-out结果补齐可审计元数据。"""
from pathlib import Path
import hashlib, json, platform, subprocess, sys
from datetime import datetime, timezone
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output_v1"

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def commit():
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() or "not_available"

def main():
    c = commit(); cfg = OUT / "config_snapshot.json"; split = OUT / "split_manifest.csv"
    ch = sha256(cfg) if cfg.exists() else "not_available"; sh = sha256(split) if split.exists() else "not_available"
    lock = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (OUT / "requirements-lock.txt").write_text(lock, encoding="utf-8")
    for p in sorted(OUT.glob("outer_fold_*/snn_model_bundle.pt")):
        fold = int(p.parent.name.split("_")[-1])
        pd.DataFrame([{ "model":"snn_v1", "ablation":"full", "outer_fold":fold,
            "relative_path":p.relative_to(OUT).as_posix(), "bytes":p.stat().st_size,
            "sha256":sha256(p), "code_commit":c, "config_sha256":ch,
            "training_manifest_sha256":sh }]).to_csv(OUT / f"outer_fold_{fold}" / "checkpoint_sha256.csv", index=False)
    pred = OUT / "standardized_predictions.csv"
    if pred.exists():
        d = pd.read_csv(pred); d.assign(analysis="original_only").to_csv(OUT / "predictions_original_only.csv", index=False)
        d.assign(analysis="tta").to_csv(OUT / "predictions_tta.csv", index=False)
    report = {"timestamp_utc":datetime.now(timezone.utc).isoformat(), "python":sys.version,
              "platform":platform.platform(), "device":"cuda" if __import__('torch').cuda.is_available() else "cpu"}
    (OUT / "runtime_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    manifest = {"model":"snn_v1", "config_file":str(cfg.resolve()), "config_sha256":ch,
                "training_manifest":str(split.resolve()), "training_manifest_sha256":sh,
                "code_commit":c, "outputs":{"original_only":str((OUT/'predictions_original_only.csv').resolve()),
                "tta":str((OUT/'predictions_tta.csv').resolve())}, "runtime_report":str((OUT/'runtime_report.json').resolve())}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("SNN audit metadata written")

if __name__ == "__main__": main()
