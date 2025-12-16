import os
import sys
import subprocess
import warnings
from pathlib import Path
import pandas as pd

from config import cfg
from data_cleaning import run_cleaning

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
env = os.environ.copy()
env["PYTHONPATH"] = str(PROJECT_ROOT)

raw_train_root = cfg.raw_train_root
train_csv = cfg.raw_train_csv
val_csv = cfg.raw_val_csv

preprocessed_dir = cfg.preprocessed_dir
splits_dir = cfg.splits_dir
ckpt_dir = cfg.ckpt_dir

for d in [preprocessed_dir, splits_dir, ckpt_dir]:
    d.mkdir(parents=True, exist_ok=True)

print("========== Step 1A: Train Preprocessing ==========")

train_pairs = []
for subdir, _, files in os.walk(raw_train_root):
    subdir = Path(subdir)
    if "t2.nii.gz" in files and "t2_anatomy_reader1.nii.gz" in files:
        train_pairs.append((subdir / "t2.nii.gz", subdir / "t2_anatomy_reader1.nii.gz"))

if len(train_pairs) == 0:
    raise RuntimeError("❌ No T2 + mask pairs found")

manifest = preprocessed_dir / "manifest.csv"
if not manifest.exists():
    run_cleaning(train_pairs, preprocessed_dir)

print("✅ Preprocessing done")

print("========== Step 1C: Train/Val splits ==========")

slice_train_csv = splits_dir / "train.csv"
slice_val_csv = splits_dir / "val.csv"

if not slice_train_csv.exists() or not slice_val_csv.exists():
    man = pd.read_csv(manifest, dtype={"case": str})
    ds_train = pd.read_csv(train_csv)
    ds_val = pd.read_csv(val_csv)

    train_cases = {f"{int(i):03d}" for i in ds_train["ID"].tolist()}
    val_cases = {f"{int(i):03d}" for i in ds_val["ID"].tolist()}

    train_df = man[man["case"].isin(train_cases)]
    val_df = man[man["case"].isin(val_cases)]

    train_df[["image"]].to_csv(slice_train_csv, index=False)
    val_df[["image"]].to_csv(slice_val_csv, index=False)

    print(f"Train slices: {len(train_df)}, Val slices: {len(val_df)}")

print("========== Step 3: Training MedSAM2 (Predictor path) ==========")

cmd_train = [
    PYTHON, str(PROJECT_ROOT / "src" / "medsam2_train.py"),
    "--manifest", str(manifest),
    "--train_split", str(slice_train_csv),
    "--val_split", str(slice_val_csv),
    "--sam2_ckpt", str(cfg.medsam2_ckpt),
    "--sam2_cfg", "configs/sam2.1_hiera_t512",
    "--ckpt_dir", str(ckpt_dir),
    "--batch_size", "4",
    "--box_jitter", "0.10",
    "--num_workers", "1",
    "--amp",
    "--resume",
]
subprocess.run(cmd_train, check=True, cwd=PROJECT_ROOT, env=env)

print("\n🎯 Training complete. Checkpoints saved to:", ckpt_dir)
