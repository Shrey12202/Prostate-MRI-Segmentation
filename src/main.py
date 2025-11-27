import os
import sys
import subprocess
import warnings
from pathlib import Path

# internal imports
from config import cfg
from data_cleaning import run_cleaning, generate_test_manifest

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------
#  Path Setup
# --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

raw_train_root = PROJECT_ROOT / "test_dataset" / "valid"
raw_test_csv  = PROJECT_ROOT / "test.csv"
raw_test_root = PROJECT_ROOT / "test_dataset" / "test"

preprocessed_dir = Path("/scratch/sbv2019/mri/experiments/preprocessed_debug")
embeddings_dir   = Path("/scratch/sbv2019/mri/experiments/embeddings_debug")
splits_dir       = Path("/scratch/sbv2019/mri/experiments/splits_debug")
ckpt_dir         = Path("/scratch/sbv2019/mri/checkpoints_debug")
pred_dir_test    = Path("/scratch/sbv2019/mri/experiments/preds_test")

for d in [preprocessed_dir, embeddings_dir, splits_dir, ckpt_dir, pred_dir_test]:
    d.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable
env = os.environ.copy()
env["PYTHONPATH"] = str(PROJECT_ROOT)

# --------------------------------------------------------------------
#  Step 1 - Preprocess TRAIN
# --------------------------------------------------------------------
print("========== Step 1A: Train Preprocessing ==========")

# discover T2 + prostate mask pairs
train_pairs = []
for subdir, _, files in os.walk(raw_train_root):
    subdir = Path(subdir)
    if "t2.nii.gz" in files and "t2_anatomy_reader1.nii.gz" in files:
        train_pairs.append((
            subdir / "t2.nii.gz",
            subdir / "t2_anatomy_reader1.nii.gz"
        ))

run_cleaning(train_pairs, preprocessed_dir)
print("✅ Training data preprocessed.")

# --------------------------------------------------------------------
#  Step 1B - Preprocess TEST (slice-level PNGs)
# --------------------------------------------------------------------
print("========== Step 1B: Test Preprocessing ==========")

test_manifest = generate_test_manifest(
    raw_test_root = PROJECT_ROOT / "test_dataset" / "test",
    out_dir = preprocessed_dir,
    image_size = cfg.image_size
)


print(f"✅ Test manifest saved to {test_manifest}\n")

# --------------------------------------------------------------------
# Step 2A - Embedding Extraction (TRAIN)
# --------------------------------------------------------------------
print("========== Step 2A: Train Embedding Extraction ==========")

cmd_embed_train = [
    PYTHON, "-m", "src.medsam_embedder",
    "--checkpoint", str(cfg.medsam_ckpt),
    "--preprocessed_dir", str(preprocessed_dir),
    "--embeddings_dir", str(embeddings_dir),
]
subprocess.run(cmd_embed_train, check=True, cwd=PROJECT_ROOT, env=env)

print("✅ Train embeddings saved.\n")

# --------------------------------------------------------------------
# Step 2B - Embedding Extraction (TEST)
# --------------------------------------------------------------------
print("========== Step 2B: Test Embedding Extraction ==========")

cmd_embed_test = [
    PYTHON, "-m", "src.medsam_embedder",
    "--checkpoint", str(cfg.medsam_ckpt),
    "--preprocessed_dir", str(preprocessed_dir / "test"),
    "--embeddings_dir", str(embeddings_dir / "test"),
]
subprocess.run(cmd_embed_test, check=True, cwd=PROJECT_ROOT, env=env)

print("✅ Test embeddings saved.\n")

# --------------------------------------------------------------------
# Step 3 - Train Decoder
# --------------------------------------------------------------------
print("========== Step 3: Training Decoder ==========")

cmd_train = [
    PYTHON, "-m", "src.train",
    "--manifest", str(preprocessed_dir / "manifest.csv"),
    "--train_split", str(splits_dir / "train.csv"),
    "--val_split", str(splits_dir / "val.csv"),
    "--embeddings_dir", str(embeddings_dir),
    "--ckpt_dir", str(ckpt_dir),
    "--amp",
]

subprocess.run(cmd_train, check=True, cwd=PROJECT_ROOT, env=env)
print(f"✅ Training complete. Checkpoints stored in {ckpt_dir}\n")

best_checkpoint = ckpt_dir / "best.pt"

# --------------------------------------------------------------------
# Step 4 - Test Inference
# --------------------------------------------------------------------
print("========== Step 4: Test Inference ==========")

cmd_infer = [
    PYTHON, "-m", "src.inference",
    "--checkpoint", str(best_checkpoint),
    "--embeddings_dir", str(embeddings_dir / "test"),
    "--manifest", str(test_manifest),
    "--out_dir", str(pred_dir_test),
]

subprocess.run(cmd_infer, check=True, cwd=PROJECT_ROOT, env=env)
print(f"✅ Test predictions saved to {pred_dir_test}\n")

# --------------------------------------------------------------------
# Step 5 - Test Evaluation
# --------------------------------------------------------------------
print("========== Step 5: Test Evaluation ==========")

cmd_test = [
    PYTHON, "-m", "src.test_eval",
    "--checkpoint", str(best_checkpoint),
    "--manifest", str(test_manifest),
    "--embeddings_dir", str(embeddings_dir / "test"),
]

subprocess.run(cmd_test, check=True, cwd=PROJECT_ROOT, env=env)
print("🎯 Test evaluation complete!")

# --------------------------------------------------------------------
print("\n🎯 Pipeline complete! All stages finished successfully.")
