import os
import sys
import subprocess
import warnings
from pathlib import Path
import pandas as pd
# internal imports
from config import cfg
from data_cleaning import run_cleaning, generate_test_manifest

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------
#  Path Setup
# --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

raw_train_root = cfg.raw_train_root     
train_csv     = cfg.raw_train_csv      
val_csv       = cfg.raw_val_csv   

raw_test_csv  = cfg.raw_test_csv
raw_test_root = cfg.raw_test_root

preprocessed_dir = cfg.preprocessed_dir
embeddings_dir   = cfg.embeddings_dir
ckpt_dir         = cfg.ckpt_dir
splits_dir       = cfg.splits_dir
pred_dir_test    = Path("/scratch/sbv2019/mri/experiments/preds_test_full")

for d in [preprocessed_dir, embeddings_dir, ckpt_dir, pred_dir_test]:
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
        
if len(train_pairs) == 0:
    raise RuntimeError("❌ No T2 + anatomy mask pairs in training folder!")

# run_cleaning(train_pairs, preprocessed_dir)
train_manifest = preprocessed_dir / "manifest.csv"

if train_manifest.exists():
    print("⚠️ Preprocessed folder already exists — skipping preprocessing.")
else:
    print("🔄 Running preprocessing (first time)...")
    run_cleaning(train_pairs, preprocessed_dir)

print("✅ Training data preprocessed.")

# --------------------------------------------------------------------
#  Step 1B - Preprocess TEST (slice-level PNGs)
# --------------------------------------------------------------------
print("========== Step 1B: Test Preprocessing ==========")

test_manifest = preprocessed_dir / "test_manifest.csv"

if test_manifest.exists():
    print("⚠️ Test preprocessing already exists — skipping.")
else:
    test_manifest = generate_test_manifest(
        raw_test_root = raw_test_root,
        out_dir = preprocessed_dir,
        image_size = cfg.image_size
    )


print(f"✅ Test manifest saved to {test_manifest}\n")

# --------------------------------------------------------------------
#  Step 1C - Build slice-level Train/Val splits from Prostate158 CSVs
# --------------------------------------------------------------------

print("========== Step 1C: Building slice-level Train/Val splits ==========")
splits_dir.mkdir(parents=True, exist_ok=True)

slice_train_csv = splits_dir / "train.csv"
slice_val_csv   = splits_dir / "val.csv"

if slice_train_csv.exists() and slice_val_csv.exists():
    print(f"⚠️ Slice-level splits already exist in {splits_dir} — skipping.")
else:
    man = pd.read_csv(train_manifest, dtype={"case": str})    # columns: image, mask, case, slice
    ds_train = pd.read_csv(train_csv)   # official Prostate158 train.csv
    ds_val   = pd.read_csv(val_csv)     # official Prostate158 valid.csv

    print("Got train csv as:",len(ds_train),"  val as:",len(ds_val),"  manifest csv as:",len(man))
    # Official 'ID' column is integers (e.g. 20) while manifest.case is "020"
    train_cases = {f"{int(i):03d}" for i in ds_train["ID"].tolist()}
    val_cases   = {f"{int(i):03d}" for i in ds_val["ID"].tolist()}
    # print("train cases",len(train_cases), train_cases)

    train_df = man[man["case"].isin(train_cases)]
    # print("train df",len(train_df))
    val_df   = man[man["case"].isin(val_cases)]

    train_df[["image"]].to_csv(slice_train_csv, index=False)
    val_df[["image"]].to_csv(slice_val_csv, index=False)

    print(f"✅ Wrote slice-level splits to {splits_dir}")
    print(f"   Train slices: {len(train_df)}, Val slices: {len(val_df)}")



# --------------------------------------------------------------------
# Step 2A - Embedding Extraction (TRAIN)
# --------------------------------------------------------------------
print("========== Step 2A: Train Embedding Extraction ==========")

existing_embs = list((embeddings_dir).glob("*.pt"))
if len(existing_embs) > 0:
    print(f"⚠️  Skipping train embeddings — found {len(existing_embs)} embeddings.")
else:
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

test_emb_dir = embeddings_dir / "test"
test_emb_dir.mkdir(parents=True, exist_ok=True)

existing_test_embs = list(test_emb_dir.glob("*.pt"))
if len(existing_test_embs) > 0:
    print(f"⚠️  Skipping test embeddings — found {len(existing_test_embs)} embeddings in {test_emb_dir}.")
else:
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

best_checkpoint = ckpt_dir / f"best_{cfg.embedding_version}.pt"

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
