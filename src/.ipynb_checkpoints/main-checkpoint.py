# import os
# import sys
# import subprocess
# import warnings
# from pathlib import Path
# import pandas as pd
# # internal imports
# from config import cfg
# from data_cleaning import run_cleaning, generate_test_manifest

# warnings.filterwarnings("ignore")

# # --------------------------------------------------------------------
# #  Path Setup
# # --------------------------------------------------------------------
# PROJECT_ROOT = Path(__file__).resolve().parent.parent

# raw_train_root = cfg.raw_train_root     
# train_csv     = cfg.raw_train_csv      
# val_csv       = cfg.raw_val_csv   

# raw_test_csv  = cfg.raw_test_csv
# raw_test_root = cfg.raw_test_root

# preprocessed_dir = cfg.preprocessed_dir
# embeddings_dir   = cfg.embeddings_dir
# ckpt_dir         = cfg.ckpt_dir
# splits_dir       = cfg.splits_dir
# pred_dir_test    = Path("/scratch/sbv2019/mri/experiments/preds_test_full")

# for d in [preprocessed_dir, embeddings_dir, ckpt_dir, pred_dir_test]:
#     d.mkdir(parents=True, exist_ok=True)

# PYTHON = sys.executable
# env = os.environ.copy()
# env["PYTHONPATH"] = str(PROJECT_ROOT)

# # --------------------------------------------------------------------
# #  Step 1 - Preprocess TRAIN
# # --------------------------------------------------------------------
# print("========== Step 1A: Train Preprocessing ==========")

# # discover T2 + prostate mask pairs
# train_pairs = []
# for subdir, _, files in os.walk(raw_train_root):
#     subdir = Path(subdir)
#     if "t2.nii.gz" in files and "t2_anatomy_reader1.nii.gz" in files:
#         train_pairs.append((
#             subdir / "t2.nii.gz",
#             subdir / "t2_anatomy_reader1.nii.gz"
#         ))
        
# if len(train_pairs) == 0:
#     raise RuntimeError("❌ No T2 + anatomy mask pairs in training folder!")

# # run_cleaning(train_pairs, preprocessed_dir)
# train_manifest = preprocessed_dir / "manifest.csv"

# if train_manifest.exists():
#     print("⚠️ Preprocessed folder already exists — skipping preprocessing.")
# else:
#     print("🔄 Running preprocessing (first time)...")
#     run_cleaning(train_pairs, preprocessed_dir)

# print("✅ Training data preprocessed.")

# # --------------------------------------------------------------------
# #  Step 1B - Preprocess TEST (slice-level PNGs)
# # --------------------------------------------------------------------
# print("========== Step 1B: Test Preprocessing ==========")

# test_manifest = preprocessed_dir / "test_manifest.csv"

# if test_manifest.exists():
#     print("⚠️ Test preprocessing already exists — skipping.")
# else:
#     test_manifest = generate_test_manifest(
#         raw_test_root = raw_test_root,
#         out_dir = preprocessed_dir,
#         image_size = cfg.image_size
#     )


# print(f"✅ Test manifest saved to {test_manifest}\n")

# # --------------------------------------------------------------------
# #  Step 1C - Build slice-level Train/Val splits from Prostate158 CSVs
# # --------------------------------------------------------------------

# print("========== Step 1C: Building slice-level Train/Val splits ==========")
# splits_dir.mkdir(parents=True, exist_ok=True)

# slice_train_csv = splits_dir / "train.csv"
# slice_val_csv   = splits_dir / "val.csv"

# if slice_train_csv.exists() and slice_val_csv.exists():
#     print(f"⚠️ Slice-level splits already exist in {splits_dir} — skipping.")
# else:
#     man = pd.read_csv(train_manifest, dtype={"case": str})    # columns: image, mask, case, slice
#     ds_train = pd.read_csv(train_csv)   # official Prostate158 train.csv
#     ds_val   = pd.read_csv(val_csv)     # official Prostate158 valid.csv

#     print("Got train csv as:",len(ds_train),"  val as:",len(ds_val),"  manifest csv as:",len(man))
#     # Official 'ID' column is integers (e.g. 20) while manifest.case is "020"
#     train_cases = {f"{int(i):03d}" for i in ds_train["ID"].tolist()}
#     val_cases   = {f"{int(i):03d}" for i in ds_val["ID"].tolist()}
#     # print("train cases",len(train_cases), train_cases)

#     train_df = man[man["case"].isin(train_cases)]
#     # print("train df",len(train_df))
#     val_df   = man[man["case"].isin(val_cases)]

#     train_df[["image"]].to_csv(slice_train_csv, index=False)
#     val_df[["image"]].to_csv(slice_val_csv, index=False)

#     print(f"✅ Wrote slice-level splits to {splits_dir}")
#     print(f"   Train slices: {len(train_df)}, Val slices: {len(val_df)}")



# # --------------------------------------------------------------------
# # Step 2A - Embedding Extraction (TRAIN)
# # --------------------------------------------------------------------
# print("========== Step 2A: Train Embedding Extraction ==========")

# existing_embs = list((embeddings_dir).glob("*.pt"))
# if len(existing_embs) > 0:
#     print(f"⚠️  Skipping train embeddings — found {len(existing_embs)} embeddings.")
# else:
#     cmd_embed_train = [
#         PYTHON, "-m", "src.medsam_embedder",
#         "--checkpoint", str(cfg.medsam_ckpt),
#         "--preprocessed_dir", str(preprocessed_dir),
#         "--embeddings_dir", str(embeddings_dir),
#     ]
    
#     subprocess.run(cmd_embed_train, check=True, cwd=PROJECT_ROOT, env=env)
#     print("✅ Train embeddings saved.\n")

# # --------------------------------------------------------------------
# # Step 2B - Embedding Extraction (TEST)
# # --------------------------------------------------------------------
# print("========== Step 2B: Test Embedding Extraction ==========")

# test_emb_dir = embeddings_dir / "test"
# test_emb_dir.mkdir(parents=True, exist_ok=True)

# existing_test_embs = list(test_emb_dir.glob("*.pt"))
# if len(existing_test_embs) > 0:
#     print(f"⚠️  Skipping test embeddings — found {len(existing_test_embs)} embeddings in {test_emb_dir}.")
# else:
#     cmd_embed_test = [
#         PYTHON, "-m", "src.medsam_embedder",
#         "--checkpoint", str(cfg.medsam_ckpt),
#         "--preprocessed_dir", str(preprocessed_dir / "test"),
#         "--embeddings_dir", str(embeddings_dir / "test"),
#     ]
#     subprocess.run(cmd_embed_test, check=True, cwd=PROJECT_ROOT, env=env)
    
#     print("✅ Test embeddings saved.\n")

# # --------------------------------------------------------------------
# # Step 3 - Train Decoder
# # --------------------------------------------------------------------
# print("========== Step 3: Training Decoder ==========")

# cmd_train = [
#     PYTHON, "-m", "src.train",
#     "--manifest", str(preprocessed_dir / "manifest.csv"),
#     "--train_split", str(splits_dir / "train.csv"),
#     "--val_split", str(splits_dir / "val.csv"),
#     "--embeddings_dir", str(embeddings_dir),
#     "--ckpt_dir", str(ckpt_dir),
#     "--amp",
#  ]

# subprocess.run(cmd_train, check=True, cwd=PROJECT_ROOT, env=env)
# print(f"✅ Training complete. Checkpoints stored in {ckpt_dir}\n")

# best_checkpoint = ckpt_dir / f"best_{cfg.embedding_version}.pt"

# # --------------------------------------------------------------------
# # Step 4 - Test Inference
# # --------------------------------------------------------------------
# print("========== Step 4: Test Inference ==========")

# cmd_infer = [
#     PYTHON, "-m", "src.inference",
#     "--checkpoint", str(best_checkpoint),
#     "--embeddings_dir", str(embeddings_dir / "test"),
#     "--manifest", str(test_manifest),
#     "--out_dir", str(pred_dir_test),
# ]

# subprocess.run(cmd_infer, check=True, cwd=PROJECT_ROOT, env=env)
# print(f"✅ Test predictions saved to {pred_dir_test}\n")

# # --------------------------------------------------------------------
# # Step 5 - Test Evaluation
# # --------------------------------------------------------------------
# print("========== Step 5: Test Evaluation ==========")

# cmd_test = [
#     PYTHON, "-m", "src.test_eval",
#     "--checkpoint", str(best_checkpoint),
#     "--manifest", str(test_manifest),
#     "--embeddings_dir", str(embeddings_dir / "test"),
# ]

# subprocess.run(cmd_test, check=True, cwd=PROJECT_ROOT, env=env)
# print("🎯 Test evaluation complete!")

# # --------------------------------------------------------------------
# print("\n🎯 Pipeline complete! All stages finished successfully.")

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


import os
import sys
import subprocess
import warnings
from pathlib import Path
import pandas as pd

from config import cfg
from data_cleaning import run_cleaning, generate_test_manifest

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
env = os.environ.copy()
env["PYTHONPATH"] = str(PROJECT_ROOT)

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
raw_train_root = cfg.raw_train_root
train_csv = cfg.raw_train_csv
val_csv = cfg.raw_val_csv

preprocessed_dir = cfg.preprocessed_dir
splits_dir = cfg.splits_dir
ckpt_dir = cfg.ckpt_dir

for d in [preprocessed_dir, splits_dir, ckpt_dir]:
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# Step 1A - Train preprocessing
# --------------------------------------------------------------------
print("========== Step 1A: Train Preprocessing ==========")

train_pairs = []
for subdir, _, files in os.walk(raw_train_root):
    subdir = Path(subdir)
    if "t2.nii.gz" in files and "t2_anatomy_reader1.nii.gz" in files:
        train_pairs.append((
            subdir / "t2.nii.gz",
            subdir / "t2_anatomy_reader1.nii.gz"
        ))

if len(train_pairs) == 0:
    raise RuntimeError("❌ No T2 + mask pairs found")

manifest = preprocessed_dir / "manifest.csv"
if not manifest.exists():
    run_cleaning(train_pairs, preprocessed_dir)

print("✅ Preprocessing done")

# --------------------------------------------------------------------
# Step 1C - Slice-level train/val splits
# --------------------------------------------------------------------
print("========== Step 1C: Train/Val splits ==========")

slice_train_csv = splits_dir / "train.csv"
slice_val_csv = splits_dir / "val.csv"

if not slice_train_csv.exists():
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

# --------------------------------------------------------------------
# Step 3 - Train MedSAM Decoder (NO EMBEDDINGS)
# --------------------------------------------------------------------
print("========== Step 3: Training MedSAM Decoder ==========")

cmd_train = [
    PYTHON, "-m", "src.train",
    "--manifest", str(manifest),
    "--train_split", str(slice_train_csv),
    "--val_split", str(slice_val_csv),
    "--sam_ckpt", str(cfg.medsam_ckpt),
    "--ckpt_dir", str(ckpt_dir),
    "--batch_size", "8",
    "--box_jitter", "0.10",
    "--unfreeze_last_blocks", "2",
    "--amp",

    # NEW: BoxNet + prompt-less
    "--train_boxnet",
    "--val_promptless",
    "--exclude_empty_val",
    "--promptless_prob", "0.30",
    "--boxnet_lr", "1e-3",
    "--boxnet_weight", "1.0",
    "--boxnet_thresh", "0.30",
]

cmd_train.append("--resume")


subprocess.run(cmd_train, check=True, cwd=PROJECT_ROOT, env=env)

print("\n🎯 Training complete. Checkpoints saved to:", ckpt_dir)

# --------------------------------------------------------------------
# Step 4 - Save qualitative results for ALL test cases (from NIfTI)
# --------------------------------------------------------------------
print("========== Step 4: Saving qualitative results (ALL test cases) ==========")

import torch
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn.functional as F
from src.boxnet import BoxUNet
from src.medsam_baseline import build_sam_baseline
from src.medsam_forward import sam_forward_logits
from src.metrics import dice_coef
from src.config import cfg

device = "cuda" if torch.cuda.is_available() else "cpu"

OUT_ROOT = Path("/scratch/sbv2019/mri/qualitative_test_nii")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------- Load test CSV ----------------
test_df = pd.read_csv(cfg.raw_test_csv)
print(f"Loaded test CSV with {len(test_df)} cases")

# ---------------- Load trained MedSAM ----------------
sam = build_sam_baseline(
    ckpt_path=str(cfg.medsam_ckpt),
    model_type="vit_b",
    device=device,
    unfreeze_last_blocks=0,
)

ckpt = torch.load(cfg.ckpt_dir / "best_sam.pt", map_location=device)
sam.mask_decoder.load_state_dict(ckpt["mask_decoder"])
sam.image_encoder.load_state_dict(ckpt["image_encoder"])
sam.eval()

print(f"Loaded trained MedSAM (epoch {ckpt['epoch']}, val Dice = {ckpt['val_dice']:.4f})")

# ---- Load BoxNet from checkpoint (if present) ----
boxnet = None
if ckpt.get("boxnet") is not None:
    boxnet = BoxUNet(in_ch=3, base=32).to(device)
    boxnet.load_state_dict(ckpt["boxnet"])
    boxnet.eval()
    print("✅ Loaded BoxNet from best_sam.pt")
else:
    print("⚠️ No BoxNet weights found in best_sam.pt (ckpt['boxnet'] is None)")


TARGET_H, TARGET_W = cfg.image_size  # should be (1024, 1024)

all_dices = []

with torch.no_grad():
    for _, row in test_df.iterrows():
        case_id = str(row["ID"]).zfill(3)

        raw_case_dir = cfg.raw_test_root / case_id
        img_path = raw_case_dir / "t2.nii.gz"
        msk_path = raw_case_dir / "t2_anatomy_reader1.nii.gz"

        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")
        if not msk_path.exists():
            raise FileNotFoundError(f"Missing mask: {msk_path}")

        print(f"Processing case {case_id}")

        case_dir = OUT_ROOT / f"case_{case_id}"
        case_dir.mkdir(parents=True, exist_ok=True)

        img_vol = nib.load(img_path).get_fdata()
        msk_vol = nib.load(msk_path).get_fdata()
        assert img_vol.shape == msk_vol.shape, "Image/mask shape mismatch"

        case_dices = []

        for z in range(img_vol.shape[2]):
            img_slice = img_vol[:, :, z].astype(np.float32)
            msk_slice = msk_vol[:, :, z].astype(np.float32)

            if msk_slice.sum() == 0:
                continue

            # ---- normalize slice ----
            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-6)

            # ---- to torch ----
            img_t = torch.from_numpy(img_slice)[None, None, ...]  # (1,1,H,W)
            msk_t = torch.from_numpy(msk_slice)[None, None, ...]  # (1,1,H,W)

            # ---- resize to 1024×1024 ----
            img_t = F.interpolate(img_t, size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False)
            msk_t = F.interpolate(msk_t, size=(TARGET_H, TARGET_W), mode="nearest")

            # ---- convert to MedSAM expected shapes ----
            img = img_t.repeat(1, 3, 1, 1).to(device)     # (1,3,1024,1024)
            msk = msk_t[:, 0].to(device)                  # (1,1024,1024)

            # logits = sam_forward_logits(
            #     sam,
            #     img,
            #     msk,                 # GT-prompted (upper bound)
            #     training=False,
            #     box_jitter_frac=0.0,
            # )
            logits = sam_forward_logits(
                sam,
                img,
                gt_masks=None,          # ✅ do NOT pass GT to model
                training=False,
                box_jitter_frac=0.0,
                boxnet=boxnet,
                boxnet_thresh=ckpt.get("boxnet_thresh", 0.30),
            )

            dice = dice_coef(logits, msk)
            case_dices.append(dice)
            all_dices.append(dice)

            print(f"Case {case_id} | Slice {z:03d} | Dice = {dice:.4f}")

            pred = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy()

            # save the *resized* views for consistent visualization
            img_np = img[0, 0].cpu().numpy()
            gt_np = msk[0].cpu().numpy()

            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(img_np, cmap="gray"); ax[0].set_title("Image (1024)")
            ax[1].imshow(gt_np, cmap="gray");  ax[1].set_title("GT (1024)")
            ax[2].imshow(pred, cmap="gray");   ax[2].set_title("Prediction")

            for a in ax:
                a.axis("off")

            fig.savefig(case_dir / f"slice_{z:03d}.png", dpi=150)
            plt.close(fig)

        if case_dices:
            print(f"Case {case_id} | Mean Dice = {sum(case_dices)/len(case_dices):.4f}")

if all_dices:
    print(f"ALL TEST CASES | Mean Dice = {sum(all_dices)/len(all_dices):.4f}")

print(f"✅ Saved qualitative results for ALL test cases to {OUT_ROOT}")
