import os
import subprocess
import warnings
from pathlib import Path

# IMPORTANT: use src.<module> for ALL internal imports
from config import cfg
from data_cleaning import run_cleaning

warnings.filterwarnings("ignore")


# ========================================================================
# Step 1 — Pair discovery (unchanged except for imports)
# ========================================================================
def discover_pairs(raw_root: Path, modality="t2", reader="reader1"):
    import os
    from pathlib import Path

    pairs = []
    for subdir, _, files in os.walk(raw_root):
        subdir_path = Path(subdir)
        for f in files:
            f_low = f.lower()

            # Identify image files (t2.nii.gz)
            if modality in f_low and not any(k in f_low for k in ["tumor", "tumour", "mask", "seg"]):
                img_path = subdir_path / f

                # Expected mask name variants
                mask_candidates = [
                    f"{modality}_tumor_{reader}.nii",
                    f"{modality}_tumor_{reader}.nii.gz",
                    f"{modality}_tumour_{reader}.nii",
                    f"{modality}_tumour_{reader}.nii.gz",
                    f"{modality}_mask_{reader}.nii.gz",
                    f"{modality}_seg_{reader}.nii.gz",
                ]

                for mask_name in mask_candidates:
                    mask_path = subdir_path / mask_name
                    if mask_path.exists():
                        pairs.append((img_path, mask_path))
                        break

    if not pairs:
        raise RuntimeError(
            f"No {modality}-{reader} pairs found under {raw_root}. "
            "Ensure each patient folder has both image and mask files."
        )

    print(f"✅ Found {len(pairs)} {modality}-{reader} pairs under {raw_root}")
    return pairs


# ========================================================================
# Step 2 — Define directories & project root (REQUIRED FIX)
# ========================================================================

# PROJECT ROOT is parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# raw dataset path
raw_data_root = PROJECT_ROOT / "test_dataset" / "valid"

# HPC experiment directories
preprocessed_dir = Path("/scratch/sbv2019/mri/experiments/preprocessed_debug")
embeddings_dir   = Path("/scratch/sbv2019/mri/experiments/embeddings_debug")
splits_dir       = Path("/scratch/sbv2019/mri/experiments/splits_debug")
ckpt_dir         = Path("/scratch/sbv2019/mri/checkpoints_debug")

for d in [preprocessed_dir, embeddings_dir, splits_dir, ckpt_dir]:
    d.mkdir(parents=True, exist_ok=True)


# Prepare environment for ALL subprocesses
env = os.environ.copy()
env["PYTHONPATH"] = str(PROJECT_ROOT)


# ========================================================================
# Step 3 — Run pipeline stages
# ========================================================================

print("========== Step 1: Data Cleaning ==========")
pairs = discover_pairs(raw_data_root, modality="t2", reader="reader1")
run_cleaning(pairs, preprocessed_dir)
print(f"✅ Preprocessed data saved to {preprocessed_dir}\n")


print("========== Step 2: Embedding Extraction ==========")
cmd_embed = [
    "python", "-m", "src.medsam_embedder",
    "--checkpoint", str(cfg.medsam_ckpt),
    "--preprocessed_dir", str(preprocessed_dir),
    "--embeddings_dir", str(embeddings_dir),
]

subprocess.run(cmd_embed, check=True, cwd=PROJECT_ROOT, env=env)
print(f"✅ Embeddings saved to {embeddings_dir}\n")


print("========== Step 3: Training Decoder ==========")

cmd_train = [
    "python", "-m", "src.train",
    "--manifest", str(preprocessed_dir / "manifest.csv"),
    "--train_split", str(splits_dir / "train.csv"),
    "--val_split", str(splits_dir / "val.csv"),
    "--embeddings_dir", str(embeddings_dir),
    "--ckpt_dir", str(ckpt_dir),
    "--amp",
]

subprocess.run(cmd_train, check=True, cwd=PROJECT_ROOT, env=env)
print(f"✅ Training complete. Checkpoints in {ckpt_dir}\n")


print("========== Step 4: Inference ==========")
cmd_infer = [
    "python", "-m", "src.inference",
    "--checkpoint", str(ckpt_dir / "best.pt"),  # FIXED PATH
]

subprocess.run(cmd_infer, check=True, cwd=PROJECT_ROOT, env=env)
print("✅ Inference complete — predictions saved under experiments/preds/\n")


print("🎯 Pipeline complete! All stages finished successfully.")
