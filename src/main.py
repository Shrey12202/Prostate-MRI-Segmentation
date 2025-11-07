"""Main pipeline runner for MedSAM → U-Net Decoder (Prostate158)."""



import os

from pathlib import Path

import subprocess

from config import cfg

from data_cleaning import run_cleaning



def discover_pairs(raw_root: Path):

    """

    Finds (image, mask) pairs for Prostate158-style dataset.

    Supports both .nii and .nii.gz, and both 'tumour'/'tumor' spellings.

    Works for nested structure like test_dataset/test/001/.

    """

    pairs = []

    for subdir, _, files in os.walk(raw_root):

        subdir_path = Path(subdir)

        for f in files:

            f_low = f.lower()



            # Look for t2 or adc MRI image files

            if ("t2" in f_low or "adc" in f_low) and not ("tumor" in f_low or "tumour" in f_low):

                img_path = subdir_path / f



                # Determine matching mask names

                if "t2" in f_low:

                    possible_masks = ["t2_tumor_reader1.nii", "t2_tumor_reader1.nii.gz",

                                      "t2_tumour_reader1.nii", "t2_tumour_reader1.nii.gz"]

                elif "adc" in f_low:

                    possible_masks = ["adc_tumor_reader1.nii", "adc_tumor_reader1.nii.gz",

                                      "adc_tumour_reader1.nii", "adc_tumour_reader1.nii.gz"]

                else:

                    continue



                # Look for mask file in same folder

                mask_path = None

                for m in possible_masks:

                    if (subdir_path / m).exists():

                        mask_path = subdir_path / m

                        break



                if mask_path:

                    pairs.append((img_path, mask_path))

                    print(f"✅ Found pair: {img_path.relative_to(raw_root)} <-> {mask_path.relative_to(raw_root)}")

                else:

                    print(f"⚠️ No mask found for {img_path.relative_to(raw_root)}")



    if not pairs:

        raise RuntimeError(f"No image/mask pairs found under {raw_root}")

    print(f"\n✅ Total pairs found: {len(pairs)}\n")

    return pairs

    

import warnings

warnings.filterwarnings("ignore")

print("========== Step 1: Data Cleaning ==========")

raw_data_root: Path = Path("/home/pg2825/Prostate-MRI-Segmentation-main/test_dataset")

# Example: Path("/scratch/<username>/Prostate158_small")



# You can keep these as debug experiment dirs

preprocessed_dir: Path = Path("/scratch/pg2825/mri/experiments/preprocessed_debug")

embeddings_dir: Path = Path("/scratch/pg2825/mri/experiments/embeddings_debug")

splits_dir: Path = Path("/scratch/pg2825/mri/experiments/splits_debug")

ckpt_dir: Path = Path("/scratch/pg2825/mri/checkpoints_debug")



pairs = discover_pairs(raw_data_root)
"""
run_cleaning(pairs, preprocessed_dir)
"""
print(f"✅ Preprocessed data saved to {preprocessed_dir}")




print("========== Step 2: Embedding Extraction ==========")

cmd_embed = [

    "python", "-m", "medsam_embedder",

    "--checkpoint", str(cfg.medsam_ckpt),

    "--preprocessed_dir", str(preprocessed_dir),

    "--embeddings_dir", str(embeddings_dir)

]

"""subprocess.run(cmd_embed, check=True)
"""
print("========== Step 3: Training Decoder ==========")

cmd_train = [

    "python", "-m", "train", "--amp",

]

subprocess.run(cmd_train, check=True)



print("========== Step 4: Inference ==========")

cmd_infer = [

    "python", "-m", "inference",

    "--checkpoint", str(cfg.ckpt_dir/"best.pt")

]

subprocess.run(cmd_infer, check=True)



print("Pipeline complete! Predictions saved under experiments/preds.")

