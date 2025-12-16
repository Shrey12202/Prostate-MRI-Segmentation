# """
# Preprocess TEST dataset safely without touching existing code.

# Strategy:
# 1. Run run_cleaning() on test data into a TEMP directory
# 2. Read temp manifest.csv
# 3. Append rows to main manifest.csv
# 4. Write combined manifest.csv

# No existing code is modified.
# """

# import os
# import shutil
# from pathlib import Path

# import pandas as pd

# from src.config import cfg
# from src.data_cleaning import run_cleaning


# def main():
#     print("========== Preprocessing TEST dataset (SAFE MODE) ==========")

#     # ------------------------------------------------------------
#     # Paths
#     # ------------------------------------------------------------
#     main_preprocessed_dir = cfg.preprocessed_dir
#     main_manifest = main_preprocessed_dir / "manifest.csv"

#     if not main_manifest.exists():
#         raise RuntimeError(
#             f"Main manifest not found: {main_manifest}\n"
#             "Run training preprocessing first."
#         )

#     tmp_preprocessed_dir = cfg.preprocessed_dir.parent / "_preprocessed_test_tmp"

#     if tmp_preprocessed_dir.exists():
#         raise RuntimeError(
#             f"Temporary directory already exists: {tmp_preprocessed_dir}\n"
#             "Delete it manually before re-running."
#         )

#     # ------------------------------------------------------------
#     # Collect test pairs
#     # ------------------------------------------------------------
#     test_pairs = []
#     for subdir, _, files in os.walk(cfg.raw_test_root):
#         subdir = Path(subdir)
#         if "t2.nii.gz" in files and "t2_anatomy_reader1.nii.gz" in files:
#             test_pairs.append((
#                 subdir / "t2.nii.gz",
#                 subdir / "t2_anatomy_reader1.nii.gz"
#             ))

#     if not test_pairs:
#         raise RuntimeError("❌ No test T2 + mask pairs found")

#     print(f"Found {len(test_pairs)} test volumes")

#     # ------------------------------------------------------------
#     # Run preprocessing into TEMP directory
#     # ------------------------------------------------------------
#     print(f"Running test preprocessing into temp dir: {tmp_preprocessed_dir}")
#     run_cleaning(
#         test_pairs,
#         tmp_preprocessed_dir
#     )

#     tmp_manifest = tmp_preprocessed_dir / "manifest.csv"
#     if not tmp_manifest.exists():
#         raise RuntimeError("Temp manifest.csv not created")

#     # ------------------------------------------------------------
#     # Merge manifests
#     # ------------------------------------------------------------
#     df_main = pd.read_csv(main_manifest)
#     df_test = pd.read_csv(tmp_manifest)

#     print(f"Main manifest rows before: {len(df_main)}")
#     print(f"Test manifest rows:        {len(df_test)}")

#     # Safety: ensure same columns
#     if list(df_main.columns) != list(df_test.columns):
#         raise RuntimeError(
#             "Manifest column mismatch between train and test.\n"
#             f"Train columns: {df_main.columns.tolist()}\n"
#             f"Test columns:  {df_test.columns.tolist()}"
#         )

#     # Optional deduplication guard
#     df_combined = pd.concat([df_main, df_test], ignore_index=True)
#     df_combined = df_combined.drop_duplicates(subset=["image", "mask"])

#     df_combined.to_csv(main_manifest, index=False)

#     print(f"Combined manifest rows after: {len(df_combined)}")

#     # ------------------------------------------------------------
#     # Cleanup temp directory
#     # ------------------------------------------------------------
#     shutil.rmtree(tmp_preprocessed_dir)
#     print("🧹 Cleaned up temporary directory")

#     print("✅ Test preprocessing appended successfully")


# if __name__ == "__main__":
#     main()

import pandas as pd
from pathlib import Path

from src.config import cfg
import re

def norm_case(x: str) -> str:
    """
    Convert various case formats to a canonical 3-digit ID.
    Examples:
      "1" -> "001"
      "001" -> "001"
      "case_12" -> "012"
      "test/007/t2.nii.gz" -> "007"  (if passed a path)
    """
    s = str(x)
    # pull the last 1-4 digit group (handles case_12, patient003 etc.)
    m = re.findall(r"\d+", s)
    if not m:
        return s  # fallback, but should not happen
    return m[-1].zfill(3)


def extract_case_from_path(p: str) -> str:
    # expected like "test/001/t2.nii.gz" -> "001"
    parts = Path(p).parts
    if len(parts) >= 2:
        return str(parts[1])
    raise ValueError(f"Unexpected path format: {p}")


def main():
    manifest_csv = cfg.preprocessed_dir / "manifest.csv"
    raw_test_csv = cfg.raw_test_csv
    out_csv = cfg.splits_dir / "test.csv"

    cfg.splits_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_csv)
    raw_test = pd.read_csv(raw_test_csv)

    # ---- Build test case list from raw_test["t2"] ----
    if "t2" not in raw_test.columns:
        raise RuntimeError(f"'t2' column not found in {raw_test_csv}")

    # raw test cases (already extracted as "001", "002", ...)
    test_cases = sorted(raw_test["t2"].astype(str).apply(extract_case_from_path).apply(norm_case).unique().tolist())
    print(f"Found {len(test_cases)} test cases (from raw test CSV). Example: {test_cases[:5]}")
    
    # normalize manifest case ids
    if "case" not in manifest.columns:
        if "image" not in manifest.columns:
            raise RuntimeError("manifest.csv missing both 'case' and 'image' columns; cannot derive case ids.")
        manifest["case"] = manifest["image"].astype(str)
    
    manifest["case_norm"] = manifest["case"].apply(norm_case)
    
    # filter
    test_df = manifest[manifest["case_norm"].isin(test_cases)].copy()
    
    if len(test_df) == 0:
        # Debug print: show a few manifest case samples to spot mismatch
        examples = manifest["case"].astype(str).head(10).tolist()
        examples_norm = manifest["case_norm"].astype(str).head(10).tolist()
        raise RuntimeError(
            "No test slices found after filtering manifest.\n"
            f"Example manifest['case'] values: {examples}\n"
            f"Example manifest normalized values: {examples_norm}\n"
            f"Expected test cases: {test_cases[:10]}"
        )
    
    # ensure required columns
    required = ["image", "mask", "case", "slice"]
    missing_cols = [c for c in required if c not in test_df.columns]
    if missing_cols:
        raise RuntimeError(f"manifest.csv missing required columns: {missing_cols}")
    
    # write split (keep original case column, but also overwrite to normalized to be consistent)
    test_df["case"] = test_df["case_norm"]
    test_df = test_df.sort_values(["case", "slice"])
    out_df = test_df[required]
    out_df.to_csv(out_csv, index=False)
    
    print(f"✅ Wrote slice-level test split: {out_csv}")
    print(f"   Slices: {len(out_df)} | Cases: {out_df['case'].nunique()}")



if __name__ == "__main__":
    main()
