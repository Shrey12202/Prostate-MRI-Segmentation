# MedSAM → U-Net Decoder (Prostate158)

This repo provides a minimal, modular pipeline to train a light U-Net **decoder** on top of **frozen MedSAM** image-encoder features for prostate MRI segmentation (Prostate158). It produces separate Python files for cleaning, augmentation, dataset, model, training, and inference.

> ⚠️ You must install MedSAM/SAM and have a compatible checkpoint to extract embeddings. The code assumes a SAM-like API with an `image_encoder` method. Adjust `src/medsam_embedder.py` if your fork differs.

## Structure

```
src/
  config.py
  medsam_embedder.py
  train.py
  inference.py
  models/
    unet_decoder.py
  data/
    data_cleaning.py
    augmentation.py
    dataset.py
  utils/
    metrics.py
    train_utils.py
experiments/
checkpoints/
logs/
```

## Workflow

1. **Prepare Prostate158 pairs** (NIfTI image + mask for each case). Update paths in `src/config.py` (`raw_data_root`) or call the cleaners with explicit paths.

2. **Clean & preprocess** to consistent 2D slices:
   - Converts NIfTI volumes to 2D PNG slices, normalizes intensities using robust percentiles, and either bbox-crops around the prostate or resizes to `cfg.image_size` (default 256×256).
   - Writes a manifest `manifest.csv` listing image/mask paths.
   - Entry point: `run_cleaning(...)` in `src/data/data_cleaning.py`.

3. **Extract MedSAM embeddings** (frozen encoder):
   ```bash
   python -m src.medsam_embedder --checkpoint /path/to/medsam_ckpt.pth \
       --preprocessed_dir experiments/preprocessed_v1 \       --embeddings_dir experiments/embeddings_v1
   ```
   Saves **per-slice** `.pt` tensors shaped `(C, H/stride, W/stride)`. Configure `embedding_channels` and `embedding_stride` in `src/config.py` to match your encoder.

4. **Create splits** (CSV lists of `image` column values from the manifest) under `experiments/splits_v1/{train,val,test}.csv`.

5. **Train decoder**:
   ```bash
   python -m src.train --amp
   ```
   - Logs to TensorBoard under `logs/`.
   - Checkpoints to `checkpoints/` (`best.pt`, `last.pt`).

6. **Inference**:
   ```bash
   python -m src.inference --checkpoint checkpoints/best.pt
   ```
   Writes PNG masks to `experiments/preds/`.

## Notes & Tips

- If your MedSAM encoder outputs downsampled features at stride 16 or with channel dim 768, set `embedding_stride` and `embedding_channels` accordingly in `config.py`.
- The decoder is intentionally **small** (two upsampling stages) assuming stride=4 inputs. Increase stages if your stride is larger.
- For 3D aggregation, you can add simple post-processing like connected-components per volume or median filtering across adjacent slices.
- Dice is optimized via a soft-dice loss; feel free to mix with BCE or focal.
- Data augmentations are mild (flips/rot-translate/scale and brightness/contrast). You can extend with elastic deforms if desired.
