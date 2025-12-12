from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    # ---------- Paths ----------
    # raw_data_root: Path = Path("/home/sbv2019/Prostate-MRI-Segmentation-main/test_dataset")
    
    raw_train_root: Path = Path("/scratch/sbv2019/mri/prostate158_full/prostate158_train/train")
    raw_train_csv: Path = Path("/scratch/sbv2019/mri/prostate158_full/prostate158_train/train.csv")
    raw_val_csv: Path = Path("/scratch/sbv2019/mri/prostate158_full/prostate158_train/valid.csv")
    raw_test_root = Path("/scratch/sbv2019/mri/prostate158_full/prostate158_test/test")
    raw_test_csv  = Path("/scratch/sbv2019/mri/prostate158_full/prostate158_test/test.csv")

    #preprocessed_dir: Path = Path("/scratch/sbv2019/mri/experiments/preprocessed_debug")
    #embeddings_dir: Path = Path("/scratch/sbv2019/mri/experiments/embeddings_debug")
    #splits_dir: Path = Path("/scratch/sbv2019/mri/experiments/splits_debug")
    #ckpt_dir: Path = Path("/scratch/sbv2019/mri/checkpoints_debug")
    
    preprocessed_dir: Path = Path("/scratch/sbv2019/mri/experiments/preprocessed_full")
    ckpt_dir:         Path = Path("/scratch/sbv2019/mri/checkpoints_full")
    pred_dir:         Path = Path("/scratch/sbv2019/mri/experiments/preds_full")
    splits_dir:       Path = Path("/scratch/sbv2019/mri/experiments/splits_full")
    
    # Choose embedding variant here
    embedding_version = "base64"   # or "base32"

    embedding_root = Path("/scratch/sbv2019/mri/experiments")

    embeddings_dir: Path = (
        embedding_root / f"embeddings_full_{embedding_version}"
    )


    log_dir: Path = Path("/scratch/sbv2019/mri/experiments/logs_debug_full")

    # ---------- MedSAM weights ----------
    medsam_ckpt: Path = Path("/scratch/sbv2019/mri/models/medsam/medsam_vit_b.pth")
    # Example alternative:
    medsam_backbone: str = "vit_b"  # ViT-B encoder backbone

    # ---------- Data ----------
    image_size: tuple[int, int] = (1024, 1024)
    robust_norm: tuple[float, float] = (1.0, 99.0)
    use_bbox_crop: bool = False
    bbox_pad_px: int = 16

    # ---------- Embeddings ----------
    embedding_channels: int = 256
    embedding_stride: int = 16

    # ---------- Training ----------
    num_classes: int = 1
    # num_classes: int = 2
    batch_size: int = 32           # small for dry run
    num_workers: int = 8        # avoids multiprocessing issues on HPC
    num_epochs: int = 40        # one epoch to test pipeline
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    amp: bool = True
    seed: int = 1337

    # ---------- Augmentation ----------
    rotation_deg: float = 10.0
    hflip_p: float = 0.5
    vflip_p: float = 0.5
    brightness: float = 0.15
    contrast: float = 0.15
    translate: float = 0.05
    scale: tuple[float, float] = (0.95, 1.05)

cfg = Config()
