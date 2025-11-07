from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    # ---------- Paths ----------
    # Point this to your current small dataset folders
    raw_data_root: Path = Path("/home/pg2825/Prostate-MRI-Segmentation-main/test_dataset")
    # Example: Path("/scratch/<username>/Prostate158_small")

    # You can keep these as debug experiment dirs
    preprocessed_dir: Path = Path("/scratch/mri/experiments/preprocessed_debug")
    embeddings_dir: Path = Path("/scratch/mri/experiments/embeddings_debug")
    splits_dir: Path = Path("/scratch/mri/experiments/splits_debug")
    ckpt_dir: Path = Path("/scratch/mri/checkpoints_debug")
    log_dir: Path = Path("logs_debug")

    # ---------- MedSAM weights ----------
    # You can place medsam_vit_b.pth inside src/ or anywhere else
    medsam_ckpt: Path = Path("medsam_vit_b.pth")
    # Example alternative:
    # medsam_ckpt: Path = Path("/scratch/<username>/checkpoints/medsam_vit_b.pth")
    medsam_backbone: str = "vit_b"  # ViT-B encoder backbone

    # ---------- Data ----------
    image_size: tuple[int, int] = (1024, 1024)
    robust_norm: tuple[float, float] = (1.0, 99.0)
    use_bbox_crop: bool = True
    bbox_pad_px: int = 16

    # ---------- Embeddings ----------
    embedding_channels: int = 256
    embedding_stride: int = 4

    # ---------- Training ----------
    num_classes: int = 2
    batch_size: int = 1          # small for dry run
    num_workers: int = 0         # avoids multiprocessing issues on HPC
    num_epochs: int = 1          # one epoch to test pipeline
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
