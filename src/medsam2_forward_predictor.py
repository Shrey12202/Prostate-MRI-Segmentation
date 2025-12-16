import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


@torch.no_grad()
def infer_mask_from_box(
    img_t,         # (3,H,W) torch
    gt_mask,       # (H,W) torch
    sam2_ckpt,
    sam2_cfg,
    device="cuda",
):
    sam = build_sam2(
        config_file=sam2_cfg,
        ckpt_path=sam2_ckpt,
        device=device,
    )
    predictor = SAM2ImagePredictor(sam)

    img_np = img_t.permute(1, 2, 0).cpu().numpy()
    if img_np.max() <= 1.5:
        img_np = (img_np * 255).astype(np.uint8)

    predictor.set_image(img_np)

    ys, xs = torch.where(gt_mask > 0)
    if ys.numel() == 0:
        box = np.array([[0, 0, img_np.shape[1] - 1, img_np.shape[0] - 1]])
    else:
        box = np.array([[xs.min(), ys.min(), xs.max(), ys.max()]])

    masks, _, _ = predictor.predict(
        box=box,
        multimask_output=False,
    )

    return torch.from_numpy(masks[0]).float()
