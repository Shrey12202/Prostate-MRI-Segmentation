import torch
from transformers import SamModel

# 1. Load MedSAM ViT-Base weights from HuggingFace
model = SamModel.from_pretrained("wanglab/medsam-vit-base")

# 2. Save the state_dict as a .pth file
torch.save(model.state_dict(), "medsam_vit_b.pth")

print("Saved medsam_vit_b.pth successfully.")
