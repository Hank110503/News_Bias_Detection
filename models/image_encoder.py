import torch.nn as nn
from transformers import CLIPModel

class ImageEncoder(nn.Module):
    def __init__(self, clip_name, freeze=True):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name)

        if freeze:
            for p in self.clip.parameters():
                p.requires_grad = False

        self.output_dim = self.clip.config.projection_dim

    def forward(self, pixel_values):
        feat = self.clip.get_image_features(pixel_values=pixel_values)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat