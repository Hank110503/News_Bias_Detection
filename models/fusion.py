import torch
import torch.nn as nn

class ConcatFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img_feat, txt_feat):
        return torch.cat([img_feat, txt_feat], dim=1)


class CrossAttentionFusion(nn.Module):
    def __init__(self, img_dim, txt_dim,hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()

        # 图像: 512 -> hidden_dim
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        
        # 文本: 1024 -> hidden_dim
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)

        self.text_to_image_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.image_to_text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, image_feat, text_feat):
        """
        text_feat  : [B, D]
        image_feat : [B, D]
        """
        # 1. 维度投影 (Projection)
        # text_feat 变成 [B, hidden_dim]
        # image_feat 变成 [B, hidden_dim]
        text_feat = self.txt_proj(text_feat)
        image_feat = self.img_proj(image_feat)

        # 2. 增加序列维度 [B, D] -> [B, 1, D]
        # 因为这里是全局特征 pooling 后的向量，不是序列，所以长度为 1
        text_feat = text_feat.unsqueeze(1)
        image_feat = image_feat.unsqueeze(1)

        # 3. Text attends to Image
        # Query: Text, Key/Value: Image
        # text attends to image
        t2i, _ = self.text_to_image_attn(
            query=text_feat,
            key=image_feat,
            value=image_feat
        )

        text_feat = self.norm1(text_feat + t2i)

        # image attends to text
        i2t, _ = self.image_to_text_attn(
            query=image_feat,
            key=text_feat,
            value=text_feat
        )

        image_feat = self.norm2(image_feat + i2t)

        fused = torch.cat([text_feat, image_feat], dim=-1)

        return fused.squeeze(1)