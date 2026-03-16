import torch
import torch.nn as nn
import torch.nn.functional as F

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
        fused_dim = hidden_dim * 2  # 因为后面做了 cat     
        self.ffn = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, fused_dim)
        )
        self.norm3 = nn.LayerNorm(fused_dim)
        
        self.dropout = nn.Dropout(dropout)

    # def forward(self, image_feat, text_feat):
    #     """
    #     输入升级：不再是 [B, D]，而是序列 [B, Seq_Len, D]
        
    #     Args:
    #         image_seq: [B, N_img_patches, img_dim] (来自 CLIP vision_model 的 last_hidden_state)
    #         text_seq:  [B, N_txt_tokens, txt_dim] (来自 XLM-R 的 last_hidden_state)
    #         attention_mask: [B, N_txt_tokens] (文本的 padding mask，用于忽略 <pad>)
            
    #     Returns:
    #         fused_feat: [B, hidden_dim * 2] (用于分类的全局特征)
    #         attn_weights: dict {'t2i': ..., 'i2t': ...} (用于可视化的注意力矩阵)
    #     """
    #     # 1. 维度投影 (Projection)
    #     # text_feat 变成 [B, hidden_dim]
    #     # image_feat 变成 [B, hidden_dim]
    #     text_feat = self.txt_proj(text_feat)
    #     image_feat = self.img_proj(image_feat)

    #     # 2. 增加序列维度 [B, D] -> [B, 1, D]
    #     # 因为这里是全局特征 pooling 后的向量，不是序列，所以长度为 1
    #     text_feat = text_feat.unsqueeze(1)
    #     image_feat = image_feat.unsqueeze(1)

    #     # 3. Text attends to Image
    #     # Query: Text, Key/Value: Image
    #     # text attends to image
    #     t2i, _ = self.text_to_image_attn(
    #         query=text_feat,
    #         key=image_feat,
    #         value=image_feat
    #     )

    #     text_feat = self.norm1(text_feat + t2i)

    #     # image attends to text
    #     i2t, _ = self.image_to_text_attn(
    #         query=image_feat,
    #         key=text_feat,
    #         value=text_feat
    #     )

    #     image_feat = self.norm2(image_feat + i2t)

    #     fused = torch.cat([text_feat, image_feat], dim=-1)

    #     return fused.squeeze(1)


    def forward(self, image_seq, text_seq, attention_mask=None):
        """
        输入升级：不再是 [B, D]，而是序列 [B, Seq_Len, D]
        
        Args:
            image_seq: [B, N_img_patches, img_dim] (来自 CLIP vision_model 的 last_hidden_state)
            text_seq:  [B, N_txt_tokens, txt_dim] (来自 XLM-R 的 last_hidden_state)
            attention_mask: [B, N_txt_tokens] (文本的 padding mask，用于忽略 <pad>)
            
        Returns:
            fused_feat: [B, hidden_dim * 2] (用于分类的全局特征)
            attn_weights: dict {'t2i': ..., 'i2t': ...} (用于可视化的注意力矩阵)
        """
        B = image_seq.size(0)

        # 1. 维度投影
        # image_seq: [B, N_img, D] -> [B, N_img, hidden_dim]
        img_proj = self.img_proj(image_seq)
        # text_seq: [B, N_txt, D] -> [B, N_txt, hidden_dim]
        txt_proj = self.txt_proj(text_seq)

        # 2. Text attends to Image (细粒度：每个词关注哪些图块)
        # Query: Text, Key/Value: Image
        # key_padding_mask 不需要给 image，因为 image patch 通常没有 padding
        t2i_attn_out, t2i_weights = self.text_to_image_attn(
            query=txt_proj,
            key=img_proj,
            value=img_proj,
            key_padding_mask=None 
        )
        
        # 残差连接 + 归一化
        txt_interact = self.norm1(txt_proj + t2i_attn_out)

        # 3. Image attends to Text (细粒度：每个图块关注哪些词)
        # Query: Image, Key/Value: Text
        # key_padding_mask: 告诉模型忽略文本中的 <pad> 位置 (True 表示忽略)
        # 注意：MultiheadAttention 的 key_padding_mask 为 True 表示 mask 掉
        if attention_mask is not None:
            # attention_mask 通常是 1 表示有效，0 表示 padding。需要反转并转为 bool
            txt_key_mask = (attention_mask == 0)
        else:
            txt_key_mask = None

        i2t_attn_out, i2t_weights = self.image_to_text_attn(
            query=img_proj,
            key=txt_proj,
            value=txt_proj,
            key_padding_mask=txt_key_mask
        )

        img_interact = self.norm2(img_proj + i2t_attn_out)

        # 4. 进一步融合与前馈 (可选，增强非线性)
        # 这里我们可以把两个交互后的序列再拼起来过一层 FFN，或者分别处理
        # 为了简单且有效，我们分别对交互后的序列做 Pooling
        
        # 对 Text 序列做 Mean Pooling (忽略 padding)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            txt_pooled = (txt_interact * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-9)
        else:
            txt_pooled = txt_interact.mean(dim=1)
            
        # 对 Image 序列做 Mean Pooling
        img_pooled = img_interact.mean(dim=1)

        # 5. 最终拼接
        fused = torch.cat([txt_pooled, img_pooled], dim=-1)
        
        # 加上 FFN 增强表达能力
        fused = fused + self.dropout(self.ffn(fused))
        fused = self.norm3(fused)

        # 准备返回注意力权重用于可视化
        # t2i_weights: [B, Num_Heads, N_txt, N_img]
        # i2t_weights: [B, Num_Heads, N_img, N_txt]
        attn_dict = {
            't2i': t2i_weights, 
            'i2t': i2t_weights
        }

        return fused, attn_dict