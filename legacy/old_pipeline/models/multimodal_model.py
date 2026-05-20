import torch.nn as nn
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder
from .fusion import ConcatFusion, CrossAttentionFusion
import torch

class MultimodalBiasModel(nn.Module):
    def __init__(self, clip_name, text_model_name,
                 num_presence_labels,
                 num_relation_classes,use_lora=False, lora_r=8, lora_alpha=16):

        super().__init__()

        self.image_encoder = ImageEncoder(clip_name, freeze=True, use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha)
        self.text_encoder = TextEncoder(text_model_name, freeze=True, use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha)
        # self.presence_fusion_weights = nn.Parameter(torch.tensor([0.33, 0.33, 0.33]))

        #使用简单拼接的方式融合图像和文本特征
        # self.fusion = ConcatFusion()
        # joint_dim = (
        #     self.image_encoder.output_dim +
        #     self.text_encoder.output_dim
        # )

        #使用注意力机制的方式融合图像和文本特征
        hidden_dim = self.text_encoder.output_dim
        self.fusion = CrossAttentionFusion(
            img_dim=self.image_encoder.output_dim,
            txt_dim=self.text_encoder.output_dim,
            hidden_dim=hidden_dim,
            num_heads=8
        )
        joint_dim = 2 * hidden_dim

        #使用DecisonFusion的方式融合图像和文本特征
        # self.image_head = nn.Sequential(
        #     nn.Linear(self.image_encoder.output_dim, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(256, num_presence_labels)
        # )

        # self.text_head = nn.Sequential(
        #     nn.Linear(self.text_encoder.output_dim, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(256, num_presence_labels)
        # )

        # self.joint_head = nn.Sequential(
        #     nn.Linear(joint_dim, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(512, num_presence_labels)
        # )



        self.presence_head = nn.Sequential(
            nn.Linear(joint_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_presence_labels)
        )

        self.relation_head = nn.Sequential(
            nn.Linear(joint_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_relation_classes)
        )

    # def forward(self, pixel_values, input_ids, attention_mask):
    #     img_feat = self.image_encoder(pixel_values)
    #     txt_feat = self.text_encoder(input_ids, attention_mask)
    #     joint_feat = self.fusion(img_feat, txt_feat)
    #     #使用前两种fusion方式的结果进行预测的返回值
    #     return self.presence_head(joint_feat), self.relation_head(joint_feat)


    def forward(self, pixel_values, input_ids, attention_mask):
        # 1. 获取序列特征
        # img_seq: [B, N_img, D]
        img_seq = self.image_encoder(pixel_values)
        
        # txt_seq: [B, N_txt, D], att_mask: [B, N_txt]
        txt_seq, txt_mask = self.text_encoder(input_ids, attention_mask)
        
        # 2. 交叉注意力融合
        # fused: [B, joint_dim], attn_weights: dict
        fused_feat, attn_weights = self.fusion(img_seq, txt_seq, attention_mask=txt_mask)
        
        # 3. 预测
        presence_logits = self.presence_head(fused_feat)
        relation_logits = self.relation_head(fused_feat)
        
        # 返回 logits 和 注意力权重 (用于可视化)
        return presence_logits, relation_logits, attn_weights