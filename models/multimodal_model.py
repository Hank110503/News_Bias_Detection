import torch.nn as nn
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder
from .fusion import ConcatFusion, CrossAttentionFusion

class MultimodalBiasModel(nn.Module):
    def __init__(self, clip_name, text_model_name,
                 num_presence_labels,
                 num_relation_classes):

        super().__init__()

        self.image_encoder = ImageEncoder(clip_name)
        self.text_encoder = TextEncoder(text_model_name)

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

    def forward(self, pixel_values, input_ids, attention_mask):
        img_feat = self.image_encoder(pixel_values)
        txt_feat = self.text_encoder(input_ids, attention_mask)
        # print(f"--- Debug Dimensions ---")
        # print(f"Image feat shape: {img_feat.shape}")  
        # print(f"Text feat shape: {txt_feat.shape}")   
        joint = self.fusion(img_feat, txt_feat)

        return self.presence_head(joint), self.relation_head(joint)