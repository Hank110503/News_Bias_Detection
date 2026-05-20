import torch.nn as nn
from transformers import CLIPModel
from peft import LoraConfig, get_peft_model
from torch.nn import functional as F


class ImageEncoder(nn.Module):
    def __init__(self, clip_name, freeze=True, use_lora=False, lora_r=8, lora_alpha=16):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name)
        self.use_lora = use_lora

        if freeze:
            for p in self.clip.parameters():
                p.requires_grad = False

        self.output_dim = self.clip.vision_model.config.hidden_size

        if use_lora:
            # 配置 LoRA
            # target_modules 需要根据具体模型架构调整，CLIP ViT 通常包含 q_proj, v_proj, k_proj, out_proj
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"], 
                lora_dropout=0.1,
                bias="none",
                modules_to_save=[], # 如果需要微调分类头可以加在这里，否则不用
            )
            # 将 LoRA 应用到 vision_model 上
            self.clip.vision_model = get_peft_model(self.clip.vision_model, config)

    def forward(self, pixel_values):
        # 获取带有 LoRA 权重的视觉编码器输出
        outputs = self.clip.vision_model(pixel_values=pixel_values)

        #直接输出特征：
        # # 提取 [CLS] 对应的 pooler 输出
        # img_feat = outputs.pooler_output

        # # 进行 L2 归一化 (CLIP 风格特征的标准操作，有助于训练稳定)
        # img_feat = F.normalize(img_feat, p=2, dim=-1)
        # return img_feat
    
        #输出序列（保留空间信息，把中提取特征交给fusion层）
        return outputs.last_hidden_state