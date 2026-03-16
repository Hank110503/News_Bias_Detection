import torch.nn as nn
from transformers import RobertaModel
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

class TextEncoder(nn.Module):
    def __init__(self, model_name, freeze=True, use_lora=False, lora_r=8, lora_alpha=16):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.use_lora = use_lora
        
        # 加载完整的 XLM-RoBERTa 模型
        self.text_model = AutoModel.from_pretrained(model_name)
        
        # 冻结原始参数
        if freeze:
            for param in self.text_model.parameters():
                param.requires_grad = False
            
        self.output_dim = self.text_model.config.hidden_size

        if use_lora:
            # 配置 LoRA
            # RoBERTa/XLM-R 的 target_modules 通常是 query, value, key, dense (或者 q_proj, v_proj 等，视具体实现而定)
            # 对于 AutoModel (Roberta/XLMR)，层名称通常是：query, key, value, intermediate, output.dense
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["query", "value", "key"], # 针对 Attention 部分
                # 或者更激进一点：["query", "value", "key", "intermediate.dense", "output.dense"]
                lora_dropout=0.1,
                bias="none",
            )
            self.text_model = get_peft_model(self.text_model, config)
            self.text_model.print_trainable_parameters()

    def forward(self, input_ids, attention_mask):
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 直接输出特征：
        # # 取 pooler_output 作为文本全局特征
        # txt_feat = outputs.pooler_output
        # return txt_feat

        # 输出序列（保留 token 级别特征，把中提取特征交给 fusion 层）
        return outputs.last_hidden_state, attention_mask

