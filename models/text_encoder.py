import torch.nn as nn
from transformers import RobertaModel

class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        self.output_dim = self.encoder.config.hidden_size
        self._frozen = False

    # -------------------------
    # 冻结
    # -------------------------
    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        self._frozen = True

    # -------------------------
    # 解冻
    # -------------------------
    def unfreeze(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
        self._frozen = False

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]