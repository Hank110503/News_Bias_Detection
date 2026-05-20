from transformers import CLIPProcessor, RobertaTokenizer
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import torch
import json

# 允许加载被截断的图片
ImageFile.LOAD_TRUNCATED_IMAGES = True
class NewsBiasDataset(Dataset):
    def __init__(self, df, relation_to_id, presence_cols,
                 clip_processor,
                 tokenizer,
                 max_text_len=512):
        self.df = df.reset_index(drop=True)
        self.relation_to_id = relation_to_id # 使用外部统一的映射
        self.max_text_len = max_text_len
        self.clip_processor = clip_processor
        self.text_tokenizer = tokenizer

        self.presence_cols = presence_cols


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image
        image = Image.open(row["image_path"]).convert("RGB")
        clip_inputs = self.clip_processor(images=image, return_tensors="pt")

        # Text
        with open(row["text_path"], "r", encoding="utf-8") as f:
            text = json.load(f).get("text", "")
        text_inputs = self.text_tokenizer(
            text, padding="max_length", truncation=True,
            max_length=self.max_text_len, return_tensors="pt"
        )

        presence = torch.tensor(row[self.presence_cols].values.astype(float), dtype=torch.float)
        
        relation_label = row["D1_Relationship_Type"]
        relation_id = self.relation_to_id[relation_label]
        relation = torch.tensor(relation_id, dtype=torch.long)

        return (
            clip_inputs["pixel_values"].squeeze(0),
            text_inputs["input_ids"].squeeze(0),
            text_inputs["attention_mask"].squeeze(0),
            presence,
            relation
        )