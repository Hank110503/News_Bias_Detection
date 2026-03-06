import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, RobertaTokenizer, RobertaModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
from sklearn.metrics import f1_score
import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 环境变量设置
os.environ["HF_HOME"] = "D:/hf_cache"

# --- Dataset ---
class NewsBiasDataset(Dataset):
    def __init__(self, df, relation_to_id, 
                 clip_name="openai/clip-vit-base-patch32",
                 text_model_name="roberta-large",
                 max_text_len=512):
        self.df = df.reset_index(drop=True)
        self.relation_to_id = relation_to_id # 使用外部统一的映射
        self.clip_processor = CLIPProcessor.from_pretrained(clip_name)
        self.text_tokenizer = RobertaTokenizer.from_pretrained(text_model_name)
        self.max_text_len = max_text_len

        self.presence_cols = [
            "V1_Salience.present", "V2_Perspective.present", "V3_Color_Lighting.present",
            "V4_Symbolism.present", "T1_Loaded_Language.present", "T2_Moral_Judgment.present",
            "J1_Role_Framing.present", "J2_Selective_Imbalance.present", "J3_Stereotyping.present"
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image
        image = Image.open(row["image_path"]).convert("RGB")
        clip_inputs = self.clip_processor(images=image, return_tensors="pt")

        # Text
        with open(row["text_path"], "r", encoding="utf-8") as f:
            text = f.read()
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

# --- Model ---
class MultimodalBiasModel(nn.Module):
    def __init__(self, clip_name, text_model_name, num_presence_labels, num_relation_classes, finetune_text=True):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name)
        self.text_encoder = RobertaModel.from_pretrained(text_model_name)

        # 冻结 CLIP (通常视觉特征已经很强)
        for p in self.clip.parameters():
            p.requires_grad = False

        if not finetune_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        img_dim = self.clip.config.projection_dim # 512
        txt_dim = self.text_encoder.config.hidden_size # 1024
        joint_dim = img_dim + txt_dim

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
        img_feat = self.clip.get_image_features(pixel_values=pixel_values)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True) # L2 Normalization

        txt_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = txt_outputs.last_hidden_state[:, 0, :] # [CLS] token

        joint = torch.cat([img_feat, txt_feat], dim=1)
        return self.presence_head(joint), self.relation_head(joint)

def compute_pos_weight(df, presence_cols):
    # 强制转为数值，非法值变 NaN
    presence_df = df[presence_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # NaN 当作 0（即“未出现”）
    presence_df = presence_df.fillna(0.0)

    counts = presence_df.sum(axis=0)
    neg = len(presence_df) - counts

    pos_weight = neg / (counts + 1e-6)

    return torch.tensor(
        pos_weight.values,
        dtype=torch.float32
    )



# --- Training Logic ---
def train_model(df, save_path, batch_size=32, num_epochs=10, lr=1e-5,unfreeze_epoch=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 统一构建标签映射
    df = df.dropna(subset=["D1_Relationship_Type"])
    unique_relations = sorted(df["D1_Relationship_Type"].unique())
    relation_to_id = {label: idx for idx, label in enumerate(unique_relations)}
    print(f"Relation Mapping: {relation_to_id}")

    # 2. Split
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["D1_Relationship_Type"]
    )

    presence_cols = [
    "V1_Salience.present", "V2_Perspective.present", "V3_Color_Lighting.present",
    "V4_Symbolism.present", "T1_Loaded_Language.present", "T2_Moral_Judgment.present",
    "J1_Role_Framing.present", "J2_Selective_Imbalance.present", "J3_Stereotyping.present"
]
    pos_weight = compute_pos_weight(train_df, presence_cols).to(device)
    print("pos_weight:", pos_weight.cpu().numpy().round(2))

    train_set = NewsBiasDataset(train_df, relation_to_id)
    val_set = NewsBiasDataset(val_df, relation_to_id)

    # Windows 下如果报错请将 num_workers 设为 0
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # 3. Model & Loss
    model = MultimodalBiasModel(
        clip_name="openai/clip-vit-base-patch32",
        text_model_name="roberta-large",
        num_presence_labels=9,
        num_relation_classes=len(unique_relations)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion_p = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_r = nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        if epoch == unfreeze_epoch:
            print("🔓 Unfreezing text encoder")
            for p in model.text_encoder.parameters():
                p.requires_grad = True
        # --- Train Phase ---
        model.train()
        total_train_loss = 0
        total_train_loss_p = 0
        total_train_loss_r = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for pixel_values, input_ids, attention_mask, presence, relation in pbar:
            pixel_values, input_ids, attention_mask = pixel_values.to(device), input_ids.to(device), attention_mask.to(device)
            presence, relation = presence.to(device), relation.to(device)

            optimizer.zero_grad()
            p_logits, r_logits = model(pixel_values, input_ids, attention_mask)
            
            loss_p = criterion_p(p_logits, presence)
            loss_r = criterion_r(r_logits, relation)
            loss = loss_p + loss_r

            
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_train_loss_p += loss_p.item()
            total_train_loss_r += loss_r.item()
            pbar.set_postfix({
                "Lp": f"{loss_p.item():.3f}",
                "Lr": f"{loss_r.item():.3f}"
            })

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        all_presence_preds = []
        all_presence_labels = []
        all_relation_preds = []
        all_relation_labels = []

        with torch.no_grad():
            for pixel_values, input_ids, attention_mask, presence, relation in val_loader:
                pixel_values, input_ids, attention_mask = pixel_values.to(device), input_ids.to(device), attention_mask.to(device)
                presence, relation = presence.to(device), relation.to(device)
                
                p_logits, r_logits = model(pixel_values, input_ids, attention_mask)
                # loss
                loss_p = criterion_p(p_logits, presence)
                loss_r = criterion_r(r_logits, relation)
                total_val_loss += (loss_p + loss_r).item()

                # presence preds
                p_pred = (torch.sigmoid(p_logits) > 0.5).int()
                all_presence_preds.append(p_pred.cpu())
                all_presence_labels.append(presence.cpu().int())

                # relation preds
                r_pred = torch.argmax(r_logits, dim=1)
                all_relation_preds.append(r_pred.cpu())
                all_relation_labels.append(relation.cpu())
                

        avg_lp = total_train_loss_p / len(train_loader)
        avg_lr = total_train_loss_r / len(train_loader)
        print(f"Epoch {epoch+1} | [Train] Lp: {avg_lp:.4f} | Lr: {avg_lr:.4f}")

        presence_preds = torch.cat(all_presence_preds).numpy()
        presence_labels = torch.cat(all_presence_labels).numpy()

        relation_preds = torch.cat(all_relation_preds).numpy()
        relation_labels = torch.cat(all_relation_labels).numpy()

        presence_macro_f1 = f1_score(
            presence_labels, presence_preds, average="macro", zero_division=0
        )

        relation_macro_f1 = f1_score(
            relation_labels, relation_preds, average="macro"
        )


        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        print(
            f"[Val] Loss: {avg_val_loss:.4f} | "
            f"Presence Macro-F1: {presence_macro_f1:.4f} | "
            f"Relation Macro-F1: {relation_macro_f1:.4f}"
        )

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print("✔ Best model saved.")

    return model

if __name__ == "__main__":
    df_data = pd.read_csv("label-gemini-flash-lite-2.5.csv")
    train_model(df_data, "multimodal_bias_model.pt")