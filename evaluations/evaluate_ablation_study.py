import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from transformers import CLIPProcessor, CLIPModel, RobertaTokenizer, RobertaModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import logging

# --- 配置与环境 ---
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["HF_HOME"] = "D:/hf_cache"  # 根据你的实际情况修改
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 设置随机种子以确保消融实验的公平性（控制变量）
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# --- 1. Dataset ---
class NewsBiasDataset(Dataset):
    def __init__(self, df, relation_to_id, 
                 clip_name="openai/clip-vit-base-patch32",
                 text_model_name="roberta-large",
                 max_text_len=512):
        self.df = df.reset_index(drop=True)
        self.relation_to_id = relation_to_id
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

        # 处理图片
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except Exception as e:
            # 容错处理：如果图片路径错误，生成全黑图片防止崩溃（并在日志中记录）
            print(f"Warning: Could not open image {row['image_path']}, using blank.")
            image = Image.new('RGB', (224, 224), color='black')
            
        clip_inputs = self.clip_processor(images=image, return_tensors="pt")

        # 处理文本
        try:
            with open(row["text_path"], "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            text = "" # 容错
            
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

# --- 2. Model (支持消融模式) ---
class AblationBiasModel(nn.Module):
    def __init__(self, clip_name, text_model_name, num_presence_labels, num_relation_classes, 
                 mode="multimodal", finetune_text=True):
        super().__init__()
        self.mode = mode  # Options: 'multimodal', 'text_only', 'image_only'
        
        # Encoders
        self.clip = CLIPModel.from_pretrained(clip_name)
        self.text_encoder = RobertaModel.from_pretrained(text_model_name)

        # 冻结 CLIP (通常保持冻结)
        for p in self.clip.parameters():
            p.requires_grad = False
        
        # 初始冻结 Text Encoder (稍后在训练循环中解冻)
        # 如果是 image_only 模式，Text Encoder 永远不需要梯度
        if not finetune_text or mode == 'image_only':
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        # 计算联合特征维度
        img_dim = self.clip.config.projection_dim # 512
        txt_dim = self.text_encoder.config.hidden_size # 1024
        
        if self.mode == "text_only":
            joint_dim = txt_dim
        elif self.mode == "image_only":
            joint_dim = img_dim
        else: # multimodal
            joint_dim = img_dim + txt_dim

        # Heads
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
        img_feat = None
        txt_feat = None
        
        # --- Image Feature Extraction ---
        if self.mode in ["multimodal", "image_only"]:
            img_feat = self.clip.get_image_features(pixel_values=pixel_values)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        # --- Text Feature Extraction ---
        if self.mode in ["multimodal", "text_only"]:
            txt_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            txt_feat = txt_outputs.last_hidden_state[:, 0, :] # [CLS] embedding

        # --- Feature Fusion ---
        if self.mode == "multimodal":
            joint = torch.cat([img_feat, txt_feat], dim=1)
        elif self.mode == "text_only":
            joint = txt_feat
        elif self.mode == "image_only":
            joint = img_feat
            
        return self.presence_head(joint), self.relation_head(joint)

# --- 3. 工具函数 ---
def compute_pos_weight(df, presence_cols):
    presence_df = df[presence_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    counts = presence_df.sum(axis=0)
    neg = len(presence_df) - counts
    pos_weight = neg / (counts + 1e-6)
    return torch.tensor(pos_weight.values, dtype=torch.float32)

# --- 4. 实验运行器 ---
def run_ablation_experiment(df, mode, relation_to_id, pos_weight, num_epochs=5):
    print(f"\n{'='*20} Running Experiment: {mode.upper()} {'='*20}")
    
    # 重新划分数据集 (random_state=42 保证每次实验的数据分割完全一致)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["D1_Relationship_Type"]
    )
    
    train_set = NewsBiasDataset(train_df, relation_to_id)
    val_set = NewsBiasDataset(val_df, relation_to_id)
    
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0) # Windows设为0
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)

    # 初始化模型
    model = AblationBiasModel(
        clip_name="openai/clip-vit-base-patch32",
        text_model_name="roberta-large",
        num_presence_labels=9,
        num_relation_classes=len(relation_to_id),
        mode=mode
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion_p = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_r = nn.CrossEntropyLoss()

    best_metrics = {"val_loss": float('inf'), "p_f1": 0.0, "r_f1": 0.0, "mode": mode}
    
    for epoch in range(num_epochs):
        # 文本解冻策略 (仅在需要文本的模式下)
        if epoch == 2 and mode != "image_only":
            print(f"   [Epoch {epoch+1}] 🔓 Unfreezing Text Encoder layers")
            for p in model.text_encoder.parameters():
                p.requires_grad = True

        # --- Train ---
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{num_epochs} [{mode}]", leave=False)
        
        for px, ids, mask, pres, rel in loop:
            px, ids, mask = px.to(DEVICE), ids.to(DEVICE), mask.to(DEVICE)
            pres, rel = pres.to(DEVICE), rel.to(DEVICE)
            
            optimizer.zero_grad()
            p_logits, r_logits = model(px, ids, mask)
            
            loss = criterion_p(p_logits, pres) + criterion_r(r_logits, rel)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validation ---
        model.eval()
        val_loss = 0
        all_p_preds, all_p_labels = [], []
        all_r_preds, all_r_labels = [], []

        with torch.no_grad():
            for px, ids, mask, pres, rel in val_loader:
                px, ids, mask = px.to(DEVICE), ids.to(DEVICE), mask.to(DEVICE)
                pres, rel = pres.to(DEVICE), rel.to(DEVICE)
                
                p_logits, r_logits = model(px, ids, mask)
                val_loss += (criterion_p(p_logits, pres) + criterion_r(r_logits, rel)).item()
                
                # Metrics Collection
                all_p_preds.append((torch.sigmoid(p_logits) > 0.5).int().cpu())
                all_p_labels.append(pres.int().cpu())
                all_r_preds.append(torch.argmax(r_logits, dim=1).cpu())
                all_r_labels.append(rel.cpu())

        avg_val_loss = val_loss / len(val_loader)
        
        p_f1 = f1_score(torch.cat(all_p_labels), torch.cat(all_p_preds), average="macro", zero_division=0)
        r_f1 = f1_score(torch.cat(all_r_labels), torch.cat(all_r_preds), average="macro", zero_division=0)

        # print(f"   Val Loss: {avg_val_loss:.4f} | P-F1: {p_f1:.4f} | R-F1: {r_f1:.4f}")

        if avg_val_loss < best_metrics["val_loss"]:
            best_metrics = {"val_loss": avg_val_loss, "p_f1": p_f1, "r_f1": r_f1, "mode": mode}
            # 可以选择在这里保存每个模式的最佳模型
            torch.save(model.state_dict(), f"best_model_{mode}.pt")

    print(f"Finished {mode}. Best Val Loss: {best_metrics['val_loss']:.4f} | P-F1: {best_metrics['p_f1']:.4f}")
    return best_metrics

# --- 5. 主程序 ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # 1. 读取数据 (请修改你的 CSV 路径)
    csv_path = "label-gemini-flash-lite-2.5.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: Dataset {csv_path} not found.")
        exit()
        
    df_data = pd.read_csv(csv_path)
    # 过滤掉标签缺失的行
    df_data = df_data.dropna(subset=["D1_Relationship_Type"])
    
    # 2. 准备 Label Mapping
    unique_relations = sorted(df_data["D1_Relationship_Type"].unique())
    relation_to_id = {label: idx for idx, label in enumerate(unique_relations)}
    print(f"Relation Classes: {len(unique_relations)}")

    # 3. 计算正样本权重 (用于不平衡处理)
    presence_cols = [
        "V1_Salience.present", "V2_Perspective.present", "V3_Color_Lighting.present",
        "V4_Symbolism.present", "T1_Loaded_Language.present", "T2_Moral_Judgment.present",
        "J1_Role_Framing.present", "J2_Selective_Imbalance.present", "J3_Stereotyping.present"
    ]
    pos_weight = compute_pos_weight(df_data, presence_cols).to(DEVICE)

    # 4. 执行消融实验循环
    # modes = ["image_only", "text_only", "multimodal"]
    modes = ["text_only", "image_only", "multimodal"] # 推荐顺序
    results = {}

    for mode in modes:
        # 每个实验运行 5-10 个 Epoch 即可看出差异
        results[mode] = run_ablation_experiment(
            df_data, mode, relation_to_id, pos_weight, num_epochs=10
        )

    # 5. 打印最终结果表
    print("\n\n")
    print("+" * 65)
    print(f"{'ABLATION STUDY RESULTS':^65}")
    print("+" * 65)
    print(f"{'Mode':<15} | {'Presence Macro-F1':<20} | {'Relation Macro-F1':<20}")
    print("-" * 65)
    
    for mode in modes:
        res = results[mode]
        print(f"{mode:<15} | {res['p_f1']:.4f}{' '*14} | {res['r_f1']:.4f}")
    print("+" * 65)

    # 6. 生成可视化图表
    try:
        p_scores = [results[m]['p_f1'] for m in modes]
        r_scores = [results[m]['r_f1'] for m in modes]
        
        x = np.arange(len(modes))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        rects1 = ax.bar(x - width/2, p_scores, width, label='Presence Task (F1)', color='skyblue')
        rects2 = ax.bar(x + width/2, r_scores, width, label='Relation Task (F1)', color='salmon')

        ax.set_ylabel('Macro F1 Score')
        ax.set_title('Ablation Study: Impact of Modalities')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in modes])
        ax.legend()
        ax.set_ylim(0, 1.1)

        # 添加数值标签
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout()
        plt.savefig("ablation_result.png", dpi=300)
        print("\n📊 Chart saved as 'ablation_result.png'")
        
    except Exception as e:
        print(f"\nCould not generate plot: {e}")