#!/usr/bin/env python3
# evaluate_model.py
# Usage example:
# python evaluate_model.py --annotation_csv label-gemini-flash-lite-2.5.csv --model_path multimodal_bias_model.pt --batch_size 16

import os
import argparse
import logging
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.model_selection import train_test_split


from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from transformers import CLIPProcessor, CLIPModel, RobertaTokenizer, RobertaModel

# -----------------------------
# Dataset (must match training)
# -----------------------------
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

        # image
        image = Image.open(row["image_path"]).convert("RGB")
        clip_inputs = self.clip_processor(images=image, return_tensors="pt")
        pixel_values = clip_inputs["pixel_values"].squeeze(0)

        # text
        with open(row["text_path"], "r", encoding="utf-8") as f:
            text = f.read()
        text_inputs = self.text_tokenizer(text,
                                          padding="max_length",
                                          truncation=True,
                                          max_length=self.max_text_len,
                                          return_tensors="pt")
        input_ids = text_inputs["input_ids"].squeeze(0)
        attention_mask = text_inputs["attention_mask"].squeeze(0)

        presence = torch.tensor(row[self.presence_cols].values.astype(float), dtype=torch.float)

        relation_label = row["D1_Relationship_Type"]
        relation_id = self.relation_to_id[relation_label]
        relation = torch.tensor(relation_id, dtype=torch.long)

        return pixel_values, input_ids, attention_mask, presence, relation

# -----------------------------
# Model (must match training)
# -----------------------------
class MultimodalBiasModel(nn.Module):
    def __init__(self, clip_name, text_model_name, num_presence_labels, num_relation_classes, finetune_text=False):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name)
        self.text_encoder = RobertaModel.from_pretrained(text_model_name)

        # freeze CLIP (same as training)
        for p in self.clip.parameters():
            p.requires_grad = False

        if not finetune_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        img_dim = self.clip.config.projection_dim  # 512
        txt_dim = self.text_encoder.config.hidden_size
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
        img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-12)
        txt_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = txt_outputs.last_hidden_state[:, 0, :]   # CLS
        joint = torch.cat([img_feat, txt_feat], dim=1)
        return self.presence_head(joint), self.relation_head(joint)

# -----------------------------
# Evaluation utilities
# -----------------------------
def get_relation_mapping(df):
    unique_relations = sorted(df["D1_Relationship_Type"].dropna().unique())
    relation_to_id = {label: idx for idx, label in enumerate(unique_relations)}
    id_to_relation = {idx: label for label, idx in relation_to_id.items()}
    return relation_to_id, id_to_relation

def collate_preds(labels_list):
    """stack a list of tensors/arrays into a 2D numpy array"""
    return np.vstack([l if isinstance(l, np.ndarray) else l.numpy() for l in labels_list])

# -----------------------------
# Main evaluate function
# -----------------------------
def evaluate(args):
    logging.basicConfig(
        filename=args.log_file,
        filemode="w",
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")

    df = pd.read_csv(args.annotation_csv)
    df = df.dropna(subset=["D1_Relationship_Type"]).reset_index(drop=True)
    logging.info(f"Samples after dropping NaN in D1_Relationship_Type: {len(df)}")
    logging.info(f"Loaded annotation csv: {args.annotation_csv}  (samples: {len(df)})")

    # build the same mapping used in training
    relation_to_id, id_to_relation = get_relation_mapping(df)
    logging.info(f"Relation mapping (sorted unique): {relation_to_id}")

    # Split — we use val as test per your choice
    _, val_df = train_test_split(df, test_size=args.test_size, random_state=42, stratify=df["D1_Relationship_Type"])
    logging.info(f"Using {len(val_df)} samples as evaluation set (test_size={args.test_size})")

    # Dataset & Dataloader
    val_set = NewsBiasDataset(val_df, relation_to_id, clip_name=args.clip_name, text_model_name=args.text_model_name, max_text_len=args.max_text_len)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    num_rel = len(relation_to_id)
    model = MultimodalBiasModel(clip_name=args.clip_name, text_model_name=args.text_model_name,
                                num_presence_labels=9, num_relation_classes=num_rel, finetune_text=False)
    model = model.to(device)

    # load weights
    state = torch.load(args.model_path, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        # try strict=False in case of minor mismatch
        logging.warning("State dict mismatch, attempting strict=False load.")
        model.load_state_dict(state, strict=False)

    model.eval()
    logging.info("Model loaded and eval mode set.")

    # containers
    all_presence_labels = []
    all_presence_preds = []
    all_relation_labels = []
    all_relation_preds = []

    # iterate
    pbar = tqdm(val_loader, desc="Evaluating", ncols=100)
    with torch.no_grad():
        for pixel_values, input_ids, attention_mask, presence, relation in pbar:
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            presence = presence.to(device)
            relation = relation.to(device)

            p_logits, r_logits = model(pixel_values, input_ids, attention_mask)

            p_probs = torch.sigmoid(p_logits)
            p_pred = (p_probs > args.threshold).int()

            r_pred = torch.argmax(r_logits, dim=1)

            all_presence_labels.append(presence.cpu())
            all_presence_preds.append(p_pred.cpu())
            all_relation_labels.append(relation.cpu())
            all_relation_preds.append(r_pred.cpu())

    # stack
    presence_labels = torch.vstack(all_presence_labels).numpy()
    presence_preds = torch.vstack(all_presence_preds).numpy()
    relation_labels = torch.cat(all_relation_labels).numpy()
    relation_preds = torch.cat(all_relation_preds).numpy()

    # Per-indicator metrics
    presence_cols = [
        "V1_Salience.present", "V2_Perspective.present", "V3_Color_Lighting.present",
        "V4_Symbolism.present", "T1_Loaded_Language.present", "T2_Moral_Judgment.present",
        "J1_Role_Framing.present", "J2_Selective_Imbalance.present", "J3_Stereotyping.present"
    ]

    per_indicator = []
    for i, col in enumerate(presence_cols):
        y_true = presence_labels[:, i]
        y_pred = presence_preds[:, i]
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        per_indicator.append({
            "indicator": col,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s) if s is not None else 0
        })

    per_indicator_df = pd.DataFrame(per_indicator)
    per_indicator_df.to_csv(args.out_prefix + "_per_indicator.csv", index=False)
    logging.info(f"Per-indicator CSV saved to {args.out_prefix + '_per_indicator.csv'}")

    # Overall multi-label metrics
    presence_macro_f1 = f1_score(presence_labels, presence_preds, average="macro", zero_division=0)
    presence_micro_f1 = f1_score(presence_labels, presence_preds, average="micro", zero_division=0)
    logging.info(f"Presence Macro-F1: {presence_macro_f1:.4f} | Presence Micro-F1: {presence_micro_f1:.4f}")

    # Relation metrics
    relation_macro_f1 = f1_score(relation_labels, relation_preds, average="macro", zero_division=0)
    relation_micro_f1 = f1_score(relation_labels, relation_preds, average="micro", zero_division=0)
    logging.info(f"Relation Macro-F1: {relation_macro_f1:.4f} | Relation Micro-F1: {relation_micro_f1:.4f}")

    # Confusion matrix (relation)
    cm = confusion_matrix(relation_labels, relation_preds)
    cm_df = pd.DataFrame(cm, index=[id_to_relation[i] for i in range(len(id_to_relation))],
                         columns=[id_to_relation[i] for i in range(len(id_to_relation))])
    cm_df.to_csv(args.out_prefix + "_relation_confusion_matrix.csv")
    logging.info(f"Confusion matrix saved to {args.out_prefix + '_relation_confusion_matrix.csv'}")

    # classification report for relation
    cls_report = classification_report(relation_labels, relation_preds, target_names=[id_to_relation[i] for i in range(len(id_to_relation))], zero_division=0, output_dict=True)
    cls_df = pd.DataFrame(cls_report).transpose()
    cls_df.to_csv(args.out_prefix + "_relation_classification_report.csv")
    logging.info(f"Relation classification report saved to {args.out_prefix + '_relation_classification_report.csv'}")

    # summary csv
    summary = {
        "presence_macro_f1": presence_macro_f1,
        "presence_micro_f1": presence_micro_f1,
        "relation_macro_f1": relation_macro_f1,
        "relation_micro_f1": relation_micro_f1,
        "num_eval_samples": len(val_df)
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(args.out_prefix + "_summary.csv", index=False)
    logging.info(f"Summary saved to {args.out_prefix + '_summary.csv'}")

    logging.info("Evaluation finished.")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_csv", type=str, required=True, help="annotations csv containing image_path,text_path and labels")
    parser.add_argument("--model_path", type=str, required=True, help="path to saved model .pt")
    parser.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--text_model_name", type=str, default="roberta-large")
    parser.add_argument("--max_text_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5, help="sigmoid threshold for multi-label")
    parser.add_argument("--test_size", type=float, default=0.2, help="val/test fraction used from the annotation csv")
    parser.add_argument("--out_prefix", type=str, default="results/eval", help="prefix for output csv files")
    parser.add_argument("--log_file", type=str, default="evaluation.log")
    args = parser.parse_args()

    evaluate(args)
