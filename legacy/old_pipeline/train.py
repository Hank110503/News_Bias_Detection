import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CLIPImageProcessor

from data.dataset import NewsBiasDataset
from legacy.old_pipeline.models.multimodal_model import MultimodalBiasModel
from legacy.old_pipeline.training.loss import build_loss
from legacy.old_pipeline.training.trainer import Trainer
from legacy.old_pipeline.utils.class_weight import compute_pos_weight
from legacy.old_pipeline.utils.logger import ExperimentLogger


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    exp_logger = ExperimentLogger(exp_name="old_dataset/base_exp")
    exp_logger.info(f"Using device: {device}")

    df = pd.read_csv("data/processed/labels/label-gemini-flash-lite-2.5.csv")
    df = df.dropna(subset=["D1_Relationship_Type"])

    unique_relations = sorted(df["D1_Relationship_Type"].unique())
    relation_to_id = {label: idx for idx, label in enumerate(unique_relations)}

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["D1_Relationship_Type"],
    )

    presence_cols = [
        "V1_Salience.present",
        "V2_Perspective.present",
        "V3_Color_Lighting.present",
        "V4_Symbolism.present",
        "T1_Loaded_Language.present",
        "T2_Moral_Judgment.present",
        "J1_Role_Framing.present",
        "J2_Selective_Imbalance.present",
    ]

    pos_weight = compute_pos_weight(train_df, presence_cols).to(device)
    exp_logger.info(f"Data loaded: Train={len(train_df)}, Val={len(val_df)}")

    clip_model_name = "openai/clip-vit-large-patch14"
    clip_processor = CLIPImageProcessor.from_pretrained(clip_model_name)

    text_model_name = "FacebookAI/xlm-roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    train_set = NewsBiasDataset(
        train_df,
        relation_to_id,
        presence_cols,
        clip_processor,
        tokenizer,
    )

    val_set = NewsBiasDataset(
        val_df,
        relation_to_id,
        presence_cols,
        clip_processor,
        tokenizer,
    )

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=12)

    model = MultimodalBiasModel(
        clip_name="openai/clip-vit-large-patch14",
        text_model_name="FacebookAI/xlm-roberta-large",
        num_presence_labels=len(presence_cols),
        num_relation_classes=len(unique_relations),
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=5e-5,
        weight_decay=0.01,
    )

    criterion_p, criterion_r = build_loss(pos_weight)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion_p=criterion_p,
        criterion_r=criterion_r,
        device=device,
        save_path=os.path.join(exp_logger.exp_dir, "best_model.pt"),
        early_stop_patience=5,
        presence_weight=1.0,
        relation_weight=0.3,
    )

    num_epochs = 15

    for epoch in range(num_epochs):
        exp_logger.info(f"--- Epoch {epoch + 1}/{num_epochs} ---")
        print(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")

        train_loss, train_lp, train_lr = trainer.train_one_epoch(train_loader)
        val_loss, presence_f1, relation_f1 = trainer.validate(val_loader)

        print(
            f"[Train] Loss: {train_loss:.4f} | "
            f"Lp: {train_lp:.4f} | Lr: {train_lr:.4f}"
        )
        print(
            f"[Val] Loss: {val_loss:.4f} | "
            f"Presence Macro-F1: {presence_f1:.4f} | "
            f"Relation Macro-F1: {relation_f1:.4f}"
        )

        exp_logger.log_metrics(
            epoch,
            {
                "Total_Loss": train_loss,
                "Presence_Loss": train_lp,
                "Relation_Loss": train_lr,
            },
            step_type="Train",
        )
        exp_logger.log_metrics(
            epoch,
            {
                "Total_Loss": val_loss,
                "Presence_F1": presence_f1,
                "Relation_F1": relation_f1,
            },
            step_type="Val",
        )

        exp_logger.info(
            f"Train Loss: {train_loss:.4f} | "
            f"Val F1 (Pres): {presence_f1:.4f} | "
            f"Val F1 (Rel): {relation_f1:.4f}"
        )

        if val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            torch.save(model.state_dict(), trainer.save_path)
            print("Best model saved.")
            exp_logger.save_checkpoint(model, is_best=True)

    exp_logger.close()


if __name__ == "__main__":
    main()
