
import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.dataset import NewsBiasDataset
from models.multimodal_model import MultimodalBiasModel
from training.trainer import Trainer
from training.loss import build_loss
from utils.class_weight import compute_pos_weight

from transformers import CLIPProcessor, RobertaTokenizer
from utils.logger import ExperimentLogger

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    exp_logger = ExperimentLogger(exp_name="multimodal_bias_v1")
    exp_logger.info(f"Using device: {device}")

    # ------------------------
    # 1. Load Data
    # ------------------------
    df = pd.read_csv("data/processed/labels/label-gemini-flash-lite-2.5.csv")
    df = df.dropna(subset=["D1_Relationship_Type"])

    unique_relations = sorted(df["D1_Relationship_Type"].unique())
    relation_to_id = {
        label: idx for idx, label in enumerate(unique_relations)
    }

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["D1_Relationship_Type"]
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
        "J3_Stereotyping.present",
    ]

    pos_weight = compute_pos_weight(train_df, presence_cols).to(device)
    exp_logger.info(f"Data loaded: Train={len(train_df)}, Val={len(val_df)}")
    # ------------------------
    # 2. Processor / Tokenizer
    # ------------------------
    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    train_set = NewsBiasDataset(
        train_df,
        relation_to_id,
        clip_processor,
        tokenizer
    )

    val_set = NewsBiasDataset(
        val_df,
        relation_to_id,
        clip_processor,
        tokenizer
    )

    train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        num_workers=12
    )

    val_loader = DataLoader(
        val_set,
        batch_size=32,
        shuffle=False,
        num_workers=12
    )

    # ------------------------
    # 3. Model
    # ------------------------
    model = MultimodalBiasModel(
        clip_name="openai/clip-vit-base-patch32",
        text_model_name="roberta-large",
        num_presence_labels=9,
        num_relation_classes=len(unique_relations)
    ).to(device)

    # 初始冻结 text encoder（和你原代码等价）
    model.text_encoder.freeze()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5
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
    )

    # ------------------------
    # 4. Training Loop
    # ------------------------
    num_epochs = 20
    unfreeze_epoch = 3

    for epoch in range(num_epochs):
        exp_logger.info(f"--- Epoch {epoch+1}/{num_epochs} ---")
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")

        # 解冻 text encoder
        if epoch == unfreeze_epoch:
            exp_logger.info("Unfreezing text encoder...")
            print("Unfreezing text encoder")
            model.text_encoder.unfreeze()

            # 重新定义 optimizer（关键！！）
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-5
            )
            trainer.optimizer = optimizer

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

        # --- 5. 记录数据 ---
        exp_logger.log_metrics(epoch, {
            "Total_Loss": train_loss, 
            "Presence_Loss": train_lp, 
            "Relation_Loss": train_lr
        }, step_type="Train")
        
        exp_logger.log_metrics(epoch, {
            "Total_Loss": val_loss, 
            "Presence_F1": presence_f1, 
            "Relation_F1": relation_f1
        }, step_type="Val")

        exp_logger.info(f"Train Loss: {train_loss:.4f} | Val F1 (Pres): {presence_f1:.4f} | Val F1 (Rel): {relation_f1:.4f}")

        # 保存最优模型

        # Save best
        if val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            torch.save(model.state_dict(), trainer.save_path)
            print("✔ Best model saved.")
            exp_logger.save_checkpoint(model, is_best=True)

    exp_logger.close()
    
if __name__ == "__main__":
    main()