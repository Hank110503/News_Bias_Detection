# training/trainer.py

import torch
from tqdm import tqdm
import numpy as np
from legacy.old_pipeline.utils.metrics import compute_metrics


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion_p,
        criterion_r,
        device,
        save_path,
        scheduler=None,
        early_stop_patience=5,
        presence_weight=1.0,
        relation_weight=1.0,
        grad_clip=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion_p = criterion_p
        self.criterion_r = criterion_r
        self.device = device
        self.save_path = save_path
        self.scheduler = scheduler
        self.grad_clip = grad_clip

        self.best_val_loss = float("inf")
        self.early_stop_patience = early_stop_patience
        self.no_improve_epochs = 0
        self.presence_weight = presence_weight
        self.relation_weight = relation_weight

    # ========================
    # Train One Epoch
    # ========================
    def train_one_epoch(self, dataloader):
        self.model.train()

        total_loss = 0
        total_loss_p = 0
        total_loss_r = 0

        pbar = tqdm(dataloader, desc="Training")

        for pixel_values, input_ids, attention_mask, presence, relation in pbar:
            pixel_values = pixel_values.to(self.device)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            presence = presence.to(self.device)
            relation = relation.to(self.device)

            self.optimizer.zero_grad()

            # 使用简单拼接融合方式
            # p_logits, r_logits = self.model(
            #     pixel_values, input_ids, attention_mask
            # )
            # 使用cross-attention融合方式
            p_logits, r_logits, attn_weights = self.model(pixel_values, input_ids, attention_mask)


            loss_p = self.criterion_p(p_logits, presence)
            loss_r = self.criterion_r(r_logits, relation)
            loss = self.presence_weight * loss_p + self.relation_weight * loss_r

            # #使用DecisonFusion的方式
            # p_logits_img, p_logits_txt, p_logits_joint, fused_logits, r_logits = self.model(
            #     pixel_values, input_ids, attention_mask
            # )

            # loss_p_img = self.criterion_p(p_logits_img, presence)
            # loss_p_txt = self.criterion_p(p_logits_txt, presence)
            # loss_p_joint = self.criterion_p(p_logits_joint, presence)
            # loss_p_fused = self.criterion_p(fused_logits, presence)
            # loss_r = self.criterion_r(r_logits, relation)

            #     # 这里可以根据需要调整各个loss的权重
            # loss_p = (
            #     0.2 * loss_p_img +
            #     0.2 * loss_p_txt +
            #     0.2 * loss_p_joint +
            #     0.4 * loss_p_fused
            # )
            # loss = loss_p + 0.3 * loss_r

            loss.backward()

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )

            self.optimizer.step()

        # 训练过程中监控注意力分布
        # 打印 t2i 的平均最大注意力值，看模型是否关注到了东西
            t2i_w = attn_weights['t2i'] # [B, Heads, Txt_Len, Img_Len]
            max_attn = t2i_w.max(dim=-1)[0].mean()


            total_loss += loss.item()
            total_loss_p += loss_p.item()
            total_loss_r += loss_r.item()

            pbar.set_postfix({
                "Lp": f"{loss_p.item():.3f}",
                "Lr": f"{loss_r.item():.3f}",
                "Max_T2I_Attn": f"{max_attn.item():.3f}"
            })

        return (
            total_loss / len(dataloader),
            total_loss_p / len(dataloader),
            total_loss_r / len(dataloader),
        )

    # ========================
    # Validation
    # ========================
    def validate(self, dataloader):
        self.model.eval()

        total_loss = 0
        all_presence_preds = []
        all_presence_labels = []
        all_relation_preds = []
        all_relation_labels = []

        with torch.no_grad():
            for pixel_values, input_ids, attention_mask, presence, relation in dataloader:
                pixel_values = pixel_values.to(self.device)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                presence = presence.to(self.device)
                relation = relation.to(self.device)


                #使用前两种融合方式
                p_logits, r_logits, attention_weights = self.model(
                    pixel_values, input_ids, attention_mask
                )

                loss_p = self.criterion_p(p_logits, presence)
                loss_r = self.criterion_r(r_logits, relation)
                total_loss += (loss_p + loss_r).item()


                # #使用DecisonFusion的方式
                # outputs = self.model(pixel_values, input_ids, attention_mask)

                # loss_img = self.criterion_p(outputs[0], presence)
                # loss_txt = self.criterion_p(outputs[1], presence)
                # loss_joint = self.criterion_p(outputs[2], presence)
                # loss_fused = self.criterion_p(outputs[3], presence)

                # loss_presence = (
                #     0.2 * loss_img +
                #     0.2 * loss_txt +
                #     0.2 * loss_joint +
                #     0.4 * loss_fused
                # )
                # loss_relation = self.criterion_r(outputs[4], relation)
                # total_loss += loss_presence.item() + 0.3 * loss_relation.item()
                # p_logits = outputs[3]
                # r_logits = outputs[4]



                # presence
                p_pred = (torch.sigmoid(p_logits) > 0.5).int()
                all_presence_preds.append(p_pred.cpu())
                all_presence_labels.append(presence.cpu().int())

                # relation
                r_pred = torch.argmax(r_logits, dim=1)
                all_relation_preds.append(r_pred.cpu())
                all_relation_labels.append(relation.cpu())

        presence_preds = torch.cat(all_presence_preds).numpy()
        presence_labels = torch.cat(all_presence_labels).numpy()

        relation_preds = torch.cat(all_relation_preds).numpy()
        relation_labels = torch.cat(all_relation_labels).numpy()

        presence_macro_f1, relation_macro_f1 = compute_metrics(
            presence_labels,
            presence_preds,
            relation_labels,
            relation_preds
        )

        return (
            total_loss / len(dataloader),
            presence_macro_f1,
            relation_macro_f1,
        )

    # ========================
    # Full Training Loop
    # ========================
    def fit(self, train_loader, val_loader, num_epochs):

        for epoch in range(num_epochs):

            print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")

            train_loss, train_lp, train_lr = self.train_one_epoch(train_loader)

            val_loss, presence_f1, relation_f1 = self.validate(val_loader)

            if self.scheduler:
                self.scheduler.step(val_loss)

            print(
                f"[Train] Loss: {train_loss:.4f} | "
                f"Lp: {train_lp:.4f} | Lr: {train_lr:.4f}"
            )

            print(
                f"[Val] Loss: {val_loss:.4f} | "
                f"Presence Macro-F1: {presence_f1:.4f} | "
                f"Relation Macro-F1: {relation_f1:.4f}"
            )

            # Save Best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.save_path)
                print("✔ Best model saved.")
                self.no_improve_epochs = 0
            else:
                self.no_improve_epochs += 1

            # Early stopping
            if self.no_improve_epochs >= self.early_stop_patience:
                print("Early stopping triggered.")
                break
