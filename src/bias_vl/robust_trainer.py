from __future__ import annotations

from typing import Any

import torch
from trl import SFTTrainer


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


class RobustSFTTrainer(SFTTrainer):
    def __init__(
        self,
        *args: Any,
        skip_oom_batches: bool = True,
        skip_oom_eval_batches: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.skip_oom_batches = skip_oom_batches
        self.skip_oom_eval_batches = skip_oom_eval_batches
        self.skipped_oom_train_batches = 0
        self.skipped_oom_eval_batches = 0
        self._last_logged_oom_train_batches = 0
        self._last_logged_oom_eval_batches = 0

    def _clear_after_oom(self) -> None:
        if hasattr(self.model, "zero_grad"):
            self.model.zero_grad(set_to_none=True)
        if hasattr(self, "optimizer") and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        try:
            return super().training_step(
                model,
                inputs,
                num_items_in_batch=num_items_in_batch,
            )
        except RuntimeError as exc:
            if not self.skip_oom_batches or not _is_oom_error(exc):
                raise

            self.skipped_oom_train_batches += 1
            self._clear_after_oom()
            print(
                "[RobustSFTTrainer] Skipping training batch due to CUDA OOM "
                f"at global_step={self.state.global_step}. "
                f"skipped_train_oom_batches={self.skipped_oom_train_batches}"
            )
            return torch.zeros((), device=self.args.device, requires_grad=True)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        try:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )
        except RuntimeError as exc:
            if not self.skip_oom_eval_batches or not _is_oom_error(exc):
                raise

            self.skipped_oom_eval_batches += 1
            self._clear_after_oom()
            print(
                "[RobustSFTTrainer] Skipping eval batch due to CUDA OOM "
                f"at global_step={self.state.global_step}. "
                f"skipped_eval_oom_batches={self.skipped_oom_eval_batches}"
            )
            return None, None, None

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        enriched_logs = dict(logs)
        enriched_logs["skipped_oom_train_batches"] = self.skipped_oom_train_batches
        enriched_logs["skipped_oom_eval_batches"] = self.skipped_oom_eval_batches
        enriched_logs["oom_train_since_last_log"] = (
            self.skipped_oom_train_batches - self._last_logged_oom_train_batches
        )
        enriched_logs["oom_eval_since_last_log"] = (
            self.skipped_oom_eval_batches - self._last_logged_oom_eval_batches
        )

        self._last_logged_oom_train_batches = self.skipped_oom_train_batches
        self._last_logged_oom_eval_batches = self.skipped_oom_eval_batches

        super().log(enriched_logs, *args, **kwargs)
