from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any, Dict

import yaml
from datasets import Dataset
from trl import SFTConfig

from .collator import QwenVLDataCollator
from .datasets import STAGE2_TASK_MODES, filter_stage2_samples, load_converted_samples, split_samples
from .modeling import apply_lora, build_trainer_kwargs, load_model_with_adapter
from .robust_trainer import RobustSFTTrainer
from .schema import model_dump_compat

ROOT_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen2.5-VL Stage 2.")
    parser.add_argument("--config", default="configs/stage2.yaml")
    parser.add_argument(
        "--adapter_path",
        default=None,
        help="Optional Stage 1 adapter path to continue training from.",
    )
    parser.add_argument(
        "--task_mode",
        choices=sorted(STAGE2_TASK_MODES),
        default=None,
        help="Stage 2 task mode. Defaults to config.task_mode or stage2.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    with (ROOT_DIR / path).resolve().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_resume_checkpoint(config: Dict[str, Any]) -> str | None:
    requested = config.get("resume_from_checkpoint")
    output_dir = (ROOT_DIR / config["output_dir"]).resolve()

    def latest_checkpoint() -> str | None:
        checkpoints = sorted(
            [
                path
                for path in output_dir.glob("checkpoint-*")
                if path.is_dir() and path.name.split("-")[-1].isdigit()
            ],
            key=lambda path: int(path.name.split("-")[-1]),
        )
        if not checkpoints:
            return None
        return str(checkpoints[-1])

    if requested in {None, "", False}:
        return None
    if requested in {True, "latest", "auto"}:
        return latest_checkpoint()

    checkpoint_path = (ROOT_DIR / str(requested)).resolve()
    if checkpoint_path.exists():
        return str(checkpoint_path)

    fallback = latest_checkpoint()
    if fallback:
        print(
            f"[train_stage2] Requested resume checkpoint not found: {checkpoint_path}. "
            f"Falling back to latest checkpoint: {fallback}"
        )
        return fallback

    print(
        f"[train_stage2] Requested resume checkpoint not found: {checkpoint_path}. "
        "Starting from scratch."
    )
    return None


def build_training_args(config: Dict[str, Any]) -> SFTConfig:
    output_dir = str((ROOT_DIR / config["output_dir"]).resolve())
    kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "per_device_train_batch_size": int(config.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(config.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 8)),
        "num_train_epochs": float(config.get("num_train_epochs", 3)),
        "max_steps": int(config.get("max_steps", -1)),
        "learning_rate": float(config.get("learning_rate", 2e-5)),
        "warmup_ratio": float(config.get("warmup_ratio", 0.03)),
        "logging_steps": int(config.get("logging_steps", 10)),
        "save_steps": int(config.get("save_steps", 200)),
        "eval_steps": int(config.get("eval_steps", 200)),
        "save_total_limit": int(config.get("save_total_limit", 2)),
        "max_length": int(config.get("max_seq_length", 2048)),
        "bf16": bool(config.get("bf16", True)),
        "remove_unused_columns": False,
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "report_to": config.get("report_to", "none"),
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", True)),
        "dataloader_num_workers": int(config.get("dataloader_num_workers", 0)),
        "save_strategy": config.get("save_strategy", "steps"),
        "logging_strategy": config.get("logging_strategy", "steps"),
    }
    signature = inspect.signature(SFTConfig.__init__)
    if "logging_dir" in signature.parameters:
        kwargs["logging_dir"] = config.get(
            "logging_dir",
            str(Path(output_dir) / "tensorboard"),
        )
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = config.get("eval_strategy", "steps")
    else:
        kwargs["evaluation_strategy"] = config.get("eval_strategy", "steps")
    return SFTConfig(**kwargs)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    resume_from_checkpoint = resolve_resume_checkpoint(config)
    adapter_path = args.adapter_path or config.get("adapter_path")
    task_mode = args.task_mode or config.get("task_mode", "stage2")
    require_observations = task_mode == "stage2"

    samples = filter_stage2_samples(
        load_converted_samples(config["dataset_path"]),
        require_observations=require_observations,
    )
    train_samples, eval_samples = split_samples(
        samples=samples,
        eval_ratio=float(config.get("eval_ratio", 0.05)),
        seed=int(config.get("seed", 42)),
        stratify_stage2=True,
    )

    model, processor = load_model_with_adapter(
        model_name=config["model_name"],
        adapter_path=adapter_path,
        torch_dtype=config.get("torch_dtype", "auto"),
        device_map=config.get("device_map", "auto"),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        is_trainable_adapter=bool(adapter_path),
    )
    if not adapter_path:
        model = apply_lora(model, config)

    collator = QwenVLDataCollator(
        processor,
        stage=task_mode,
        repo_root=ROOT_DIR,
        max_length=int(config.get("max_seq_length", 2048)),
        max_image_pixels=(
            int(config["max_image_pixels"])
            if config.get("max_image_pixels") is not None
            else None
        ),
    )
    training_args = build_training_args(config)

    trainer_kwargs = build_trainer_kwargs(
        trainer_cls=RobustSFTTrainer,
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list([model_dump_compat(sample) for sample in train_samples]),
        eval_dataset=Dataset.from_list([model_dump_compat(sample) for sample in eval_samples]),
        data_collator=collator,
        processor=processor,
    )
    trainer = RobustSFTTrainer(
        **trainer_kwargs,
        skip_oom_batches=bool(config.get("skip_oom_batches", True)),
        skip_oom_eval_batches=bool(config.get("skip_oom_eval_batches", True)),
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    final_adapter_dir = (ROOT_DIR / config["output_dir"] / "final_adapter").resolve()
    trainer.model.save_pretrained(final_adapter_dir)
    processor.save_pretrained(final_adapter_dir)
    print(f"Stage 2 adapter saved to {final_adapter_dir} (task_mode={task_mode})")


if __name__ == "__main__":
    main()
