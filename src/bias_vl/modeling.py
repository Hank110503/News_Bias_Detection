from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
import peft.peft_model as peft_model_module
from transformers import AutoProcessor

ROOT_DIR = Path(__file__).resolve().parents[2]


def _normalize_no_split_module_classes(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(str(item) for item in value)
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        normalized: list[Any] = []
        for item in value:
            if isinstance(item, set):
                normalized.extend(sorted(str(sub_item) for sub_item in item))
            else:
                normalized.append(item)
        return normalized
    return value


def _ensure_peft_device_map_compat() -> None:
    if not getattr(peft_model_module.get_balanced_memory, "_bias_vl_compat", False):
        original_get_balanced_memory = peft_model_module.get_balanced_memory

        def compatible_get_balanced_memory(*args: Any, **kwargs: Any) -> Any:
            if "no_split_module_classes" in kwargs:
                kwargs["no_split_module_classes"] = _normalize_no_split_module_classes(
                    kwargs["no_split_module_classes"]
                )
            return original_get_balanced_memory(*args, **kwargs)

        compatible_get_balanced_memory._bias_vl_compat = True  # type: ignore[attr-defined]
        peft_model_module.get_balanced_memory = compatible_get_balanced_memory

    if not getattr(peft_model_module.infer_auto_device_map, "_bias_vl_compat", False):
        original_infer_auto_device_map = peft_model_module.infer_auto_device_map

        def compatible_infer_auto_device_map(*args: Any, **kwargs: Any) -> Any:
            if "no_split_module_classes" in kwargs:
                kwargs["no_split_module_classes"] = _normalize_no_split_module_classes(
                    kwargs["no_split_module_classes"]
                )
            return original_infer_auto_device_map(*args, **kwargs)

        compatible_infer_auto_device_map._bias_vl_compat = True  # type: ignore[attr-defined]
        peft_model_module.infer_auto_device_map = compatible_infer_auto_device_map


def resolve_torch_dtype(torch_dtype: str | None) -> Any:
    if not torch_dtype or torch_dtype == "auto":
        return "auto"
    if not hasattr(torch, torch_dtype):
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
    return getattr(torch, torch_dtype)


def load_processor(model_name: str) -> Any:
    return AutoProcessor.from_pretrained(model_name, trust_remote_code=True)


def load_qwen_vl_model(
    model_name: str,
    torch_dtype: str | None = "auto",
    device_map: str | None = "auto",
    gradient_checkpointing: bool = False,
) -> Any:
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
    }
    resolved_dtype = resolve_torch_dtype(torch_dtype)
    if resolved_dtype != "auto":
        model_kwargs["torch_dtype"] = resolved_dtype
    if device_map:
        model_kwargs["device_map"] = device_map

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs,
        )
    except ImportError:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            **model_kwargs,
        )

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def apply_lora(model: Any, config: Dict[str, Any]) -> Any:
    lora_config = LoraConfig(
        r=int(config.get("lora_r", 32)),
        lora_alpha=int(config.get("lora_alpha", 64)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        target_modules=config.get("target_modules", "all-linear"),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def load_adapter(model: Any, adapter_path: str | Path | None) -> Any:
    return load_adapter_with_mode(model, adapter_path, is_trainable=False)


def load_adapter_with_mode(
    model: Any,
    adapter_path: str | Path | None,
    *,
    is_trainable: bool = False,
) -> Any:
    if not adapter_path:
        return model
    adapter_dir = (ROOT_DIR / adapter_path).resolve()
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_dir}")
    _ensure_peft_device_map_compat()
    return PeftModel.from_pretrained(model, adapter_dir, is_trainable=is_trainable)


def load_model_with_adapter(
    model_name: str,
    adapter_path: str | Path | None = None,
    torch_dtype: str | None = "auto",
    device_map: str | None = "auto",
    gradient_checkpointing: bool = False,
    is_trainable_adapter: bool = False,
) -> Tuple[Any, Any]:
    processor = load_processor(model_name)
    model = load_qwen_vl_model(
        model_name=model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        gradient_checkpointing=gradient_checkpointing,
    )
    if adapter_path:
        model = load_adapter_with_mode(
            model,
            adapter_path,
            is_trainable=is_trainable_adapter,
        )
    return model, processor


def build_trainer_kwargs(
    trainer_cls: Any,
    model: Any,
    args: Any,
    train_dataset: Any,
    eval_dataset: Any,
    data_collator: Any,
    processor: Any,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }

    signature = inspect.signature(trainer_cls.__init__)
    if "tokenizer" in signature.parameters:
        kwargs["tokenizer"] = processor.tokenizer
    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = processor
    return kwargs
