from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageFile, UnidentifiedImageError

from .datasets import (
    STAGE2_TASK_MODES,
    build_stage1_messages,
    build_stage2_messages_for_mode,
)
from .schema import ConvertedSample

ImageFile.LOAD_TRUNCATED_IMAGES = True


class QwenVLDataCollator:
    def __init__(
        self,
        processor: Any,
        stage: str,
        repo_root: str | Path,
        max_length: int,
        max_image_pixels: int | None = None,
    ):
        if stage not in {"stage1", *STAGE2_TASK_MODES}:
            raise ValueError(f"Unsupported stage: {stage}")
        self.processor = processor
        self.stage = stage
        self.repo_root = Path(repo_root)
        self.max_length = max_length
        self.max_image_pixels = max_image_pixels
        self.text_max_length = max(256, max_length // 2)
        self.observation_max_length = max(128, max_length // 8)
        self._warned_image_paths: set[str] = set()

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        if not text:
            return text
        tokenized = self.processor.tokenizer(
            text,
            truncation=True,
            max_length=max_tokens,
            add_special_tokens=False,
        )
        return self.processor.tokenizer.decode(
            tokenized["input_ids"],
            skip_special_tokens=True,
        )

    def _prepare_sample(self, sample: ConvertedSample) -> ConvertedSample:
        payload = sample.dict()
        payload["text"] = self._truncate_text(payload["text"], self.text_max_length)

        if self.stage == "stage2":
            payload["visual_observation"] = self._truncate_text(
                payload["visual_observation"],
                self.observation_max_length,
            )
            payload["textual_observation"] = self._truncate_text(
                payload["textual_observation"],
                self.observation_max_length,
            )
            payload["joint_mechanism"] = self._truncate_text(
                payload["joint_mechanism"],
                self.observation_max_length,
            )

        return ConvertedSample(**payload)

    def _prepare_image(self, image_path: str | Path) -> Image.Image:
        resolved_path = self.repo_root / image_path
        try:
            image = Image.open(resolved_path)
            image.load()
            image = image.convert("RGB")
        except (FileNotFoundError, OSError, UnidentifiedImageError) as exc:
            key = str(image_path)
            if key not in self._warned_image_paths:
                print(
                    f"[QwenVLDataCollator] Falling back to a blank image for {resolved_path}: {exc}"
                )
                self._warned_image_paths.add(key)
            fallback_edge = int((self.max_image_pixels or 262144) ** 0.5)
            return Image.new("RGB", (fallback_edge, fallback_edge), color=(128, 128, 128))

        if not self.max_image_pixels:
            return image

        width, height = image.size
        total_pixels = width * height
        if total_pixels <= self.max_image_pixels:
            return image

        scale = (self.max_image_pixels / float(total_pixels)) ** 0.5
        resized = image.resize(
            (
                max(1, int(width * scale)),
                max(1, int(height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
        return resized

    def _build_messages(self, sample: ConvertedSample) -> List[Dict[str, Any]]:
        if self.stage == "stage1":
            return build_stage1_messages(sample)
        return build_stage2_messages_for_mode(sample, self.stage)

    def _processor_call(
        self,
        *,
        texts: List[str],
        images: List[Image.Image] | None = None,
        padding: bool = True,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "text": texts,
            "padding": padding,
            "return_tensors": "pt",
        }
        if images is not None:
            kwargs["images"] = images
        return self.processor(**kwargs)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        samples = [item if isinstance(item, ConvertedSample) else ConvertedSample(**item) for item in batch]

        full_texts: List[str] = []
        prompt_texts: List[str] = []
        images: List[Image.Image] | None = [] if self.stage != "stage2_text_only" else None

        for sample in samples:
            prepared_sample = self._prepare_sample(sample)
            messages = self._build_messages(prepared_sample)
            full_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_text = self.processor.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_texts.append(full_text)
            prompt_texts.append(prompt_text)
            if images is not None:
                images.append(self._prepare_image(sample.image))

        model_inputs = self._processor_call(
            texts=full_texts,
            images=images,
            padding=True,
        )

        labels = model_inputs["input_ids"].clone()
        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)

        for index, prompt_text in enumerate(prompt_texts):
            prompt_images = None if images is None else [images[index]]
            prompt_inputs = self._processor_call(
                texts=[prompt_text],
                images=prompt_images,
                padding=False,
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]
            labels[index, :prompt_len] = -100

        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        model_inputs["labels"] = labels
        return model_inputs
