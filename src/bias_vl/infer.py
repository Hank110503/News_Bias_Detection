from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from PIL import Image

from .datasets import (
    DEFAULT_STAGE1_SYSTEM_PROMPT,
    DEFAULT_STAGE2_DIRECT_SYSTEM_PROMPT,
    DEFAULT_STAGE2_SYSTEM_PROMPT,
    DEFAULT_STAGE2_TEXT_ONLY_SYSTEM_PROMPT,
    STAGE2_TASK_MODES,
)
from .modeling import load_model_with_adapter
from .schema import Stage1Target, Stage2Target, model_dump_compat

ROOT_DIR = Path(__file__).resolve().parents[2]
DEBUG_OUTPUT_DIR = ROOT_DIR / "outputs" / "debug"
DEFAULT_MAX_NEW_TOKENS = 2048
MAX_JSON_RETRY_TOKENS = 4096


class JsonExtractionError(ValueError):
    def __init__(
        self,
        decoded_text: str,
        *,
        is_truncated: bool = False,
        debug_path: Path | None = None,
    ) -> None:
        preview = decoded_text[:500]
        guidance_parts: list[str] = [f" Decoded length: {len(decoded_text)} chars."]
        if is_truncated:
            guidance_parts.append(
                " The model output looks truncated; try increasing --max_new_tokens."
            )
        if debug_path is not None:
            guidance_parts.append(f" Raw output saved to: {debug_path}.")
        guidance = "".join(guidance_parts)
        super().__init__(f"No valid JSON object found in model output: {preview}{guidance}")
        self.decoded_text = decoded_text
        self.is_truncated = is_truncated
        self.debug_path = debug_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for the new Qwen2.5-VL pipeline.")
    parser.add_argument("--image")
    parser.add_argument("--text", help="Inline article text.")
    parser.add_argument("--text-file", help="Optional text file to read article text from.")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--stage", choices=["stage1", "stage2", "full"], default="full")
    parser.add_argument(
        "--task_mode",
        choices=sorted(STAGE2_TASK_MODES),
        default="stage2",
        help="Stage 2 task mode. full is only valid with stage2.",
    )
    parser.add_argument("--observation_json", help="Required for stage2 if not running full pipeline.")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    return parser.parse_args()


def read_text_arg(text: str | None, text_file: str | None) -> str:
    if text:
        return text
    if text_file:
        return Path(text_file).read_text(encoding="utf-8").strip()
    raise ValueError("Either --text or --text-file must be provided.")


def resolve_image_path(image_path: str) -> Path:
    candidate = Path(image_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (ROOT_DIR / candidate).resolve()


def strip_markdown_fence(decoded_text: str) -> str:
    stripped = decoded_text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def looks_like_truncated_json(decoded_text: str) -> bool:
    start = decoded_text.find("{")
    if start < 0:
        return False

    depth = 0
    in_string = False
    escape = False
    for char in decoded_text[start:]:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)

    return depth > 0 or in_string


def persist_invalid_json_output(decoded_text: str) -> Path | None:
    try:
        DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEBUG_OUTPUT_DIR / "latest_invalid_model_output.txt"
        output_path.write_text(decoded_text, encoding="utf-8")
        return output_path
    except OSError:
        return None


def extract_json_blob(decoded_text: str) -> Dict[str, Any]:
    cleaned_text = strip_markdown_fence(decoded_text)
    decoder = json.JSONDecoder()
    for start, char in enumerate(cleaned_text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned_text[start:])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue
    raise JsonExtractionError(
        decoded_text,
        is_truncated=looks_like_truncated_json(cleaned_text),
    )


def generate_text(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    image: Image.Image | None,
    max_new_tokens: int,
) -> str:
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    processor_kwargs: Dict[str, Any] = {
        "text": [prompt],
        "return_tensors": "pt",
    }
    if image is not None:
        processor_kwargs["images"] = [image]
    inputs = processor(**processor_kwargs)
    device = getattr(model, "device", None)
    if device is None or str(device) == "meta":
        device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return processor.batch_decode(
        generated[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )[0]


def generate_json(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    image: Image.Image | None,
    max_new_tokens: int,
) -> Dict[str, Any]:
    retry_tokens = min(max(max_new_tokens * 2, DEFAULT_MAX_NEW_TOKENS), MAX_JSON_RETRY_TOKENS)
    attempt_budgets = [max_new_tokens]
    if retry_tokens > max_new_tokens:
        attempt_budgets.append(retry_tokens)

    last_error: JsonExtractionError | None = None
    last_generated_text = ""
    for index, token_budget in enumerate(attempt_budgets):
        generated_text = generate_text(model, processor, messages, image, token_budget)
        last_generated_text = generated_text
        try:
            return extract_json_blob(generated_text)
        except JsonExtractionError as error:
            last_error = error
            should_retry = error.is_truncated and index + 1 < len(attempt_budgets)
            if should_retry:
                continue
            debug_path = persist_invalid_json_output(generated_text)
            raise JsonExtractionError(
                generated_text,
                is_truncated=error.is_truncated,
                debug_path=debug_path,
            ) from error

    if last_error is not None:
        debug_path = persist_invalid_json_output(last_generated_text)
        raise JsonExtractionError(
            last_generated_text,
            is_truncated=last_error.is_truncated,
            debug_path=debug_path,
        ) from last_error
    raise ValueError("Model generation finished without producing JSON output.")


def infer_stage1(
    model: Any,
    processor: Any,
    text: str,
    image: Image.Image,
    max_new_tokens: int,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": DEFAULT_STAGE1_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Text:\n{text}"},
            ],
        },
    ]
    result = generate_json(model, processor, messages, image, max_new_tokens)
    return model_dump_compat(Stage1Target(**result))


def infer_stage2(
    model: Any,
    processor: Any,
    text: str,
    image: Image.Image | None,
    observations: Dict[str, Any] | None,
    max_new_tokens: int,
    task_mode: str = "stage2",
) -> Dict[str, Any]:
    if task_mode == "stage2":
        if image is None:
            raise ValueError("stage2 task_mode requires an image.")
        if observations is None:
            raise ValueError("stage2 task_mode requires observations.")
        observation_text = (
            f"[Observations]\n"
            f"visual_observation: {observations['visual_observation']}\n"
            f"textual_observation: {observations['textual_observation']}\n"
            f"joint_mechanism: {observations['joint_mechanism']}\n"
        )
        messages = [
            {"role": "system", "content": DEFAULT_STAGE2_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Text:\n{text}\n\n{observation_text}"},
                ],
            },
        ]
    elif task_mode == "stage2_direct":
        if image is None:
            raise ValueError("stage2_direct task_mode requires an image.")
        messages = [
            {"role": "system", "content": DEFAULT_STAGE2_DIRECT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Text:\n{text}"},
                ],
            },
        ]
    elif task_mode == "stage2_text_only":
        image = None
        messages = [
            {"role": "system", "content": DEFAULT_STAGE2_TEXT_ONLY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Text:\n{text}"},
                ],
            },
        ]
    else:
        raise ValueError(f"Unsupported Stage 2 task_mode: {task_mode}")

    result = generate_json(model, processor, messages, image, max_new_tokens)
    return model_dump_compat(Stage2Target(**result))


def infer_full(
    model: Any,
    processor: Any,
    text: str,
    image: Image.Image,
    max_new_tokens: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    observations = infer_stage1(model, processor, text, image, max_new_tokens)
    stage2_result = infer_stage2(
        model,
        processor,
        text,
        image,
        observations,
        max_new_tokens=max_new_tokens,
        task_mode="stage2",
    )
    return observations, stage2_result


def main() -> None:
    args = parse_args()
    if args.stage == "full" and args.task_mode != "stage2":
        raise ValueError("--stage full is only supported with --task_mode stage2.")
    text = read_text_arg(args.text, args.text_file)
    image = None
    if args.stage in {"stage1", "full"} or args.task_mode != "stage2_text_only":
        if not args.image:
            raise ValueError("--image is required for the selected stage/task_mode.")
        image = Image.open(resolve_image_path(args.image)).convert("RGB")

    model, processor = load_model_with_adapter(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    if args.stage == "stage1":
        print(
            json.dumps(
                infer_stage1(model, processor, text, image, args.max_new_tokens),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.stage == "stage2":
        if args.task_mode == "stage2" and not args.observation_json:
            raise ValueError("--observation_json is required for stage2 inference.")
        observations = None
        if args.observation_json:
            observations = json.loads(Path(args.observation_json).read_text(encoding="utf-8"))
        print(
            json.dumps(
                infer_stage2(
                    model,
                    processor,
                    text,
                    image if args.task_mode != "stage2_text_only" else None,
                    observations,
                    args.max_new_tokens,
                    task_mode=args.task_mode,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    observations, final_result = infer_full(
        model,
        processor,
        text,
        image,
        args.max_new_tokens,
    )
    print(
        json.dumps(
            {
                "observations": observations,
                "final_result": final_result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
