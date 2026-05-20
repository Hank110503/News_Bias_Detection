from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import yaml
from PIL import Image, ImageFile, UnidentifiedImageError
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Any = None, *args: Any, **kwargs: Any) -> Any:
        return iterable

from .datasets import filter_stage1_samples, load_converted_samples, split_samples
from .infer import infer_stage1
from .modeling import load_model_with_adapter

ROOT_DIR = Path(__file__).resolve().parents[2]
STAGE1_FIELDS = [
    "visual_observation",
    "textual_observation",
    "joint_mechanism",
]

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch Stage 1 inference and compute lightweight automatic metrics."
    )
    parser.add_argument("--config", default="configs/stage1.yaml")
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--split", choices=["train", "eval", "all"], default="eval")
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--predictions_output",
        default="outputs/stage1_eval_predictions.jsonl",
    )
    parser.add_argument(
        "--report_output",
        default="outputs/stage1_eval_report.json",
    )
    parser.add_argument(
        "--markdown_output",
        default="outputs/stage1_eval_report.md",
    )
    parser.add_argument(
        "--failures_output",
        default="outputs/stage1_eval_failures.jsonl",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    with (ROOT_DIR / path).resolve().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def select_rows(
    rows: Sequence[Any],
    split: str,
    eval_ratio: float,
    seed: int,
) -> List[Any]:
    if split == "all":
        return list(rows)

    train_rows, eval_rows = split_samples(
        samples=rows,
        eval_ratio=eval_ratio,
        seed=seed,
        stratify_stage2=False,
    )
    return train_rows if split == "train" else eval_rows


def maybe_subsample(rows: List[Any], num_samples: int, seed: int) -> List[Any]:
    if num_samples <= 0 or num_samples >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, num_samples)


def safe_open_image(image_path: str | Path) -> tuple[Image.Image, str | None]:
    resolved_path = (ROOT_DIR / image_path).resolve()
    try:
        image = Image.open(resolved_path)
        image.load()
        return image.convert("RGB"), None
    except (FileNotFoundError, OSError, UnidentifiedImageError) as exc:
        return Image.new("RGB", (512, 512), color=(128, 128, 128)), str(exc)


def tokenize(text: str) -> List[str]:
    return [token for token in text.lower().split() if token]


def token_f1(gold: str, pred: str) -> float:
    gold_tokens = tokenize(gold)
    pred_tokens = tokenize(pred)
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0
    gold_counter = Counter(gold_tokens)
    pred_counter = Counter(pred_tokens)
    overlap = sum((gold_counter & pred_counter).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    curr = [0] * (len(b) + 1)
    for token_a in a:
        for j, token_b in enumerate(b, 1):
            if token_a == token_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (len(b) + 1)
    return prev[-1]


def rouge_l_f1(gold: str, pred: str) -> float:
    gold_tokens = tokenize(gold)
    pred_tokens = tokenize(pred)
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0
    lcs = lcs_length(gold_tokens, pred_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def collect_metrics(
    gold_rows: List[Dict[str, Any]],
    pred_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not gold_rows or not pred_rows:
        empty_scores = {field: 0.0 for field in STAGE1_FIELDS}
        return {
            "matched_rows": 0,
            "field_non_empty_rate": empty_scores,
            "field_token_f1": empty_scores,
            "field_rouge_l_f1": empty_scores,
            "field_avg_pred_tokens": empty_scores,
            "macro_token_f1": 0.0,
            "macro_rouge_l_f1": 0.0,
        }

    field_non_empty: Dict[str, float] = {}
    field_token_f1: Dict[str, float] = {}
    field_rouge_l: Dict[str, float] = {}
    field_pred_length: Dict[str, float] = {}

    for field in STAGE1_FIELDS:
        field_non_empty[field] = float(
            np.mean([bool(str(row.get(field) or "").strip()) for row in pred_rows])
        )
        field_token_f1[field] = float(
            np.mean(
                [
                    token_f1(str(gold.get(field) or ""), str(pred.get(field) or ""))
                    for gold, pred in zip(gold_rows, pred_rows)
                ]
            )
        )
        field_rouge_l[field] = float(
            np.mean(
                [
                    rouge_l_f1(str(gold.get(field) or ""), str(pred.get(field) or ""))
                    for gold, pred in zip(gold_rows, pred_rows)
                ]
            )
        )
        field_pred_length[field] = float(
            np.mean([len(tokenize(str(row.get(field) or ""))) for row in pred_rows])
        )

    return {
        "matched_rows": len(pred_rows),
        "field_non_empty_rate": field_non_empty,
        "field_token_f1": field_token_f1,
        "field_rouge_l_f1": field_rouge_l,
        "field_avg_pred_tokens": field_pred_length,
        "macro_token_f1": float(np.mean(list(field_token_f1.values()))),
        "macro_rouge_l_f1": float(np.mean(list(field_rouge_l.values()))),
    }


def build_markdown_report(
    metrics: Dict[str, Any],
    prediction_path: Path,
    report_path: Path,
    failures_path: Path,
    examples: List[Dict[str, Any]],
) -> str:
    lines = [
        "# Stage 1 Batch Evaluation",
        "",
        f"- prediction_file: `{prediction_path}`",
        f"- report_file: `{report_path}`",
        f"- failures_file: `{failures_path}`",
        f"- matched_rows: `{metrics['matched_rows']}`",
        f"- failed_rows: `{metrics['failed_rows']}`",
        f"- macro_token_f1: `{metrics['macro_token_f1']:.4f}`",
        f"- macro_rouge_l_f1: `{metrics['macro_rouge_l_f1']:.4f}`",
        "",
        "## Field Metrics",
        "",
        "| Field | Non-empty | Token F1 | ROUGE-L F1 | Avg Pred Tokens |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for field in STAGE1_FIELDS:
        lines.append(
            "| "
            + f"{field} | "
            + f"{metrics['field_non_empty_rate'][field]:.4f} | "
            + f"{metrics['field_token_f1'][field]:.4f} | "
            + f"{metrics['field_rouge_l_f1'][field]:.4f} | "
            + f"{metrics['field_avg_pred_tokens'][field]:.2f} |"
        )

    if examples:
        lines.extend(["", "## Sample Examples", ""])
        for idx, example in enumerate(examples, 1):
            lines.extend(
                [
                    f"### Example {idx}",
                    f"- news_id: `{example['news_id']}`",
                    "",
                    "#### Gold",
                    "",
                    "```json",
                    json.dumps(example["gold"], ensure_ascii=False, indent=2),
                    "```",
                    "",
                    "#### Prediction",
                    "",
                    "```json",
                    json.dumps(example["prediction"], ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_path = args.dataset_path or config["dataset_path"]
    adapter_path = args.adapter_path or str(Path(config["output_dir"]) / "final_adapter")
    model_name = args.model_name or config["model_name"]

    samples = filter_stage1_samples(load_converted_samples(dataset_path))
    samples = select_rows(
        rows=samples,
        split=args.split,
        eval_ratio=float(config.get("eval_ratio", 0.05)),
        seed=int(config.get("seed", args.seed)),
    )
    samples = maybe_subsample(samples, args.num_samples, args.seed)

    model, processor = load_model_with_adapter(
        model_name=model_name,
        adapter_path=adapter_path,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    predictions_output = (ROOT_DIR / args.predictions_output).resolve()
    report_output = (ROOT_DIR / args.report_output).resolve()
    markdown_output = (ROOT_DIR / args.markdown_output).resolve()
    failures_output = (ROOT_DIR / args.failures_output).resolve()
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    failures_output.parent.mkdir(parents=True, exist_ok=True)

    gold_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    example_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    with (
        predictions_output.open("w", encoding="utf-8") as writer,
        failures_output.open("w", encoding="utf-8") as failure_writer,
    ):
        for sample in tqdm(samples, desc="Stage1 Eval", unit="sample"):
            image, image_error = safe_open_image(sample.image)
            meta = dict(sample.meta)
            if image_error:
                meta["image_fallback_error"] = image_error
            try:
                prediction = infer_stage1(
                    model=model,
                    processor=processor,
                    text=sample.text,
                    image=image,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as exc:
                failure = {
                    "meta": meta,
                    "image": sample.image,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                failure_writer.write(json.dumps(failure, ensure_ascii=False) + "\n")
                failures.append(failure)
                if hasattr(tqdm, "write"):
                    tqdm.write(
                        "[eval_stage1_batch] Skipping failed sample "
                        f"news_id={meta.get('news_id')} error={type(exc).__name__}: {exc}"
                    )
                continue

            output_row = {
                "visual_observation": prediction.get("visual_observation", ""),
                "textual_observation": prediction.get("textual_observation", ""),
                "joint_mechanism": prediction.get("joint_mechanism", ""),
                "meta": meta,
            }
            writer.write(json.dumps(output_row, ensure_ascii=False) + "\n")

            gold_row = {
                "visual_observation": sample.visual_observation,
                "textual_observation": sample.textual_observation,
                "joint_mechanism": sample.joint_mechanism,
                "meta": meta,
            }
            gold_rows.append(gold_row)
            pred_rows.append(output_row)

            if len(example_rows) < 5:
                example_rows.append(
                    {
                        "news_id": meta.get("news_id"),
                        "gold": {
                            field: gold_row[field]
                            for field in STAGE1_FIELDS
                        },
                        "prediction": {
                            field: output_row[field]
                            for field in STAGE1_FIELDS
                        },
                    }
                )

    metrics = collect_metrics(gold_rows, pred_rows)
    metrics.update(
        {
            "split": args.split,
            "dataset_path": str((ROOT_DIR / dataset_path).resolve()),
            "adapter_path": str((ROOT_DIR / adapter_path).resolve()),
            "predictions_output": str(predictions_output),
            "failures_output": str(failures_output),
            "requested_samples": len(samples),
            "num_samples": len(pred_rows),
            "failed_rows": len(failures),
            "max_new_tokens": args.max_new_tokens,
        }
    )

    report_output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_output.write_text(
        build_markdown_report(
            metrics,
            predictions_output,
            report_output,
            failures_output,
            example_rows,
        ),
        encoding="utf-8",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Stage 1 predictions written to {predictions_output}")
    print(f"Stage 1 failures written to {failures_output}")
    print(f"Stage 1 metrics written to {report_output}")
    print(f"Stage 1 markdown report written to {markdown_output}")


if __name__ == "__main__":
    main()
