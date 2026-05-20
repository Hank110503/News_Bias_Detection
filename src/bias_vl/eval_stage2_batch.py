from __future__ import annotations

import argparse
import json
import random
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

from .datasets import STAGE2_TASK_MODES, filter_stage2_samples, load_converted_samples, split_samples
from .evaluate import evaluate_stage2, load_jsonl
from .infer import infer_stage2
from .modeling import load_model_with_adapter
from .schema import CANONICAL_CUE_ORDER, model_dump_compat

ROOT_DIR = Path(__file__).resolve().parents[2]

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch Stage 2 inference and compute automatic metrics."
    )
    parser.add_argument("--config", default="configs/stage2_v1.yaml")
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument(
        "--task_mode",
        choices=sorted(STAGE2_TASK_MODES),
        default=None,
        help="Stage 2 task mode. Defaults to config.task_mode or stage2.",
    )
    parser.add_argument("--predictions_input", default=None)
    parser.add_argument(
        "--resume_existing",
        action="store_true",
        help="Resume interrupted evaluation by appending to existing predictions/failures files.",
    )
    parser.add_argument("--split", choices=["train", "eval", "all"], default="eval")
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--predictions_output",
        default="outputs/stage2_eval_predictions.jsonl",
    )
    parser.add_argument(
        "--report_output",
        default="outputs/stage2_eval_report.json",
    )
    parser.add_argument(
        "--markdown_output",
        default="outputs/stage2_eval_report.md",
    )
    parser.add_argument(
        "--failures_output",
        default="outputs/stage2_eval_failures.jsonl",
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
        stratify_stage2=True,
    )
    return train_rows if split == "train" else eval_rows


def maybe_subsample(rows: List[Any], num_samples: int, seed: int) -> List[Any]:
    if num_samples <= 0 or num_samples >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, num_samples)


def resolve_key(meta: Dict[str, Any] | None, fallback: int) -> str:
    return str((meta or {}).get("news_id") or fallback)


def safe_open_image(image_path: str | Path) -> tuple[Image.Image, str | None]:
    resolved_path = (ROOT_DIR / image_path).resolve()
    try:
        image = Image.open(resolved_path)
        image.load()
        return image.convert("RGB"), None
    except (FileNotFoundError, OSError, UnidentifiedImageError) as exc:
        return Image.new("RGB", (512, 512), color=(128, 128, 128)), str(exc)


def cue_reason_non_empty_rate(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    values: List[bool] = []
    for row in rows:
        cues = row.get("cues", {}) or {}
        for cue_name in CANONICAL_CUE_ORDER:
            cue_value = cues.get(cue_name, {}) or {}
            values.append(bool(str(cue_value.get("reason") or "").strip()))
    if not values:
        return 0.0
    return float(np.mean(values))


def consistency_reason_non_empty_rate(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    values = [
        bool(str(((row.get("consistency_dimension") or {}).get("reason") or "")).strip())
        for row in rows
    ]
    return float(np.mean(values))


def build_markdown_report(
    metrics: Dict[str, Any],
    prediction_path: Path,
    report_path: Path,
    failures_path: Path,
    examples: List[Dict[str, Any]],
) -> str:
    lines = [
        "# Stage 2 Batch Evaluation",
        "",
        f"- prediction_file: `{prediction_path}`",
        f"- report_file: `{report_path}`",
        f"- failures_file: `{failures_path}`",
        f"- matched_rows: `{metrics['matched_rows']}`",
        f"- failed_rows: `{metrics['failed_rows']}`",
        f"- cue_present_macro_f1: `{metrics['cue_present_macro_f1']:.4f}`",
        f"- cue_score_accuracy: `{metrics['cue_score_accuracy']:.4f}`",
        f"- cue_score_mae: `{metrics['cue_score_mae']:.4f}`",
        f"- relation_macro_f1: `{metrics['relation_macro_f1']:.4f}`",
        f"- cue_reason_non_empty_rate: `{metrics['cue_reason_non_empty_rate']:.4f}`",
        (
            f"- consistency_reason_non_empty_rate: "
            f"`{metrics['consistency_reason_non_empty_rate']:.4f}`"
        ),
        "",
        "## Sample Examples",
        "",
    ]

    if not examples:
        lines.append("No successful examples were available.")
        return "\n".join(lines)

    for index, example in enumerate(examples, 1):
        lines.extend(
            [
                f"### Example {index}",
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


def build_metrics(
    *,
    gold_rows: List[Dict[str, Any]],
    pred_rows: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    samples: Sequence[Any],
    dataset_path: str | Path,
    adapter_path: str | Path,
    predictions_output: Path,
    failures_output: Path,
    split: str,
    max_new_tokens: int,
    task_mode: str,
    status: str,
    fatal_error: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metrics = evaluate_stage2(gold_rows, pred_rows)
    metrics.update(
        {
            "status": status,
            "split": split,
            "dataset_path": str((ROOT_DIR / dataset_path).resolve()),
            "adapter_path": str((ROOT_DIR / adapter_path).resolve()),
            "predictions_output": str(predictions_output),
            "failures_output": str(failures_output),
            "requested_samples": len(samples),
            "num_samples": len(pred_rows),
            "failed_rows": len(failures),
            "max_new_tokens": max_new_tokens,
            "task_mode": task_mode,
            "cue_reason_non_empty_rate": cue_reason_non_empty_rate(pred_rows),
            "consistency_reason_non_empty_rate": consistency_reason_non_empty_rate(pred_rows),
        }
    )
    if fatal_error is not None:
        metrics["fatal_error"] = fatal_error
    return metrics


def write_reports(
    *,
    metrics: Dict[str, Any],
    predictions_output: Path,
    report_output: Path,
    markdown_output: Path,
    failures_output: Path,
    example_rows: List[Dict[str, Any]],
) -> None:
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


def build_gold_rows(samples: Sequence[Any]) -> List[Dict[str, Any]]:
    gold_rows: List[Dict[str, Any]] = []
    for sample in samples:
        meta = dict(sample.meta)
        sample_payload = model_dump_compat(sample)
        gold_rows.append(
            {
                "cues": sample_payload.get("cues", {}),
                "consistency_dimension": sample_payload.get("consistency_dimension", {}),
                "meta": meta,
            }
        )
    return gold_rows


def build_example_rows(
    gold_rows: List[Dict[str, Any]],
    pred_rows: List[Dict[str, Any]],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    pred_by_id = {
        str((row.get("meta") or {}).get("news_id") or idx): row
        for idx, row in enumerate(pred_rows)
    }
    examples: List[Dict[str, Any]] = []
    for idx, gold in enumerate(gold_rows):
        key = str((gold.get("meta") or {}).get("news_id") or idx)
        prediction = pred_by_id.get(key)
        if prediction is None:
            continue
        examples.append(
            {
                "news_id": (gold.get("meta") or {}).get("news_id"),
                "gold": gold,
                "prediction": prediction,
            }
        )
        if len(examples) >= limit:
            break
    return examples


def filter_pending_samples(
    samples: Sequence[Any],
    pred_rows: Sequence[Dict[str, Any]],
    failures: Sequence[Dict[str, Any]],
) -> List[Any]:
    completed_keys = {
        resolve_key(row.get("meta"), idx)
        for idx, row in enumerate(pred_rows)
    }
    completed_keys.update(
        resolve_key(row.get("meta"), idx)
        for idx, row in enumerate(failures)
    )
    pending: List[Any] = []
    for idx, sample in enumerate(samples):
        sample_key = resolve_key(dict(sample.meta), idx)
        if sample_key not in completed_keys:
            pending.append(sample)
    return pending


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_path = args.dataset_path or config["dataset_path"]
    adapter_path = args.adapter_path or str(Path(config["output_dir"]) / "final_adapter")
    model_name = args.model_name or config["model_name"]
    task_mode = args.task_mode or config.get("task_mode", "stage2")
    require_observations = task_mode == "stage2"

    samples = filter_stage2_samples(
        load_converted_samples(dataset_path),
        require_observations=require_observations,
    )
    samples = select_rows(
        rows=samples,
        split=args.split,
        eval_ratio=float(config.get("eval_ratio", 0.05)),
        seed=int(config.get("seed", args.seed)),
    )
    samples = maybe_subsample(samples, args.num_samples, args.seed)

    predictions_output = (ROOT_DIR / args.predictions_output).resolve()
    report_output = (ROOT_DIR / args.report_output).resolve()
    markdown_output = (ROOT_DIR / args.markdown_output).resolve()
    failures_output = (ROOT_DIR / args.failures_output).resolve()
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    failures_output.parent.mkdir(parents=True, exist_ok=True)

    all_gold_rows: List[Dict[str, Any]] = build_gold_rows(samples)
    gold_rows: List[Dict[str, Any]] = list(all_gold_rows)
    pred_rows: List[Dict[str, Any]] = []
    example_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    status = "completed"
    fatal_error: Dict[str, Any] | None = None
    save_every = 100

    if args.predictions_input:
        predictions_input = (ROOT_DIR / args.predictions_input).resolve()
        pred_rows = load_jsonl(predictions_input)
        predictions_output = predictions_input
        if failures_output.exists():
            failures = load_jsonl(failures_output)
        example_rows = build_example_rows(all_gold_rows, pred_rows)
    else:
        if args.resume_existing:
            if predictions_output.exists():
                pred_rows = load_jsonl(predictions_output)
            if failures_output.exists():
                failures = load_jsonl(failures_output)
            samples = filter_pending_samples(samples, pred_rows, failures)
            if hasattr(tqdm, "write"):
                tqdm.write(
                    "[eval_stage2_batch] Resume mode enabled. "
                    f"Loaded {len(pred_rows)} predictions, {len(failures)} failures, "
                    f"{len(samples)} samples remaining."
                )

        model, processor = load_model_with_adapter(
            model_name=model_name,
            adapter_path=adapter_path,
            torch_dtype="auto",
            device_map="auto",
        )
        model.eval()

        file_mode = "a" if args.resume_existing else "w"
        with (
            predictions_output.open(file_mode, encoding="utf-8") as writer,
            failures_output.open(file_mode, encoding="utf-8") as failure_writer,
        ):
            try:
                for index, sample in enumerate(tqdm(samples, desc="Stage2 Eval", unit="sample"), 1):
                    meta = dict(sample.meta)
                    image = None
                    image_error = None
                    if task_mode != "stage2_text_only":
                        image, image_error = safe_open_image(sample.image)
                    if image_error:
                        meta["image_fallback_error"] = image_error

                    observations = None
                    if task_mode == "stage2":
                        observations = {
                            "visual_observation": sample.visual_observation,
                            "textual_observation": sample.textual_observation,
                            "joint_mechanism": sample.joint_mechanism,
                        }

                    try:
                        prediction = infer_stage2(
                            model=model,
                            processor=processor,
                            text=sample.text,
                            image=image,
                            observations=observations,
                            max_new_tokens=args.max_new_tokens,
                            task_mode=task_mode,
                        )
                    except Exception as exc:
                        failure = {
                            "meta": meta,
                            "image": sample.image,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                        failure_writer.write(json.dumps(failure, ensure_ascii=False) + "\n")
                        failure_writer.flush()
                        failures.append(failure)
                        if hasattr(tqdm, "write"):
                            tqdm.write(
                                "[eval_stage2_batch] Skipping failed sample "
                                f"news_id={meta.get('news_id')} error={type(exc).__name__}: {exc}"
                            )
                        continue

                    output_row = {
                        "cues": prediction.get("cues", {}),
                        "consistency_dimension": prediction.get("consistency_dimension", {}),
                        "meta": meta,
                    }
                    writer.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                    writer.flush()

                    pred_rows.append(output_row)

                    if index % save_every == 0:
                        partial_metrics = build_metrics(
                            gold_rows=all_gold_rows,
                            pred_rows=pred_rows,
                            failures=failures,
                            samples=all_gold_rows,
                            dataset_path=dataset_path,
                            adapter_path=adapter_path,
                            predictions_output=predictions_output,
                            failures_output=failures_output,
                            split=args.split,
                            max_new_tokens=args.max_new_tokens,
                            task_mode=task_mode,
                            status="running",
                        )
                        example_rows = build_example_rows(all_gold_rows, pred_rows)
                        write_reports(
                            metrics=partial_metrics,
                            predictions_output=predictions_output,
                            report_output=report_output,
                            markdown_output=markdown_output,
                            failures_output=failures_output,
                            example_rows=example_rows,
                        )
            except KeyboardInterrupt:
                status = "interrupted"
                if hasattr(tqdm, "write"):
                    tqdm.write("[eval_stage2_batch] Interrupted. Writing partial reports.")
            except Exception as exc:
                status = "failed"
                fatal_error = {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                if hasattr(tqdm, "write"):
                    tqdm.write(
                        "[eval_stage2_batch] Fatal error encountered. "
                        f"Writing partial reports: {type(exc).__name__}: {exc}"
                    )

    example_rows = build_example_rows(all_gold_rows, pred_rows)
    metrics = build_metrics(
        gold_rows=all_gold_rows,
        pred_rows=pred_rows,
        failures=failures,
        samples=all_gold_rows,
        dataset_path=dataset_path,
        adapter_path=adapter_path,
        predictions_output=predictions_output,
        failures_output=failures_output,
        split=args.split,
        max_new_tokens=args.max_new_tokens,
        task_mode=task_mode,
        status=status,
        fatal_error=fatal_error,
    )
    write_reports(
        metrics=metrics,
        predictions_output=predictions_output,
        report_output=report_output,
        markdown_output=markdown_output,
        failures_output=failures_output,
        example_rows=example_rows,
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Stage 2 predictions written to {predictions_output}")
    print(f"Stage 2 failures written to {failures_output}")
    print(f"Stage 2 metrics written to {report_output}")
    print(f"Stage 2 markdown report written to {markdown_output}")


if __name__ == "__main__":
    main()
