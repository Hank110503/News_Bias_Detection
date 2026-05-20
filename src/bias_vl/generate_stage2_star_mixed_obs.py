from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml
from PIL import Image, ImageFile, UnidentifiedImageError

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Any = None, *args: Any, **kwargs: Any) -> Any:
        return iterable

from .datasets import filter_stage2_samples, load_converted_samples, split_samples
from .infer import infer_stage2
from .modeling import load_model_with_adapter
from .schema import CANONICAL_CUE_ORDER, model_dump_compat

ROOT_DIR = Path(__file__).resolve().parents[2]
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Stage 2 training dataset with STaR-style quality-aware "
            "mixing: keep predicted observations only when a frozen teacher agrees "
            "with gold labels, otherwise fall back to gold observations."
        )
    )
    parser.add_argument("--stage2_config", default="configs/stage2_v4.yaml")
    parser.add_argument(
        "--gold_dataset",
        default="data/processed/converted/bishe_schema_complete_cues_reason_merged.jsonl",
    )
    parser.add_argument(
        "--predicted_dataset",
        default="data/processed/converted/bishe_schema_stage2_predicted_obs_reason_merged.jsonl",
    )
    parser.add_argument("--teacher_adapter_path", default="outputs/stage2_v1/final_adapter")
    parser.add_argument("--model_name", default=None)
    parser.add_argument(
        "--output_dataset",
        default="data/processed/converted/bishe_schema_stage2_star_mixed_obs_7to3_reason_merged.jsonl",
    )
    parser.add_argument(
        "--teacher_diagnostics_output",
        default="data/processed/converted/bishe_schema_stage2_star_mixed_obs_7to3_reason_merged_diagnostics.jsonl",
    )
    parser.add_argument(
        "--report_output",
        default="data/processed/converted/bishe_schema_stage2_star_mixed_obs_7to3_reason_merged_report.json",
    )
    parser.add_argument(
        "--predicted_ratio",
        type=float,
        default=0.7,
        help="Among teacher-accepted rows, probability of keeping predicted observations.",
    )
    parser.add_argument(
        "--min_cue_present_f1",
        type=float,
        default=0.8,
        help="Minimum sample-level cue-present F1 required to trust predicted observations.",
    )
    parser.add_argument(
        "--max_cue_score_mae",
        type=float,
        default=0.4,
        help="Maximum sample-level cue-score MAE allowed to trust predicted observations.",
    )
    parser.add_argument(
        "--eval_observation_source",
        choices=["predicted", "gold"],
        default="predicted",
        help="Observation source for the held-out eval split embedded in the dataset.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    with (ROOT_DIR / path).resolve().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def safe_open_image(image_path: str | Path) -> tuple[Image.Image, str | None]:
    resolved_path = (ROOT_DIR / image_path).resolve()
    try:
        image = Image.open(resolved_path)
        image.load()
        return image.convert("RGB"), None
    except (FileNotFoundError, OSError, UnidentifiedImageError) as exc:
        return Image.new("RGB", (512, 512), color=(128, 128, 128)), str(exc)


def sample_news_id(sample: Any) -> str:
    return str(((sample.meta or {}).get("news_id") or "")).strip()


def load_processed_ids(path: Path) -> set[str]:
    processed_ids: set[str] = set()
    if not path.exists():
        return processed_ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            news_id = str(((payload.get("meta") or {}).get("news_id") or "")).strip()
            if news_id:
                processed_ids.add(news_id)
    return processed_ids


def maybe_subsample(rows: List[Any], num_samples: int, seed: int) -> List[Any]:
    if num_samples <= 0 or num_samples >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, num_samples)


def binary_f1(gold: Sequence[int], pred: Sequence[int]) -> float:
    tp = sum(1 for gold_item, pred_item in zip(gold, pred) if gold_item == 1 and pred_item == 1)
    fp = sum(1 for gold_item, pred_item in zip(gold, pred) if gold_item == 0 and pred_item == 1)
    fn = sum(1 for gold_item, pred_item in zip(gold, pred) if gold_item == 1 and pred_item == 0)
    if tp == 0:
        return 1.0 if fp == 0 and fn == 0 else 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_teacher_prediction(gold_sample: Any, teacher_prediction: Dict[str, Any]) -> Dict[str, Any]:
    gold_present: List[int] = []
    pred_present: List[int] = []
    gold_scores: List[int] = []
    pred_scores: List[int] = []

    gold_cues = gold_sample.cues or {}
    pred_cues = teacher_prediction.get("cues", {}) or {}
    for cue_name in CANONICAL_CUE_ORDER:
        gold_value = gold_cues.get(cue_name)
        pred_value = pred_cues.get(cue_name, {}) or {}
        gold_present.append(int(bool(getattr(gold_value, "present", False))))
        pred_present.append(int(bool(pred_value.get("present", False))))
        gold_scores.append(int(getattr(gold_value, "score", 0)))
        pred_scores.append(int(pred_value.get("score", 0)))

    cue_present_f1 = binary_f1(gold_present, pred_present)
    cue_score_mae = sum(
        abs(gold_score - pred_score)
        for gold_score, pred_score in zip(gold_scores, pred_scores)
    ) / len(CANONICAL_CUE_ORDER)
    relation_match = (
        str(gold_sample.consistency_dimension.D1_Relationship_Type or "")
        == str(((teacher_prediction.get("consistency_dimension") or {}).get("D1_Relationship_Type") or ""))
    )
    return {
        "cue_present_f1": cue_present_f1,
        "cue_score_mae": cue_score_mae,
        "relation_match": relation_match,
    }


def build_output_payload(
    *,
    gold_sample: Any,
    observation_sample: Any,
    observation_source: str,
    gold_observations: Dict[str, str],
    predicted_observations: Dict[str, str] | None,
    diagnostics: Dict[str, Any],
    teacher_adapter_path: str,
    image_error: str | None,
) -> Dict[str, Any]:
    payload = model_dump_compat(gold_sample)
    payload["visual_observation"] = observation_sample.visual_observation
    payload["textual_observation"] = observation_sample.textual_observation
    payload["joint_mechanism"] = observation_sample.joint_mechanism
    payload["meta"] = dict(payload.get("meta") or {})
    payload["meta"]["gold_observations"] = gold_observations
    if predicted_observations is not None:
        payload["meta"]["predicted_observations"] = predicted_observations
    payload["meta"]["star_observation_source"] = observation_source
    payload["meta"]["star_teacher_adapter_path"] = teacher_adapter_path
    payload["meta"]["star_teacher_accept"] = bool(diagnostics.get("teacher_accept", False))
    payload["meta"]["star_teacher_relation_match"] = diagnostics.get("relation_match")
    payload["meta"]["star_teacher_cue_present_f1"] = diagnostics.get("cue_present_f1")
    payload["meta"]["star_teacher_score_mae"] = diagnostics.get("cue_score_mae")
    payload["meta"]["star_split_role"] = diagnostics.get("split_role")
    if diagnostics.get("teacher_error_type"):
        payload["meta"]["star_teacher_error_type"] = diagnostics["teacher_error_type"]
        payload["meta"]["star_teacher_error_message"] = diagnostics.get("teacher_error_message", "")
    if image_error:
        payload["meta"]["image_fallback_error"] = image_error
    return payload


def main() -> None:
    args = parse_args()
    stage2_config = load_config(args.stage2_config)
    model_name = args.model_name or stage2_config["model_name"]

    gold_dataset_path = (ROOT_DIR / args.gold_dataset).resolve()
    predicted_dataset_path = (ROOT_DIR / args.predicted_dataset).resolve()
    output_dataset_path = (ROOT_DIR / args.output_dataset).resolve()
    diagnostics_output_path = (ROOT_DIR / args.teacher_diagnostics_output).resolve()
    report_output_path = (ROOT_DIR / args.report_output).resolve()
    output_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_path.parent.mkdir(parents=True, exist_ok=True)

    gold_samples = filter_stage2_samples(load_converted_samples(gold_dataset_path))
    predicted_samples = filter_stage2_samples(load_converted_samples(predicted_dataset_path))
    predicted_by_id = {
        sample_news_id(sample): sample
        for sample in predicted_samples
        if sample_news_id(sample)
    }

    gold_samples = maybe_subsample(gold_samples, args.num_samples, args.seed)
    train_samples, eval_samples = split_samples(
        samples=gold_samples,
        eval_ratio=float(stage2_config.get("eval_ratio", 0.05)),
        seed=int(stage2_config.get("seed", args.seed)),
        stratify_stage2=True,
    )
    train_ids = {sample_news_id(sample) for sample in train_samples}
    eval_ids = {sample_news_id(sample) for sample in eval_samples}

    processed_ids: set[str] = set()
    skipped_existing = 0
    if args.resume:
        processed_ids = load_processed_ids(output_dataset_path)
        original_count = len(gold_samples)
        gold_samples = [
            sample for sample in gold_samples if sample_news_id(sample) not in processed_ids
        ]
        skipped_existing = original_count - len(gold_samples)

    model, processor = load_model_with_adapter(
        model_name=model_name,
        adapter_path=args.teacher_adapter_path,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    counts: Counter[str] = Counter()
    diagnostics_examples: List[Dict[str, Any]] = []
    rng = random.Random(args.seed)
    mode = "a" if args.resume else "w"

    with (
        output_dataset_path.open(mode, encoding="utf-8") as writer,
        diagnostics_output_path.open(mode, encoding="utf-8") as diagnostics_writer,
    ):
        for gold_sample in tqdm(gold_samples, desc="Stage2 STaR", unit="sample"):
            news_id = sample_news_id(gold_sample)
            split_role = "eval" if news_id in eval_ids else "train"
            predicted_sample = predicted_by_id.get(news_id)

            gold_observations = {
                "visual_observation": gold_sample.visual_observation,
                "textual_observation": gold_sample.textual_observation,
                "joint_mechanism": gold_sample.joint_mechanism,
            }
            predicted_observations = None
            if predicted_sample is not None:
                predicted_observations = {
                    "visual_observation": predicted_sample.visual_observation,
                    "textual_observation": predicted_sample.textual_observation,
                    "joint_mechanism": predicted_sample.joint_mechanism,
                }

            image, image_error = safe_open_image(gold_sample.image)
            diagnostics: Dict[str, Any] = {
                "split_role": split_role,
                "teacher_accept": False,
                "cue_present_f1": None,
                "cue_score_mae": None,
                "relation_match": None,
            }

            if split_role == "eval":
                if args.eval_observation_source == "predicted" and predicted_sample is not None:
                    observation_sample = predicted_sample
                    observation_source = "eval_predicted"
                    counts["eval_predicted_rows"] += 1
                else:
                    observation_sample = gold_sample
                    observation_source = (
                        "eval_gold" if args.eval_observation_source == "gold" else "eval_gold_fallback"
                    )
                    counts["eval_gold_rows"] += 1
            elif predicted_sample is None:
                observation_sample = gold_sample
                observation_source = "train_gold_missing_predicted"
                counts["train_missing_predicted_rows"] += 1
            else:
                try:
                    teacher_prediction = infer_stage2(
                        model=model,
                        processor=processor,
                        text=gold_sample.text,
                        image=image,
                        observations=predicted_observations,
                        max_new_tokens=args.max_new_tokens,
                    )
                    teacher_metrics = evaluate_teacher_prediction(gold_sample, teacher_prediction)
                    diagnostics.update(teacher_metrics)
                    accepted = (
                        teacher_metrics["relation_match"]
                        and teacher_metrics["cue_present_f1"] >= args.min_cue_present_f1
                        and teacher_metrics["cue_score_mae"] <= args.max_cue_score_mae
                    )
                    diagnostics["teacher_accept"] = accepted
                    if accepted and rng.random() < args.predicted_ratio:
                        observation_sample = predicted_sample
                        observation_source = "train_star_predicted"
                        counts["train_star_predicted_rows"] += 1
                    else:
                        observation_sample = gold_sample
                        observation_source = (
                            "train_star_gold_mix" if accepted else "train_star_gold_rejected"
                        )
                        counts[observation_source] += 1
                except Exception as exc:
                    diagnostics["teacher_error_type"] = type(exc).__name__
                    diagnostics["teacher_error_message"] = str(exc)
                    observation_sample = gold_sample
                    observation_source = "train_gold_teacher_failure"
                    counts["train_teacher_failures"] += 1

                diagnostics_row = {
                    "meta": gold_sample.meta,
                    "observation_source": observation_source,
                    "teacher_accept": diagnostics.get("teacher_accept", False),
                    "cue_present_f1": diagnostics.get("cue_present_f1"),
                    "cue_score_mae": diagnostics.get("cue_score_mae"),
                    "relation_match": diagnostics.get("relation_match"),
                    "teacher_error_type": diagnostics.get("teacher_error_type"),
                    "teacher_error_message": diagnostics.get("teacher_error_message"),
                }
                diagnostics_writer.write(json.dumps(diagnostics_row, ensure_ascii=False) + "\n")
                diagnostics_writer.flush()
                if len(diagnostics_examples) < 20:
                    diagnostics_examples.append(diagnostics_row)

            payload = build_output_payload(
                gold_sample=gold_sample,
                observation_sample=observation_sample,
                observation_source=observation_source,
                gold_observations=gold_observations,
                predicted_observations=predicted_observations,
                diagnostics=diagnostics,
                teacher_adapter_path=args.teacher_adapter_path,
                image_error=image_error,
            )
            writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
            writer.flush()
            counts["written_rows"] += 1

    report = {
        "stage2_config": str((ROOT_DIR / args.stage2_config).resolve()),
        "gold_dataset": str(gold_dataset_path),
        "predicted_dataset": str(predicted_dataset_path),
        "output_dataset": str(output_dataset_path),
        "teacher_diagnostics_output": str(diagnostics_output_path),
        "report_output": str(report_output_path),
        "teacher_adapter_path": str((ROOT_DIR / args.teacher_adapter_path).resolve()),
        "model_name": model_name,
        "seed": args.seed,
        "resume": args.resume,
        "skipped_existing": skipped_existing,
        "predicted_ratio": args.predicted_ratio,
        "min_cue_present_f1": args.min_cue_present_f1,
        "max_cue_score_mae": args.max_cue_score_mae,
        "eval_observation_source": args.eval_observation_source,
        "max_new_tokens": args.max_new_tokens,
        "counts": dict(counts),
        "gold_samples": len(gold_samples) + skipped_existing,
        "predicted_samples": len(predicted_samples),
        "train_split_size": len(train_ids),
        "eval_split_size": len(eval_ids),
        "diagnostics_examples": diagnostics_examples,
    }
    report_output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"STaR-mixed dataset written to {output_dataset_path}")
    print(f"Teacher diagnostics written to {diagnostics_output_path}")
    print(f"Report written to {report_output_path}")


if __name__ == "__main__":
    main()
