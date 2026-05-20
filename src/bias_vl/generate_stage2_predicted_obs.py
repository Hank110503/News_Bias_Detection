from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import yaml
from PIL import Image, ImageFile, UnidentifiedImageError
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Any = None, *args: Any, **kwargs: Any) -> Any:
        return iterable

from .datasets import filter_stage2_samples, load_converted_samples, split_samples
from .infer import infer_stage1
from .modeling import load_model_with_adapter
from .schema import model_dump_compat

ROOT_DIR = Path(__file__).resolve().parents[2]
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Stage 2 dataset whose observations come from a frozen Stage 1 model."
    )
    parser.add_argument("--stage1_config", default="configs/stage1.yaml")
    parser.add_argument("--stage2_config", default="configs/stage2_v1.yaml")
    parser.add_argument("--input_dataset", default=None)
    parser.add_argument("--stage1_adapter_path", default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument(
        "--output_dataset",
        default="data/processed/converted/bishe_schema_stage2_predicted_obs_reason_merged.jsonl",
    )
    parser.add_argument(
        "--failures_output",
        default="data/processed/converted/bishe_schema_stage2_predicted_obs_reason_merged_failures.jsonl",
    )
    parser.add_argument(
        "--report_output",
        default="data/processed/converted/bishe_schema_stage2_predicted_obs_reason_merged_report.json",
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--split", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry_failures_from", default=None)
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


def select_rows(
    rows: List[Any],
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


def load_processed_ids(*paths: Path) -> set[str]:
    processed_ids: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
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


def main() -> None:
    args = parse_args()
    stage1_config = load_config(args.stage1_config)
    stage2_config = load_config(args.stage2_config)

    input_dataset = args.input_dataset or stage2_config["dataset_path"]
    stage1_adapter_path = args.stage1_adapter_path or str(
        Path(stage1_config["output_dir"]) / "final_adapter"
    )
    model_name = args.model_name or stage1_config["model_name"]

    samples = filter_stage2_samples(load_converted_samples(input_dataset))
    samples = select_rows(
        rows=samples,
        split=args.split,
        eval_ratio=float(stage2_config.get("eval_ratio", 0.05)),
        seed=int(stage2_config.get("seed", args.seed)),
    )
    if 0 < args.num_samples < len(samples):
        rng = random.Random(args.seed)
        samples = rng.sample(samples, args.num_samples)

    retry_failure_ids: set[str] = set()
    if args.retry_failures_from:
        retry_failure_ids = load_processed_ids((ROOT_DIR / args.retry_failures_from).resolve())
        samples = [
            sample
            for sample in samples
            if str(((sample.meta or {}).get("news_id") or "")).strip() in retry_failure_ids
        ]

    output_dataset = (ROOT_DIR / args.output_dataset).resolve()
    failures_output = (ROOT_DIR / args.failures_output).resolve()
    report_output = (ROOT_DIR / args.report_output).resolve()
    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    failures_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)

    processed_ids: set[str] = set()
    skipped_existing = 0
    if args.resume:
        processed_ids = load_processed_ids(output_dataset, failures_output)
        original_count = len(samples)
        samples = [
            sample
            for sample in samples
            if str(((sample.meta or {}).get("news_id") or "")).strip() not in processed_ids
        ]
        skipped_existing = original_count - len(samples)

    model, processor = load_model_with_adapter(
        model_name=model_name,
        adapter_path=stage1_adapter_path,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    kept = 0
    failed = 0
    failure_examples: List[Dict[str, Any]] = []
    mode = "a" if args.resume else "w"

    with (
        output_dataset.open(mode, encoding="utf-8") as writer,
        failures_output.open(mode, encoding="utf-8") as failure_writer,
    ):
        for sample in tqdm(samples, desc="Stage2 ObsGen", unit="sample"):
            image, image_error = safe_open_image(sample.image)
            try:
                predicted_obs = infer_stage1(
                    model=model,
                    processor=processor,
                    text=sample.text,
                    image=image,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as exc:
                failed += 1
                failure = {
                    "meta": sample.meta,
                    "image": sample.image,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                if image_error:
                    failure["image_fallback_error"] = image_error
                failure_writer.write(json.dumps(failure, ensure_ascii=False) + "\n")
                if len(failure_examples) < 20:
                    failure_examples.append(failure)
                if hasattr(tqdm, "write"):
                    tqdm.write(
                        "[generate_stage2_predicted_obs] Skipping failed sample "
                        f"news_id={(sample.meta or {}).get('news_id')} "
                        f"error={type(exc).__name__}: {exc}"
                    )
                continue

            payload = model_dump_compat(sample)
            payload["visual_observation"] = predicted_obs.get("visual_observation", "")
            payload["textual_observation"] = predicted_obs.get("textual_observation", "")
            payload["joint_mechanism"] = predicted_obs.get("joint_mechanism", "")
            payload["meta"] = dict(payload.get("meta") or {})
            payload["meta"]["gold_observations"] = {
                "visual_observation": sample.visual_observation,
                "textual_observation": sample.textual_observation,
                "joint_mechanism": sample.joint_mechanism,
            }
            payload["meta"]["observation_source"] = "stage1_predicted"
            payload["meta"]["stage1_adapter_path"] = stage1_adapter_path
            if image_error:
                payload["meta"]["image_fallback_error"] = image_error

            writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
            writer.flush()
            kept += 1

    final_processed_ids = load_processed_ids(output_dataset, failures_output)
    report = {
        "input_dataset": str((ROOT_DIR / input_dataset).resolve()),
        "output_dataset": str(output_dataset),
        "failures_output": str(failures_output),
        "model_name": model_name,
        "stage1_adapter_path": str((ROOT_DIR / stage1_adapter_path).resolve()),
        "requested_samples": len(samples),
        "split": args.split,
        "resume": args.resume,
        "retry_failures_from": (
            str((ROOT_DIR / args.retry_failures_from).resolve())
            if args.retry_failures_from
            else None
        ),
        "retry_failure_ids": len(retry_failure_ids),
        "skipped_existing": skipped_existing,
        "kept_samples": kept,
        "failed_samples": failed,
        "total_processed_ids": len(final_processed_ids),
        "failure_examples": failure_examples,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Predicted-observation dataset written to {output_dataset}")
    print(f"Failures written to {failures_output}")
    print(f"Report written to {report_output}")


if __name__ == "__main__":
    main()
