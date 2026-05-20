from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .datasets import filter_stage2_samples, load_converted_samples
from .schema import model_dump_compat

ROOT_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Stage 2 dataset with mixed predicted/gold observations."
    )
    parser.add_argument("--stage2_config", default="configs/stage2_v2.yaml")
    parser.add_argument(
        "--gold_dataset",
        default="data/processed/converted/bishe_schema_complete_cues_reason_merged.jsonl",
    )
    parser.add_argument(
        "--predicted_dataset",
        default="data/processed/converted/bishe_schema_stage2_predicted_obs_reason_merged.jsonl",
    )
    parser.add_argument(
        "--output_dataset",
        default="data/processed/converted/bishe_schema_stage2_mixed_obs_7to3_reason_merged.jsonl",
    )
    parser.add_argument(
        "--report_output",
        default="data/processed/converted/bishe_schema_stage2_mixed_obs_7to3_reason_merged_report.json",
    )
    parser.add_argument(
        "--predicted_ratio",
        type=float,
        default=0.7,
        help="Probability of using predicted observations when a predicted row exists.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    with (ROOT_DIR / path).resolve().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sample_news_id(sample: Any) -> str:
    return str(((sample.meta or {}).get("news_id") or "")).strip()


def main() -> None:
    args = parse_args()
    stage2_config = load_config(args.stage2_config)

    gold_dataset_path = (ROOT_DIR / args.gold_dataset).resolve()
    predicted_dataset_path = (ROOT_DIR / args.predicted_dataset).resolve()
    output_dataset_path = (ROOT_DIR / args.output_dataset).resolve()
    report_output_path = (ROOT_DIR / args.report_output).resolve()
    output_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_path.parent.mkdir(parents=True, exist_ok=True)

    gold_samples = filter_stage2_samples(load_converted_samples(gold_dataset_path))
    predicted_samples = filter_stage2_samples(load_converted_samples(predicted_dataset_path))

    predicted_by_id = {
        sample_news_id(sample): sample
        for sample in predicted_samples
        if sample_news_id(sample)
    }

    rng = random.Random(args.seed)
    counts = Counter()

    with output_dataset_path.open("w", encoding="utf-8") as writer:
        for gold_sample in gold_samples:
            news_id = sample_news_id(gold_sample)
            predicted_sample = predicted_by_id.get(news_id)

            use_predicted = (
                predicted_sample is not None and rng.random() < args.predicted_ratio
            )
            source_sample = predicted_sample if use_predicted else gold_sample

            payload = model_dump_compat(source_sample)
            payload["meta"] = dict(payload.get("meta") or {})
            payload["meta"]["gold_observations"] = {
                "visual_observation": gold_sample.visual_observation,
                "textual_observation": gold_sample.textual_observation,
                "joint_mechanism": gold_sample.joint_mechanism,
            }
            payload["meta"]["mixed_observation_source"] = (
                "stage1_predicted" if use_predicted else "gold"
            )
            payload["meta"]["mixed_predicted_ratio"] = args.predicted_ratio

            if predicted_sample is None:
                counts["fallback_missing_predicted"] += 1
            if use_predicted:
                counts["predicted_observation_rows"] += 1
            else:
                counts["gold_observation_rows"] += 1

            writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
            counts["written_rows"] += 1

    report = {
        "gold_dataset": str(gold_dataset_path),
        "predicted_dataset": str(predicted_dataset_path),
        "output_dataset": str(output_dataset_path),
        "stage2_config": str((ROOT_DIR / args.stage2_config).resolve()),
        "predicted_ratio": args.predicted_ratio,
        "seed": args.seed,
        "gold_samples": len(gold_samples),
        "predicted_samples": len(predicted_samples),
        "counts": dict(counts),
        "stage2_output_dir": str((ROOT_DIR / stage2_config["output_dir"]).resolve()),
    }
    report_output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Mixed-observation dataset written to {output_dataset_path}")
    print(f"Report written to {report_output_path}")


if __name__ == "__main__":
    main()
