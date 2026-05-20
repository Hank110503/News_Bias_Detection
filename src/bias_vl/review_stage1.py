from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import yaml
from PIL import Image

from .infer import infer_stage1
from .modeling import load_model_with_adapter

ROOT_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run qualitative Stage 1 review on a few samples and write a markdown report."
    )
    parser.add_argument("--config", default="configs/stage1.yaml")
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output", default="outputs/stage1_review.md")
    parser.add_argument(
        "--news_id",
        action="append",
        dest="news_ids",
        help="Optional news_id to inspect. Repeat the flag to pass multiple ids.",
    )
    parser.add_argument("--text_preview_chars", type=int, default=400)
    return parser.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    with (ROOT_DIR / path).resolve().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with (ROOT_DIR / path).resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def choose_rows(
    rows: List[Dict[str, Any]],
    news_ids: List[str] | None,
    num_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if news_ids:
        wanted = {str(item) for item in news_ids}
        selected = [
            row
            for row in rows
            if str((row.get("meta") or {}).get("news_id")) in wanted
        ]
        if not selected:
            raise ValueError(f"No rows matched news_id values: {sorted(wanted)}")
        return selected

    rng = random.Random(seed)
    if len(rows) <= num_samples:
        return rows
    return rng.sample(rows, num_samples)


def to_pretty_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_report(
    rows: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    text_preview_chars: int,
) -> str:
    sections: List[str] = ["# Stage 1 Review", ""]

    for index, (row, prediction) in enumerate(zip(rows, predictions), 1):
        meta = row.get("meta") or {}
        preview = (row.get("text") or "").strip().replace("\n", " ")
        if len(preview) > text_preview_chars:
            preview = preview[:text_preview_chars].rstrip() + "..."

        gold = {
            "visual_observation": row.get("visual_observation", ""),
            "textual_observation": row.get("textual_observation", ""),
            "joint_mechanism": row.get("joint_mechanism", ""),
        }

        sections.extend(
            [
                f"## Sample {index}",
                f"- news_id: `{meta.get('news_id', '')}`",
                f"- image: `{row.get('image', '')}`",
                f"- title: {meta.get('title', '')}",
                "",
                "### Text Preview",
                "",
                preview or "(empty)",
                "",
                "### Gold",
                "",
                "```json",
                to_pretty_json(gold),
                "```",
                "",
                "### Prediction",
                "",
                "```json",
                to_pretty_json(prediction),
                "```",
                "",
            ]
        )

    return "\n".join(sections)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    dataset_path = args.dataset_path or config["dataset_path"]
    adapter_path = args.adapter_path or str(
        Path(config["output_dir"]) / "final_adapter"
    )
    model_name = args.model_name or config["model_name"]

    rows = load_jsonl(dataset_path)
    selected_rows = choose_rows(
        rows=rows,
        news_ids=args.news_ids,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    model, processor = load_model_with_adapter(
        model_name=model_name,
        adapter_path=adapter_path,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    predictions: List[Dict[str, Any]] = []
    for row in selected_rows:
        image = Image.open((ROOT_DIR / row["image"]).resolve()).convert("RGB")
        predictions.append(
            infer_stage1(
                model=model,
                processor=processor,
                text=row["text"],
                image=image,
                max_new_tokens=args.max_new_tokens,
            )
        )

    report = build_report(
        rows=selected_rows,
        predictions=predictions,
        text_preview_chars=args.text_preview_chars,
    )
    output_path = (ROOT_DIR / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Stage 1 review written to {output_path}")


if __name__ == "__main__":
    main()
