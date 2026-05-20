from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .schema import CANONICAL_CUE_ORDER


ROOT_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter converted samples to only rows with all canonical cue fields."
    )
    parser.add_argument(
        "--input",
        default="data/processed/converted/bishe_schema_reason_merged.jsonl",
        help="Input converted JSONL path.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/converted/bishe_schema_complete_cues_reason_merged.jsonl",
        help="Output filtered JSONL path.",
    )
    parser.add_argument(
        "--report",
        default="data/processed/converted/complete_cues_report_reason_merged.json",
        help="Output report JSON path.",
    )
    return parser.parse_args()


def has_complete_cues(row: Dict[str, Any]) -> tuple[bool, List[str]]:
    cues = row.get("cues") or {}
    missing = [cue_name for cue_name in CANONICAL_CUE_ORDER if cues.get(cue_name) is None]
    return (not missing), missing


def main() -> None:
    args = parse_args()
    input_path = (ROOT_DIR / args.input).resolve()
    output_path = (ROOT_DIR / args.output).resolve()
    report_path = (ROOT_DIR / args.report).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    missing_counter = Counter()
    dropped_examples: List[Dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as reader, output_path.open(
        "w", encoding="utf-8"
    ) as writer:
        for line_number, line in enumerate(reader, 1):
            row = json.loads(line)
            counts["total"] += 1
            complete, missing = has_complete_cues(row)
            if complete:
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["kept"] += 1
                continue

            counts["dropped"] += 1
            missing_counter.update(missing)
            if len(dropped_examples) < 20:
                dropped_examples.append(
                    {
                        "line": line_number,
                        "news_id": (row.get("meta") or {}).get("news_id"),
                        "missing_cues": missing,
                    }
                )

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_samples": counts["total"],
        "kept_samples": counts["kept"],
        "dropped_samples": counts["dropped"],
        "missing_cue_counts": dict(missing_counter),
        "dropped_examples": dropped_examples,
    }

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
