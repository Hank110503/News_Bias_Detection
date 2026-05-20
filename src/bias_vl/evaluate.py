from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stage outputs against gold JSONL.")
    parser.add_argument("--gold", required=True, help="Gold JSONL path.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL path.")
    parser.add_argument("--stage", choices=["stage1", "stage2"], default="stage2")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def field_coverage(rows: List[Dict[str, Any]], fields: List[str]) -> float:
    if not rows:
        return 0.0
    covered = 0
    for row in rows:
        if all(str(row.get(field) or "").strip() for field in fields):
            covered += 1
    return covered / len(rows)


def aligned_rows(
    gold_rows: List[Dict[str, Any]],
    pred_rows: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    pred_by_id = {
        str((row.get("meta") or {}).get("news_id") or idx): row
        for idx, row in enumerate(pred_rows)
    }
    pairs = []
    for idx, gold in enumerate(gold_rows):
        key = str((gold.get("meta") or {}).get("news_id") or idx)
        if key in pred_by_id:
            pairs.append((gold, pred_by_id[key]))
    return pairs


def evaluate_stage1(gold_rows: List[Dict[str, Any]], pred_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "json_parse_rate": 1.0,
        "field_coverage": field_coverage(
            pred_rows,
            ["visual_observation", "textual_observation", "joint_mechanism"],
        ),
        "matched_rows": len(aligned_rows(gold_rows, pred_rows)),
    }


def evaluate_stage2(gold_rows: List[Dict[str, Any]], pred_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    pairs = aligned_rows(gold_rows, pred_rows)
    if not pairs:
        return {
            "json_parse_rate": 1.0,
            "matched_rows": 0,
            "cue_present_macro_f1": 0.0,
            "cue_score_accuracy": 0.0,
            "cue_score_mae": 0.0,
            "relation_macro_f1": 0.0,
        }

    gold_present: List[int] = []
    pred_present: List[int] = []
    gold_scores: List[int] = []
    pred_scores: List[int] = []
    gold_rel: List[str] = []
    pred_rel: List[str] = []

    for gold, pred in pairs:
        gold_cues = gold.get("cues", {}) or {}
        pred_cues = pred.get("cues", {}) or {}
        for cue_name, gold_value in gold_cues.items():
            pred_value = pred_cues.get(cue_name, {})
            gold_present.append(int(bool(gold_value.get("present", False))))
            pred_present.append(int(bool(pred_value.get("present", False))))
            gold_scores.append(int(gold_value.get("score", 0)))
            pred_scores.append(int(pred_value.get("score", 0)))

        gold_cons = gold.get("consistency_dimension", {}) or {}
        pred_cons = pred.get("consistency_dimension", {}) or {}
        gold_rel.append(str(gold_cons.get("D1_Relationship_Type", "")))
        pred_rel.append(str(pred_cons.get("D1_Relationship_Type", "")))

    score_accuracy = float(np.mean(np.array(gold_scores) == np.array(pred_scores)))
    score_mae = float(np.mean(np.abs(np.array(gold_scores) - np.array(pred_scores))))
    relation_macro_f1 = f1_score(gold_rel, pred_rel, average="macro", zero_division=0)
    cue_present_macro_f1 = f1_score(
        gold_present,
        pred_present,
        average="macro",
        zero_division=0,
    )

    return {
        "json_parse_rate": 1.0,
        "matched_rows": len(pairs),
        "cue_present_macro_f1": cue_present_macro_f1,
        "cue_score_accuracy": score_accuracy,
        "cue_score_mae": score_mae,
        "relation_macro_f1": relation_macro_f1,
    }


def main() -> None:
    args = parse_args()
    gold_rows = load_jsonl(args.gold)
    pred_rows = load_jsonl(args.predictions)
    if args.stage == "stage1":
        result = evaluate_stage1(gold_rows, pred_rows)
    else:
        result = evaluate_stage2(gold_rows, pred_rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
