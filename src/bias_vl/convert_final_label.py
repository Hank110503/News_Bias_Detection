from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .schema import (
    CANONICAL_CUE_ORDER,
    ConsistencyDimension,
    ConvertedSample,
    CueItem,
    model_dump_compat,
    validate_relation_type,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = ROOT_DIR / "data" / "processed" / "imgs" / "war_imgs_total"
DEFAULT_TEXT_DIR = ROOT_DIR / "data" / "processed" / "texts" / "articles_texts"
CUE_TO_GROUP = {
    "V1_Salience": "visual_only",
    "V2_Color_Polarity": "visual_only",
    "V3_Power_Angle": "visual_only",
    "V4_Visual_Selectivity": "visual_only",
    "T1_Agent_Label": "text_only",
    "T2_Causal_Attribution": "text_only",
    "M1_Affect_Mismatch": "cross_modal",
    "M2_Binary_Roles": "cross_modal",
    "M3_Symbol_Decontex": "cross_modal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert final_label.json into flattened bishe-schema jsonl."
    )
    parser.add_argument(
        "--input",
        default="data/processed/labels/final_label.json",
        help="Path to the source final_label.json file.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/converted/bishe_schema_reason_merged.jsonl",
        help="Path to the converted JSONL file.",
    )
    parser.add_argument(
        "--report",
        default="data/processed/converted/conversion_report_reason_merged.json",
        help="Path to the conversion report JSON file.",
    )
    return parser.parse_args()


def normalize_media_path(raw_path: str, base_dir: Path) -> Path:
    return base_dir / Path(raw_path).name


def load_article_text(text_path: Path) -> Optional[str]:
    try:
        with text_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

    if isinstance(payload, dict):
        text = payload.get("text", "")
    else:
        text = ""
    return str(text or "").strip()


def _is_cue_payload(value: Any) -> bool:
    return isinstance(value, dict) and any(
        key in value for key in ("present", "reason", "score", "evidence")
    )


def _normalize_reason_part(value: Any) -> str:
    return str(value or "").strip()


def _join_reason_parts(parts: List[str]) -> str:
    seen: set[str] = set()
    normalized_parts: List[str] = []
    for part in parts:
        cleaned = _normalize_reason_part(part)
        if not cleaned or cleaned in seen:
            continue
        normalized_parts.append(cleaned)
        seen.add(cleaned)
    return "\n\n".join(normalized_parts)


def build_cue_reason(cue_name: str, raw_value: Dict[str, Any]) -> str:
    if cue_name.startswith("M"):
        return _join_reason_parts(
            [
                str(value)
                for key, value in raw_value.items()
                if "reason" in key.lower()
            ]
        )
    return _normalize_reason_part(raw_value.get("reason"))


def build_consistency_reason(consistency_raw: Dict[str, Any]) -> str:
    return _join_reason_parts(
        [
            consistency_raw.get("D1_text_stance", ""),
            consistency_raw.get("D1_image_stance", ""),
            consistency_raw.get("D1_reason", ""),
        ]
    )


def normalize_cue_groups(raw_cues: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    normalized: Dict[str, Dict[str, Any]] = {
        "visual_only": dict(raw_cues.get("visual_only", {}) or {}),
        "text_only": dict(raw_cues.get("text_only", {}) or {}),
        "cross_modal": dict(raw_cues.get("cross_modal", {}) or {}),
    }
    fixes: List[str] = []

    for key, value in raw_cues.items():
        if key in normalized:
            continue
        if key == "consistency_dimension":
            continue
        if key in CUE_TO_GROUP and _is_cue_payload(value):
            normalized[CUE_TO_GROUP[key]][key] = value
            fixes.append(f"re-homed:{key}")

    return normalized, fixes


def normalize_cue_item(cue_name: str, raw_value: Dict[str, Any]) -> Tuple[CueItem, Dict[str, Any]]:
    reason = build_cue_reason(cue_name, raw_value)
    present = bool(raw_value.get("present", False))
    raw_score = raw_value.get("score", None)
    meta = {"imputed_score": False}

    if raw_score is None:
        score = 1 if present else 0
        meta["imputed_score"] = True
    else:
        try:
            score = int(raw_score)
        except (TypeError, ValueError):
            score = 1 if present else 0
            meta["imputed_score"] = True

    if score == 0:
        present = False

    return CueItem(reason=reason, present=present, score=score), meta


def build_converted_sample(
    raw_sample: Dict[str, Any],
    source_index: int,
    image_dir: Path,
    text_dir: Path,
    report: Counter,
    missing_detail: Dict[str, List[Any]],
) -> Optional[ConvertedSample]:
    raw_image_path = str(raw_sample.get("image_path", "") or "")
    raw_text_path = str(raw_sample.get("text_path", "") or "")
    image_path = normalize_media_path(raw_image_path, image_dir)
    text_path = normalize_media_path(raw_text_path, text_dir)

    if not image_path.exists():
        report["dropped_missing_image"] += 1
        missing_detail["missing_images"].append(
            {"index": source_index, "image_path": raw_image_path}
        )
        return None

    if not text_path.exists():
        report["dropped_missing_text_file"] += 1
        missing_detail["missing_text_files"].append(
            {"index": source_index, "text_path": raw_text_path}
        )
        return None

    text = load_article_text(text_path)
    if not text:
        report["dropped_unreadable_text"] += 1
        missing_detail["unreadable_text"].append(
            {"index": source_index, "text_path": str(text_path)}
        )
        return None

    analysis_chain = raw_sample.get("analysis_chain", {}) or {}
    raw_cues = raw_sample.get("cues", {}) or {}
    normalized_groups, fixes = normalize_cue_groups(raw_cues)

    cues: Dict[str, CueItem] = {}
    cue_meta: Dict[str, Dict[str, Any]] = {}
    missing_reasons: List[str] = []

    for cue_name in CANONICAL_CUE_ORDER:
        group_name = CUE_TO_GROUP[cue_name]
        raw_value = normalized_groups.get(group_name, {}).get(cue_name)
        if not isinstance(raw_value, dict):
            continue

        cue_item, cue_item_meta = normalize_cue_item(cue_name, raw_value)
        if not cue_item.reason:
            missing_reasons.append(cue_name)

        cues[cue_name] = cue_item
        cue_meta[cue_name] = cue_item_meta
        if cue_item_meta["imputed_score"]:
            report["imputed_scores"] += 1

    consistency_raw = raw_sample.get("consistency_dimension", {}) or {}
    relation = validate_relation_type(consistency_raw.get("D1_Relationship_Type", ""))
    consistency = ConsistencyDimension(
        reason=build_consistency_reason(consistency_raw),
        D1_Relationship_Type=relation or "",
    )

    visual_observation = str(analysis_chain.get("visual_observation") or "").strip()
    textual_observation = str(analysis_chain.get("textual_observation") or "").strip()
    joint_mechanism = str(analysis_chain.get("joint_mechanism") or "").strip()

    if all([visual_observation, textual_observation, joint_mechanism]):
        report["stage1_eligible"] += 1
    else:
        missing_detail["stage1_missing_observations"].append(
            {
                "index": source_index,
                "news_id": raw_sample.get("news_id"),
                "missing": [
                    key
                    for key, value in {
                        "visual_observation": visual_observation,
                        "textual_observation": textual_observation,
                        "joint_mechanism": joint_mechanism,
                    }.items()
                    if not value
                ],
            }
        )

    if consistency.reason and consistency.D1_Relationship_Type:
        report["stage2_has_consistency"] += 1
    else:
        missing_detail["stage2_missing_consistency"].append(
            {
                "index": source_index,
                "news_id": raw_sample.get("news_id"),
            }
        )

    if fixes:
        report["schema_drift_fixed"] += 1

    if missing_reasons:
        report["cues_missing_reason"] += len(missing_reasons)
        missing_detail["cue_missing_reason"].append(
            {
                "index": source_index,
                "news_id": raw_sample.get("news_id"),
                "missing_cues": missing_reasons,
            }
        )

    sample = ConvertedSample(
        text=text,
        image=str(image_path.relative_to(ROOT_DIR)),
        visual_observation=visual_observation,
        textual_observation=textual_observation,
        joint_mechanism=joint_mechanism,
        cues=cues,
        consistency_dimension=consistency,
        meta={
            "source_index": source_index,
            "news_id": raw_sample.get("news_id"),
            "title": raw_sample.get("title"),
            "source_image_path": raw_image_path,
            "source_text_path": raw_text_path,
            "schema_fixes": fixes,
            "cue_meta": cue_meta,
        },
    )

    report["kept_samples"] += 1
    return sample


def convert_samples(
    input_path: Path,
    output_path: Path,
    report_path: Path,
) -> Dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {input_path}, got {type(payload).__name__}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = Counter(total_samples=len(payload))
    missing_detail: Dict[str, List[Any]] = defaultdict(list)

    with output_path.open("w", encoding="utf-8") as writer:
        for index, raw_sample in enumerate(payload):
            sample = build_converted_sample(
                raw_sample=raw_sample,
                source_index=index,
                image_dir=DEFAULT_IMAGE_DIR,
                text_dir=DEFAULT_TEXT_DIR,
                report=report,
                missing_detail=missing_detail,
            )
            if sample is None:
                continue
            writer.write(
                json.dumps(model_dump_compat(sample), ensure_ascii=False) + "\n"
            )

    report["stage2_eligible"] = sum(
        1
        for item in iterate_jsonl(output_path)
        if has_stage2_targets(item)
    )

    report_payload = {
        "input": str(input_path),
        "output": str(output_path),
        "summary": dict(report),
        "details": missing_detail,
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, ensure_ascii=False, indent=2)
    return report_payload


def has_stage2_targets(sample: Dict[str, Any]) -> bool:
    cues = sample.get("cues", {}) or {}
    has_any_cue_reason = any(
        isinstance(value, dict) and str(value.get("reason") or "").strip()
        for value in cues.values()
    )
    consistency = sample.get("consistency_dimension", {}) or {}
    has_consistency = bool(
        str(consistency.get("reason") or "").strip()
        and str(consistency.get("D1_Relationship_Type") or "").strip()
    )
    observations = all(
        str(sample.get(key) or "").strip()
        for key in (
            "visual_observation",
            "textual_observation",
            "joint_mechanism",
        )
    )
    return has_any_cue_reason and has_consistency and observations


def iterate_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    args = parse_args()
    report = convert_samples(
        input_path=(ROOT_DIR / args.input).resolve(),
        output_path=(ROOT_DIR / args.output).resolve(),
        report_path=(ROOT_DIR / args.report).resolve(),
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
