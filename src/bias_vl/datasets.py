from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from sklearn.model_selection import train_test_split

from .schema import (
    ConvertedSample,
    Stage1Target,
    Stage2Target,
    model_dump_compat,
    ordered_stage1_target,
    ordered_stage2_target,
)


STAGE2_TASK_MODES = {
    "stage2",
    "stage2_direct",
    "stage2_text_only",
}


DEFAULT_STAGE1_SYSTEM_PROMPT = """You are an expert in multimodal media bias analysis.

Analyze the given news image and text, then produce a valid JSON object with:
1. visual_observation
2. textual_observation
3. joint_mechanism
"""


DEFAULT_STAGE2_SYSTEM_PROMPT = """You are an expert in multimodal media bias analysis.

Use the given image, text, and observations to predict cues and consistency_dimension.
For each cue, the JSON key order must be:
1. reason
2. present
3. score
For the consistency_dimension, the JSON key order must be:
1. reason
2. D1_Relationship_Type
"""

DEFAULT_STAGE2_DIRECT_SYSTEM_PROMPT = """You are an expert in multimodal media bias analysis.

Use the given news image and text to directly predict cues and consistency_dimension.
Do not generate intermediate observations.
For each cue, the JSON key order must be:
1. reason
2. present
3. score
For the consistency_dimension, the JSON key order must be:
1. reason
2. D1_Relationship_Type
"""


DEFAULT_STAGE2_TEXT_ONLY_SYSTEM_PROMPT = """You are an expert in multimodal media bias analysis.

Use only the given news text to directly predict cues and consistency_dimension.
No image or intermediate observations are available.
For each cue, the JSON key order must be:
1. reason
2. present
3. score
For the consistency_dimension, the JSON key order must be:
1. reason
2. D1_Relationship_Type
"""


def load_converted_samples(dataset_path: str | Path) -> List[ConvertedSample]:
    path = Path(dataset_path)
    samples: List[ConvertedSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            samples.append(ConvertedSample(**payload))
    return samples


def split_samples(
    samples: Sequence[ConvertedSample],
    eval_ratio: float,
    seed: int,
    stratify_stage2: bool = False,
) -> Tuple[List[ConvertedSample], List[ConvertedSample]]:
    if not samples:
        return [], []

    stratify_labels = None
    if stratify_stage2:
        labels = [sample.consistency_dimension.D1_Relationship_Type for sample in samples]
        non_empty = [label for label in labels if label]
        label_counts = {label: labels.count(label) for label in set(labels)}
        enough_per_class = all(count >= 2 for count in label_counts.values())
        if len(set(non_empty)) > 1 and len(non_empty) == len(labels) and enough_per_class:
            stratify_labels = labels

    train_samples, eval_samples = train_test_split(
        list(samples),
        test_size=eval_ratio,
        random_state=seed,
        stratify=stratify_labels,
    )
    return train_samples, eval_samples


def filter_stage1_samples(samples: Iterable[ConvertedSample]) -> List[ConvertedSample]:
    return [
        sample
        for sample in samples
        if sample.visual_observation
        and sample.textual_observation
        and sample.joint_mechanism
    ]


def filter_stage2_samples(
    samples: Iterable[ConvertedSample],
    require_observations: bool = True,
) -> List[ConvertedSample]:
    filtered: List[ConvertedSample] = []
    for sample in samples:
        if require_observations and not (
            sample.visual_observation
            and sample.textual_observation
            and sample.joint_mechanism
        ):
            continue
        if not (
            sample.consistency_dimension.reason
            and sample.consistency_dimension.D1_Relationship_Type
        ):
            continue
        if not any(cue.reason for cue in sample.cues.values()):
            continue
        filtered.append(sample)
    return filtered


def build_stage1_messages(
    sample: ConvertedSample,
    system_prompt: str = DEFAULT_STAGE1_SYSTEM_PROMPT,
) -> List[Dict[str, Any]]:
    target = Stage1Target(**ordered_stage1_target(sample))
    target_json = json.dumps(model_dump_compat(target), ensure_ascii=False)
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Text:\n{sample.text}"},
            ],
        },
        {"role": "assistant", "content": target_json},
    ]


def build_stage2_messages(
    sample: ConvertedSample,
    system_prompt: str = DEFAULT_STAGE2_SYSTEM_PROMPT,
) -> List[Dict[str, Any]]:
    target = Stage2Target(**ordered_stage2_target(sample))
    observation_text = (
        f"[Observations]\n"
        f"visual_observation: {sample.visual_observation}\n"
        f"textual_observation: {sample.textual_observation}\n"
        f"joint_mechanism: {sample.joint_mechanism}\n"
    )
    target_json = json.dumps(model_dump_compat(target), ensure_ascii=False)
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": f"Text:\n{sample.text}\n\n{observation_text}",
                },
            ],
        },
        {"role": "assistant", "content": target_json},
    ]


def build_stage2_direct_messages(
    sample: ConvertedSample,
    system_prompt: str = DEFAULT_STAGE2_DIRECT_SYSTEM_PROMPT,
) -> List[Dict[str, Any]]:
    target = Stage2Target(**ordered_stage2_target(sample))
    target_json = json.dumps(model_dump_compat(target), ensure_ascii=False)
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Text:\n{sample.text}"},
            ],
        },
        {"role": "assistant", "content": target_json},
    ]


def build_stage2_text_only_messages(
    sample: ConvertedSample,
    system_prompt: str = DEFAULT_STAGE2_TEXT_ONLY_SYSTEM_PROMPT,
) -> List[Dict[str, Any]]:
    target = Stage2Target(**ordered_stage2_target(sample))
    target_json = json.dumps(model_dump_compat(target), ensure_ascii=False)
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Text:\n{sample.text}"},
            ],
        },
        {"role": "assistant", "content": target_json},
    ]


def build_stage2_messages_for_mode(
    sample: ConvertedSample,
    task_mode: str,
) -> List[Dict[str, Any]]:
    if task_mode == "stage2":
        return build_stage2_messages(sample)
    if task_mode == "stage2_direct":
        return build_stage2_direct_messages(sample)
    if task_mode == "stage2_text_only":
        return build_stage2_text_only_messages(sample)
    raise ValueError(f"Unsupported Stage 2 task mode: {task_mode}")
