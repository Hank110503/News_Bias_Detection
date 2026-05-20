from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


CANONICAL_CUE_ORDER: List[str] = [
    "V1_Salience",
    "V2_Color_Polarity",
    "V3_Power_Angle",
    "V4_Visual_Selectivity",
    "T1_Agent_Label",
    "T2_Causal_Attribution",
    "M1_Affect_Mismatch",
    "M2_Binary_Roles",
    "M3_Symbol_Decontex",
]

VALID_RELATION_TYPES = {"Reinforcing", "Contrastive", "Supplementary"}


class CueItem(BaseModel):
    reason: str = ""
    present: bool = False
    score: int = 0


class ConsistencyDimension(BaseModel):
    reason: str = ""
    D1_Relationship_Type: str = ""


class Stage1Target(BaseModel):
    visual_observation: str = ""
    textual_observation: str = ""
    joint_mechanism: str = ""


class Stage2Target(BaseModel):
    cues: Dict[str, CueItem] = Field(default_factory=dict)
    consistency_dimension: ConsistencyDimension = Field(
        default_factory=ConsistencyDimension
    )

    @field_validator("cues", mode="before")
    @classmethod
    def sanitize_cues(cls, value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        return {key: item for key, item in value.items() if item is not None}


class ConvertedSample(BaseModel):
    text: str
    image: str
    visual_observation: str = ""
    textual_observation: str = ""
    joint_mechanism: str = ""
    cues: Dict[str, CueItem] = Field(default_factory=dict)
    consistency_dimension: ConsistencyDimension = Field(
        default_factory=ConsistencyDimension
    )
    meta: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("cues", mode="before")
    @classmethod
    def sanitize_cues(cls, value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        return {key: item for key, item in value.items() if item is not None}


def ordered_stage1_target(sample: ConvertedSample) -> OrderedDict[str, str]:
    return OrderedDict(
        [
            ("visual_observation", sample.visual_observation),
            ("textual_observation", sample.textual_observation),
            ("joint_mechanism", sample.joint_mechanism),
        ]
    )


def ordered_stage2_target(sample: ConvertedSample) -> OrderedDict[str, Any]:
    cues = OrderedDict()
    for cue_name in CANONICAL_CUE_ORDER:
        if cue_name in sample.cues:
            cue = sample.cues[cue_name]
            cues[cue_name] = OrderedDict(
                [
                    ("reason", cue.reason),
                    ("present", cue.present),
                    ("score", cue.score),
                ]
            )

    consistency = OrderedDict(
        [
            ("reason", sample.consistency_dimension.reason),
            (
                "D1_Relationship_Type",
                sample.consistency_dimension.D1_Relationship_Type,
            ),
        ]
    )

    return OrderedDict(
        [
            ("cues", cues),
            ("consistency_dimension", consistency),
        ]
    )


def model_dump_json_compat(model: BaseModel, **kwargs: Any) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(**kwargs)
    return model.json(**kwargs)


def model_dump_compat(model: BaseModel, **kwargs: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(**kwargs)
    return model.dict(**kwargs)


def validate_relation_type(value: str) -> Optional[str]:
    value = (value or "").strip()
    if value in VALID_RELATION_TYPES:
        return value
    return None
