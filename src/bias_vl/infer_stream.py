from __future__ import annotations

import json
from typing import Any

from PIL import Image

from .infer import (
    infer_stage1,
    infer_stage2,
    parse_args,
    read_text_arg,
    resolve_image_path,
)
from .modeling import load_model_with_adapter


def emit(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def main() -> None:
    try:
        args = parse_args()
        if args.stage == "full" and args.task_mode != "stage2":
            raise ValueError("--stage full is only supported with --task_mode stage2.")

        emit(
            {
                "type": "status",
                "stage": "boot",
                "message": "正在加载本地多模态模型与 adapter",
            }
        )

        text = read_text_arg(args.text, args.text_file)
        image = None
        if args.stage in {"stage1", "full"} or args.task_mode != "stage2_text_only":
            if not args.image:
                raise ValueError("--image is required for the selected stage/task_mode.")
            image = Image.open(resolve_image_path(args.image)).convert("RGB")

        model, processor = load_model_with_adapter(
            model_name=args.model_name,
            adapter_path=args.adapter_path,
            torch_dtype="auto",
            device_map="auto",
        )
        model.eval()

        if args.stage == "stage1":
            emit(
                {
                    "type": "status",
                    "stage": "stage1",
                    "message": "正在生成 observations",
                }
            )
            observations = infer_stage1(model, processor, text, image, args.max_new_tokens)
            emit(
                {
                    "type": "partial",
                    "stage": "stage1",
                    "data": observations,
                    "sourceLabel": "真实本地模型推理",
                }
            )
            emit(
                {
                    "type": "final",
                    "data": {
                        "observations": observations,
                        "final_result": {
                            "cues": {},
                            "consistency_dimension": {
                                "reason": "",
                                "D1_Relationship_Type": "",
                            },
                        },
                    },
                    "sourceLabel": "真实本地模型推理",
                }
            )
            return

        if args.stage == "stage2":
            if args.task_mode == "stage2" and not args.observation_json:
                raise ValueError("--observation_json is required for stage2 inference.")
            observations = None
            if args.observation_json:
                with open(args.observation_json, "r", encoding="utf-8") as handle:
                    observations = json.load(handle)
            emit(
                {
                    "type": "status",
                    "stage": "stage2",
                    "message": "正在生成 cues 与 consistency",
                }
            )
            final_result = infer_stage2(
                model,
                processor,
                text,
                image if args.task_mode != "stage2_text_only" else None,
                observations,
                args.max_new_tokens,
                task_mode=args.task_mode,
            )
            emit(
                {
                    "type": "final",
                    "data": {
                        "observations": observations
                        or {
                            "visual_observation": "",
                            "textual_observation": "",
                            "joint_mechanism": "",
                        },
                        "final_result": final_result,
                    },
                    "sourceLabel": "真实本地模型推理",
                }
            )
            return

        emit(
            {
                "type": "status",
                "stage": "stage1",
                "message": "正在生成 observations",
            }
        )
        observations = infer_stage1(model, processor, text, image, args.max_new_tokens)
        emit(
            {
                "type": "partial",
                "stage": "stage1",
                "data": observations,
                "sourceLabel": "真实本地模型推理",
            }
        )

        emit(
            {
                "type": "status",
                "stage": "stage2",
                "message": "正在生成 cues 与 consistency",
            }
        )
        final_result = infer_stage2(
            model,
            processor,
            text,
            image,
            observations,
            args.max_new_tokens,
            task_mode="stage2",
        )
        emit(
            {
                "type": "final",
                "data": {
                    "observations": observations,
                    "final_result": final_result,
                },
                "sourceLabel": "真实本地模型推理",
            }
        )
    except Exception as error:
        emit(
            {
                "type": "error",
                "message": str(error),
            }
        )
        raise


if __name__ == "__main__":
    main()
