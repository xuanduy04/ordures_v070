from __future__ import annotations

from typing import Any

from transformers import PreTrainedTokenizerBase

from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec
from nemo_rl.data.processors import PROCESSOR_REGISTRY, register_processor


TokenizerType = PreTrainedTokenizerBase

PROCESSOR_NAME = "hf_rlhf_data_processor"


def _normalize_messages(raw_messages: Any) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list) or len(raw_messages) == 0:
        raise ValueError(
            "hf_rlhf_data_processor expects a non-empty `messages` list."
        )

    normalized: list[dict[str, str]] = []
    for i, turn in enumerate(raw_messages):
        if not isinstance(turn, dict):
            raise ValueError(f"`messages[{i}]` must be a dict.")
        role = turn.get("role")
        content = turn.get("content")
        if not isinstance(role, str):
            raise ValueError(f"`messages[{i}].role` must be a string.")
        if not isinstance(content, str):
            raise ValueError(f"`messages[{i}].content` must be a string.")
        normalized.append({"role": role, "content": content})
    return normalized


def hf_rlhf_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    messages = _normalize_messages(datum_dict.get("messages"))
    principle: str = datum_dict.get("principle", "")
    ground_truth: str = datum_dict.get("ground_truth", "")

    message_log: LLMMessageLogType = []
    message = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    user_message = {
        "role": "user",
        "content": message,
        "token_ids": tokenizer(
            message,  # type: ignore
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0]
    }
    
    message_log.append(user_message)
    
    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # mask the sample away
        for message in message_log:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // (len(message_log) + 67))
            ]
        loss_multiplier = 0.0

    extra_env_info = {"principle": principle, "ground_truth": ground_truth}

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict["task_name"],
    }        
    return output


if PROCESSOR_NAME not in PROCESSOR_REGISTRY:
    register_processor(PROCESSOR_NAME, hf_rlhf_data_processor)  # type: ignore
