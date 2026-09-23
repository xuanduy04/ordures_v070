from __future__ import annotations

import json
import re
from typing import Any, NotRequired, TypedDict

import ray

from examples.custom_rewards.base_llm_judge import (
    BaseLLMJudgeEnvConfig,
    BaseLLMJudgeEnvironment,
    BaseLLMJudgeVerifyWorker,
)


class RLHFMetadata(TypedDict):
    # 'principle' contains the custom system prompt for this specific sample
    principle: str
    # 'ground_truth' contains the reference/model answer that GenRM compares the rollout against.
    # When absent, GenRM is called with only the rollout (no second generation).
    ground_truth: NotRequired[str]


class RLHFEnvConfig(BaseLLMJudgeEnvConfig):
    pass


DEFAULT_JUDGE_SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants "
    "to the user prompt. Begin your evaluation by generating your own answer to the prompt. You must provide "
    "your answer before judging any answers. When evaluating the assistants' answers, compare both assistants' "
    "answers with your answer. You must identify and correct any mistakes or inaccurate information. Then "
    "consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly "
    "responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than "
    "one interpretation, it is more helpful and appropriate to ask for clarifications or more information from "
    "the user than providing an answer based on assumptions. Relevant means all parts of the response closely "
    "connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or "
    "excessive. Then consider the creativity and novelty of the assistant's answers when needed. Finally, "
    "identify any missing important information in the assistants' answers that would be beneficial to include "
    "when responding to the user prompt."
)

# GenRM protocol bounds: score_1/score_2 are on a 1-5 scale and ranking on a
# 1-6 scale. When scores tie, the ranking swings each score by up to
# |3.5 - 1| = |3.5 - 6| = 2.5, so the achievable reward range is derived from
# those bounds (mirrors resources_servers/genrm_compare).
RANKING_MIDPOINT = 3.5
GENRM_SCORE_MIN = 1.0
GENRM_SCORE_MAX = 5.0
GENRM_RANKING_MIN = 1.0
GENRM_RANKING_MAX = 6.0
GENRM_TIEBREAK_SWING = max(
    abs(RANKING_MIDPOINT - GENRM_RANKING_MIN),
    abs(RANKING_MIDPOINT - GENRM_RANKING_MAX),
)
GENRM_REWARD_MIN = GENRM_SCORE_MIN - GENRM_TIEBREAK_SWING
GENRM_REWARD_MAX = GENRM_SCORE_MAX + GENRM_TIEBREAK_SWING
# Neutral defaults used when the GenRM output cannot be parsed.
GENRM_DEFAULT_SCORE = 3.0
GENRM_DEFAULT_RANKING = 3.5


def _find_json_objects(text: str) -> list[str]:
    """Find top-level JSON objects in text using brace counting."""
    results: list[str] = []
    i: int = 0
    while i < len(text):
        if text[i] == "{":
            depth = 0
            start = i
            in_string = False
            escape_next = False
            for j in range(i, len(text)):
                ch = text[j]
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\" and in_string:
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        results.append(text[start : j + 1])
                        i = j
                        break
        i += 1
    return results


def _try_parse_genrm_json(json_str: str) -> tuple[float, float, float] | None:
    """Attempt to parse a JSON string into (score_1, score_2, ranking)."""
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    # Handle nested format: {"rubric_evaluations": [...], "overall": {...}}
    if "overall" in parsed and isinstance(parsed["overall"], dict):
        parsed = parsed["overall"]

    # Must have at least one expected key
    if not any(k in parsed for k in ("score_1", "score_2", "ranking")):
        return None

    try:
        return (
            float(parsed.get("score_1", GENRM_DEFAULT_SCORE)),
            float(parsed.get("score_2", GENRM_DEFAULT_SCORE)),
            float(parsed.get("ranking", GENRM_DEFAULT_RANKING)),
        )
    except (TypeError, ValueError):
        return None


def _parse_genrm_output(output: str) -> tuple[float, float, float]:
    """Parse GenRM output to extract (score_1, score_2, ranking).

    Searches for JSON in the output text, trying fenced JSON blocks first and
    then any top-level {...} object (taking the last valid one). Falls back to
    the neutral defaults when no parseable JSON is found.
    """
    # Strategy 1: Look for fenced JSON blocks (```json ... ```)
    for match in re.finditer(r"```json\s*([\s\S]*?)\s*```", output, flags=re.IGNORECASE):
        for json_str in _find_json_objects(match.group(1)):
            result = _try_parse_genrm_json(json_str)
            if result is not None:
                return result

    # Strategy 2: Find all top-level {...} and take the last valid one.
    last_valid: tuple[float, float, float] | None = None
    for json_str in _find_json_objects(output):
        result = _try_parse_genrm_json(json_str)
        if result is not None:
            last_valid = result
    if last_valid is not None:
        return last_valid

    return GENRM_DEFAULT_SCORE, GENRM_DEFAULT_SCORE, GENRM_DEFAULT_RANKING


def _genrm_reward(score_1: float, score_2: float, ranking: float, pairwise: bool) -> float:
    """Convert a GenRM comparison into a [0, 1] reward for the rollout (response_1).

    Mirrors `genrm_compare`: when both scores tie, `ranking` breaks the tie
    (ranking < 3.5 favors response_1), then the score is clipped to the derived
    reward bounds and mapped linearly onto [0, 1].
    """
    if pairwise and score_1 == score_2:
        score_1 = score_1 + (RANKING_MIDPOINT - ranking)
    clipped = min(max(score_1, GENRM_REWARD_MIN), GENRM_REWARD_MAX)
    return (clipped - GENRM_REWARD_MIN) / (GENRM_REWARD_MAX - GENRM_REWARD_MIN)


@ray.remote(max_restarts=-1, max_task_retries=-1)
class RLHFVerifyWorker(BaseLLMJudgeVerifyWorker):
    _ENV_NAME: str = "RLHF"

    def _build_messages(
        self,
        judge_prompt: str,
        principle: str = "",
        response_1: str = "",
        response_2: str = "",
        **kwargs: Any,
    ) -> list[dict[str, str]]:
        # GenRM is a pairwise model: when the data has no `ground_truth`, it is
        # called without a second generation (pointwise scoring of `response_1`).
        messages: list[dict[str, str]] = [
            {"role": "user", "content": judge_prompt},
            {"role": "principle", "content": principle or self.default_prompt},
            {"role": "response_1", "content": response_1},
        ]
        if response_2:
            messages.append({"role": "response_2", "content": response_2})
        return messages

    def _parse_score(self, judge_text: str, pairwise: bool = False, **kwargs: Any) -> float:
        score_1, score_2, ranking = _parse_genrm_output(judge_text)
        return _genrm_reward(score_1, score_2, ranking, pairwise=pairwise)


@ray.remote(max_restarts=-1, max_task_retries=-1)
class RLHFEnvironment(BaseLLMJudgeEnvironment[RLHFMetadata, RLHFEnvConfig]):
    """Single-turn reward environment that delegates scoring to LLM judge workers."""

    _ENV_NAME: str = "RLHF"
    _VERIFY_WORKER_CLS = RLHFVerifyWorker
    _DEBUG: bool = False
    DEFAULT_JUDGE_SYSTEM_PROMPT: str = DEFAULT_JUDGE_SYSTEM_PROMPT

    def _get_judge_prompt_batch(
        self,
        conversation_history: str,
        assistant_answer: str,
        env_info: RLHFMetadata,
    ) -> tuple[dict[str, Any], dict | None]:
        # GenRM only takes the final answer, not the reasoning trace.
        if "</think>" in assistant_answer:
            assistant_answer = assistant_answer.split("</think>")[-1]
        # In normal usage, env_info is a dict (that might have "principle").
        # Thus we guard malformed rows to avoid hard crashes.
        principle = str(env_info.get("principle", "")) if isinstance(env_info, dict) else ""
        # `ground_truth` (when present) is the model answer compared against the rollout.
        ground_truth = str(env_info.get("ground_truth", "")) if isinstance(env_info, dict) else ""

        return (
            {
                "judge_prompt": conversation_history,
                "principle": principle,
                "response_1": assistant_answer,
                "response_2": ground_truth,
                "pairwise": bool(ground_truth),
            },
            None,
        )
