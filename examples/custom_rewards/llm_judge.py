from __future__ import annotations

from typing import Any, NotRequired, TypedDict

import ray

from examples.custom_rewards.utils.format_utils import verify_think_format

from examples.custom_rewards.base_llm_judge import BaseLLMJudgeEnvConfig, BaseLLMJudgeVerifyWorker, BaseLLMJudgeEnvironment


class LLMJudgeMetadata(TypedDict):
    # `math_hf_data_processor` writes the expected answer under this key.
    # We reuse the same schema so this environment can drop into existing examples.
    ground_truth: str


class LLMJudgeEnvConfig(BaseLLMJudgeEnvConfig):
    # Whether to only reward answers that have correct thinking tags (default: True)
    strict_format_reward: NotRequired[bool]
    boxed_answer_only: NotRequired[bool]
    answer_colon_only: NotRequired[bool]


@ray.remote(max_restarts=-1, max_task_retries=-1)
class LLMJudgeVerifyWorker(BaseLLMJudgeVerifyWorker):
    _ENV_NAME: str = "LLMJudge"

    def _parse_score(self, judge_text: str) -> float:
        raw_score: float = super()._parse_score(judge_text)
    
        # As fallback method(s) will never be reliable, we indirectly mask away these samples and accept some noise in the reward function.
        return raw_score if 0.0 <= raw_score <= 1.0 else 0.0


@ray.remote(max_restarts=-1, max_task_retries=-1)
class LLMJudgeEnvironment(BaseLLMJudgeEnvironment[LLMJudgeMetadata, LLMJudgeEnvConfig]):
    _ENV_NAME: str = "LLMJudge"
    _VERIFY_WORKER_CLS = LLMJudgeVerifyWorker
    _DEBUG: bool = False
    DEFAULT_JUDGE_SYSTEM_PROMPT: str = r"""You are an impartial evaluator.

You are given:
- The problem statement
- The ground truth answer
- A model-generated answer

Your task is to determine whether the answer is factually consistent with the gold label.

Rules:
- Return 1 inside \boxed{}, i.e. \boxed{1} if the answer conveys the same meaning as the gold label.
- Otherwise, return 0 inside \boxed{}, i.e. \boxed{0} if the answer contradicts, is inconsistent, incomplete, or adds incorrect information.
- Minor wording differences are acceptable if the meaning is the same.
- Do not use external knowledge, only compare the answer to the gold label.
- Accept the answer regardless of formatting or structural differences, as long as the core factual content aligns with the gold label.
- If the model-generated answer contains a clear final choice (e.g., "A", "B", "Answer: x=2", ...), prioritize that choice over any additional text, even if extra, redundant, or noisy content appears. For example, "C. Not wrong, Wrong" should be interpreted as the model selecting C."""

    def __init__(self, cfg) -> None:
        super().__init__(cfg=cfg)

        self.strict_format_reward = bool(cfg.get("strict_format_reward", True))
        self.boxed_answer_only = bool(cfg.get("boxed_answer_only", False))
        self.answer_colon_only = bool(cfg.get("answer_colon_only", False))

        if self.boxed_answer_only and self.answer_colon_only:
            raise ValueError("`boxed_answer_only` and `answer_colon_only` are mutually exclusive but both are True")

        if not self.boxed_answer_only and not self.answer_colon_only:
            print(f"[INFO] {self._ENV_NAME}: Neither `boxed_answer_only` nor `answer_colon_only` is set; "
                  "will try extracting both (boxed first, then answer_colon).")

    def _get_judge_prompt_batch(
        self,
        problem_statement: str,
        assistant_answer: str,
        env_info: LLMJudgeMetadata
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        ground_truth = str(env_info["ground_truth"])

        extract_both = not self.boxed_answer_only and not self.answer_colon_only
        format_check_result = verify_think_format(
            assistant_answer,
            extract_boxed=self.boxed_answer_only or extract_both,
            extract_answer_colon=self.answer_colon_only or extract_both,
        )
        skip_query_llm_judge = (
            (self.strict_format_reward and not format_check_result.is_correct_format) or
            ((self.boxed_answer_only or self.answer_colon_only) and not format_check_result.extracted_answer)
        )
        if skip_query_llm_judge:
            judge_prompt = ""
        else:
            if format_check_result.extracted_answer:
                assistant_answer = format_check_result.extracted_answer
            else:
                # The assistant answer that the judge has access to should only be the part after the thinking.
                if "</think>" in assistant_answer:
                    assistant_answer = assistant_answer.split("</think>")[-1]

            judge_prompt = (f"[Problem Statement]\n{problem_statement}\n\n"
                            f"[Ground Truth Answer]\n{ground_truth}\n\n"
                            f"[Model Generated Answer]\n{assistant_answer}\n")
        return (
            {"skip_query_llm_judge": skip_query_llm_judge, "judge_prompt": judge_prompt},
            {"skip_query_llm_judge": skip_query_llm_judge, "format_check_result": format_check_result}
        )

    def _build_observation(self, judge_result: tuple[float, str, str | None], extra_obs_info: Any | None = None) -> dict[str, str]:
        observation = super()._build_observation(judge_result, extra_obs_info)
        
        format_check_result = extra_obs_info["format_check_result"]
        observation["content"] += f"\n\t- Format: {'correct' if format_check_result.is_correct_format else 'incorrect'}"
        if not format_check_result.is_correct_format:
            issues = "\n".join(format_check_result.issues)
            observation["content"] += f"\n\t  Formatting issues:\n{issues}"
        return observation
