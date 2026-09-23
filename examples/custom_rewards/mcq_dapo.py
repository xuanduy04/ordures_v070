from typing import Any, NotRequired, TypedDict

import ray
import torch
from tqdm.auto import tqdm
# LLMMessageLogType is the conversation structure used by NeMo RL.
# It is effectively:
#   list[dict[str, Union[str, torch.Tensor]]]
# where each dict is a turn like {"role": "user"|"assistant"|"environment", "content": "..."}.
from examples.custom_rewards.mcq_exact_match import MCQMetadata
from examples.custom_rewards.format_reward import verify_format
from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


from examples.custom_rewards.utils.tqdm_utils import maybe_tqdm


# ============================= DAPO math verifier ============================= #
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://raw.githubusercontent.com/verl-project/verl/refs/heads/main/verl/utils/reward_score/math_dapo.py

import re

def last_boxed_only_string(string: str) -> str | None:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def is_correct_minerva(
    pred: str, groud_truth: str, answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)"
) -> tuple[bool, str]:
    """Check if the solution is correct according to Minerva criteria.

    Args:
        pred: The solution string to check
        groud_truth: The ground truth answer
        answer_pattern: Regex pattern to extract the answer

    Returns:
        Tuple of (is_correct, normalized_prediction)
    """
    # Extract answer from solution
    match = re.findall(answer_pattern, pred)
    if match:
        extracted_answer = match[-1]
        pred = normalize_final_answer(extracted_answer)
        groud_truth = normalize_final_answer(groud_truth)
    else:
        raise ValueError("Could not extract answer from solution string")

    return bool(pred.strip().lower() == groud_truth.strip().lower()), pred


def is_correct_strict_box(
    pred: str, groud_truth: str, pause_tokens_index: list[int] | None = None
) -> tuple[int, str | None]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        groud_truth: The ground truth answer

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    if boxed_pred is None:
        raise ValueError("Could not extract answer from solution string")
    extracted_pred = remove_boxed(boxed_pred)

    return bool(extracted_pred.strip().lower() == groud_truth.strip().lower()), extracted_pred


def dapo_verify(
    pred: str, groud_truth: str, strict_box_verify: bool
) -> tuple[bool, str]:
    """Verify if the solution is correct.

    Args:
        pred: The solution string to verify
        groud_truth: The ground truth answer
        strict_box_verify: Whether to strictly use box verification (otherwise verifies with both '\\boxed{}' and 'Answer: '

    Returns:
        True if the solution is correct, False otherwise
    """
    try:
        correct_box, pred_box = is_correct_strict_box(pred, groud_truth)
    except:
        correct_box, pred_box = False, None
    
    if strict_box_verify:
        return correct_box

    try:
        correct_minerva, pred_minerva = is_correct_minerva(pred, groud_truth)
    except:
        correct_minerva, pred_minerva = False, None
    
    return bool(correct_box or correct_minerva)

# ============================= End of DAPO math verifier ============================= #


class MCQDAPOEnvConfig(TypedDict):
    # Whether to only reward answers that have correct thinking tags (default: True)
    strict_format_reward: NotRequired[bool]

    # Whether to only parse answers from \boxed{} or also additionally allow 'Answer: ' (default: False)
    strict_box_verify: NotRequired[bool]

    # verbosity just controls the progress bar. (default: False)
    # The class variable `_DEBUG` is the one that prints debug outputs.
    verbose: NotRequired[bool]


@ray.remote(max_restarts=-1, max_task_retries=-1)
class MCQDAPOEnvironment(EnvironmentInterface[MCQMetadata]):
    """Single-turn MCQ environment with exact-matching rewards but passed through DAPO-style filtering.

    Note:
    - We always mark each sample as terminated in one step.
    - The model responds once per prompt and we score immediately.
    """
    def __init__(self, cfg: MCQDAPOEnvConfig | dict[str, Any]):
        self.cfg = cfg

        self.strict_format_reward = bool(cfg.get("strict_format_reward", True))
        self.strict_box_verify = bool(cfg.get("strict_box_verify", False))

        self.verbose = bool(cfg.get("verbose", False))

    def __repr__(self):
        return "MCQDAPOEnvironment"

    def verify(self, predicted: str, groud_truth: str) -> bool:
        # Apply required normalization to both prediction and target.
        return dapo_verify(pred=predicted, groud_truth=groud_truth, strict_box_verify=self.strict_box_verify)

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[MCQMetadata],
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn[MCQMetadata]:
        # We build batched return fields progressively.
        rewards: list[float] = []
        observations: list[dict[str, str]] = []
        # answers: list[str] = []

        # Iterate batch-wise over conversations and their metadata because this is 100% deterministic.
        #     This `for` loop will never be the bottleneck. (current speed: 80,000 iterations per sec)
        # zip(...) is safe because NeMo RL provides matched batch lengths.
        for conversation, env_info in maybe_tqdm(zip(message_log_batch, metadata), use_tqdm=self.verbose, total=len(message_log_batch), desc="Environment 'MCQ' scoring samples"):
            # Extract the assistant output from the full conversation.
            # We concatenate all assistant turns so this still works if more than one
            # assistant turn exists in a conversation.
            assistant_response: str = "".join(
                str(turn["content"])
                for turn in conversation
                if turn.get("role") == "assistant"
            )

            format_check_result = verify_format(assistant_response)
            if self.strict_format_reward and (not format_check_result.is_correct_format):
                is_correct = False
            else:
                is_correct = self.verify(assistant_response, str(env_info["ground_truth"]))

            # Reward shape is scalar float per sample.
            if self.strict_format_reward:
                reward_value = (1.0 if is_correct else 0.0)
            else:
                reward_value = (1.0 if format_check_result.is_correct_format else 0.9) if is_correct else 0.0
            rewards.append(reward_value)
            
            # Observation is optional but useful for debug logging.
            observation = (
                f"Environment 'MCQ':\n"
                f"\t- Answer: {'correct' if is_correct else 'incorrect'}\n"
                f"\t- Format: {'correct' if format_check_result.is_correct_format else 'incorrect'}"
            )
            if not format_check_result.is_correct_format:
                issues = "\n".join(format_check_result.issues)
                observation += f"\n\t  Formatting issues:\n{issues}"

            observations.append({"role": "environment", "content": observation})
            # answers.append(format_check_result.extracted_answer)

        # Convert Python list -> CPU tensor because the environment contract expects tensor rewards in batched form.
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).cpu()
        # Single-turn environment:
        #     terminated = 1 for every sample in this batch.
        #     We match reward tensor shape for convenience.
        terminated_tensor = torch.ones_like(rewards_tensor).cpu()

        return EnvironmentReturn(
            # Environment text feedback for each sample.
            observations=observations,
            # Pass metadata through unchanged.
            metadata=metadata,
            # No dynamic stop strings for the next turn since we terminate immediately.
            next_stop_strings=[None] * len(message_log_batch),
            # Batched scalar rewards.
            rewards=rewards_tensor,
            # Batched done flags (all done after one step).
            terminateds=terminated_tensor,
            answers=None  # answers if return_extracted_answer else None,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float]]:
        # Emit an easy-to-read metric at env level:
        # average reward == exact-match accuracy for this binary reward.
        metrics = {"accuracy": float(batch["rewards"].float().mean().item())}
        # Return batch unchanged + computed metrics.
        return batch, metrics
