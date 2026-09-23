from typing import Any, TypedDict

import ray
import torch

# LLMMessageLogType is the conversation structure used by NeMo RL.
# It is effectively:
#   list[dict[str, Union[str, torch.Tensor]]]
# where each dict is a turn like {"role": "user"|"assistant"|"environment", "content": "..."}.
from examples.custom_rewards.format_reward import verify_format
from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


class MCQMetadata(TypedDict):
    # This is the metadata payload we expect per sample from the data processor.
    # For this minimal env, we only need one value: the expected answer label/text.
    # Example:
    #   {"ground_truth": "b"}
    ground_truth: str


class MCQExactMatchEnvConfig(TypedDict):
    # Whether to only reward answers that have correct thinking tags (default: True)
    # NOTE: answer will ALWAYS be parsed from the last-occuring \boxed{}, regardless of env configurations.
    strict_format_reward: bool


@ray.remote(max_restarts=-1, max_task_retries=-1)
class MCQExactMatchEnvironment(EnvironmentInterface[MCQMetadata]):
    """Single-turn MCQ environment with exact-match rewards.

    Note:
    - We always mark each sample as terminated in one step.
    - The model responds once per prompt and we score immediately.
    """
    def __init__(self, cfg: MCQExactMatchEnvConfig | dict[str, Any]):
        self.cfg = cfg

        self.strict_format_reward = bool(cfg.get("strict_format_reward", True))

    def _exact_match(self, predicted: str, expected: str) -> bool:
        # Apply required normalization to both prediction and target.
        return self._normalize_answer(predicted) == self._normalize_answer(expected)

    @staticmethod
    def _normalize_answer(text: str) -> str:
        # Basic MCQ text normalization
        return text.strip().lower()

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[MCQMetadata],
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn[MCQMetadata]:
        # We build batched return fields progressively.
        rewards: list[float] = []
        observations: list[dict[str, str]] = []
        answers: list[str] = []

        # Iterate batch-wise over conversations and their metadata because this is 100% deterministic.
        #     This for loop will never be the bottleneck.
        # zip(...) is safe because NeMo RL provides matched batch lengths.
        for conversation, env_info in zip(message_log_batch, metadata):
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
                is_correct = self._exact_match(format_check_result.extracted_answer, str(env_info["ground_truth"]))

            # Reward shape is scalar float per sample.
            rewards.append(1.0 if is_correct else 0.0)
            
            # Observation is optional but useful for debug logging.
            observation = (
                f"Environment 'MCQ':\n"
                f"\t- Answer: {'correct' if is_correct else 'incorrect'}\n"
                f"\t- Format: {'correct' if format_check_result.is_correct_format else 'incorrect'}"
            )
            if not format_check_result.is_correct_format:
                issues = "\n".join(format_check_result.issues)
                observation += f"\n\tFormatting issues:\n{issues}"

            observations.append({"role": "environment", "content": observation})
            answers.append(format_check_result.extracted_answer)

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
            answers=answers if return_extracted_answer else None,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float]]:
        # Emit an easy-to-read metric at env level:
        # average reward == exact-match accuracy for this binary reward.
        metrics = {"accuracy": float(batch["rewards"].float().mean().item())}
        # Return batch unchanged + computed metrics.
        return batch, metrics
