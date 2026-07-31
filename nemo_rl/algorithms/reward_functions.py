from collections import defaultdict
from typing import (
    NotRequired,
    TypedDict,
    TypeVar,
)

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

Tensor = TypeVar("Tensor", bound=torch.Tensor)


class RewardShapingConfig(TypedDict):
    """Configuration for reward function processing.

    This configuration enables custom reward shaping, supporting N-gram repetition
    penalties, overlong (DAPO-style) penalties, and stop-properly penalties for
    truncated responses.
    """

    enabled: bool

    # The length of the buffer to penalize responses that exceed the maximum response length threshold.
    # Responses of length greater than overlong_buffer_length + max_response_length will
    # receive the maximum penalty.
    overlong_buffer_length: NotRequired[int]

    # The penalty for responses that exceed the maximum response length threshold.
    overlong_buffer_penalty: NotRequired[float]

    # The maximum response length threshold. Responses exceeding this length will be penalized.
    max_response_length: NotRequired[int]

    # Stop properly penalty: scale factor for rewards of truncated responses (0-1).
    # When set to 0, truncated responses get zero reward.
    # When set to 1, no penalty is applied (default behavior).
    stop_properly_penalty_coef: NotRequired[float | None]

    # N-gram repetition penalty: penalize responses with over-repeated n-gram token patterns.
    # Size of the n-grams (e.g., 5 for 5-gram).
    n_gram_size: NotRequired[int]
    # Frequency threshold above which an n-gram is considered "over-repeated" (e.g., 5).
    n_gram_threshold: NotRequired[int]
    # Weight for the repetition penalty: updated_reward = original_reward + weight * penalty.
    n_gram_repetition_weight: NotRequired[float]


def _compute_n_gram_repetition_penalty(
    token_ids: torch.Tensor, n: int, threshold: int
) -> float:
    """Compute N-gram repetition penalty for a single response sequence.

    Given a sequence of token IDs, constructs the multiset of n-grams and
    identifies over-repeated n-grams (frequency > threshold). The penalty is
    -max(|G_over| / |G|, max_freq / (seq_len / n)), or 0 if nothing is
    over-repeated.

    Similar to Phi-4-reasoning (https://arxiv.org/pdf/2504.21318) and GFPO (https://openreview.net/pdf?id=UKOqoULbZS)
    """
    seq_len = token_ids.numel()
    if seq_len < n:
        return 0.0

    total_n_grams = seq_len - n + 1
    tokens = token_ids.tolist()

    freq: dict[tuple[int, ...], int] = defaultdict(int)
    max_freq = 0
    over_count = 0

    for i in range(total_n_grams):
        gram = tuple(tokens[i : i + n])
        count = freq[gram] + 1
        freq[gram] = count

        # Track max frequency and over-threshold count incrementally
        if count > max_freq:
            max_freq = count

        if count == threshold + 1:
            over_count += 1

    if over_count == 0:
        return 0.0

    fraction_over = over_count / total_n_grams
    normalized_max = max_freq / (seq_len / n)

    return max(fraction_over, normalized_max)


def apply_reward_shaping(
    batch: BatchedDataDict, cfg: RewardShapingConfig
) -> BatchedDataDict:
    """Process rewards by applying custom reward shaping penalties.

    Currently supports N-gram repetition penalties, overlong (DAPO-style) penalties, and
    stop-properly penalties.
    """
    if not cfg["enabled"]:
        return batch
    rewards = batch["total_reward"]

    # Preserve the pre-shaping reward so downstream consumers (e.g. DAPO
    # dynamic sampling) can filter prompt groups on the raw task metric
    # rather than on length-dependent shaped rewards.
    batch["unshaped_total_reward"] = rewards.clone()

    # Apply N-gram repetition penalty and/or overlong/DAPO penalty if configured
    use_n_gram_penalty = (
        cfg.get("n_gram_repetition_weight") is not None
        and cfg.get("n_gram_size") is not None
        and cfg.get("n_gram_threshold") is not None
    )
    use_overlong_penalty = (
        cfg.get("overlong_buffer_length") is not None
        and cfg.get("overlong_buffer_penalty") is not None
        and cfg.get("max_response_length") is not None
    )
    if use_n_gram_penalty or use_overlong_penalty:
        if use_n_gram_penalty:
            n_gram_repetition_weight = abs(cfg["n_gram_repetition_weight"])
            n_gram_size = cfg["n_gram_size"]
            n_gram_threshold = cfg["n_gram_threshold"]

        if use_overlong_penalty:
            overlong_buffer_length = cfg["overlong_buffer_length"]
            overlong_buffer_penalty = abs(cfg["overlong_buffer_penalty"])
            max_response_length = cfg["max_response_length"]
            # Calculate the expected response length
            expected_response_length = max_response_length - overlong_buffer_length

        assert len(batch["message_log"]) == len(rewards), (
            "The number of messages in the batch must match the number of rewards"
        )

        for i, message_log in enumerate(batch["message_log"]):
            # Get the assistant response
            assistant_token_ids = None
            for message in message_log:
                if message["role"] == "assistant":
                    assistant_token_ids = message["token_ids"]
                    break
            assert assistant_token_ids is not None, (
                "Assistant response not found during reward shaping"
            )

            if use_n_gram_penalty:
                penalty = _compute_n_gram_repetition_penalty(
                    assistant_token_ids, n_gram_size, n_gram_threshold
                )
                rewards[i] = rewards[i] - n_gram_repetition_weight * penalty

            if use_overlong_penalty:
                response_length = assistant_token_ids.shape[0]
                exceed_length = response_length - expected_response_length
                overlong_reward = min(
                    -exceed_length / overlong_buffer_length * overlong_buffer_penalty, 0
                )
                rewards[i] = rewards[i] + overlong_reward

        # Update the rewards in the batch
        batch["total_reward"] = rewards

    # Apply stop properly penalty if configured
    stop_properly_penalty_coef = cfg.get("stop_properly_penalty_coef", None)
    if stop_properly_penalty_coef is not None:
        stop_properly_penalty_coef = abs(stop_properly_penalty_coef)  # normalize coefficient to ensure penalization
        truncated = batch.get("truncated")
        assert truncated is not None, "truncated field not found in batch"
        if isinstance(truncated, list):
            truncated = torch.tensor(truncated, dtype=torch.bool, device=rewards.device)
        else:
            truncated = truncated.to(device=rewards.device)

        num_truncated = truncated.sum().item()
        if num_truncated > 0:
            original_rewards = rewards.clone()
            # For truncated samples, shift the reward by stop_properly_penalty_coef
            rewards = rewards - truncated.to(rewards.dtype) * stop_properly_penalty_coef
            batch["total_reward"] = rewards
            print(
                f"[INFO] stop properly penalty applied: {num_truncated}/{len(truncated)} samples truncated, "
                f"coef={stop_properly_penalty_coef}, "
                f"original_reward_mean={original_rewards[truncated].mean().item():.4f}, "
                f"shaped_reward_mean={rewards[truncated].mean().item():.4f}",
                flush=True,
            )
        else:
            print(
                "[INFO] stop properly penalty: no truncated samples (truncation_rate=0)",
                flush=True,
            )

    return batch
