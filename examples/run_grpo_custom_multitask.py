from __future__ import annotations

from typing import Any

from nemo_rl.environments.utils import register_env
import nemo_rl.data.utils as data_utils
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES


from examples.run_grpo_custom import main


def _extract_env_names_multidataset(data_config: dict[str, Any]) -> list[str]:
    """Extract unique env names from train/validation/default for dict-or-list splits."""
    env_names: set[str] = set()

    for split_key in ("train", "validation", "default"):
        split_cfg = data_config.get(split_key)
        if split_cfg is None:
            continue

        # Normalize to list so both single-dataset and multi-dataset forms work.
        split_entries = split_cfg if isinstance(split_cfg, list) else [split_cfg]

        for entry in split_entries:
            if not isinstance(entry, dict):
                continue
            if "env_name" in entry and entry["env_name"] is not None:
                env_names.add(str(entry["env_name"]))

    return list(env_names)


# Patch the helper used by setup_response_data in nemo_rl.data.utils.
data_utils.extract_necessary_env_names = _extract_env_names_multidataset

# Register extra envs for multi-task.


def register_custom_env(env_name: str, actor_class_fqn: str) -> None:
    print(f"Registering '{env_name}'... ", end="")

    ACTOR_ENVIRONMENT_REGISTRY[actor_class_fqn] = PY_EXECUTABLES.SYSTEM

    register_env(
        env_name=env_name,
        actor_class_fqn=actor_class_fqn,
    )
    print("Done.")


for env_name, actor_class_fqn in [
    # (
    #     "code_with_test_cases",
    #     "examples.custom_rewards.code_with_test_cases.CodeEvalEnvironment",
    # ),
    (
        "llm_judge",
        "examples.custom_rewards.llm_judge.LLMJudgeEnvironment",
    ),
    # (
    #     "mock_llm_judge",
    #     "examples.custom_rewards.llm_judge.MockLLMJudgeEnvironment",
    # ),
    # (
    #     "mcq_exact_match",
    #     "examples.custom_rewards.mcq_exact_match.MCQExactMatchEnvironment",
    # ),
    (
        "mcq_dapo",
        "examples.custom_rewards.mcq_dapo.MCQDAPOEnvironment",
    ),
    (
        "rlhf",
        "examples.custom_rewards.rlhf_reward.RLHFEnvironment",
    ),
]:
    register_custom_env(env_name, actor_class_fqn)


if __name__ == "__main__":
    main()

    print("TRAINING HAS FINISHED!!!!!!!")
