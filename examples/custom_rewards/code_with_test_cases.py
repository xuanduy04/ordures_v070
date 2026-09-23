import os
import re
import subprocess
import sys
import tempfile
from typing import Any, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from examples.custom_rewards.format_reward import verify_format


class CodeEvalMetadata(TypedDict):
    inputs: list[str]
    outputs: list[str]


@ray.remote(max_restarts=-1, max_task_retries=-1)
class CodeEvalEnvironment(EnvironmentInterface[CodeEvalMetadata]):
    """Single-turn Python Code Evaluation environment.

    Extracts the last Python code block from the assistant's response
    and evaluates it against hidden inputs/outputs using subprocesses.
    """

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        # You can pass the timeout through the config, defaulting to 2.0s
        self.timeout = self.cfg.get("timeout", 2.0)

    @staticmethod
    def _extract_last_python_code(text: str) -> str:
        """Extracts the final ```python ... ``` block from the text."""
        pattern = r"```python\s*(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if not matches:
            return ""
        # Return only the last matched code block
        return matches[-1].strip()

    @staticmethod
    def _evaluate_code(
        code: str, inputs: list[str], outputs: list[str], timeout: float = 2.0
    ) -> float:
        """Runs the code against inputs and expected outputs, returning a pass rate (0.0 to 1.0)."""
        if not inputs or not outputs:
            return 0.0

        assert len(inputs) == len(outputs), "inputs and outputs must have same length"

        correct = 0
        total = len(inputs)

        # Write code to a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            for inp, expected in zip(inputs, outputs):
                try:
                    result = subprocess.run(
                        [sys.executable, temp_path],
                        input=inp,
                        text=True,
                        capture_output=True,
                        timeout=timeout,
                    )

                    # Normalize outputs (strip trailing whitespace/newlines)
                    pred = result.stdout.strip()
                    expected = expected.strip()

                    if pred == expected:
                        correct += 1

                except subprocess.TimeoutExpired:
                    # Treat timeout as incorrect
                    continue
                except Exception:
                    # Catch any other runtime execution errors (e.g., syntax errors, crashes)
                    continue

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return correct / total if total > 0 else 0.0

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[CodeEvalMetadata],
    ) -> EnvironmentReturn[CodeEvalMetadata]:
        rewards: list[float] = []
        observations: list[dict[str, str]] = []

        for conversation, env_info in zip(message_log_batch, metadata):
            # 1. Extract the assistant's full string response
            assistant_response = "".join(
                str(turn["content"])
                for turn in conversation
                if turn.get("role") == "assistant"
            )

            # 1.5. Verify the format
            format_check_result = verify_format(
                assistant_response, answer_wrapper="```python"
            )

            # 2. Parse the final Python code block
            if format_check_result.is_correct_format:
                code = self._extract_last_python_code(assistant_response)

                # 3. Evaluate code and calculate reward
                if not code:
                    # Penalize if no valid Python code block is found
                    reward = 0.0
                    obs_text = "Environment: No valid Python code block found."
                else:
                    inputs = env_info["inputs"]
                    outputs = env_info["outputs"]
                    reward = self._evaluate_code(
                        code, inputs, outputs, timeout=self.timeout
                    )
                    obs_text = (
                        f"Environment: Code evaluation pass rate: {reward * 100:.2f}%"
                    )
                obs_text += "\nNo formatting issues."
            else:
                reward = 0.0
                issues = "\n".join(format_check_result.issues)
                obs_text = (
                    f"Environment: incorrect format.\nFormatting issues:\n{issues}"
                )

            rewards.append(reward)
            observations.append(
                {
                    "role": "environment",
                    "content": obs_text,
                }
            )

        # Convert to CPU tensors for NeMo RL
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).cpu()
        terminated_tensor = torch.ones_like(rewards_tensor).cpu()

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=rewards_tensor,
            terminateds=terminated_tensor,
            answers=None,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float]]:
        # Emit average pass rate across the entire batch
        metrics = {"avg_pass_rate": float(batch["rewards"].float().mean().item())}
        return batch, metrics
