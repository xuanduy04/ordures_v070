from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from typing import Any, Generic, NotRequired, TYPE_CHECKING, TypedDict, TypeVar
from urllib.parse import urlparse

import random
import ray
import requests
import torch
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.utils import chunk_list_to_workers
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn, MetadataT

from examples.custom_rewards.utils.llm_judge_utils import (
    clean_html_response_text,
    extract_judge_text_from_response_json,
    extract_role_content,
    fetch_json_once,
    normalize_url,
    NUMBER_PATTERN
)

from examples.custom_rewards.utils.tqdm_utils import REMOTE_TQDM


if TYPE_CHECKING:
    from ray.experimental import tqdm_ray


class BaseLLMJudgeEnvConfig(TypedDict):
    # Base vLLM server URL (just host:port style is valid too).
    vllm_server_url: str
    # Judge's model name, passed as `model` in the JSON payload.
    model: str
    # Judge's default rubric/ Likert scale prompt. (default: the class-specific `default_prompt`)
    default_prompt: NotRequired[str]
    # Number of Ray workers to judge in parallel. (default: 1)
    num_workers: NotRequired[int]
    # Inference temperature of the judge. (default: 0.2)
    temperature: NotRequired[float]
    # Number of output tokens for the judge. (default: 0 (no limit))
    max_judge_output_tokens: NotRequired[int]
    # verbosity just controls the progress bar. (default: True)
    # The class variable `_DEBUG` is the one that prints debug outputs.
    verbose: NotRequired[bool]


class BaseLLMJudgeVerifyWorker(ABC):
    _RETRYABLE_STATUS_CODES = {
        429, # Too many requests
        500, # Internal server error (should check backend)
        502, # Bad gateway
        503, # Service Unavailable (usually due to backend overload)
        504, # Gateway Timeout
    }
    _MAX_RETRIES = 67
    _WAIT_SECONDS_BEFORE_RETRY = 5
    _REQUEST_TIMEOUT_SECONDS = 600

    def __init__(
        self,
        vllm_chat_completions_url: str,
        model: str,
        default_prompt: str,
        temperature: float,
        max_judge_output_tokens: int,
        _ENV_NAME: str,
    ) -> None:
        # No verification as we assume the associated environment has done all the dirty work
        self.vllm_chat_completions_url = vllm_chat_completions_url
        self.model = model
        self.default_prompt = default_prompt
        self.temperature = temperature
        self.max_judge_output_tokens = max_judge_output_tokens
        # This is non-standard, a class-protected variable initialized every instance
        # but since every VerifyWorker should only ever be called from its corresponding Environment, this ensures minimal changes when inheriting
        self._ENV_NAME = _ENV_NAME
        
        self._session = requests.Session()
        self._session.verify = False
        self._SESSION_HEADERS = {"Content-Type": "application/json", "Expect": ""}

    def _parse_score(self, judge_text: str, **kwargs) -> float:
        r"""
        Parses and extracts a numerical score from a judge's text response.

        NOTE: Subclasses are strongly encouraged to guard this method's output.
        The fallback regex is greedy and prone to extracting unrelated values
        (e.g. dates, math equations,...) if the `\boxed{...}` is missing.
    
        The function attempts to extract the score using two strategies:
        1. Primary: Looks for the rightmost occurrence of a LaTeX `\boxed{...}` 
           marker and extracts the contents inside the braces.
        2. Fallback: If the boxed marker is missing or malformed, it falls back 
           to extracting the very last numerical value found in the text using 
           a regex pattern (`r'-?\d+(?:\.\d+)?'`).
    
        Args:
            judge_text (str): A string containing the raw text response from the judge.
    
        Returns:
            The extracted score (float). Returns `0.0` if no valid number 
            can be parsed from the text using either method.
        """
         # Try to extract from \boxed{...}
        marker = r"\boxed{"
        idx = judge_text.rfind(marker)  # rightmost occurrence
        after = idx + len(marker)  # index of char immediately after "{"
        if idx != -1 and after < len(judge_text):
            try:
                end = judge_text.index("}", after)  # first "}" after the marker
                boxed_content = judge_text[after:end].strip()
                try:
                    return float(boxed_content)
                except:
                    # Fallback 1: Last scalar number in \boxed{}
                    matches = NUMBER_PATTERN.findall(boxed_content)
                    if matches:
                        return float(matches[-1])
            except:
                pass  # Did not find closing "}" or number in \boxed{}
        
        # Fallback 2: Last scalar number in output.
        matches = NUMBER_PATTERN.findall(judge_text)
        return float(matches[-1]) if matches else 0.0

    def _build_messages(self, judge_prompt: str, **kwargs) -> list[dict[str, str]]:
        """Builds the judge request's `messages` list."""
        return [
            {"role": "user",
             "content": f"{self.default_prompt}\n\n\n{judge_prompt}"},
        ]

    def _query_llm_judge(self, judge_prompt: str, skip_query_llm_judge: bool = False, **kwargs) -> tuple[float, str, str | None]:
        """Sends a singular query to the LLM Judge, with status logging if query fails.

        Args:
            judge_prompt (str): The prompt string to be evaluated by the LLM judge.
            skip_query_llm_judge (bool): whether to skip the judge query (returns 0.0 score) 
            **kwargs: Extra keyword arguments passed as additional input 
                to the `self._build_messages()` and `self._parse_score()` calls.

        Returns:
            A tuple containing:
                - score (float): The reward or score assigned by the judge.
                - judge_text (str): The complete raw text response from the judge.
                - error_message (str | None): The complete error message as a 
                  string if an error occurred, otherwise None.
        """
        if skip_query_llm_judge:
            return 0.0, "", "Answer was specifically instructed to be skipped. LLM Judge was not queried."
        
        request_payload: dict[str, Any] = {
            "messages": self._build_messages(judge_prompt=judge_prompt, **kwargs),
            "model": self.model,
            "temperature": self.temperature,
            "stream": False,
        }
        if self.max_judge_output_tokens > 0:
            request_payload["max_tokens"] = self.max_judge_output_tokens

        attempt: int = 0
        while attempt < self._MAX_RETRIES:
            attempt += 1
            try:
                response = self._session.post(
                    self.vllm_chat_completions_url,
                    json=request_payload,
                    timeout=self._REQUEST_TIMEOUT_SECONDS,
                    headers=self._SESSION_HEADERS
                )
                if response.status_code in self._RETRYABLE_STATUS_CODES:
                    print(f"[WARNING] Environment '{self._ENV_NAME}' at {attempt=}: judge request returned status code ({response.status_code}) "
                          f"with cleaned body ({clean_html_response_text(response.text)}); "
                          f"retrying in {self._WAIT_SECONDS_BEFORE_RETRY} second(s)...")
                    time.sleep(self._WAIT_SECONDS_BEFORE_RETRY + random.uniform(0, 1))
                    continue
                
                response.raise_for_status()
                judge_text = extract_judge_text_from_response_json(response.json())
                score = self._parse_score(judge_text=judge_text, **kwargs)
                return score, judge_text, None
            except Exception as exc:
                print(f"[WARNING] Environment '{self._ENV_NAME}' at {attempt=}: judge request failed with error ({exc}); "
                      f"retrying in {self._WAIT_SECONDS_BEFORE_RETRY} second(s)...")
                time.sleep(self._WAIT_SECONDS_BEFORE_RETRY + random.uniform(0, 1))
        print(f"[WARNING] Environment '{self._ENV_NAME}' at {attempt=}: Maximum retries reached; skipping this sample via returning 0 reward.")
        return 0.0, f"Maximum retires ({self._MAX_RETRIES}) reached", f"Maximum retires ({self._MAX_RETRIES}) reached"

    def verify(
        self, judge_prompt_batch: list[dict[str, Any]], pbar: 'tqdm_ray.tqdm' | None = None
    ) -> list[tuple[float, str, str | None]]:
        if pbar is None:
            return [self._query_llm_judge(**kwargs) for kwargs in judge_prompt_batch]

        def _query_llm_judge_with_pbar_update(**kwargs) -> tuple[float, str, str | None]:
            try:
                return self._query_llm_judge(**kwargs)
            finally:
                pbar.update.remote(1)
        return [_query_llm_judge_with_pbar_update(**kwargs) for kwargs in judge_prompt_batch]

    def shutdown(self) -> None:
        self._session.close()


BaseLLMJudgeEnvConfigT = TypeVar("BaseLLMJudgeEnvConfigT", bound=BaseLLMJudgeEnvConfig)
class BaseLLMJudgeEnvironment(
    EnvironmentInterface[MetadataT],
    Generic[MetadataT, BaseLLMJudgeEnvConfigT],
):
    _ENV_NAME: str = "BaseLLMJudge"
    _VERIFY_WORKER_CLS: type[BaseLLMJudgeVerifyWorker] | None = None
    _DEBUG: bool = False
    DEFAULT_JUDGE_SYSTEM_PROMPT: str | None = None

    def __init__(self, cfg: BaseLLMJudgeEnvConfigT | dict[str, Any]) -> None:
        if self._ENV_NAME == "BaseLLMJudge":
            print("[WARNING] Subclasses should override the default `_ENV_NAME` for better logging")
        if self._VERIFY_WORKER_CLS is None:
            raise NotImplementedError(f"`_VERIFY_WORKER_CLS` must be set by subclasses, and must be a class inherited from `BaseLLMJudgeVerifyWorker` with an `@ray.remote` decorator ({self._ENV_NAME=})")
        if self.DEFAULT_JUDGE_SYSTEM_PROMPT is None:
            raise NotImplementedError(f"`DEFAULT_JUDGE_SYSTEM_PROMPT` must be set by subclasses ({self._ENV_NAME=})")

        self.cfg = cfg

        # vllm_server_url
        self.vllm_server_url = str(cfg.get("vllm_server_url", "")).strip()
        if not self.vllm_server_url:
            raise ValueError(F"{self._ENV_NAME} config requires `vllm_server_url`.")
        self.vllm_server_url = normalize_url(self.vllm_server_url)
        self.vllm_chat_completions_url =f"{self.vllm_server_url}/v1/chat/completions"

        # model
        self.model = str(cfg.get("model", "")).strip()
        if not self.model:
            raise ValueError(f"{self._ENV_NAME} config requires non-empty `model`.")
        # Checking if this model exists on the vllm server
        v1_models_response = fetch_json_once(f"{self.vllm_server_url}/v1/models")
        if "status_code" in v1_models_response:
            raise ValueError(
                f"Could not query '{self.vllm_server_url}/v1/models'. "
                f"Query attempt(s) returned: {v1_models_response}"
            )
        available_models = [model.get('id', '') for model in v1_models_response.get('data', [])]
        if self.model not in available_models:
            raise ValueError(
                f"Judge model '{self.model}' NOT found on server '{self.vllm_server_url}'. "
                f"Available models: {available_models}"
            )

        # default_prompt
        self.default_prompt = str(cfg.get("default_prompt", "")).strip()
        if not self.default_prompt:
            print(f"{self._ENV_NAME}Environment's `default_prompt` is empty, using {self._ENV_NAME}'s default `default_prompt`.")
            self.default_prompt = self.DEFAULT_JUDGE_SYSTEM_PROMPT

        # num_workers
        num_workers = cfg.get("num_workers", 1)
        try:
            self.num_workers = int(num_workers)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{self._ENV_NAME} config requires integer `num_workers`, got {num_workers!r}.") from exc
        if self.num_workers <= 0:
            raise ValueError(f"{self._ENV_NAME} config requires positive `num_workers`, got {self.num_workers}.")

        # temperature
        temperature = cfg.get("temperature", 0.2)
        try:
            self.temperature = float(temperature)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{self._ENV_NAME} config requires float `temperature`, got {temperature!r}.") from exc
        if self.temperature < 0.0:
            raise ValueError(f"{self._ENV_NAME} config requires non-negative `temperature`, got {self.temperature}.")
        if self.temperature > 2.0:
            print(f"[WARNING] {self._ENV_NAME} config get `temperature`={self.temperature}, are you sure this is correct?")

        # max_judge_output_tokens
        max_judge_output_tokens = cfg.get("max_judge_output_tokens", 0)
        try:
            self.max_judge_output_tokens = int(max_judge_output_tokens)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{self._ENV_NAME} config requires integer `max_judge_output_tokens`, got {max_judge_output_tokens!r}.") from exc
        if 0 < self.max_judge_output_tokens < 1024:
            print(f"[WARNING] {self._ENV_NAME} config got `max_judge_output_tokens`={self.max_judge_output_tokens}, are you sure this is correct?")

        # verbose
        self.verbose = cfg.get("verbose", True)
        
        # workers
        self.workers = [
            self._VERIFY_WORKER_CLS.remote(
                vllm_chat_completions_url=self.vllm_chat_completions_url,
                model=self.model,
                default_prompt=self.default_prompt,
                temperature=self.temperature,
                max_judge_output_tokens=self.max_judge_output_tokens,
                _ENV_NAME=self._ENV_NAME
            )
            for _ in range(self.num_workers)
        ]
        print(f"{self._ENV_NAME}Environment found 0 issues during initialization.")
    
    def log(self, msg: str) -> None:
        """Logs the input `msg` if `self._DEBUG` is set to True"""
        if self._DEBUG:
            assert msg
            print(f"\t[{self._ENV_NAME} Reward Class] {msg}")

    @abstractmethod
    def _get_judge_prompt_batch(
        self,
        conversation_history: str,
        assistant_answer: str,
        env_info: MetadataT
    ) -> tuple[dict[str, Any], dict | None]:
        """Generates the prompt payload and metadata required to query the LLM judge.

        Args:
            conversation_history: The raw or formatted text of the prior conversation turns.
            assistant_answer: The final response generated by the assistant to be evaluated.
            env_info: Metadata of the current sample.

        Returns:
            A tuple containing:
                - dict: The LLM call configurations. Must include 'judge_prompt' (the 
                  judge's user prompt). May include 'skip_query_llm_judge' (bool) 
                  determining whether the judge call should be skipped.
                - Optional extra observation metadata (dict), or `None` if not applicable.
                  (you may want to also pass 'skip_query_llm_judge' to this 
                  for better observation logging)
        """

    def _build_observation(self, judge_result: tuple[float, str, str | None], extra_obs_info: dict | None = None) -> dict[str, str]:
        """Constructs a formatted environment turn (i.e. observation) for a sample 

        Args:
            judge_result: A tuple containing three elements:
                - score (float): The numerical score assigned by the judge.
                - judge_text (str): The explanatory text or rationale from the judge.
                - judge_error (str | None): An error message if the judge failed, 
                  otherwise None.
            extra_obs_info: Optional metadata or configuration dictionary. 
                Can contain a "skip_query_llm_judge" boolean flag denoting if the 
                llm judge query has been skipped. Defaults to None.

        Returns:
            A dictionary representing the environment turn with the following keys:
                - "role": "environment"
                - "content": A formatted string describing the outcome (e.g., the score 
                  and its justification text, an error message, or a skipped status).
        """
        score, judge_text, judge_error = judge_result
        skip_query_llm_judge = isinstance(extra_obs_info, dict) and extra_obs_info.get("skip_query_llm_judge", False)

        if skip_query_llm_judge:
            content = f"Environment '{self._ENV_NAME}': Skipped Judge Model"
        elif judge_error:
            content = f"Environment '{self._ENV_NAME}': judge model errored ({judge_error})"
        else:
            content = (
                f"Environment '{self._ENV_NAME}': {self._ENV_NAME}_score={score:.3f}; "
                f"judge_text='{judge_text.replace("\n", " ").strip()}'"
            )
        return {"role": "environment", "content": content}

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[MetadataT],
        return_extracted_answer: bool = False
    ) -> EnvironmentReturn[MetadataT]:
        # self.log(f"\tBegin of a {self._ENV_NAME}Environment's `step()` call.")
        judge_prompt_batch = []
        extra_obs_info_batch = []
        for conversation, env_info in zip(message_log_batch, metadata):
            judge_prompt, extra_obs_info = self._get_judge_prompt_batch(
                extract_role_content(conversation, role="user"),
                extract_role_content(conversation, role="assistant"),
                env_info
            )
            judge_prompt_batch.append(judge_prompt)
            extra_obs_info_batch.append(extra_obs_info)

        chunked_judge_prompt_batch = chunk_list_to_workers(judge_prompt_batch, self.num_workers)
        pbar = REMOTE_TQDM.remote(total=len(judge_prompt_batch)) if self.verbose else None
        futures = [
            self.workers[i].verify.remote(chunk, pbar=pbar)
            for i, chunk in enumerate(chunked_judge_prompt_batch)
        ]
        worker_results = ray.get(futures)
        
        judge_results: list[tuple[float, str, str | None]] = []
        for worker_result in worker_results:
            judge_results.extend(worker_result)
        
        # self.log(f"\t{self._ENV_NAME}'s judge results returned! Parsing results...")
        rewards = [score for score, _, _ in judge_results]

        observations = [
            self._build_observation(judge_result, extra_obs_info)
            for judge_result, extra_obs_info in zip(judge_results, extra_obs_info_batch)
        ]

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).cpu()
        terminated_tensor = torch.ones_like(rewards_tensor).cpu()

        # self.log(f"\t  End of a {self._ENV_NAME}Environment's `step()` call.")
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
        # Mean judge score is the main metric to track for this reward.
        metrics = {
            f"{self._ENV_NAME}_mean_reward": float(batch["rewards"].float().mean().item())
        }
        return batch, metrics

    def shutdown(self) -> None:
        # Shutdown all judge workers.
        ray.get([worker.shutdown.remote() for worker in self.workers])
        for worker in self.workers:
            ray.kill(worker)
