from __future__ import annotations

import re
import requests
from typing import Any
from urllib.parse import urlparse
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import ray
from ray.experimental import tqdm_ray

from nemo_rl.data.interfaces import LLMMessageLogType


HTML_TAG_PATTERN = re.compile(r'</?(?:html|head|body|center)>', re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r'\s+')


def clean_html_response_text(response_text: str) -> str:
    """Cleans the `response.text` field for logging output"""
    # Replace all newlines and multiple spaces with a single space
    response_text = WHITESPACE_PATTERN.sub(' ', response_text.strip()).strip()

    # Strip and remove the specific HTML tags in one pass
    return HTML_TAG_PATTERN.sub(' ', response_text).strip()


def extract_judge_text_from_response_json(response_json: dict[str, Any]) -> str:
    """Extracts the judge's output from `response.json`"""
    # OpenAI-compatible shape:
    #   response_json["choices"][0]["message"]["content"]
    choices = response_json.get("choices", [])
    if not choices:
        return ""
    first_choice = choices[0]
    message = first_choice.get("message", {})
    content = message.get("content", "")

    # Some providers may return content as structured chunks instead of a string.
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


def extract_role_content(conversation: LLMMessageLogType, role: str) -> str:
    """Extracts and concatenates all message contents for a specific role.

    Iterates through a conversation log, filters for messages matching the 
    specified role, and joins their contents together. This ensures robust 
    handling of multi-turn conversations where a single role appears 
    multiple times.

    Args:
        conversation: A list of dictionaries representing the LLM message 
            log, where each dictionary (turn) contains "role" and "content" keys.
        role: The specific role to filter by (e.g., "user", "assistant", "system").

    Returns:
        A single string containing all concatenated message contents for the 
        given role, separated by newlines and stripped of leading/trailing whitespace.
    """
    # Message logs are lists of role/content dictionaries.
    # We concatenate all turns of the requested role to keep behavior robust.
    extracted_parts: list[str] = [
        str(turn.get("content", ""))
        for turn in conversation
        if turn.get("role") == role
    ]
    return "\n".join(extracted_parts).strip()


def fetch_json_once(server_url: str, timeout: int = 10, max_attempts: int = 6) -> dict:
    """Fetches JSON data from a server with a retry mechanism.

    Attempts to send a GET request to the specified URL. If successful, returns
    the parsed JSON response immediately. If an HTTP error or exception occurs,
    it logs the status and response text (if available) and retries until
    `max_attempts` is reached. If all attempts fail, returns a summary dictionary
    of all attempted status codes and response texts.

    Args:
        server_url: The URL of the server to fetch data from.
        timeout: The maximum time in seconds to wait for the server to respond
          per attempt. Defaults to 10.
        max_attempts: The maximum number of times to attempt the request before
          giving up. Defaults to 6.

    Returns:
        A dictionary containing the successfully parsed JSON data if any attempt
        succeeds. If all attempts fail, returns a dictionary containing a history
        of the failures structured as:
        {"status_code": [list of ints/Nones], "text": [list of cleaned strs/Nones]}
    """
    responses = {"status_code": [], "text": []}
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        response = None
        try:
            response = requests.get(
                server_url,
                headers={"Accept": "application/json", "Connection": "close"},
                timeout=timeout,
                stream=False,
                verify=False
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Querying '{server_url}', attempt {attempt}/{max_attempts}: {type(e).__name__}: {e}")
        finally:
            responses["status_code"].append(response.status_code if response is not None else None)
            responses["text"].append(response.text if response is not None else None)

            if response is not None:
                response.close()

    return responses


def normalize_url(vllm_server_url: str) -> str:
    """Normalize server URL

    Accepts:
    - localhost:8000
    - http://localhost:8000
    - http://localhost:8000/extra/stuff/behind
    """
    vllm_server_url = vllm_server_url.strip().strip("/")
    if not vllm_server_url:
        raise ValueError("RLHF config requires non-empty `vllm_server_url`.")

    if "://" not in vllm_server_url:
        vllm_server_url = f"http://{vllm_server_url}"

    parsed_url = urlparse(vllm_server_url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError("Expected `vllm_server_url` like 'localhost:8000' or 'http://localhost:8000'.")
    return f"{parsed_url.scheme}://{parsed_url.netloc}"


NUMBER_PATTERN = re.compile(r'-?\d+(?:\.\d+)?')
