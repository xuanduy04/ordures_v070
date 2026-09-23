import re
from dataclasses import dataclass

from nemo_rl.evals.answer_parsing import MULTILINGUAL_ANSWER_REGEXES


@dataclass
class FormatResult:
    is_correct_format: bool
    format_issues: list[str]
    extracted_answer: str | None


def last_boxed_only_string(string: str) -> str:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression (without the box) or empty string ("") if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return ""

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

    return string[idx + 7: right_brace_idx].strip() if right_brace_idx is not None else ""


_ANSWER_COLON_PATTERN = re.compile(
    rf"(?i)(?:{'|'.join(f'(?:{r})' for r in sorted(MULTILINGUAL_ANSWER_REGEXES, key=len, reverse=True))})[ \t]*",
)


def last_answer_colon_string(string: str) -> str:
    """Extract the content after the last multilingual "Answer:"-style marker.

    Searches for the last occurrence of any multilingual "Answer:" pattern
    (as defined in nemo_rl.evals.answer_parsing.MULTILINGUAL_ANSWER_REGEXES)
    and returns everything after it until the end of the string.

    Args:
        string: Input string to search.

    Returns:
        The content after the last "Answer:"-style marker, or "" if none found.
    """
    if not string:
        return ""
    matches = list(_ANSWER_COLON_PATTERN.finditer(string))
    if not matches:
        return ""
    return string[matches[-1].end():].strip()


def verify_think_format(
    response: str,
    extract_boxed: bool = False,
    extract_answer_colon: bool = False,
) -> FormatResult:
    r"""Verifies whether the <think> formatting is correct, additionally extracts answer.

    - When only ``extract_boxed`` is True, extracts the last ``\boxed{...}`` expression.
    - When only ``extract_answer_colon`` is True, extracts the content after the last
    multilingual "Answer:"-style marker (until end of string).
    - When both are True, boxed extraction is tried first; if no boxed answer
    is found, answer-colon extraction is used as a fallback.
    - When neither is True, ``extracted_answer`` is ``""``.

    The format check (``format_issues``) verifies:
    - Exactly one ``<think>`` / ``</think>`` pair.
    - ``<think>`` appears before ``</think>``.
    - When ``extract_boxed`` and/or ``extract_answer_colon`` is True, the
      corresponding answer signal must appear after ``</think>`` (at least
      one when both are True). When neither is True, no answer-signal check
      is performed.
    """
    text = response or ""
    format_issues: list[str] = []

    open_think = text.count("<think>")
    close_think = text.count("</think>")

    if open_think != 1 or close_think != 1:
        format_issues.append(
            f"There should only be 1 think pair ({open_think=}, {close_think=})"
        )
    else:
        if text.find("<think>") > text.find("</think>"):
            format_issues.append("<think> appears after </think>")

        if (extract_boxed or extract_answer_colon) and "</think>" in text:
            think_end_pos = text.rfind("</think>") + len("</think>")
            after_think = text[think_end_pos:]

            boxed_found = "\\boxed{" in after_think if extract_boxed else False
            answer_colon_found = bool(_ANSWER_COLON_PATTERN.search(after_think)) if extract_answer_colon else False

            if extract_boxed and extract_answer_colon:
                if not (boxed_found or answer_colon_found):
                    format_issues.append("There must be a '\\boxed{{' or 'Answer:' (multilingual) that appears after </think>")
            elif extract_boxed:
                if not boxed_found:
                    format_issues.append("There must be a '\\boxed{{' that appears after </think>")
            elif extract_answer_colon:
                if not answer_colon_found:
                    format_issues.append("There must be an 'Answer:' (multilingual) that appears after </think>")

    extracted_answer = ""
    if extract_boxed:
        extracted_answer = last_boxed_only_string(text)
    if extract_answer_colon and not extracted_answer:
        extracted_answer = last_answer_colon_string(text)

    return FormatResult(
        is_correct_format=bool(len(format_issues) == 0),
        format_issues=format_issues,
        extracted_answer=extracted_answer,
    )