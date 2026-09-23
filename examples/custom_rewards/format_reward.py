from dataclasses import dataclass


@dataclass
class FormatResult:
    is_correct_format: bool
    issues: list[str]
    extracted_answer: str | None


def parse_final_boxed(answer: str) -> str:
    r"""Return the contents inside the final \boxed{...} in `answer`.

    If no complete \boxed{...} is found, return empty string.
    Handles nested braces inside the boxed content.
    """
    if len(answer) == 0:
        return ""

    marker = r"\boxed{"
    start = answer.rfind(marker)
    if start == -1:
        return ""

    i = start + len(marker)
    depth = 1
    content_start = i

    while i < len(answer):
        ch = answer[i]

        # Skip escaped characters like \{ or \}
        if ch == "\\" and i + 1 < len(answer):
            i += 2
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return answer[content_start:i]

        i += 1

    return ""


def verify_format(response: str, answer_wrapper:str = r"\boxed{") -> FormatResult:
    r"""Verifies whether the <think> formatting is correct, additionally exacts answer inside \boxed{}."""
    text = response or ""
    issues: list[str] = []

    open_think = text.count("<think>")
    close_think = text.count("</think>")

    if open_think != 1 or close_think != 1:
        issues.append(
            f"There should only be 1 think pair ({open_think=}, {close_think=})"
        )
    else:
        if text.find("<think>") > text.find("</think>"):
            issues.append("<think> appears after </think>")
        if answer_wrapper != "</think>" and "</think>" in text and (text.rfind("</think>") + 7) > text.rfind(answer_wrapper):
            issues.append(f"{answer_wrapper} must appear after </think>")

    return FormatResult(
        is_correct_format=len(issues) == 0,
        issues=issues,
        extracted_answer=parse_final_boxed(text),
    )


def parse_answer(answer: str, answer_signal: str = r"\boxed", answer_opener: str = "{", answer_closer: str = "}") -> str:
    r"""Return the contents inside the last occurrence of the marker in `answer`.

    If no complete block is found, return empty string ("").
    Handles nested openers/closers inside the block.
    
    Args:
        answer: The string to search.
        answer_signal: The marker prefix (default r"\boxed").
                       If this is an empty string (""), returns `answer` unchanged.
        answer_opener: Single character that opens the answer block (default "{").
        answer_closer: Single character that closes the answer block (default "}").
                       If this is an empty string (""), returns everything after the opener.
    
    Returns:
        The contents between the opener and its matching closer, or empty
        string ("") if no complete block is found.
    """
    if len(answer) == 0:
        return ""
    if len(answer_signal) == 0:
        return answer

    marker = answer_signal + answer_opener
    start = answer.rfind(marker)
    if start == -1:
        return ""
    
    i = start + len(marker)
    depth = 1
    content_start = i
    if answer_closer == "":
        return answer[content_start:]

    while i < len(answer):
        ch = answer[i]

        # Skip escaped characters "\" (like in "\{" or "\}")
        if ch == "\\" and i + 1 < len(answer):
            i += 2
            continue

        if ch == answer_opener:
            depth += 1
        elif ch == answer_closer:
            depth -= 1
            if depth == 0:
                return answer[content_start:i]

        i += 1

    return ""
    

def verify_think_format(
    response: str,
    answer_signal: str = r"\boxed",
    answer_opener: str = "{",
    answer_closer: str = "}",
    extract_answer: bool = False,
) -> FormatResult:
    r"""Verifies whether the <think> formatting is correct, additionally exacts answer."""
    text = response or ""
    issues: list[str] = []

    open_think = text.count("<think>")
    close_think = text.count("</think>")

    if open_think != 1 or close_think != 1:
        issues.append(
            f"There should only be 1 think pair ({open_think=}, {close_think=})"
        )
    else:
        if text.find("<think>") > text.find("</think>"):
            issues.append("<think> appears after </think>")
        if (
            answer_signal != "</think>"
            and "</think>" in text
            and (text.rfind("</think>") + 8) > text.rfind(answer_signal + answer_opener)
        ):
            issues.append(f"There must be a(n) '{answer_signal + answer_opener}' that appears after </think>")

    return FormatResult(
        is_correct_format=bool(len(issues) == 0),
        issues=issues,
        extracted_answer=parse_answer(text, answer_signal, answer_opener, answer_closer) if extract_answer and len(issues) else "",
    )