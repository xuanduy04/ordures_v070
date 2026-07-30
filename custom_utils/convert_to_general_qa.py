"""Convert a JSON/JSONL dataset to general_qa format.

Each input entry must have a question field and an answer field.
The output is a JSONL file with entries containing ``agent_ref``,
``responses_create_params``, ``question``, ``expected_answer``, and
``should_use_judge``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INSTRUCTION_PREFIX = "Answer the following question. Put your final answer inside \\boxed{}.\n\n"


def _preprocess_underscore_args(argv: list[str]) -> list[str]:
    """Replace underscores with hyphens in ``--arg_name`` / ``--arg_name=val`` prefixes.

    argparse normalises ``-`` → ``_`` internally, so ``--skip-on-error`` and
    ``--skip_on_error`` would normally map to different destinations.  This
    function rewrites ``_`` to ``-`` in the leading ``--key`` portion so that
    argparse treats both spellings identically.
    """
    out: list[str] = []
    for arg in argv:
        if arg.startswith("--") and "=" in arg:
            key, _, val = arg.partition("=")
            arg = key.replace("_", "-") + "=" + val
        elif arg.startswith("--"):
            arg = arg.replace("_", "-")
        out.append(arg)
    return out


def _load_input(input_path: Path) -> list[dict]:
    """Load entries from *input_path* as either a JSON array or JSONL."""
    text = input_path.read_text(encoding="utf-8")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, list):
        logger.info("Detected JSON array with %d entries.", len(data))
        return data
    if isinstance(data, dict):
        logger.info("Detected single JSON object — wrapping in a list.")
        return [data]

    logger.info("Detected JSONL — parsing line-by-line.")
    entries: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            entries.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {lineno}: invalid JSON: {exc}") from exc
    return entries


def _convert_entry(entry: dict, question_field: str, answer_field: str, use_judge: bool, agent_name: str) -> dict:
    """Convert one raw entry to the general_qa JSONL schema."""
    question = str(entry.get(question_field, "")).strip()
    answer = str(entry.get(answer_field, "")).strip()

    if not question:
        raise ValueError(f"Entry has empty or missing '{question_field}' field")
    if not answer:
        raise ValueError(f"Entry has empty or missing '{answer_field}' field")

    content = INSTRUCTION_PREFIX + question

    out: dict = {
        "agent_ref": {"name": agent_name},
        "responses_create_params": {
            "input": [
                {"role": "user", "content": content},
            ]
        },
        "question": question,
        "expected_answer": answer,
        "should_use_judge": use_judge,
    }

    return out


def _validate_output_path(output_path: Path) -> None:
    """Ensure *output_path* ends with ``.jsonl``, parent dir exists, and file does not already exist."""
    if output_path.suffix.lower() != ".jsonl":
        raise ValueError(f"Output path must end with .jsonl, got: {output_path}")

    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    sys.argv = _preprocess_underscore_args(sys.argv)

    parser = argparse.ArgumentParser(
        description="Convert a JSON/JSONL dataset to general_qa format.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Absolute path to the input data file (.json or .jsonl).",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Absolute path to the output JSONL file (must end with .jsonl).",
    )
    parser.add_argument(
        "--question-field",
        default="question",
        help="Field name for the question (default: %(default)s).",
    )
    parser.add_argument(
        "--answer-field",
        default="answer",
        help="Field name for the expected answer (default: %(default)s).",
    )
    parser.add_argument(
        "--should-use-judge",
        action="store_true",
        default=False,
        help="'should_use_judge: true' in each output entry (default: false).",
    )
    parser.add_argument(
        "--agent-name",
        default="general_qa_simple_agent",
        help="agent_ref.name for each row (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-on-error",
        action="store_true",
        default=False,
        help="Skip invalid entries with a warning instead of aborting (default: false).",
    )

    args = parser.parse_args()

    input_path: Path = args.input
    output_path: Path = args.output
    question_field: str = args.question_field
    answer_field: str = args.answer_field
    should_use_judge: bool = args.should_use_judge
    agent_name: str = args.agent_name
    skip_on_error: bool = args.skip_on_error

    if not input_path.is_absolute():
        parser.error(f"Input path must be absolute, got: {input_path}")
    if not output_path.is_absolute():
        parser.error(f"Output path must be absolute, got: {output_path}")
    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")

    _validate_output_path(output_path)

    logger.info("Loading input: %s", input_path)
    raw_entries = _load_input(input_path)
    logger.info("Loaded %d raw entries.", len(raw_entries))

    converted: list[dict] = []
    skipped = 0
    for i, entry in enumerate(tqdm(raw_entries, desc="Converting", unit="entry")):
        try:
            converted.append(_convert_entry(entry, question_field, answer_field, should_use_judge, agent_name))
        except Exception as exc:
            if skip_on_error:
                logger.warning("Skipping entry %d: %s", i, exc)
                skipped += 1
                continue
            raise

    if not converted:
        raise ValueError("No valid entries found — nothing to write.")

    if skipped:
        logger.info("Skipped %d invalid entries.", skipped)

    logger.info("Writing %d entries to: %s", len(converted), output_path)
    with open(output_path, "w", encoding="utf-8") as fh:
        for entry in converted:
            json.dump(entry, fh, ensure_ascii=False)
            fh.write("\n")

    logger.info("Done — wrote %d entries.", len(converted))


if __name__ == "__main__":
    main()
