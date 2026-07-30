#!/usr/bin/env bash
# use_custom_opt.sh — Override NeMo RL packages with custom implementations via PYTHONPATH.
# Source this file in your shell before running NeMo RL to use custom opt code.
#
# Usage:
#   source custom_utils/use_custom_opt.sh [CUSTOM_NEMO_RL_DIR]
#
#   CUSTOM_NEMO_RL_DIR: path to custom nemo-rl tree (default: ${SCRIPT_DIR}/nemo-rl)
#
# Behaviour:
#   Walks the custom tree (max depth 6) for directories containing pyproject.toml,
#   setup.py, or setup.cfg — each is a Python project root.  Prepends the unique
#   set to PYTHONPATH so they take priority over installed packages.
#
#   Nested 3rdparty directories (e.g. Megatron-Bridge/3rdparty/Megatron-LM) are
#   skipped because they contain stale submodule checkouts that shadow the primary
#   3rdparty sources via the megatron namespace package.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUSTOM_NEMO_RL_DIR="${1:-${SCRIPT_DIR}/nemo-rl}"

if [[ ! -d "$CUSTOM_NEMO_RL_DIR" ]]; then
    echo "ERROR: Custom NeMo RL directory not found: $CUSTOM_NEMO_RL_DIR" >&2
    return 1 2>/dev/null || exit 1
fi

echo "Adding custom NeMo RL overrides from '$CUSTOM_NEMO_RL_DIR'"

python_paths=()

# Walk the tree for Python project roots, skipping nested 3rdparty dirs.
while IFS= read -r -d '' proj_file; do
    proj_dir="$(dirname "$proj_file")"

    # We don't skip nested "3rdparty" path-segments as they have been cleaned
    # threeparty_count=$(echo "$proj_dir" | tr '/' '\n' | grep -c '^3rdparty$'  || true)
    # if [[ $threeparty_count -gt 1 ]]; then
    #    echo "  [SKIPPED] $proj_dir (nested /3rdparty/)"
    #    continue
    # fi
    # python_paths+=("$proj_dir")
    echo "  [add] $proj_dir"
done < <(find "$CUSTOM_NEMO_RL_DIR" -maxdepth 6 \
    \( -name pyproject.toml -o -name setup.py -o -name setup.cfg \) \
    -print0 2>/dev/null)

if [[ ${#python_paths[@]} -eq 0 ]]; then
    echo "WARNING: No Python packages or projects found under $CUSTOM_NEMO_RL_DIR" >&2
    return 0 2>/dev/null || exit 0
fi

# Deduplicate (a dir may have both pyproject.toml and setup.py) and prepend.
unique_paths=($(printf '%s\n' "${python_paths[@]}" | sort -u))
export PYTHONPATH="$(IFS=:; echo "${unique_paths[*]}")${PYTHONPATH:+:$PYTHONPATH}"

echo "PYTHONPATH updated (${#unique_paths[@]} directories added)"

