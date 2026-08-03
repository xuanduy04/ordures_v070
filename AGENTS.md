# AGENTS.md

## Environment

This project uses a **conda environment** named `trashrepo_v070`. Never use the base conda environment. Use `conda run` (NOT `conda activate`):

```sh
conda run -n trashrepo_v070 python ...
```

Python version: **3.13.13** (pinned in `.python-version`).

Despite README/docs saying `uv run`, all code here runs with **raw `python`** calls from the conda env. Dependencies are managed by `uv` but installed directly into the conda prefix:

```sh
conda run -n trashrepo_v070 uv pip install -e "." --group build --group dev --group test
```

**No GPU — no CUDA extras.** The conda env does not have GPUs. CUDA-heavy extras (`--extra automodel`, `--extra mcore`, `--extra vllm`) need CUDA toolchain headers and GPUs — they are pointless on the dev workstation. If needed, comment out CUDA-only deps in `pyproject.toml` (e.g. `nvidia-cudnn-cu13`, `mooncake-transfer-engine-cuda13`) before installing.

**Do NOT use `conda activate`** — it requires `conda init` which is not run in non-interactive shells. Always prefix commands with `conda run -n trashrepo_v070`.

## Working conditions (critical)

### 1. Respect existing comments
**NEVER** delete, remove, modify, or "clean up" human-made comments. Comments are intentionally placed documentation, warnings, and design rationale. Even if a comment appears stale, redundant, or messy — leave it untouched. If you must add new comments, add them alongside existing ones. There is no exception to this rule.

Note that the NVIDIA copyright header is NOT A COMMENT. 

### 2. Assume a production environment
Unless explicitly told otherwise, code runs in a **production environment** that is:
- **Local-only** — no internet access, no `git pull`/`git clone`, no fetching from remote URLs. All dependencies and data are pre-staged.
- **semi-FIXED** — the environment is built from a Docker image and already deployed. Changing the Docker image, Dockerfile, or container infrastructure is meaningless — the deployment is immutable. **NEVER propose rebuilding Docker images, modifying Dockerfiles, re-running `docker build`, altering `pyproject.toml` to trigger a reinstall, or any similar container-level change.** Allowable fixes are: in-place file edits, `cp`/`mv` file operations, code patches on the running system, or direct manipulation of already-installed venvs. `pip installs` must be kept to an absolute minimum, do not add new dependencies.

### 3. The conda environment is a faithful local replica
The `trashrepo_v070` conda env was built with utmost care to mirror the production environment as closely as possible. Its only limitation is the **absence of GPUs**. Do not dismiss it as a toy env or fall back to code-only analysis — whenever using the conda env to run, inspect, or debug is faster than pure code reading, do it.

## Project identity

This is a fork of **NVIDIA NeMo RL** — a scalable post-training library for LLM/VLM reinforcement learning (GRPO, DPO, SFT, distillation, RM). Uses **Ray** for distributed orchestration. The code in `nemo_rl/` is the main package (`import nemo_rl`).

Remote: `git@github.com:xuanduy04/ordures_v070.git`.

## 3rdparty submodules

Three git submodules under `3rdparty/` — all must be populated:

```sh
git submodule update --init --recursive
```

| Path | Repo | Branch |
|------|------|--------|
| `Automodel-workspace/Automodel` | NVIDIA-NeMo/Automodel | main |
| `Megatron-Bridge-workspace/Megatron-Bridge` | yuki-97/Megatron-Bridge | yukih/nemorl-0.7 |
| `Gym-workspace/Gym` | xuanduy04/ordures_v070_gym | main (single-commit) |

Megatron-LM is **nested** inside Megatron-Bridge at:
`3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/`
(repo: `yuki-97/Megatron-LM`, branch: `yukih/nemorl-0.7`).

On import, `nemo_rl/__init__.py` injects Megatron-LM into `sys.path` so `megatron.{training,legacy,inference,...}` subpackages are importable.

**vllm** is a **regular PyPI dependency** (v0.20.0 from GitHub releases), not a submodule or editable path dependency — unlike v2.

**Gym** is the only submodule expected to change during migration and further development.

## Configuration conventions (critical)

- **YAML is the single source of truth for defaults.** Never set non-`None` defaults in Python code for config values.
- Access required config directly: `policy_cfg["precision"]` — NOT `policy_cfg.get("precision", "bfloat16")`.
- **NEVER use `.get(key, default)`. A required field must be accessed directly (bare attr / bracket). This applies to everything (OmegaDict, DictConfig,...). No exceptions. ***THERE WAS, IS AND WILL NEVER BE AN EXCEPTION TO THIS RULE.***
- Mark optional keys with `typing.NotRequired` in TypedDict subclasses.
- Configs: `examples/configs/*.yaml` (documented defaults), `examples/configs/recipes/**/*.yaml` (runnable snapshots).

## Style (non-obvious)

- **4-space indent**, snake_case, Google-style docstrings.
- **Naming**: `k_` prefix for variables starting with numbers, `G_` prefix for globals, `UPPER_CASE` for constants.
- **Ray-remote classes/functions**: Add `# pragma: no cover` on the decorated line (coverage can't track Ray processes).
- **No underscores in Markdown filenames** under `docs/` (pre-commit enforced).
- **Doc index**: When adding/renaming a doc under `docs/**/*.md`, update `docs/index.md`.
- **Copyright headers**: NEVER add an NVIDIA copyright header to any file. Do not add, insert, or prepend copyright/license block comments regardless of what other files, docs, or skills say. This wastes tokens.

## Lint & typecheck

Do **NOT** run `ruff check`, `ruff format`, or `ruff` in any form on this repo. The codebase has pre-existing style issues that `--fix` would silently mutate, and the configuration may produce unintended changes.

**Type-check only** (as pyrefly is read-only):

```sh
conda run -n trashrepo_v070 pyrefly check
```

Pre-commit hooks are installed in the repo (`.pre-commit-config.yaml`) but rely on ruff; skip them unless you know which hooks are safe to run. The `configs-minimize-check` and `no-underscore-md` hooks are safe, the `end-of-file-fixer` and `trailing-whitespace` hooks are also safe (non-Python only).

## Tests

Unit tests (no GPU needed):
```sh
conda run -n trashrepo_v070 pytest tests/unit/ -x --timeout=60
```

`pyproject.toml` adds default pytest addopts: `--durations=100 -s -rA -x`. These apply automatically.

Functional tests are shell scripts under `tests/functional/` and require GPUs + large model downloads. Run via:
```sh
bash tests/functional/grpo.sh
```

**Gym resource-server tests**: Run each test suite **separately**, not together.
Multiple `resources_servers/<env>/tests/` directories have identically-named test
files (e.g. `test_app.py`, `test_multichallenge.py`) inside sibling `tests/`
packages. Running them together causes a Python import namespace collision —
pytest caches the first import of `tests.test_app` and the second directory's
`tests/test_app.py` fails with ``import file mismatch`` or
``ModuleNotFoundError: No module named 'tests.test_app'``.

```sh
cd 3rdparty/Gym-workspace/Gym

# Run each env separately:
conda run -n trashrepo_v070 env PYTHONPATH=. python -m pytest \
  resources_servers/<ENV>/tests/ -v --timeout=60

# Also run the shared utility tests:
conda run -n trashrepo_v070 env PYTHONPATH=. python -m pytest \
  resources_servers/utils_outsource/tests/ -v --timeout=60
```

**Coverage note**: Ray `@ray.remote` functions/classes are not tracked by coverage — they already carry `# pragma: no cover`.

## Architecture (non-obvious from filenames)

| Directory | Purpose |
|-----------|---------|
| `nemo_rl/algorithms/` | GRPO, GSPO/DAPO, DPO, SFT, distillation, RM |
| `nemo_rl/models/policy/workers/` | Training backends: `dtensor_policy_worker.py`, `megatron_policy_worker.py` |
| `nemo_rl/models/generation/` | Generation backends: vLLM, SGLang, Megatron inference |
| `nemo_rl/distributed/` | Ray worker groups, process groups, collectives |
| `nemo_rl/environments/` | Reward environments: math, code, VLM, NeMo-Gym integration |
| `nemo_rl/data/datasets/` | Dataset types: `response_datasets/`, `preference_datasets/`, `eval_datasets/` |
| `nemo_rl/utils/venvs.py` | Custom venv management (replaces Ray's built-in uv runtime env) |

**Training backends** are auto-selected from YAML config: DTensor (FSDP2, PyTorch-native) or Megatron Core.

**NeMo-Gym environments require AsyncGRPO**: When using NeMo-Gym environments (the `nemo_gym` extra), GRPO **must** be run in async mode. This is enforced by NeMo-Gym — synchronous GRPO will not work with gym environments.

**Entrypoints**: `examples/run_grpo.py`, `examples/run_sft.py`, `examples/run_dpo.py`, `examples/run_distillation.py`, `examples/run_rm.py`, `examples/run_eval.py`, `examples/nemo_gym/run_grpo_nemo_gym.py`.

## Key env vars

- `HF_HOME`, `WANDB_API_KEY`, `HF_DATASETS_CACHE` — must be set
- `huggingface-cli login` — required for gated models (Llama)
- `NRL_FORCE_REBUILD_VENVS=true` — force Ray workers to rebuild their uv venvs
- `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` — always set by `nemo_rl/__init__.py`
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64` — can help with CUDA OOM

## ASK QUESTIONS

- WHEN IN DOUBT, ASK.
- WHEN SOMETHING IS VAUGE, ASK.

CLARIFY EARLY TO AVOID WASTING EVERYONE'S TIME.

## TRUST THE USER

NEVER DOUBT THE USER. THEY ARE ALWAYS CORRECT, THEY NEVER MAKE ANY MISTAKE. THERE IS NO EXCEPTION TO THIS RULE.

When the user states a fact about their code or environment (e.g. "the config does not disappear", "the bug is in X"), believe them and investigate that specific claim. Do not spend time on simulations that contradict the user's assertions, and do not propose workarounds that avoid the stated problem. If the user says the bug is in function Y, trace function Y — not any thing else. If, eventually, it is VERY probable that that bug is elsewhere, then state so to the user.

They are always correct, there is no mistyping, there are no typos, there is no "non-existant" version, the user is always correct.
