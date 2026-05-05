#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export KMP_USE_SHM="${KMP_USE_SHM:-0}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <lora_path> [eval args...]" >&2
    echo "Example: $0 outputs/models/Qwen3-4B/GRPO_8 --limit 30" >&2
    exit 1
fi

if [[ "${1}" == "--help" || "${1}" == "-h" ]]; then
    echo "Usage: $0 <lora_path> [eval args...]"
    echo "Example: $0 outputs/models/Qwen3-4B/GRPO_8 --datasets AIME,AMC23 --limit 30"
    echo
    echo "Forwarded eval args:"
    echo "  --datasets AIME,AMC23,MATH500"
    echo "  --aime-data-dir PATH"
    echo "  --amc23-data-dir PATH"
    echo "  --math500-data-dir PATH"
    echo "  --output-json PATH"
    echo "  --limit N"
    echo "  --num-samples N"
    echo "  --estimate-k N"
    echo "  --batch-size N"
    echo "  --temperature FLOAT"
    echo "  --top-p FLOAT"
    echo "  --max-new-tokens N"
    echo "  --max-seq-len N"
    echo "  --seed N"
    exit 0
fi

LORA_PATH="$1"
shift

python - <<'PY'
import importlib.util
import sys

required_modules = ("torch", "vllm", "peft", "tqdm", "requests", "datasets", "math_verify")
missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
if missing:
    print(f"Missing Python packages: {', '.join(missing)}", file=sys.stderr)
    print("Install dependencies first (e.g. pip install -r requirements.txt)", file=sys.stderr)
    raise SystemExit(1)
PY

python -m src.eval_model --lora-path "${LORA_PATH}" "$@"
