#!/usr/bin/env bash
# LoRA SFT driver for scripts/train_stage1.sh: one training command, a time cap,
# the "halve the samples and rerun" rule, and the merge that vLLM serving needs.
# The wrapper sets STAGE, DEFAULT_MAX_HOURS and DEFAULT_HALVE, then sources this file.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

VENV_BIN="$ROOT/.venv_train/bin"
export PATH="$VENV_BIN:$PATH"  # llamafactory-cli launches torchrun by name for multi-GPU runs
BASE="${VLM_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
CONFIG="src/train/configs/nep_agent_tool_evidence.yaml"
DATASET_DIR=""  # default: dataset_dir of the config
OUTPUT=""       # default: output_dir of the config
LOG=""          # default: logs/train_<config name>.log
GPU=2
MAX_HOURS="$DEFAULT_MAX_HOURS"
HALVE="$DEFAULT_HALVE"
MAX_SAMPLES=0
MAX_STEPS=0
MERGE=1

usage() {
    cat <<USAGE
Usage: $0 [--config YAML] [--gpu N|N,M,...] [--max-hours H] [--halve-on-timeout 0|1] [--max-samples N] [--max-steps N]
          [--dataset-dir DIR] [--output DIR] [--log FILE] [--base DIR] [--no-merge]
  --config       LLaMA-Factory training config (default src/train/configs/nep_agent_tool_evidence.yaml)
  --dataset-dir  folder with train.json and dataset_info.json (default: dataset_dir of the config)
  --output       adapter folder (default: output_dir of the config); the merged model is written to <output>_merged
  --base         base model path or hub id (default: VLM_PATH, else Qwen/Qwen3-VL-8B-Instruct)
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --max-hours) MAX_HOURS="$2"; shift 2 ;;
        --halve-on-timeout) HALVE="$2"; shift 2 ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --log) LOG="$2"; shift 2 ;;
        --base) BASE="$2"; shift 2 ;;
        --no-merge) MERGE=0; shift ;;
        --config) CONFIG="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -f "$CONFIG" ]]; then
    echo "missing config $CONFIG" >&2
    exit 2
fi
config_value() { awk -v key="$1:" '$1 == key { print $2; exit }' "$CONFIG"; }
[[ -n "$DATASET_DIR" ]] || DATASET_DIR="$(config_value dataset_dir)"
[[ -n "$OUTPUT" ]] || OUTPUT="$(config_value output_dir)"
[[ -n "$LOG" ]] || LOG="logs/train_$(basename "$CONFIG" .yaml).log"

if [[ ",$GPU," == *",0,"* && "${ALLOW_GPU0:-0}" != "1" ]]; then  # set ALLOW_GPU0=1 on hosts where GPU 0 may be used
    echo "GPU 0 is reserved; set ALLOW_GPU0=1 to use it" >&2
    exit 2
fi
# Several GPUs (e.g. --gpu 1,2,3) run data-parallel through torchrun.
if [[ "$GPU" == *,* ]]; then
    export FORCE_TORCHRUN=1
fi
if [[ ! -x "$VENV_BIN/llamafactory-cli" ]]; then
    echo "missing $VENV_BIN/llamafactory-cli; create .venv_train first" >&2
    exit 2
fi
if [[ ! -f "$DATASET_DIR/train.json" ]]; then
    echo "missing $DATASET_DIR/train.json; build the dataset first" >&2
    exit 2
fi

mkdir -p "$(dirname "$LOG")" "$(dirname "$OUTPUT")"
LIMIT_SECONDS="$(awk -v hours="$MAX_HOURS" 'BEGIN { printf "%d", hours * 3600 }')"
total_samples() {
    "$VENV_BIN/python" -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$DATASET_DIR/train.json"
}

note() { echo "[$(date '+%F %T')] stage${STAGE}: $*" | tee -a "$LOG"; }

STARTED=$SECONDS
SAMPLES="$MAX_SAMPLES"
ATTEMPT=1
while :; do
    ARGS=("model_name_or_path=$BASE" "dataset_dir=$DATASET_DIR" "output_dir=$OUTPUT")
    [[ "$SAMPLES" -gt 0 ]] && ARGS+=("max_samples=$SAMPLES")
    [[ "$MAX_STEPS" -gt 0 ]] && ARGS+=("max_steps=$MAX_STEPS")
    note "attempt $ATTEMPT, gpu $GPU, samples ${SAMPLES:-all} (0=all), cap ${LIMIT_SECONDS}s"
    attempt_started=$SECONDS
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_DISABLED=true PYTHONPATH="$ROOT" \
        timeout --signal=TERM --kill-after=120 "$LIMIT_SECONDS" \
        "$VENV_BIN/llamafactory-cli" train "$CONFIG" "${ARGS[@]}" 2>&1 | tee -a "$LOG"
    code=${PIPESTATUS[0]}
    note "attempt $ATTEMPT exited with $code after $((SECONDS - attempt_started))s"

    if [[ "$code" -eq 0 ]]; then
        break
    fi
    if [[ "$code" -ne 124 && "$code" -ne 137 ]]; then
        note "training failed; see the log above"
        exit "$code"
    fi
    if [[ "$HALVE" -ne 1 || "$ATTEMPT" -ge 2 ]]; then
        note "time cap reached and no rerun is allowed; stopping without an adapter"
        exit "$code"
    fi
    [[ "$SAMPLES" -gt 0 ]] || SAMPLES="$(total_samples)"
    SAMPLES=$((SAMPLES / 2))
    ATTEMPT=$((ATTEMPT + 1))
    note "time cap reached; rerunning with half the samples ($SAMPLES)"
done

if [[ ! -f "$OUTPUT/adapter_model.safetensors" ]]; then
    note "training finished but $OUTPUT/adapter_model.safetensors is missing"
    exit 1
fi

"$VENV_BIN/python" - "$OUTPUT/train_meta.json" "$STAGE" "$GPU" "$ATTEMPT" "$SAMPLES" "$((SECONDS - STARTED))" "$MAX_HOURS" <<'PY'
import json, sys

path, stage, gpu, attempt, samples, seconds, cap = sys.argv[1:]
meta = {"stage": int(stage), "gpu": gpu, "attempts": int(attempt), "max_samples": int(samples),
        "seconds": int(seconds), "cap_hours": float(cap)}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(meta, handle, indent=1)
PY

if [[ "$MERGE" -eq 1 ]]; then
    note "merging adapter into ${OUTPUT%/}_merged"
    PYTHONPATH="$ROOT" "$VENV_BIN/python" -m src.train.merge_adapter \
        --base "$BASE" --adapter "$OUTPUT" --out "${OUTPUT%/}_merged" --force 2>&1 | tee -a "$LOG"
    code=${PIPESTATUS[0]}
    if [[ "$code" -ne 0 ]]; then
        note "merge failed"
        exit "$code"
    fi
fi
note "done in $((SECONDS - STARTED))s"
