#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
PRETRAINED="${PRETRAINED:-GSAI-ML/LLaDA-V}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-s32_len256_block256}"
GEN_LENGTH="${GEN_LENGTH:-256}"
BLOCK_LENGTH="${BLOCK_LENGTH:-256}"
STEPS="${STEPS:-32}"
TEMPERATURE="${TEMPERATURE:-0}"
RUN_MMMU="${RUN_MMMU:-1}"
RUN_HALLUSION="${RUN_HALLUSION:-1}"
RUN_POPE="${RUN_POPE:-1}"
POPE_SPLITS="${POPE_SPLITS:-random,popular,adversarial}"
POPE_SAMPLES="${POPE_SAMPLES:-1000}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p exp logs
LOG_FILE="logs/visual_margin_${RUN_ID}.log"

run() {
  local name="$1"
  shift
  echo
  echo "===== START $name ====="
  date
  "$@"
  echo "===== DONE $name ====="
  date
}

common=(
  --pretrained "$PRETRAINED"
  --device cuda:0
  --gen_length "$GEN_LENGTH"
  --block_length "$BLOCK_LENGTH"
  --steps "$STEPS"
  --temperature "$TEMPERATURE"
)

{
  echo "Model: $PRETRAINED"
  echo "Setting: gen=$GEN_LENGTH block=$BLOCK_LENGTH steps=$STEPS"

  if [[ "$RUN_MMMU" == "1" ]]; then
    mmmu_data=()
    if [[ -n "${MMMU_ARROW:-}" ]]; then
      mmmu_data=(--dataset_arrow "$MMMU_ARROW")
    fi
    run "MMMU Native" "$PYTHON_BIN" eval_mmmu_native.py \
      "${common[@]}" "${mmmu_data[@]}" \
      --output_dir "exp/mmmu_native_${RUN_ID}"
    run "MMMU VisualMargin" "$PYTHON_BIN" eval_mmmu_visual_margin.py \
      "${common[@]}" "${mmmu_data[@]}" \
      --output_dir "exp/mmmu_visual_margin_${RUN_ID}"
  fi

  if [[ "$RUN_HALLUSION" == "1" ]]; then
    hallusion_data=()
    if [[ -n "${HALLUSION_ARROW:-}" ]]; then
      hallusion_data=(--dataset_arrow "$HALLUSION_ARROW")
    fi
    run "HallusionBench Native" "$PYTHON_BIN" eval_hallusion.py \
      --method native "${common[@]}" "${hallusion_data[@]}" \
      --output_dir "exp/hallusion_native_${RUN_ID}"
    run "HallusionBench VisualMargin" "$PYTHON_BIN" eval_hallusion.py \
      --method visual_margin "${common[@]}" "${hallusion_data[@]}" \
      --output_dir "exp/hallusion_visual_margin_${RUN_ID}"
  fi

  if [[ "$RUN_MMMU" == "1" && "$RUN_HALLUSION" == "1" ]]; then
    run "MMMU/Hallusion comparison" "$PYTHON_BIN" compare_mmmu_hallusion.py \
      --mmmu_native "exp/mmmu_native_${RUN_ID}/mmmu_results.json" \
      --mmmu_margin "exp/mmmu_visual_margin_${RUN_ID}/mmmu_results.json" \
      --hallusion_native "exp/hallusion_native_${RUN_ID}/hallusion_native_results.json" \
      --hallusion_margin "exp/hallusion_visual_margin_${RUN_ID}/hallusion_visual_margin_results.json" \
      --output "exp/mmmu_hallusion_${RUN_ID}_comparison.json"
  fi

  if [[ "$RUN_POPE" == "1" ]]; then
    if [[ -z "${POPE_ROOT:-}" ]]; then
      echo "POPE_ROOT is required when RUN_POPE=1" >&2
      exit 2
    fi
    IFS=',' read -ra splits <<< "$POPE_SPLITS"
    for split in "${splits[@]}"; do
      native_dir="exp/pope_${split}_native_${RUN_ID}_${POPE_SAMPLES}"
      margin_dir="exp/pope_${split}_visual_margin_${RUN_ID}_${POPE_SAMPLES}"
      run "POPE $split Native" "$PYTHON_BIN" eval_pope_native.py \
        "${common[@]}" --pope_root "$POPE_ROOT" --split "$split" \
        --max_samples "$POPE_SAMPLES" --output_dir "$native_dir"
      run "POPE $split VisualMargin" "$PYTHON_BIN" eval_pope_visual_margin.py \
        "${common[@]}" --pope_root "$POPE_ROOT" --split "$split" \
        --max_samples "$POPE_SAMPLES" --output_dir "$margin_dir"
      run "POPE $split comparison" "$PYTHON_BIN" compare_pope.py \
        --native "$native_dir/pope_results.json" \
        --method "$margin_dir/pope_results.json" \
        --output "$margin_dir/comparison.json"
    done
  fi
} 2>&1 | tee -a "$LOG_FILE"
