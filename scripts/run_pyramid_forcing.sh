#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

child_pid=""

cleanup_on_interrupt() {
    trap - INT TERM
    echo
    echo "[WARN] Interrupted. Terminating torchrun and all child processes..." >&2

    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill -TERM -- "-$child_pid" 2>/dev/null || kill -TERM "$child_pid" 2>/dev/null || true

        for _ in {1..10}; do
            if ! kill -0 "$child_pid" 2>/dev/null; then
                break
            fi
            sleep 1
        done

        if kill -0 "$child_pid" 2>/dev/null; then
            echo "[WARN] Escalating to SIGKILL for remaining processes..." >&2
            kill -KILL -- "-$child_pid" 2>/dev/null || kill -KILL "$child_pid" 2>/dev/null || true
        fi

        wait "$child_pid" 2>/dev/null || true
    fi

    exit 130
}

trap cleanup_on_interrupt INT TERM

# Defaults
config=configs/pyramid-forcing.yaml
checkpoint=checkpoints/self_forcing_dmd.pt
prompts=prompts/MovieGenVideoBench_num128.txt
output_dir=videos/Exp_pyramid_forcing
num_frames=120
num_gpus=""
master_port="${MASTER_PORT:-29500}"

# Progress-bar output. tqdm is the default since the per-block bar now
# counts denoise + clean pass correctly (5/5) and the per-prompt status
# line goes through tqdm.write(). Override with PYRAMIDKV_LOG_MODE=silent
# when batching over slow SSH where the bar flushes block the host
# thread, or PYRAMIDKV_LOG_MODE=print for plain log-style output.
export PYRAMIDKV_LOG_MODE="${PYRAMIDKV_LOG_MODE:-tqdm}"

# CUDA refresh fast path: replaces the per-segment scatter_copy + Python
# descriptor build with a single mega-kernel pulled in via the JIT extension.
# Validated at d4a94bb (pack 11960 → 7063 ms, -41% Python pack-side time).
# On 123f cpp-HEAD bench (a313d07): pack_ms 26931 → 15325 (-11.6s CPU).
# Wall-clock DiT only -0.6s today because CPU time is overlapped with GPU,
# but the CPU savings will materialise under M2.2 CUDA Graph capture where
# graph replay eliminates CPU dispatch entirely. Default-on; override with 0.
export PYRAMIDKV_CUDA_REFRESH="${PYRAMIDKV_CUDA_REFRESH:-1}"

# MegaCache path: when set to 1, _initialize_kv_cache builds the C++/CUDA
# PyramidKVCacheManager-backed MegaCache instead of Python's AdaptiveKVCache.
# Default-on now that the raw-K refactor + Plan-B buffer shrink land
# parity with the Python reference at ~30% faster wall-clock. Override
# with PYRAMIDKV_USE_MEGA_CACHE=0 to fall back to the Python path (e.g. when
# debugging a kernel regression). Requires the JIT extension to build
# (see pyramidkv/_scatter_ext.py).
export PYRAMIDKV_USE_MEGA_CACHE="${PYRAMIDKV_USE_MEGA_CACHE:-1}"

# JIT build verbosity for debugging: when set to 1, _try_load emits the full
# nvcc compile log + traceback on failure instead of swallowing the error.
export PYRAMIDKV_VERBOSE_BUILD="${PYRAMIDKV_VERBOSE_BUILD:-0}"

infer_visible_gpu_count() {
    local visible="${CUDA_VISIBLE_DEVICES:-}"
    local count=0
    local gpu=""

    if [[ -z "$visible" ]]; then
        printf '0\n'
        return 0
    fi

    IFS=',' read -r -a visible_gpus <<< "$visible"
    for gpu in "${visible_gpus[@]}"; do
        gpu="${gpu//[[:space:]]/}"
        if [[ -z "$gpu" || "$gpu" == "-1" ]]; then
            continue
        fi
        count=$((count + 1))
    done

    printf '%s\n' "$count"
}

count_nonempty_prompts() {
    local prompts_path="$1"
    grep -cve '^[[:space:]]*$' "$prompts_path"
}

# Parse CLI args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)     config="$2";     shift 2 ;;
        --checkpoint) checkpoint="$2"; shift 2 ;;
        --prompts)    prompts="$2";    shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --num-frames) num_frames="$2"; shift 2 ;;
        --num-gpus)   num_gpus="$2";   shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ! -f "$config" ]]; then
    echo "Config not found: $config" >&2
    exit 1
fi
if [[ ! -f "$checkpoint" ]]; then
    echo "Checkpoint not found: $checkpoint" >&2
    exit 1
fi
if [[ ! -f "$prompts" ]]; then
    echo "Prompts not found: $prompts" >&2
    exit 1
fi

prompt_count="$(count_nonempty_prompts "$prompts")"
if [[ "$prompt_count" -le 0 ]]; then
    echo "Prompts file contains no non-empty prompts: $prompts" >&2
    exit 1
fi

visible_gpu_count="$(infer_visible_gpu_count)"
if [[ -z "$num_gpus" ]]; then
    if [[ "$visible_gpu_count" -gt 0 ]]; then
        num_gpus="$visible_gpu_count"
        echo "[INFO] Auto-detected --num-gpus=$num_gpus from CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    else
        num_gpus=4
    fi
fi

if ! [[ "$num_gpus" =~ ^[0-9]+$ ]] || [[ "$num_gpus" -le 0 ]]; then
    echo "Invalid --num-gpus value: $num_gpus" >&2
    exit 1
fi

if [[ "$visible_gpu_count" -gt 0 && "$num_gpus" -gt "$visible_gpu_count" ]]; then
    echo "Invalid --num-gpus=$num_gpus: only $visible_gpu_count GPU(s) visible via CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
fi

if [[ "$num_gpus" -gt "$prompt_count" ]]; then
    echo "[INFO] Reducing --num-gpus from $num_gpus to $prompt_count because prompts file has only $prompt_count non-empty prompt(s)." >&2
    num_gpus="$prompt_count"
fi

mkdir -p "$output_dir"

# 保存实验配置快照
cp "$config" "$output_dir/config.yaml"

# 生成 prompts.csv
python -c "
import csv, sys
with open('$prompts') as f:
    lines = [l.strip() for l in f if l.strip()]
with open('$output_dir/prompts.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['index', 'prompt'])
    for i, p in enumerate(lines):
        w.writerow([i, p])
print(f'Wrote {len(lines)} prompts to $output_dir/prompts.csv')
"

# 推理
# torchrun adds ~500MB of NCCL workspace + a process-group barrier even
# at --nproc_per_node=1. Skip it entirely on a single GPU and launch
# Python directly so the steady-state VRAM matches a `uv run python
# inference.py` baseline (~500MB instead of ~1000MB at startup).
if [[ "$num_gpus" -eq 1 ]]; then
    setsid python inference.py \
        --config_path "$config" \
        --output_folder "$output_dir" \
        --checkpoint_path "$checkpoint" \
        --data_path "$prompts" \
        --use_ema \
        --num_output_frames "$num_frames" \
        --fixed_prefix video_ &
else
    setsid torchrun --master_port="$master_port" --nproc_per_node="$num_gpus" inference.py \
        --config_path "$config" \
        --output_folder "$output_dir" \
        --checkpoint_path "$checkpoint" \
        --data_path "$prompts" \
        --use_ema \
        --num_output_frames "$num_frames" \
        --fixed_prefix video_ &
fi

child_pid=$!
wait "$child_pid"
