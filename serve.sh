#!/bin/bash
# Generated at release time from the model's settings launch (the container runtime; tools
# release/checklist.py generate-serve). The release checklist regenerates it and refuses any
# difference: do not edit. The caller gives LAYOUT, and may give PORT, HOST and
# CUDA_VISIBLE_DEVICES; every other inherited name but process plumbing is cleared.
# DRY=1 prints the environment and the command and exits before exec.
# The recipe start may also give VLLM_*, NCCL_*, EXTRA_ARGS, TMP, TEMP, the user inputs API_KEY, EXTRA_ARGS, HOST, MAX_BATCHED, MAX_SEQS, PORT, SERVED_MODEL_NAME and the locations CUDA_CACHE_PATH, FLASHINFER_WORKSPACE_BASE, HF_HOME, HOME, TILELANG_CACHE_DIR, TILELANG_TMP_DIR, TMPDIR, TORCHINDUCTOR_CACHE_DIR, TORCH_EXTENSIONS_DIR, TRITON_CACHE_DIR, VLLM_CACHE_ROOT, XDG_CACHE_HOME.
# From: release/checklist.py generate-serve --model dsv41-flash --base rc1-1022.toml
# Base rc1-1022.toml sha256 bb3b6ddb8b759a93e57b0f3d70c3c32af646337bcb9a94b26e12fa30d253e449
# Variants, in order: candidate hybrid
# set args.compilation-config="{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[4,5,6,7,8,12,16,18,24,28,32,40]}"
# set args.max-num-batched-tokens=2048
# set args.max-num-partial-prefills=0
# set args.prefill-chunk-with-decodes=0
# set args.speculative-config="{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"num_speculative_tokens_per_batch_size\":[[1,3,5],[4,8,3]],\"enable_adaptive_verification\":false}"
# set env.VLLM_CUDAGRAPH_ESTIMATE_SPLIT="1"
# set env.VLLM_CUSTOM_ALLREDUCE_FLAGS_MAX_BYTES="409600"
# set env.VLLM_DEEPSEEK_V41_ATTN_PIPELINE="1"
# set env.VLLM_DEEPSEEK_V41_DRAFT_PROLOGUE_FUSED="1"
# set env.VLLM_DEEPSEEK_V41_ENGRAM_EARLY_GATHER="1"
# set env.VLLM_DEEPSEEK_V41_ENGRAM_WKV_REPLICATED_PREFILL="1"
# set env.VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE="1"
# set env.VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2="1"
# set env.VLLM_DEEPSEEK_V41_FUSED_ROUTE="1"
# set env.VLLM_DEEPSEEK_V41_GATE_GEMV="1"
# set env.VLLM_DEEPSEEK_V41_MHC_SIDE_STREAM="1"
# set env.VLLM_DEEPSEEK_V41_MOE_PREFILL_TC="1"
# set env.VLLM_DEEPSEEK_V41_PREFILL_CE_ALLREDUCE="1"
# set env.VLLM_DEEPSEEK_V41_PREFILL_SPLIT_REDUCE="1"
# set env.VLLM_DEEPSEEK_V41_ROUTER_MMA="1"
# set env.VLLM_DEEPSEEK_V41_SELECT_ONE_LAUNCH="1"
# set env.VLLM_DEEPSEEK_V41_SWA_META_BATCH="1"
# set env.VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM="1"
# set env.VLLM_DEEPSEEK_V41_WO_B_BF16_PREFILL_ALLREDUCE="1"
# set env.VLLM_DSPARK_PER_K_WIDTH="0"
# set env.VLLM_DSPARK_SHARD_ARGMAX="1"
# set env.VLLM_ENGRAM_PLACEMENT="ram"
# set env.VLLM_ENGRAM_ROW_CACHE_GIB="0"
# set env.VLLM_EXL3_MOE_PREFILL_UNCHUNKED="1"
# set env.VLLM_EXL3_MOE_V2_DRAFT="1"
# set env.VLLM_EXL3_MOE_V2_FOLD="1"
# set env.VLLM_EXL3_MOE_V2_RING="1"
# set env.VLLM_SM80_TARGET_SHARD_ARGMAX="1"
set -euo pipefail
[ $# -eq 0 ] || { echo "the reference launcher takes no arguments: $*" >&2; exit 2; }
layout=${LAYOUT:?the caller names the layout}
port=${PORT:-}
host=${HOST:-}
cvd_set=${CUDA_VISIBLE_DEVICES+1}
cvd=${CUDA_VISIBLE_DEVICES:-}
dry=${DRY:-${DRY_RUN:-0}}
for _name in $(compgen -e); do
  case "$_name" in VLLM_*|NCCL_*|EXTRA_ARGS|TMP|TEMP|CUDA_CACHE_PATH|FLASHINFER_WORKSPACE_BASE|HF_HOME|HOME|TILELANG_CACHE_DIR|TILELANG_TMP_DIR|TMPDIR|TORCHINDUCTOR_CACHE_DIR|TORCH_EXTENSIONS_DIR|TRITON_CACHE_DIR|VLLM_CACHE_ROOT|XDG_CACHE_HOME|API_KEY|HOST|MAX_BATCHED|MAX_SEQS|PORT|SERVED_MODEL_NAME) ;; *) unset "$_name" 2>/dev/null || true ;; esac
done
[[ -z "${MAX_BATCHED:-}" || "$MAX_BATCHED" =~ ^[0-9]+$ ]] || { echo "MAX_BATCHED=$MAX_BATCHED: not an integer" >&2; exit 2; }
[[ -z "${MAX_SEQS:-}" || "$MAX_SEQS" =~ ^[0-9]+$ ]] || { echo "MAX_SEQS=$MAX_SEQS: not an integer" >&2; exit 2; }
[[ -z "${PORT:-}" || "$PORT" =~ ^[0-9]+$ ]] || { echo "PORT=$PORT: not an integer" >&2; exit 2; }
case "$layout" in
  tp4)
    export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-/cache/.nv/ComputeCache}
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_HOME=/usr/local/cuda
    export CUDA_VERSION=13.3.1
    export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-/cache}
    export HF_HOME=${HF_HOME:-/cache/huggingface}
    export HOME=${HOME:-/cache}
    export LC_CTYPE=C.UTF-8
    export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
    export LIBRARY_PATH=/usr/local/cuda/lib64/stubs
    export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL-SYS}
    export NCCL_VERSION=${NCCL_VERSION-2.30.7-1}
    export NVARCH=x86_64
    export NVIDIA_PRODUCT_NAME=CUDA
    export NV_CUDA_CUDART_DEV_VERSION=13.3.29-1
    export NV_CUDA_CUDART_VERSION=13.3.29-1
    export NV_CUDA_LIB_VERSION=13.3.1-1
    export NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE=cuda-nsight-compute-13-3=13.3.1-1
    export NV_CUDA_NSIGHT_COMPUTE_VERSION=13.3.1-1
    export NV_LIBCUBLAS_DEV_PACKAGE=libcublas-dev-13-3=13.6.0.2-1
    export NV_LIBCUBLAS_DEV_PACKAGE_NAME=libcublas-dev-13-3
    export NV_LIBCUBLAS_DEV_VERSION=13.6.0.2-1
    export NV_LIBCUBLAS_PACKAGE=libcublas-13-3=13.6.0.2-1
    export NV_LIBCUBLAS_PACKAGE_NAME=libcublas-13-3
    export NV_LIBCUBLAS_VERSION=13.6.0.2-1
    export NV_LIBCUSPARSE_DEV_VERSION=12.8.2.51-1
    export NV_LIBCUSPARSE_VERSION=12.8.2.51-1
    export NV_LIBNCCL_DEV_PACKAGE=libnccl-dev=2.30.7-1+cuda13.3
    export NV_LIBNCCL_DEV_PACKAGE_NAME=libnccl-dev
    export NV_LIBNCCL_DEV_PACKAGE_VERSION=2.30.7-1
    export NV_LIBNCCL_PACKAGE=libnccl2=2.30.7-1+cuda13.3
    export NV_LIBNCCL_PACKAGE_NAME=libnccl2
    export NV_LIBNCCL_PACKAGE_VERSION=2.30.7-1
    export NV_LIBNPP_DEV_PACKAGE=libnpp-dev-13-3=13.1.2.81-1
    export NV_LIBNPP_DEV_VERSION=13.1.2.81-1
    export NV_LIBNPP_PACKAGE=libnpp-13-3=13.1.2.81-1
    export NV_LIBNPP_VERSION=13.1.2.81-1
    export NV_NVML_DEV_VERSION=13.3.29-1
    export NV_NVTX_VERSION=13.3.29-1
    export PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONNOUSERSITE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
    export TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR:-/cache/.tilelang/cache}
    export TILELANG_TMP_DIR=${TILELANG_TMP_DIR:-/cache/.tilelang/cache/tmp}
    export TMPDIR=${TMPDIR:-/cache/tmp}
    export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/cache/torchinductor}
    export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/cache/.cache/torch_extensions}
    export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/cache/.triton/cache}
    export VIRTUAL_ENV=/opt/venv
    export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT-/cache/.cache/vllm}
    export VLLM_CUDAGRAPH_ESTIMATE_SPLIT=${VLLM_CUDAGRAPH_ESTIMATE_SPLIT-1}
    export VLLM_CUSTOM_ALLREDUCE_ALGO=${VLLM_CUSTOM_ALLREDUCE_ALGO-2stage}
    export VLLM_CUSTOM_ALLREDUCE_FLAGS=${VLLM_CUSTOM_ALLREDUCE_FLAGS-1}
    export VLLM_CUSTOM_ALLREDUCE_FLAGS_MAX_BYTES=${VLLM_CUSTOM_ALLREDUCE_FLAGS_MAX_BYTES-409600}
    export VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=${VLLM_CUSTOM_ALLREDUCE_PCIE_P2P-1}
    export VLLM_DEEPSEEK_V41_ATTN_PIPELINE=${VLLM_DEEPSEEK_V41_ATTN_PIPELINE-1}
    export VLLM_DEEPSEEK_V41_DRAFT_PROLOGUE_FUSED=${VLLM_DEEPSEEK_V41_DRAFT_PROLOGUE_FUSED-1}
    export VLLM_DEEPSEEK_V41_ENGRAM_EARLY_GATHER=${VLLM_DEEPSEEK_V41_ENGRAM_EARLY_GATHER-1}
    export VLLM_DEEPSEEK_V41_ENGRAM_WKV_REPLICATED_PREFILL=${VLLM_DEEPSEEK_V41_ENGRAM_WKV_REPLICATED_PREFILL-1}
    export VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE=${VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE-1}
    export VLLM_DEEPSEEK_V41_FUSED_MHC=${VLLM_DEEPSEEK_V41_FUSED_MHC-1}
    export VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH=${VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH-1}
    export VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2=${VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2-1}
    export VLLM_DEEPSEEK_V41_FUSED_ROUTE=${VLLM_DEEPSEEK_V41_FUSED_ROUTE-1}
    export VLLM_DEEPSEEK_V41_GATE_GEMV=${VLLM_DEEPSEEK_V41_GATE_GEMV-1}
    export VLLM_DEEPSEEK_V41_MHC_SIDE_STREAM=${VLLM_DEEPSEEK_V41_MHC_SIDE_STREAM-1}
    export VLLM_DEEPSEEK_V41_MOE_PREFILL_TC=${VLLM_DEEPSEEK_V41_MOE_PREFILL_TC-1}
    export VLLM_DEEPSEEK_V41_MOE_ROUND_ONCE_ALLREDUCE=${VLLM_DEEPSEEK_V41_MOE_ROUND_ONCE_ALLREDUCE-0}
    export VLLM_DEEPSEEK_V41_PREFILL_CE_ALLREDUCE=${VLLM_DEEPSEEK_V41_PREFILL_CE_ALLREDUCE-1}
    export VLLM_DEEPSEEK_V41_PREFILL_SPLIT_REDUCE=${VLLM_DEEPSEEK_V41_PREFILL_SPLIT_REDUCE-1}
    export VLLM_DEEPSEEK_V41_ROUTER_MMA=${VLLM_DEEPSEEK_V41_ROUTER_MMA-1}
    export VLLM_DEEPSEEK_V41_SELECT_ONE_LAUNCH=${VLLM_DEEPSEEK_V41_SELECT_ONE_LAUNCH-1}
    export VLLM_DEEPSEEK_V41_SWA_META_BATCH=${VLLM_DEEPSEEK_V41_SWA_META_BATCH-1}
    export VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM=${VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM-1}
    export VLLM_DEEPSEEK_V41_WO_B_BF16_ALLREDUCE=${VLLM_DEEPSEEK_V41_WO_B_BF16_ALLREDUCE-1}
    export VLLM_DEEPSEEK_V41_WO_B_BF16_PREFILL_ALLREDUCE=${VLLM_DEEPSEEK_V41_WO_B_BF16_PREFILL_ALLREDUCE-1}
    export VLLM_DEV_AB_SWITCH=${VLLM_DEV_AB_SWITCH-''}
    export VLLM_DSPARK_PER_K_WIDTH=${VLLM_DSPARK_PER_K_WIDTH-0}
    export VLLM_DSPARK_SHARD_ARGMAX=${VLLM_DSPARK_SHARD_ARGMAX-1}
    export VLLM_ENGRAM_PLACEMENT=${VLLM_ENGRAM_PLACEMENT-ram}
    export VLLM_ENGRAM_ROW_CACHE_GIB=${VLLM_ENGRAM_ROW_CACHE_GIB-0}
    export VLLM_EXL3_EXPERTS_CONTIGUOUS=${VLLM_EXL3_EXPERTS_CONTIGUOUS-1}
    export VLLM_EXL3_EXPERT_PLACEMENT=${VLLM_EXL3_EXPERT_PLACEMENT-''}
    export VLLM_EXL3_EXPERT_PLACEMENT_SHA256=${VLLM_EXL3_EXPERT_PLACEMENT_SHA256-''}
    export VLLM_EXL3_HYBRID=${VLLM_EXL3_HYBRID-1}
    export VLLM_EXL3_MOE_CHUNK_ROWS=${VLLM_EXL3_MOE_CHUNK_ROWS-2048}
    export VLLM_EXL3_MOE_PREFILL_UNCHUNKED=${VLLM_EXL3_MOE_PREFILL_UNCHUNKED-1}
    export VLLM_EXL3_MOE_V2=${VLLM_EXL3_MOE_V2-1}
    export VLLM_EXL3_MOE_V2_DRAFT=${VLLM_EXL3_MOE_V2_DRAFT-1}
    export VLLM_EXL3_MOE_V2_FOLD=${VLLM_EXL3_MOE_V2_FOLD-1}
    export VLLM_EXL3_MOE_V2_RING=${VLLM_EXL3_MOE_V2_RING-1}
    export VLLM_HOST_ALLREDUCE=${VLLM_HOST_ALLREDUCE-1}
    export VLLM_ROUND_ONCE_ALLREDUCE_GROUPED=${VLLM_ROUND_ONCE_ALLREDUCE_GROUPED-0}
    export VLLM_SM80_CAPTURE_DIR=${VLLM_SM80_CAPTURE_DIR-''}
    export VLLM_SM80_TARGET_SHARD_ARGMAX=${VLLM_SM80_TARGET_SHARD_ARGMAX-1}
    export VLLM_USE_BREAKABLE_CUDAGRAPH=${VLLM_USE_BREAKABLE_CUDAGRAPH-1}
    export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/cache/.cache}
    if [ -n "${API_KEY+1}" ]; then export VLLM_API_KEY="$API_KEY"; fi
    if [ -n "$cvd_set" ]; then export CUDA_VISIBLE_DEVICES="$cvd"; else export CUDA_VISIBLE_DEVICES=0,1,2,3; fi
    argv=(/opt/venv/bin/vllm \
      serve \
      /models/DeepSeek-V4.1-Flash-1017-hybrid-exl3-fp8 \
      --host "${host:-0.0.0.0}" \
      --served-model-name "${SERVED_MODEL_NAME:-dsv41-flash}" \
      --max-model-len 262144 \
      --gpu-memory-utilization 0.95 \
      --max-num-seqs "${MAX_SEQS:-8}" \
      --port "${port:-8001}" \
      --hf-overrides '{"engram_table_dir":"/models/DeepSeek-V4.1-Flash-1017-engram"}' \
      --load-format exl3 \
      --enable-expert-parallel \
      --mm-encoder-tp-mode data \
      --kv-cache-dtype bfloat16 \
      --reasoning-parser deepseek_v41 \
      --enable-auto-tool-choice \
      --tool-call-parser deepseek_v41 \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,5,6,7,8,12,16,18,24,28,32,40]}' \
      --max-num-batched-tokens "${MAX_BATCHED:-2048}" \
      --max-num-partial-prefills 0 \
      --prefill-chunk-with-decodes 0 \
      --speculative-config '{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,3,5],[4,8,3]],"enable_adaptive_verification":false}' \
      --tensor-parallel-size 4 \
      --pipeline-parallel-size 1)
    if [ -n "${EXTRA_ARGS:-}" ]; then read -ra _extra <<< "$EXTRA_ARGS"; argv+=("${_extra[@]}"); fi
    unset API_KEY EXTRA_ARGS HOST MAX_BATCHED MAX_SEQS PORT SERVED_MODEL_NAME
    if [ "$dry" = 1 ]; then
      echo "reference serve.sh: DRY=1, not starting the server; environment and command follow" >&2
      env | sort
      printf -v _line '%q ' "${argv[@]}"
      echo "DRY_RUN_ARGV_SHELL=${_line% }"
      exit 0
    fi
    exec "${argv[@]}"
    ;;
  *) echo "layout $layout: not a layout of this recipe" >&2; exit 2 ;;
esac
