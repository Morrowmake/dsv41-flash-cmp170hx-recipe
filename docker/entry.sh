#!/usr/bin/env bash
# entry.sh: the server's entry point inside the engine image, for ./start.sh
# and for Docker Compose (docker/prepare.sh serve). It prepares what the
# launch needs at run time and then runs the image's serve.sh, which holds the
# release's launch: the engine command and every serving default, generated
# from the release settings and baked into the image at /opt/recipe/serve.sh.
#
#   1. caches   a private temporary directory, the optional compilation-cache
#               seed (VLLM_IMAGE_CACHE_SEED=1), and every compiler and download
#               cache under the mounted /cache
#   2. engram   where the Engram tables live (engram_placement.sh): under
#               VLLM_ENGRAM_PLACEMENT=auto, the default, host RAM when the
#               memory this container can use covers NEED_RAM_GIB, else the
#               SSD with a row cache sized from that memory
#   3. p2p      the peer-access content check (p2p_check.sh): peer transports
#               only after it passes; P2P=off or P2P=force skip it
#   4. serve    serve.sh, under the two-request boot check (boot_check.py)
#               unless BOOT_CHECK=0 or start.sh runs that check itself
#
# EXTRA_ARGS may not set an argument the release fixes (extra_args_guard);
# with speculative decoding off in it, the boot check reads no draft counters.
#
# DRY=1 checks nothing on the cards and lets serve.sh print the environment
# and the command instead of starting the server.
set -euo pipefail
# This recipe's files are mounted at /recipe; the image's venv is /opt/venv.
REPO_ROOT="${RECIPE_DIR:-/recipe}"
VENV="${VENV:-/opt/venv}"
SERVE_SH="${SERVE_SH:-/opt/recipe/serve.sh}"
[ -f "$SERVE_SH" ] || { echo "entry.sh: this image has no $SERVE_SH" >&2; exit 1; }
# One layout: tensor-parallel 4, routed experts split across the cards.
export LAYOUT="${LAYOUT:-tp4}"
TP=4
PP=1

# Speculative decoding switched off through EXTRA_ARGS
# (`--speculative-config None`, `null` or `off`, also written with `=`).
# Without the drafter the adaptive verify depth has no state to work on and
# the server stops while capturing its CUDA graphs, and the boot check has no
# draft counters to read. drafting_off returns 0 when drafting is off; the
# entry and start.sh then clear VLLM_DSPARK_ADAPTIVE_DEPTH and run the boot
# check without its draft-counter checks.
drafting_off() {
    local -a args
    local i value
    read -ra args <<< "${EXTRA_ARGS:-}"
    for ((i = 0; i < ${#args[@]}; i++)); do
        case "${args[i]}" in
            --speculative-config=*) value="${args[i]#*=}" ;;
            --speculative-config) value="${args[i+1]:-}" ;;
            *) continue ;;
        esac
        case "${value,,}" in none|null|off|''|\'\'|'""') return 0 ;; esac
    done
    return 1
}
DRAFTING_OFF_LINE="Speculative decoding: off (EXTRA_ARGS); VLLM_DSPARK_ADAPTIVE_DEPTH cleared, boot check without draft counters"
# With drafting off, the settings that only the drafter reads are cleared.
drafting_off_clear() {
    export VLLM_DSPARK_ADAPTIVE_DEPTH=
}

# EXTRA_ARGS adds options to the server's command line. It may not set an
# argument this release fixes (the checkpoint, the parallel layout, the
# quantisation, the memory budget and the KV cache): extra_args_guard prints
# the first such argument and returns 2. The engine's parser accepts a unique
# prefix of a long option, `_` for `-`, `--no-` before a switch and dotted
# JSON keys, so each of those is matched too; so are the short aliases.
# Switching speculative decoding off (drafting_off's forms) is always allowed.
EXTRA_ARGS_CONTROLLED="config model hf-config-path hf-overrides model-class-overrides tensor-parallel-size pipeline-parallel-size data-parallel-size data-parallel-size-local decode-context-parallel-size prefill-context-parallel-size gpu-memory-utilization device-memory-utilization kv-cache-memory-bytes num-gpu-blocks-override quantization quantization-config kv-cache-dtype kv-cache-dtype-skip-layers mamba-cache-dtype mamba-ssm-cache-dtype cpu-offload-gb cpu-offload-params offload-backend offload-group-size offload-num-in-group offload-prefetch-step offload-params kv-offloading-size kv-offloading-backend kv-transfer-config"
EXTRA_ARGS_CONTROLLED_SHORT="-q -pp -tp -dcp -pcp -dp -dpl"
extra_args_guard() {
    local -a args
    local i arg option controlled
    read -ra args <<< "${EXTRA_ARGS:-}"
    for ((i = 0; i < ${#args[@]}; i++)); do
        arg="${args[i]}"
        case "$arg" in
            --speculative-config=*)
                case "${arg#*=}" in [Nn][Oo][Nn][Ee]|[Nn][Uu][Ll][Ll]|[Oo][Ff][Ff]|''|\'\'|'""') continue ;; esac ;;
            --speculative-config)
                case "${args[i+1]:-}" in [Nn][Oo][Nn][Ee]|[Nn][Uu][Ll][Ll]|[Oo][Ff][Ff]|''|\'\'|'""') i=$((i + 1)); continue ;; esac ;;
        esac
        case "$arg" in
            --)
                echo "EXTRA_ARGS cannot end the options with --: $arg" >&2; return 2 ;;
            --*)
                option="${arg%%=*}"; option="${option%%.*}"; option="${option//_/-}"; option="${option#--}"
                for controlled in $EXTRA_ARGS_CONTROLLED; do
                    if [[ "$controlled" == "$option"* || "no-$controlled" == "$option"* ]]; then
                        echo "EXTRA_ARGS cannot set --$controlled, which this release fixes: $arg" >&2; return 2
                    fi
                done ;;
            -?*)
                option="${arg%%=*}"; option="${option%%.*}"
                for controlled in $EXTRA_ARGS_CONTROLLED_SHORT; do
                    if [[ "$option" == "$controlled"* || "$controlled" == "$option"* ]]; then
                        echo "EXTRA_ARGS cannot set $controlled, which this release fixes: $arg" >&2; return 2
                    fi
                done ;;
        esac
    done
    return 0
}
extra_args_guard || exit 2

# 1. Caches. The server runs as the user who started it; everything it writes
# goes under /cache, which that user owns. Compilers use a private temporary
# directory there.
if [ -n "${RECIPE_CACHE_ROOT:-}" ] && [ "${DRY:-0}" != 1 ]; then
    [ "$RECIPE_CACHE_ROOT" = /cache ] || { echo '[boot-check] FAIL: container cache root must be /cache' >&2; exit 1; }
    temp_base="${TMPDIR:-/cache/tmp}"
    case "$temp_base/" in /cache/*) ;; *) echo '[boot-check] FAIL: container TMPDIR must be under /cache' >&2; exit 1 ;; esac
    mkdir -p "$temp_base"
    temp_base="$(realpath "$temp_base")"
    case "$temp_base/" in /cache/*) ;; *) echo '[boot-check] FAIL: container TMPDIR resolves outside /cache' >&2; exit 1 ;; esac
    TMPDIR="$(mktemp -d "$temp_base/run.XXXXXXXX")"
    export TMPDIR
    export TMP="$TMPDIR" TEMP="$TMPDIR"
    unset temp_base
fi
# Cache seeds are opt-in, and only in an image that carries one.
if [ "${VLLM_IMAGE_CACHE_SEED:-0}" = 1 ] && [ "${DRY:-0}" = 1 ]; then
    echo "[image-cache] enabled (dry run)"
elif [ "${VLLM_IMAGE_CACHE_SEED:-0}" = 1 ]; then
    [ -f /opt/image-tools/cache_seed.py ] || { echo "[image-cache] enabled but image helper missing" >&2; exit 1; }
    cache_exports="$("$VENV/bin/python" /opt/image-tools/cache_seed.py seed /opt/cache-seed)"
    eval "$cache_exports"
    unset cache_exports
else
    echo "[image-cache] disabled"
fi
# Explicit compiler destinations, with or without a seed: each must be under
# /cache and writable.
if [ -n "${RECIPE_CACHE_ROOT:-}" ] && [ "${DRY:-0}" != 1 ]; then
    export HOME="${HOME:-/cache}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
    export HF_HOME="${HF_HOME:-/cache/huggingface}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
    export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}"
    export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
    export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$HOME/.tilelang/cache}"
    export TILELANG_TMP_DIR="${TILELANG_TMP_DIR:-$HOME/.tilelang/cache/tmp}"
    export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$HOME/.nv/ComputeCache}"
    export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$HOME}"
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/torchinductor}"
    for cache_path in "$HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$VLLM_CACHE_ROOT" "$TILELANG_CACHE_DIR" "$TILELANG_TMP_DIR" "$CUDA_CACHE_PATH" "$FLASHINFER_WORKSPACE_BASE" "$TORCHINDUCTOR_CACHE_DIR"; do
        case "$cache_path/" in /cache/*) ;; *) echo "[boot-check] FAIL: compiler cache outside /cache: $cache_path" >&2; exit 1 ;; esac
        mkdir -p "$cache_path"
        cache_path="$(realpath "$cache_path")"
        case "$cache_path/" in /cache/*) ;; *) echo "[boot-check] FAIL: cache symlink outside /cache: $cache_path" >&2; exit 1 ;; esac
        [ -w "$cache_path" ] || { echo "[boot-check] FAIL: cache is not writable: $cache_path" >&2; exit 1; }
    done
    unset cache_path
fi

# 2. The Engram tables: ram or ssd, decided once here and passed to serve.sh;
# under ssd also the row cache size, its fill at start and direct reads,
# unless they are set.
source "$REPO_ROOT/engram_placement.sh"
engram_placement || exit 2
export VLLM_ENGRAM_PLACEMENT="$ENGRAM_PLACEMENT"
if [ "$ENGRAM_PLACEMENT" = ssd ]; then
    export VLLM_ENGRAM_ROW_CACHE_GIB="$ENGRAM_ROW_CACHE_GIB"
    export VLLM_ENGRAM_ROW_CACHE_PRELOAD="$ENGRAM_PRELOAD"
    export VLLM_ENGRAM_DIRECT_READ="$ENGRAM_DIRECT_READ"
fi
echo "$ENGRAM_LINE"
if drafting_off; then
    drafting_off_clear
    echo "$DRAFTING_OFF_LINE"
fi

# 3. Peer access: every peer transport off unless the check passes
# (p2p_check.sh sets P2P_ENABLED and the transports).
source "$REPO_ROOT/p2p_check.sh"
echo "entry.sh: [p2p] enabled=${P2P_ENABLED:-0}"

# 4. The server.
BOOT_CHECK="${BOOT_CHECK:-1}"
case "$BOOT_CHECK" in 0|1) ;; *) echo "[boot-check] FAIL: BOOT_CHECK must be 0 or 1" >&2; exit 2 ;; esac
if [ "${DRY:-0}" = 1 ]; then
    echo "[boot-check] BOOT_CHECK=$BOOT_CHECK (dry run)"
    exec /bin/bash "$SERVE_SH"
fi
if [ "$BOOT_CHECK" = 0 ]; then
    echo "[boot-check] disabled (BOOT_CHECK=0)"
elif [ "${RECIPE_BOOT_CHECK_OWNER:-}" = start ]; then
    echo "[boot-check] delegated to start.sh after health"
else
    boot_args=()
    [ -z "${RECIPE_CACHE_ROOT:-}" ] || boot_args+=(--container)
    ! drafting_off || boot_args+=(--no-draft)
    exec "$VENV/bin/python" "$REPO_ROOT/boot_check.py" "${boot_args[@]}" \
        --base "http://127.0.0.1:${PORT:-8000}" \
        --model "${SERVED_MODEL_NAME:-dsv41-flash}" \
        --ready-timeout "${READY_TIMEOUT:-3600}" --serve /bin/bash "$SERVE_SH"
fi
exec /bin/bash "$SERVE_SH"
