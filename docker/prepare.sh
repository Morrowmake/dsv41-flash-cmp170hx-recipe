#!/usr/bin/env bash
# prepare.sh: storage and checkpoints for the server, run inside the engine
# image (this repository mounts it at /recipe/prepare.sh).
#
#   prepare.sh prepare           Docker Compose's prepare service. Started as
#                                root: makes the cache and the three model
#                                directories writable by the serving user, then
#                                runs `models` as that user.
#   prepare.sh models            as the current user (./start.sh download):
#                                downloads each checkpoint unless it is
#                                complete (REFRESH_WEIGHTS=1: checks it with the
#                                Hub again), then builds the hybrid checkpoint
#                                unless it is already built. Hugging Face
#                                resumes a partial download; a complete
#                                checkpoint is skipped without contacting the
#                                Hub.
#   prepare.sh serve             Docker Compose's server: runs docker/entry.sh
#                                as the owner of /cache, the serving user chosen
#                                by prepare, never as root.
#   prepare.sh check DIR NAME    exit 0 when DIR holds a complete checkpoint
#                                NAME (export, official or hybrid); runs on the
#                                host too (./start.sh).
#   prepare.sh required NAME     the files checkpoint NAME needs besides the
#                                shards its index names.
#
# The served checkpoint is a hybrid: the routed experts of the EXL3 export,
# every other weight from DeepSeek's official FP8 checkpoint at its own
# precision. It is a directory of links into the two downloads (no weight is
# copied), plus the config, index and quantization config written for it, built
# with the engine's tools/sm80/build_hybrid_checkpoint.py. The Engram tables
# are read from the official checkpoint's last two shards. Inside the container
# the three directories are always at the paths below; the links point there.
#
# The serving user is the owner of the checkout (mounted at /checkout), or
# HOST_UID and HOST_GID when set, so `docker compose up` needs no settings for
# any user. ./start.sh runs the server as the user who runs it.
#
# Ownership: Docker creates a missing bind-mount directory as root. The cache
# is this recipe's own, so everything in it is given to the serving user. A
# model directory changes owner only while it is owned by root and empty (what
# Docker just created); existing checkpoints and other files are never changed
# or removed.
set -euo pipefail

# The checkpoints, at the Hub revisions this release was validated with.
EXPORT_REPO="${EXPORT_REPO:-LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw}"
EXPORT_REVISION="${EXPORT_REVISION:-c178a01284fc5c0e859d8668cffaac81ca8f4042}"
OFFICIAL_REPO="${OFFICIAL_REPO:-deepseek-ai/DeepSeek-V4.1-Flash-1017}"
OFFICIAL_REVISION="${OFFICIAL_REVISION:-1bae0e1088993b9b82de08f59de5be5a34e42170}"
# Their directories in the container (the release launch names them).
EXPORT_DIR=/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw
OFFICIAL_DIR=/models/DeepSeek-V4.1-Flash-1017-engram
HYBRID_DIR=/models/DeepSeek-V4.1-Flash-1017-hybrid-exl3-fp8
# The hybrid build: EXL3 for the routed experts (target model and drafter
# layers), the official FP8 method for the rest; the Engram embedding tables
# stay out of the weights (the engine reads them from the official shards).
HYBRID_EXL3_MODULES='^(layers|mtp)\.\d+\.ffn\.experts(\.|$)'
HYBRID_DROP='\.engram\.embed\.'
HYBRID_METHOD=deepseek_v4_fp8
# sha256 of the files the build writes, as validated. The same inputs always
# give the same bytes; anything else is not the validated checkpoint.
HYBRID_SHA256="config.json=ebc5ca822e4af47f63f7ac5707ea09a5f5af4ea00284912ba0f92a7c3bb7a78f
model.safetensors.index.json=e96523840003f6ca8f071250343f8188937fd9ed684dba1f713217b9c0574aba
quantization_config.json=c6cdf9f915cb7fa48b7564243fd179f9b8530a9f2bd5d17b91c6dbb4f8ec1e15"
NEED_DISK_GB="${NEED_DISK_GB:-680}"
CACHE=/cache
CHECKOUT=/checkout
HF=/opt/venv/bin/hf
PY=/opt/venv/bin/python
BUILDER=/opt/vllm-src/tools/sm80/build_hybrid_checkpoint.py

say() { printf 'prepare: %s\n' "$*"; }
fail() { printf 'prepare: %s\n' "$*" >&2; exit 1; }

# required NAME: the files checkpoint NAME needs besides the shards its index
# names. The server reads every one of them.
required() {
    case "$1" in
        export) echo config.json model.safetensors.index.json quantization_config.json \
                    tokenizer.json tokenizer_config.json ;;
        official) echo config.json model.safetensors.index.json ;;
        hybrid) echo config.json model.safetensors.index.json quantization_config.json \
                    tokenizer.json tokenizer_config.json hybrid-build.json ;;
        *) return 1 ;;
    esac
}

# The download patterns of each repository: what `required` and the index need.
includes() {
    case "$1" in
        export) echo config.json model.safetensors.index.json quantization_config.json \
                    tokenizer.json tokenizer_config.json 'model-*.safetensors' ;;
        official) echo config.json model.safetensors.index.json 'model-*.safetensors' ;;
    esac
}

# shards DIR: every shard file the index names.
shards() {
    grep -o '"[^"]*\.safetensors"' "$1/model.safetensors.index.json" | tr -d '"' | sort -u
}

# complete DIR NAME: every required file and every shard the index names, and
# no partial download left by an interrupted fetch. The hybrid's shards are
# links to container paths, so outside the container only the links are
# checked; its written files must have the validated bytes everywhere.
complete() {
    local dir="$1" name="$2" file shard line
    [ -d "$dir" ] || return 1
    for file in $(required "$name"); do
        [ -f "$dir/$file" ] || return 1
    done
    if [ "$name" = hybrid ]; then
        while IFS= read -r line; do
            [ "$(sha256sum <"$dir/${line%%=*}" | cut -d' ' -f1)" = "${line#*=}" ] || return 1
        done <<<"$HYBRID_SHA256"
        while IFS= read -r shard; do
            [ -L "$dir/$shard" ] || return 1
            if [ -d "$OFFICIAL_DIR" ]; then [ -f "$dir/$shard" ] || return 1; fi
        done < <(shards "$dir")
        return 0
    fi
    if [ -d "$dir/.cache/huggingface/download" ] \
        && [ -n "$(find "$dir/.cache/huggingface/download" -name '*.incomplete' -print -quit)" ]; then
        return 1
    fi
    while IFS= read -r shard; do
        [ -f "$dir/$shard" ] || return 1
    done < <(shards "$dir")
}

# present DIR NAME: complete, and not asked to check it with the Hub again.
present() {
    [ "${REFRESH_WEIGHTS:-0}" != 1 ] && complete "$1" "$2"
}

fetch() {
    local name="$1" repo="$2" revision="$3" dir="$4"
    [ -x "$HF" ] || fail "no Hugging Face client at $HF"
    [ -w "$dir" ] || fail "$dir is not writable by uid $(id -u); give it to that user, or set HOST_UID and HOST_GID to its owner"
    say "downloading $repo @ $revision (resumes a partial download)"
    # shellcheck disable=SC2046
    "$HF" download "$repo" --revision "$revision" --local-dir "$dir" --include $(includes "$name")
    complete "$dir" "$name" || fail "$repo: the download ended but $dir is still incomplete; run the same command again to resume"
    say "$repo: complete"
}

# hybrid: build the link directory, or confirm the one there. A directory that
# holds anything but a complete, validated build is left alone: delete it (it
# holds only links and three small files) and run again.
hybrid() {
    if complete "$HYBRID_DIR" hybrid; then
        say "$HYBRID_DIR: built, skipping"
        return 0
    fi
    [ -z "$(ls -A "$HYBRID_DIR" 2>/dev/null)" ] \
        || fail "$HYBRID_DIR holds a different or partial hybrid build; delete that directory's contents (links and three small files, no weights) and run again"
    [ -w "$HYBRID_DIR" ] || fail "$HYBRID_DIR is not writable by uid $(id -u)"
    [ -f "$BUILDER" ] || fail "this image has no $BUILDER"
    say "building the hybrid checkpoint (links, no copies)"
    "$PY" "$BUILDER" --export "$EXPORT_DIR" --source "$OFFICIAL_DIR" \
        --exl3-modules "$HYBRID_EXL3_MODULES" --drop "$HYBRID_DROP" \
        --method "$HYBRID_METHOD" --out "$HYBRID_DIR" >/dev/null
    complete "$HYBRID_DIR" hybrid || fail "the hybrid build in $HYBRID_DIR does not have the validated bytes (config, index or quantization config); check the two downloads are at the pinned revisions"
    say "hybrid checkpoint: built"
}

models() {
    local used=0 avail
    if ! present "$EXPORT_DIR" export || ! present "$OFFICIAL_DIR" official; then
        complete "$EXPORT_DIR" export || used=$((used + $(du -s -BG "$EXPORT_DIR" | cut -dG -f1)))
        complete "$OFFICIAL_DIR" official || used=$((used + $(du -s -BG "$OFFICIAL_DIR" | cut -dG -f1)))
        avail="$(df -BG --output=avail "$OFFICIAL_DIR" | tail -1 | tr -dc '0-9')"
        if [ -n "$avail" ] && [ "$((avail + used))" -lt "$NEED_DISK_GB" ] \
            && ! { complete "$EXPORT_DIR" export && complete "$OFFICIAL_DIR" official; }; then
            fail "only ${avail} GiB free under the model directory and ${used} GiB already downloaded; the two checkpoints need ~${NEED_DISK_GB} GiB"
        fi
    fi
    if present "$EXPORT_DIR" export; then say "$EXPORT_REPO: complete, skipping"
    else fetch export "$EXPORT_REPO" "$EXPORT_REVISION" "$EXPORT_DIR"; fi
    if present "$OFFICIAL_DIR" official; then say "$OFFICIAL_REPO: complete, skipping"
    else fetch official "$OFFICIAL_REPO" "$OFFICIAL_REVISION" "$OFFICIAL_DIR"; fi
    hybrid
    say "checkpoints ready"
}

# claim <dir>: give <dir> to the serving user when root owns it and it is empty.
claim() {
    [ "$(stat -c %u "$1")" = 0 ] || return 0
    [ -z "$(ls -A "$1")" ] || return 0
    chown "$SERVE_UID:$SERVE_GID" "$1"
    say "$1: now owned by $SERVE_UID:$SERVE_GID"
}

serving_ids() {
    if [ -n "${HOST_UID:-}" ]; then
        SERVE_UID="$HOST_UID"
        SERVE_GID="${HOST_GID:-$HOST_UID}"
    else
        [ -d "$CHECKOUT" ] || fail "mount the checkout at $CHECKOUT, or set HOST_UID and HOST_GID"
        SERVE_UID="$(stat -c %u "$CHECKOUT")"
        SERVE_GID="$(stat -c %g "$CHECKOUT")"
    fi
    [[ "$SERVE_UID" =~ ^[0-9]+$ && "$SERVE_GID" =~ ^[0-9]+$ ]] \
        || fail "HOST_UID and HOST_GID must be numeric ids"
}

# as_serving_user CMD...: from root, run CMD as the serving user in place of
# this process. Returns when the serving user is root itself.
as_serving_user() {
    [ "$SERVE_UID" != 0 ] || return 0
    exec setpriv --reuid="$SERVE_UID" --regid="$SERVE_GID" --clear-groups -- "$@"
}

prepare() {
    local dir
    if [ "$(id -u)" = 0 ]; then
        serving_ids
        say "serving user: $SERVE_UID:$SERVE_GID"
        mkdir -p "$CACHE"
        find "$CACHE" -xdev \( ! -uid "$SERVE_UID" -o ! -gid "$SERVE_GID" \) \
            -exec chown -h "$SERVE_UID:$SERVE_GID" {} +
        for dir in "$EXPORT_DIR" "$OFFICIAL_DIR" "$HYBRID_DIR"; do
            claim "$dir"
        done
        as_serving_user /bin/bash "$0" models
    fi
    models
}

serve() {
    if [ "$(id -u)" = 0 ]; then
        SERVE_UID="$(stat -c %u "$CACHE")"
        SERVE_GID="$(stat -c %g "$CACHE")"
        as_serving_user /bin/bash /recipe/entry.sh
    fi
    exec /bin/bash /recipe/entry.sh
}

case "${1:-}" in
    prepare) prepare ;;
    models) models ;;
    serve) serve ;;
    check) [ "$#" = 3 ] || fail "usage: prepare.sh check DIR NAME"; complete "$2" "$3" ;;
    required) [ "$#" = 2 ] || fail "usage: prepare.sh required NAME"; required "$2" ;;
    *) fail "usage: prepare.sh prepare | models | serve | check DIR NAME | required NAME" ;;
esac
