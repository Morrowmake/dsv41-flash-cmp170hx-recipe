#!/usr/bin/env bash
# Build an OCI image without a container runtime or GPU. Stable Python,
# dependency and native-library layers precede the editable engine layer.
# Needs Linux x86_64, uv, GNU tar, pigz, crane and a CUDA 13.x toolkit.
# STEP=assemble reuses a finalized, audited STAGE. Choose a new OUT each time.
# BASE_OCI permits assembly from a retained local base without registry access.
# IMAGE_LAYERED=0 keeps one runtime layer; IMAGE_PARALLEL_COMPRESSION=0 uses gzip.
# IMAGE_CACHE_SEED_DIR optionally supplies a GPU-generated, matching cache bundle.
# NATIVE_DIR is native_build.py's output for VLLM_COMMIT (our compiled extensions,
# built once in BASE and validated natively from those exact files); the image
# packs them by hash and never compiles them. NATIVE_MANIFEST_SHA256 overrides the pin.
# SERVE_SH is the launch the image carries as /opt/recipe/serve.sh, generated from
# the release settings; default this checkout's serve.sh.
set -euo pipefail
umask 022
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRANE="${CRANE:-crane}"
UV="${UV:-uv}"
BUILD_CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
STAGE="${STAGE:-/var/tmp/cmp170hx-image/stage}"
OUT="${OUT:-$HERE/out}"
BASE="nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87"
VLLM_REPO="https://github.com/Morrowmake/vllm-cmp170hx.git"
VLLM_BRANCH="${VLLM_BRANCH:-ampere}"
RELEASE_VLLM_COMMIT='a0c57b52376efd67c57735bc949d4bafa22ae29e'
VLLM_COMMIT="${VLLM_COMMIT:-$RELEASE_VLLM_COMMIT}"
# Upstream compiled extensions: the nightly wheel of the pin's upstream base
# 4e0681ed56 itself. Our own extensions come from NATIVE_DIR (above).
VLLM_WHEEL_COMMIT="4e0681ed5654c1d8afb3c104d4381afd08529de8"
# Upstream release tag the pin's base descends from (sets the version string).
VERSION_TAG="v0.32.0rc0"
# sha256 of the validated native_build.py manifest.json for RELEASE_VLLM_COMMIT.
RELEASE_NATIVE_MANIFEST_SHA256='aa940260d8787ab9d596892693d730ccf6e82855dd2f17a7116c04a31e1c2717'
NATIVE_MANIFEST_SHA256="${NATIVE_MANIFEST_SHA256:-$RELEASE_NATIVE_MANIFEST_SHA256}"
SERVE_SH="${SERVE_SH:-$HERE/../serve.sh}"
UPSTREAM_REPO="https://github.com/vllm-project/vllm.git"
PYTHON_VERSION="3.12"
PYTHON_INSTALL_VERSION="3.12.14"
CONSTRAINTS="${CONSTRAINTS:-$HERE/CONSTRAINTS}"
export UV_CONSTRAINT="$CONSTRAINTS" CUDA_VISIBLE_DEVICES=""
log_features() { printf "[image] layered=%s parallel_compression=%s cache_seed=%s\n" "${IMAGE_LAYERED:-1}" "${IMAGE_PARALLEL_COMPRESSION:-1}" "$([ -n "${IMAGE_CACHE_SEED_DIR:-}" ] && echo included || echo disabled)"; }
TAG="$(python3 "$HERE/image_identity.py" tag "$VLLM_COMMIT")"
PIP=(--extra-index-url https://flashinfer.ai/whl/)
export UV_LINK_MODE=copy PYTHONDONTWRITEBYTECODE=1 UV_NO_CONFIG=1
# git records an identity in reflogs; keep the image free of the builder's
export GIT_AUTHOR_NAME=Morrowmake GIT_AUTHOR_EMAIL=noreply@github.com
export GIT_COMMITTER_NAME=Morrowmake GIT_COMMITTER_EMAIL=noreply@github.com
export GIT_CONFIG_GLOBAL=/dev/null
mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
require_native() {
    [ -n "$NATIVE_MANIFEST_SHA256" ] || { echo "No validated native manifest pinned (RELEASE_NATIVE_MANIFEST_SHA256 or NATIVE_MANIFEST_SHA256)" >&2; exit 1; }
    [ -f "${NATIVE_DIR:-}/manifest.json" ] || { echo "Set NATIVE_DIR to the output of: python3 docker/native_build.py build --repo <fork clone> --commit $VLLM_COMMIT --base-oci <crane pull of BASE> --work <new dir> --out <new dir>" >&2; exit 1; }
}
# The launch the image carries: a generated serve.sh, never an empty or missing file.
require_serve() {
    [ -s "$SERVE_SH" ] || { echo "SERVE_SH=$SERVE_SH: no launch to bake (generate it from the release settings)" >&2; exit 1; }
    head -1 "$SERVE_SH" | grep -q '^#!/bin/bash' || { echo "SERVE_SH=$SERVE_SH is not a bash launcher" >&2; exit 1; }
}

clone_source() {
    log "engine source @ $VLLM_COMMIT"
    # Local clone optimisations copy unrelated objects, even with single-branch.
    # Use transport negotiation and retain only the explicit version tag.
    git clone --no-local --no-tags --single-branch --branch "${CLONE_BRANCH:-$VLLM_BRANCH}" \
        "${CLONE_FROM:-$VLLM_REPO}" "$STAGE/opt/vllm-src"
    git -C "$STAGE/opt/vllm-src" remote set-url origin "$VLLM_REPO"
    git -C "$STAGE/opt/vllm-src" checkout -B "$VLLM_BRANCH" "$VLLM_COMMIT"
    if [ "${CLONE_BRANCH:-$VLLM_BRANCH}" != "$VLLM_BRANCH" ]; then
        git -C "$STAGE/opt/vllm-src" branch -D "$CLONE_BRANCH"
        git -C "$STAGE/opt/vllm-src" update-ref -d "refs/remotes/origin/$CLONE_BRANCH"
        git -C "$STAGE/opt/vllm-src" remote set-branches origin "$VLLM_BRANCH"
    fi
    git -C "$STAGE/opt/vllm-src" update-ref "refs/remotes/origin/$VLLM_BRANCH" "$VLLM_COMMIT"
    git -C "$STAGE/opt/vllm-src" symbolic-ref refs/remotes/origin/HEAD "refs/remotes/origin/$VLLM_BRANCH"
    # The installed version string comes from the nearest upstream release tag
    # (setuptools-scm); make sure the clone has it even when CLONE_FROM does not.
    if ! git -C "$STAGE/opt/vllm-src" rev-parse -q --verify "refs/tags/$VERSION_TAG" >/dev/null; then
        git -C "$STAGE/opt/vllm-src" fetch --no-tags "$UPSTREAM_REPO" \
            "refs/tags/$VERSION_TAG:refs/tags/$VERSION_TAG"
    fi
    git -C "$STAGE/opt/vllm-src" merge-base --is-ancestor "$VERSION_TAG" HEAD \
        || { echo "$VERSION_TAG is not an ancestor of $VLLM_COMMIT" >&2; exit 1; }
    git -C "$STAGE/opt/vllm-src" submodule update --init --recursive --depth 1
    # A pin older than the branch tip must not retain the later objects.
    git -C "$STAGE/opt/vllm-src" reflog expire --expire=now --all
    git -C "$STAGE/opt/vllm-src" repack -ad
    git -C "$STAGE/opt/vllm-src" prune --expire=now
}

build_tree() {
    rm -rf "$STAGE"; mkdir -p "$STAGE/opt"
    log "python $PYTHON_VERSION"
    UV_PYTHON_INSTALL_DIR="$STAGE/python-dl" "$UV" python install "$PYTHON_INSTALL_VERSION"
    mv "$STAGE"/python-dl/cpython-${PYTHON_VERSION}.*-linux-x86_64-gnu "$STAGE/opt/python"
    rm -rf "$STAGE/python-dl"
    "$UV" venv --relocatable --python "$STAGE/opt/python/bin/python$PYTHON_VERSION" "$STAGE/opt/venv"
    clone_source

    export VIRTUAL_ENV="$STAGE/opt/venv"
    log "locked dependencies"
    "$UV" pip install "${PIP[@]}" -r "$CONSTRAINTS"
    log "vllm (upstream precompiled extensions)"
    VLLM_BUILD_AMPERE_MARLIN=0 VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT="$VLLM_WHEEL_COMMIT" \
        VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 CUDA_HOME="$BUILD_CUDA_HOME" \
        "$UV" pip install --no-deps -e "$STAGE/opt/vllm-src"
    log "the fork's plugin entry points (plugins/sm80; vLLM refuses to start without them)"
    "$UV" pip install --no-deps "$STAGE/opt/vllm-src/plugins/sm80"
    "$UV" pip freeze > "$STAGE/opt/venv/requirements.lock.txt"
    python3 - "$STAGE/opt" "$CONSTRAINTS" "$HERE" <<'LOCK'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from payload import check_lock
check_lock(Path(sys.argv[1]), Path(sys.argv[2]))
LOCK

    log "our compiled extensions (packed by hash from NATIVE_DIR; not compiled here)"
    require_native
    python3 "$HERE/native_build.py" pack --native "$NATIVE_DIR" --venv-check \
        --manifest-sha256 "$NATIVE_MANIFEST_SHA256" --commit "$VLLM_COMMIT" --opt "$STAGE/opt"
    log "verify imports (no GPU)"
    NATIVE_MANIFEST_SHIPPED="$STAGE/opt/venv/.native-manifest.json" \
    CUDA_VISIBLE_DEVICES= "$STAGE/opt/venv/bin/python" - <<'PY' | tee "$OUT/verify.txt"
import importlib, sys, torch
print(f"python      {sys.version.split()[0]}")
print(f"torch       {torch.__version__} (cuda {torch.version.cuda})")
import vllm
print(f"vllm        {vllm.__version__}")
import vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch, vllm._custom_ops  # noqa: F401
for mod in ("triton", "tilelang", "flashinfer", "humming"):
    m = importlib.import_module(mod)
    print(f"{mod:<11} {getattr(m, '__version__', 'ok')}")
from vllm.sm80.attention import triton_mla_sparse                      # noqa: F401
from vllm.sm80.kernels.attention import triton_mqa_logits, triton_e4m3  # noqa: F401
from vllm.sm80.distributed import host_shm_all_reduce                  # noqa: F401
import vllm.plugins
vllm.plugins.load_general_plugins()
from vllm.sm80 import plugin
from vllm.sm80.guard import check_plugin_loaded
if not plugin.is_registered():
    raise SystemExit("the fork's plugin sm80 is not registered")
check_plugin_loaded()
print("sm80 plugin  registered")
import json, os
artefacts = json.load(open(os.environ["NATIVE_MANIFEST_SHIPPED"]))["artefacts"]
if "vllm/_exl3_C.abi3.so" not in artefacts:
    raise SystemExit("the native manifest ships no vllm/_exl3_C.abi3.so; this model's EXL3 experts need it")
importlib.import_module("vllm._exl3_C")
print("sm_80 EXL3 extension loaded")
if "vllm/_ampere_marlin_C.abi3.so" in artefacts:
    from vllm.sm80.kernels.marlin_moe.extension import require_extension
    require_extension()
    print("sm_80 patch modules and optional Marlin registrations ok")
else:
    print("sm_80 patch modules ok; optional Marlin not shipped")
PY
    unset VIRTUAL_ENV
}

relocate() {
    log "relocate $STAGE/opt -> /opt"
    find "$STAGE/opt" -name __pycache__ -type d -prune -exec rm -rf {} +
    rm -rf "$STAGE/opt/vllm-src/.git/logs"
    find "$STAGE/opt/vllm-src" -path '*/.git/modules/*/logs' -type d -prune -exec rm -rf {} + 2>/dev/null || true
    # text files that name the staging path (uv writes the interpreter's
    # install location into sysconfig, so that one maps to /opt/python)
    grep -rlI --null -F "$STAGE/python-dl/" "$STAGE/opt" \
        | xargs -0 -r sed -i -E "s#$STAGE/python-dl/cpython-[^/\"' ]+#/opt/python#g" || true
    grep -rlI --null -F "$STAGE" "$STAGE/opt" | xargs -0 -r sed -i "s#$STAGE/opt#/opt#g" || true
    # symlinks that point into the staging path
    find "$STAGE/opt" -type l -print0 | while IFS= read -r -d '' l; do
        t="$(readlink "$l")"
        case "$t" in "$STAGE"/*) ln -sfn "${t#"$STAGE"}" "$l" ;; esac
    done
    # nothing may still name the staging path, a home directory or this host
    local bad=0
    if grep -rl -F -a "$STAGE" "$STAGE/opt" | head -5 | grep .; then bad=1; fi
    if grep -rl -a -F "$HOME/" "$STAGE/opt" | head -5 | grep .; then bad=1; fi
    if find "$STAGE/opt" -type l -lname '/home/*' | grep .; then bad=1; fi
    # the host name, reported for review (it can occur inside unrelated binaries)
    grep -rlI -w -F "$(hostname -s)" "$STAGE/opt" --exclude-dir=.git | head -20 > "$OUT/hostname-hits.txt" || true
    [ "$bad" = 0 ] || { echo "relocate: leftover host paths (above)" >&2; exit 1; }
}

finalize_tree() {
    python3 "$HERE/runtime_tree.py" finalize "$STAGE/opt" "$VLLM_COMMIT" \
        --forbid "$HOME/" --forbid "$STAGE/"
}

check_native_tree() {
    python3 "$HERE/native_build.py" check-tree --manifest "$STAGE/opt/venv/.native-manifest.json" \
        --manifest-sha256 "$NATIVE_MANIFEST_SHA256" --commit "$VLLM_COMMIT" --tree "$STAGE/opt/vllm-src"
}
layer() {
    require_serve
    check_native_tree
    python3 "$HERE/runtime_tree.py" check "$STAGE/opt" "$VLLM_COMMIT" --forbid "$HOME/" --forbid "$STAGE/"
    [ ! -e "$OUT/payload" ] || { echo "Choose a new OUT (payload already exists)" >&2; exit 1; }
    python3 "$HERE/payload.py" "$STAGE/opt" "$OUT/payload" "$CONSTRAINTS"
    mkdir -p "$OUT/payload/engine/opt/image-tools" "$OUT/payload/engine/opt/recipe"
    cp "$HERE/cache_seed.py" "$HERE/rootfs_smoke.py" "$HERE/bwrap_targets.py" "$HERE/../LICENSE" "$HERE/THIRD_PARTY_NOTICES" "$OUT/payload/engine/opt/image-tools/"
    install -m 0755 "$SERVE_SH" "$OUT/payload/engine/opt/recipe/serve.sh"
    sha256sum "$OUT/payload/engine/opt/recipe/serve.sh" | cut -d' ' -f1 > "$OUT/serve-sh.sha256"
    log "baked /opt/recipe/serve.sh sha256 $(cat "$OUT/serve-sh.sha256") from $SERVE_SH"
    mkdir -p "$OUT/payload/engine/opt/cache-seed" "$OUT/payload/engine/recipe" \
        "$OUT/payload/engine/cache" "$OUT/payload/engine/dev/shm" \
        "$OUT/payload/engine/proc" "$OUT/payload/engine/tmp" \
        "$OUT/payload/engine/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw" \
        "$OUT/payload/engine/models/DeepSeek-V4.1-Flash-1017-engram" \
        "$OUT/payload/engine/models/DeepSeek-V4.1-Flash-1017-hybrid-exl3-fp8"
    chmod 1777 "$OUT/payload/engine/tmp"
    python3 "$HERE/p2p_build.py" "$HERE/../p2p_probe.cu" \
        "$OUT/payload/engine/opt/image-tools/p2p" --nvcc "$BUILD_CUDA_HOME/bin/nvcc"
    if [ -n "${IMAGE_CACHE_SEED_DIR:-}" ]; then
        python3 "$HERE/cache_seed.py" check "$IMAGE_CACHE_SEED_DIR" "$VLLM_COMMIT" "$STAGE/opt/venv/requirements.lock.txt"
        mkdir -p "$OUT/payload/cache-seed/opt/cache-seed"
        cp -a "$IMAGE_CACHE_SEED_DIR/." "$OUT/payload/cache-seed/opt/cache-seed/"
    fi
}

assemble() {
    local base_oci="${BASE_OCI:-$OUT/base-oci}"
    if [ -z "${BASE_OCI:-}" ]; then
        "$CRANE" pull --platform linux/amd64 --format oci "$BASE" "$base_oci"
    fi
    python3 "$HERE/oci_layers.py" "$base_oci" "$OUT/payload" "$OUT/oci" "$VLLM_COMMIT" "$TAG"
    python3 "$HERE/native_build.py" check-oci --oci "$OUT/oci" --require-shipped-manifest \
        --manifest "$STAGE/opt/venv/.native-manifest.json" | tee "$OUT/native-check.json"
    log "local OCI complete (not pushed)"
}

log_features

case "${STEP:-all}" in
    all) require_serve; build_tree; relocate; finalize_tree; layer; assemble ;;
    relocate) relocate; finalize_tree; layer; assemble ;;
    assemble) layer; assemble ;;
    *) echo "unknown STEP" >&2; exit 2 ;;
esac
