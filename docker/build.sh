#!/usr/bin/env bash
# Build the release image without a container runtime or GPU: an assembled engine.
# Nothing of the engine is compiled here. /opt/assembly holds three pinned trees:
#   engine/   the fork's patch queue at RELEASE_QUEUE_COMMIT (tag QUEUE_TAG) with the
#             compiled extensions of the upstream vLLM wheel of RELEASE_UPSTREAM_COMMIT
#             (pinned by URL and sha256)
#   plugin/   the sm_80 plugin at RELEASE_PLUGIN_COMMIT (tag PLUGIN_TAG) with its natives
#             (native_build.py build --plugin; they must be the pinned build's bytes,
#             native-manifest.json, RELEASE_NATIVE_MANIFEST_SHA256) and its metadata
#   assembly.json  the release's record of them (docker/assembly.json, checked against
#             this build field by field)
# beside Python in /opt/python and the dependency lock (CONSTRAINTS) in /opt/venv; the
# image runs with PYTHONPATH=/opt/assembly/engine:/opt/assembly/plugin and the Env the
# plugin's launcher declares. Needs Linux x86_64, git, curl, uv, GNU tar, pigz,
# bubblewrap (the native build), a CUDA 13.x toolkit (the P2P probe) and, without
# BASE_OCI, crane.
# CLONE_FROM: a git repository to fetch the queue and plugin from instead of the public
# fork (the release tags, or the pinned commits when it has no such tags).
# NATIVE_DIR: a native_build.py build --plugin output to use instead of building it here.
# WHEEL: a local copy of the upstream wheel (checked by its pinned sha256).
# STEP=relocate|assemble resumes a STAGE this script built. Choose a new OUT each time.
# BASE_OCI permits assembly from a retained local base without registry access.
# IMAGE_LAYERED=0 keeps one runtime layer; IMAGE_PARALLEL_COMPRESSION=0 uses gzip.
# SERVE_SH is the launch the image carries as /opt/recipe/serve.sh, generated from
# the release settings; default this checkout's serve.sh.
set -euo pipefail
# The assembly's trees are Git checkouts made with umask 002; the image holds their modes.
umask 002
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRANE="${CRANE:-crane}"
UV="${UV:-uv}"
BUILD_CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
STAGE="${STAGE:-/var/tmp/cmp170hx-image/stage}"
OUT="${OUT:-$HERE/out}"
BASE="nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87"
VLLM_REPO="https://github.com/Morrowmake/vllm-cmp170hx.git"
RELEASE_UPSTREAM_COMMIT='330c5bffeeb48725c0502a351e8ead2b54bd81eb'
RELEASE_QUEUE_COMMIT='0af5991c009e0c3ff4181f026a8e7dffe1370b45'
RELEASE_PLUGIN_COMMIT='0b34823c39ba0fd5c0d4690887882b4cfa69034e'
RELEASE_NATIVE_BUILD='native-6a7c62b43c-t2.14.0'
QUEUE_TAG="dsv41-flash-recipe-1.0.1-queue"
PLUGIN_TAG="dsv41-flash-recipe-1.0.1-plugin"
# The upstream nightly wheel of RELEASE_UPSTREAM_COMMIT: its compiled extensions.
WHEEL_FILE="vllm-0.32.1rc1.dev515+g330c5bffe-cp38-abi3-manylinux_2_28_x86_64.whl"
WHEEL_URL="https://wheels.vllm.ai/330c5bffeeb48725c0502a351e8ead2b54bd81eb/vllm-0.32.1rc1.dev515%2Bg330c5bffe-cp38-abi3-manylinux_2_28_x86_64.whl"
WHEEL_SHA256="91ab870c88b285645b74f9e83ee680ff56ec22a5f682b0c7c957c5f403675245"
# sha256 of the validated native build's manifest.json (docker/native-manifest.json).
RELEASE_NATIVE_MANIFEST_SHA256='59c9a4aee6024bd5772b46bab823c177f8977bdcc83b8e9f19d0cc3c01c06a3f'
NATIVE_MANIFEST="$HERE/native-manifest.json"
RECORD="$HERE/assembly.json"
SERVE_SH="${SERVE_SH:-$HERE/../serve.sh}"
PYTHON_VERSION="3.12"
PYTHON_INSTALL_VERSION="3.12.14"
CONSTRAINTS="${CONSTRAINTS:-$HERE/CONSTRAINTS}"
SRC_GIT="$STAGE/src.git"
FILES="$STAGE/assembly-files.json"
export UV_CONSTRAINT="$CONSTRAINTS" CUDA_VISIBLE_DEVICES=""
log_features() { printf "[image] layered=%s parallel_compression=%s\n" "${IMAGE_LAYERED:-1}" "${IMAGE_PARALLEL_COMPRESSION:-1}"; }
# The image tag names all three pins: engine-<upstream 10>-<queue 10>-<plugin 10>.
TAG="$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); from image_identity import engine_tag; print(engine_tag(*sys.argv[2:]))' \
    "$HERE" "$RELEASE_QUEUE_COMMIT" "$RELEASE_UPSTREAM_COMMIT" "$RELEASE_PLUGIN_COMMIT")"
PIP=(--extra-index-url https://flashinfer.ai/whl/)
export UV_LINK_MODE=copy PYTHONDONTWRITEBYTECODE=1 UV_NO_CONFIG=1
# git records an identity in reflogs; keep the build free of the builder's
export GIT_AUTHOR_NAME=Morrowmake GIT_AUTHOR_EMAIL=noreply@github.com
export GIT_COMMITTER_NAME=Morrowmake GIT_COMMITTER_EMAIL=noreply@github.com
export GIT_CONFIG_GLOBAL=/dev/null
mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
# The launch the image carries: a generated serve.sh, never an empty or missing file.
require_serve() {
    [ -s "$SERVE_SH" ] || { echo "SERVE_SH=$SERVE_SH: no launch to bake (generate it from the release settings)" >&2; exit 1; }
    head -1 "$SERVE_SH" | grep -q '^#!/bin/bash' || { echo "SERVE_SH=$SERVE_SH is not a bash launcher" >&2; exit 1; }
}
base_oci() {
    BASE_LAYOUT="${BASE_OCI:-$OUT/base-oci}"
    if [ -z "${BASE_OCI:-}" ] && [ ! -e "$BASE_LAYOUT/index.json" ]; then
        "$CRANE" pull --platform linux/amd64 --format oci "$BASE" "$BASE_LAYOUT"
    fi
}

# One pinned commit into $SRC_GIT: by its release tag (TAG empty: by the commit itself),
# or, from a CLONE_FROM without that tag, by the commit. The commit fetched must be the pin.
fetch_pin() {
    local tag="$1" commit="$2" from="${CLONE_FROM:-$VLLM_REPO}" ref got
    ref="$commit"
    if [ -n "$tag" ]; then
        ref="refs/tags/$tag"
        if [ -n "${CLONE_FROM:-}" ] && ! git ls-remote --exit-code "$from" "$ref" >/dev/null; then
            ref="$commit"
        fi
    fi
    git -C "$SRC_GIT" fetch -q --no-tags --depth=1 "$from" "$ref"
    got="$(git -C "$SRC_GIT" rev-parse 'FETCH_HEAD^{commit}')"
    [ "$got" = "$commit" ] || { echo "${tag:-$commit} from $from is $got, not the pinned $commit" >&2; exit 1; }
    git -C "$SRC_GIT" update-ref "refs/pins/$commit" "$commit"
}

fetch_sources() {
    log "sources: queue $RELEASE_QUEUE_COMMIT, plugin $RELEASE_PLUGIN_COMMIT, upstream $RELEASE_UPSTREAM_COMMIT"
    git init -q --bare "$SRC_GIT"
    fetch_pin "$QUEUE_TAG" "$RELEASE_QUEUE_COMMIT"
    fetch_pin "$PLUGIN_TAG" "$RELEASE_PLUGIN_COMMIT"
    # The queue holds the upstream commit; its file list decides which wheel members are build products.
    fetch_pin "" "$RELEASE_UPSTREAM_COMMIT"
    # The queue may not change what the wheel's extensions are built from (they would be stale).
    local changed
    changed="$(git -C "$SRC_GIT" diff --name-only "$RELEASE_UPSTREAM_COMMIT" "$RELEASE_QUEUE_COMMIT" \
        -- CMakeLists.txt cmake csrc rust setup.py)"
    [ -z "$changed" ] || { echo "the queue changes native inputs of the upstream wheel:" >&2; echo "$changed" >&2; exit 1; }
    log "upstream wheel $WHEEL_FILE"
    mkdir -p "$STAGE/wheel"
    if [ -n "${WHEEL:-}" ]; then
        cp "$WHEEL" "$STAGE/wheel/$WHEEL_FILE"
    else
        curl -fsSL --retry 3 -o "$STAGE/wheel/$WHEEL_FILE" "$WHEEL_URL"
    fi
    echo "$WHEEL_SHA256  $STAGE/wheel/$WHEEL_FILE" | sha256sum -c --quiet
}

# The plugin's natives: NATIVE_DIR, or built here in the pinned base image.
natives() {
    if [ -n "${NATIVE_DIR:-}" ]; then
        [ -f "$NATIVE_DIR/manifest.json" ] || { echo "NATIVE_DIR=$NATIVE_DIR holds no manifest.json" >&2; exit 1; }
        return
    fi
    base_oci
    log "the plugin's natives (native_build.py build --plugin, in the pinned base image)"
    python3 "$HERE/native_build.py" build --plugin --repo "$SRC_GIT" --commit "$RELEASE_PLUGIN_COMMIT" \
        --base-oci "$BASE_LAYOUT" --constraints "$CONSTRAINTS" --work "$STAGE/native-work" \
        --out "$STAGE/native" --jobs "${NATIVE_JOBS:-8}" --uv "$(command -v "$UV")"
    NATIVE_DIR="$STAGE/native"
}

# A commit's files into DIR, as a checkout writes them (modes, links), without Git data.
checkout() {
    local commit="$1" dir="$2" index="$STAGE/checkout.index"
    mkdir -p "$dir"
    GIT_INDEX_FILE="$index" git --git-dir="$SRC_GIT" --work-tree="$dir" read-tree "$commit"
    GIT_INDEX_FILE="$index" git --git-dir="$SRC_GIT" --work-tree="$dir" checkout-index -a -f
    rm -f "$index"
}

build_tree() {
    local tree="$STAGE/opt/assembly"
    log "python $PYTHON_VERSION"
    UV_PYTHON_INSTALL_DIR="$STAGE/python-dl" "$UV" python install "$PYTHON_INSTALL_VERSION"
    mv "$STAGE"/python-dl/cpython-${PYTHON_VERSION}.*-linux-x86_64-gnu "$STAGE/opt/python"
    rm -rf "$STAGE/python-dl"
    "$UV" venv --relocatable --python "$STAGE/opt/python/bin/python$PYTHON_VERSION" "$STAGE/opt/venv"

    export VIRTUAL_ENV="$STAGE/opt/venv"
    log "locked dependencies (the engine and the plugin are not installed: PYTHONPATH names them)"
    "$UV" pip install "${PIP[@]}" -r "$CONSTRAINTS"
    "$UV" pip freeze > "$STAGE/opt/venv/requirements.lock.txt"
    unset VIRTUAL_ENV
    python3 - "$STAGE/opt" "$CONSTRAINTS" "$HERE" <<'LOCK'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from payload import check_lock
check_lock(Path(sys.argv[1]), Path(sys.argv[2]))
LOCK

    log "engine: the queue's files and the upstream wheel's build products"
    checkout "$RELEASE_QUEUE_COMMIT" "$tree/engine"
    python3 "$HERE/assembly.py" wheel --wheel "$STAGE/wheel/$WHEEL_FILE" --sha256 "$WHEEL_SHA256" \
        --tree "$tree/engine" --src "$SRC_GIT" --queue "$RELEASE_QUEUE_COMMIT" --upstream "$RELEASE_UPSTREAM_COMMIT"
    log "plugin: the plugin's files, its natives (the pinned build's bytes) and its metadata"
    checkout "$RELEASE_PLUGIN_COMMIT" "$tree/plugin"
    python3 "$HERE/assembly.py" plugin --tree "$tree/plugin" --native "$NATIVE_DIR" \
        --pinned-manifest "$NATIVE_MANIFEST" --manifest-sha256 "$RELEASE_NATIVE_MANIFEST_SHA256" \
        --src "$SRC_GIT" --plugin "$RELEASE_PLUGIN_COMMIT"
    python3 "$HERE/assembly.py" record --record "$RECORD" --opt "$STAGE/opt" --src "$SRC_GIT" \
        --constraints "$CONSTRAINTS" --pinned-manifest "$NATIVE_MANIFEST" \
        --manifest-sha256 "$RELEASE_NATIVE_MANIFEST_SHA256" --upstream "$RELEASE_UPSTREAM_COMMIT" \
        --queue "$RELEASE_QUEUE_COMMIT" --plugin "$RELEASE_PLUGIN_COMMIT" --native-build "$RELEASE_NATIVE_BUILD" \
        --wheel-file "$WHEEL_FILE" --wheel-url "$WHEEL_URL" --wheel-sha256 "$WHEEL_SHA256"
    install -m 0664 "$RECORD" "$tree/assembly.json"
    python3 "$HERE/assembly.py" files --opt "$STAGE/opt" --src "$SRC_GIT" --queue "$RELEASE_QUEUE_COMMIT" \
        --plugin "$RELEASE_PLUGIN_COMMIT" --out "$FILES"

    log "verify imports (no GPU)"
    (cd /tmp && PYTHONPATH="$tree/engine:$tree/plugin" CUDA_VISIBLE_DEVICES= \
        "$STAGE/opt/venv/bin/python" - "$tree/engine" "$tree/plugin" <<'PY') | tee "$OUT/verify.txt"
import importlib, importlib.metadata as metadata, json, os, sys
engine, plugin = sys.argv[1:3]
import torch
print(f"python      {sys.version.split()[0]}")
print(f"torch       {torch.__version__} (cuda {torch.version.cuda})")
found = metadata.distribution("vllm-sm80")
if os.path.realpath(str(found.locate_file(""))) != os.path.realpath(plugin):
    raise SystemExit(f"vllm-sm80 metadata found in {found.locate_file('')}, not the assembly's plugin")
points = {(e.group, e.value) for e in found.entry_points}
for group in ("vllm.general_plugins", "vllm.platform_plugins"):
    if not any(g == group and v.startswith("vllm_sm80.plugin:") for g, v in points):
        raise SystemExit(f"vllm-sm80 declares no {group} entry point")
natives = json.load(open(os.path.join(os.path.dirname(engine), "assembly.json")))["natives"]
for rel in sorted(natives):
    torch.ops.load_library(os.path.join(plugin, rel))
    print(f"native      {rel} loaded")
assert not torch.cuda.is_initialized()
import vllm
if not os.path.realpath(vllm.__file__).startswith(os.path.realpath(engine) + os.sep):
    raise SystemExit(f"vllm imported from {vllm.__file__}, not the assembly's engine")
print(f"vllm        {vllm.__version__}")
import vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch, vllm._custom_ops  # noqa: F401
for mod in ("triton", "tilelang", "flashinfer", "humming"):
    m = importlib.import_module(mod)
    print(f"{mod:<11} {getattr(m, '__version__', 'ok')}")
import vllm.plugins
vllm.plugins.load_general_plugins()
if not importlib.import_module("vllm_sm80.plugin").is_registered():
    raise SystemExit("general plugins loaded; the sm_80 plugin did not register")
importlib.import_module("vllm_sm80.guard").check_plugin_loaded()
print("sm80 plugin  registered")
PY
}

relocate() {
    log "relocate $STAGE/opt -> /opt"
    find "$STAGE/opt" -name __pycache__ -type d -prune -exec rm -rf {} +
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
    grep -rlI -w -F "$(hostname -s)" "$STAGE/opt" | head -20 > "$OUT/hostname-hits.txt" || true
    [ "$bad" = 0 ] || { echo "relocate: leftover host paths (above)" >&2; exit 1; }
}

layer() {
    require_serve
    # /opt/assembly is the assembly's files exactly, as built (relocation leaves it alone)
    python3 "$HERE/assembly.py" check --files "$FILES" --opt "$STAGE/opt"
    [ ! -e "$OUT/payload" ] || { echo "Choose a new OUT (payload already exists)" >&2; exit 1; }
    python3 "$HERE/payload.py" "$STAGE/opt" "$OUT/payload" "$CONSTRAINTS"
    # /opt/image-tools: the engine's licence, the notices and the P2P probe; then the mount points
    local tools="$OUT/payload/engine/opt/image-tools"
    mkdir -p "$tools"
    install -m 0644 "$HERE/THIRD_PARTY_NOTICES" "$tools/THIRD_PARTY_NOTICES"
    install -m 0644 "$STAGE/opt/assembly/engine/LICENSE" "$tools/LICENSE"
    for mount in recipe cache models hf-hub dev/shm proc tmp; do
        mkdir -p "$OUT/payload/engine/$mount"
    done
    chmod 1777 "$OUT/payload/engine/tmp"
    python3 "$HERE/p2p_build.py" "$HERE/../p2p_probe.cu" "$tools/p2p" --nvcc "$BUILD_CUDA_HOME/bin/nvcc"
    mkdir -p "$OUT/payload/engine/opt/recipe"
    install -m 0755 "$SERVE_SH" "$OUT/payload/engine/opt/recipe/serve.sh"
    sha256sum "$OUT/payload/engine/opt/recipe/serve.sh" | cut -d' ' -f1 > "$OUT/serve-sh.sha256"
    log "baked /opt/recipe/serve.sh sha256 $(cat "$OUT/serve-sh.sha256") from $SERVE_SH"
}

assemble() {
    base_oci
    local flags=()
    [ "${IMAGE_LAYERED:-1}" = 1 ] || flags+=(--no-layers)
    [ "${IMAGE_PARALLEL_COMPRESSION:-1}" = 1 ] || flags+=(--no-parallel)
    python3 "$HERE/assembly.py" oci --opt "$STAGE/opt" --base "$BASE_LAYOUT" --payload "$OUT/payload" \
        --out "$OUT/oci" --tag "$TAG" --threads "${IMAGE_THREADS:-16}" "${flags[@]}"
    python3 "$HERE/assembly.py" check --files "$FILES" --oci "$OUT/oci"
    python3 "$HERE/native_build.py" check-oci --oci "$OUT/oci" --manifest "$NATIVE_MANIFEST" \
        --prefix /opt/assembly/plugin | tee "$OUT/native-check.json"
    log "local OCI $(cat "$OUT/digest.txt") complete (not pushed)"
}

log_features

case "${STEP:-all}" in
    all)
        require_serve
        rm -rf "$STAGE"; mkdir -p "$STAGE/opt"
        fetch_sources; natives; build_tree; relocate; layer; assemble ;;
    relocate) relocate; layer; assemble ;;
    assemble) layer; assemble ;;
    *) echo "unknown STEP" >&2; exit 2 ;;
esac
