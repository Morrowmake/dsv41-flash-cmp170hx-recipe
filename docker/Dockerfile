# vLLM (Morrowmake/vllm-cmp170hx, branch ampere) for DeepSeek-V4.1-Flash
# on 4x NVIDIA CMP 170HX, with the sm_80 EXL3 and Marlin extensions.
#
# The engine is installed the way the recipe's native install does it: Python
# 3.12, torch 2.13.0 (CUDA 13.0 build), the fork checked out at the pinned
# commit and installed editable with upstream's precompiled extensions
# (VLLM_USE_PRECOMPILED=1), then the runtime extras the engine pins in
# requirements/cuda.txt (FlashInfer 0.7.0.post1, humming-kernels 0.1.16, TileLang
# 0.1.12). The pin sits on upstream 4e0681ed56, whose own nightly wheel
# supplies the upstream compiled extensions (VLLM_WHEEL_COMMIT). Our own compiled
# extensions are not compiled here: docker/native_build.py builds them once in
# this base image, native validation runs those exact files, and this build
# packs them by hash from docker/native/ (its output; NATIVE_MANIFEST_SHA256
# pins the manifest). Model weights are not in the image; mount them (see
# README.md in this folder for the run command).
#
# The CUDA 13.3.1 devel base matches the toolkit the release was validated
# with (Triton, TileLang, FlashInfer and the host-memory all-reduce extension
# compile against it at runtime, so nvcc and g++ must be present).
# NVIDIA_REQUIRE_CUDA is relaxed to cuda>=13.0 so the container starts on any
# driver 580 or newer, the same requirement as the recipe.
#
# Entry point: the vllm CLI. The image carries the release's launch as
# /opt/recipe/serve.sh (SERVE_SH, the generated serve.sh of this checkout by
# default); the recipe runs it through its docker/entry.sh, which checks peer
# access and runs the boot check first (container.env in this folder).
#
#   bash docker/build-docker.sh vllm-cmp170hx:engine-{{PIN}}
# The wrapper validates the final layer count; Dockerfile instructions cannot
# inspect the inherited base image manifest.

ARG BASE=nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87

ARG IMAGE_LAYERED=1

ARG VLLM_COMMIT={{PIN}}
ARG NATIVE_MANIFEST_SHA256=aa940260d8787ab9d596892693d730ccf6e82855dd2f17a7116c04a31e1c2717

FROM ${BASE} AS build
ARG PYTHON_VERSION=3.12
ARG PYTHON_INSTALL_VERSION=3.12.14
ARG UV_VERSION=0.12.11
COPY docker/CONSTRAINTS /tmp/CONSTRAINTS
ENV UV_CONSTRAINT=/tmp/CONSTRAINTS
ENV UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python-dl \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_INSTALL_DIR=/usr/local/bin sh
RUN uv python install "${PYTHON_INSTALL_VERSION}" \
 && mv /opt/python-dl/cpython-${PYTHON_VERSION}.*-linux-x86_64-gnu /opt/python \
 && rm -rf /opt/python-dl \
 && grep -rlI --null /opt/python-dl/ /opt/python \
    | xargs -0 -r sed -i -E "s#/opt/python-dl/cpython-[^/\"' ]+#/opt/python#g" \
 && uv venv --relocatable --python /opt/python/bin/python${PYTHON_VERSION} /opt/venv
ENV VIRTUAL_ENV=/opt/venv
# Dependencies are independent of the engine pin and install from the lock.
RUN uv pip install --extra-index-url https://flashinfer.ai/whl/ -r /tmp/CONSTRAINTS
ARG VLLM_REPO=https://github.com/Morrowmake/vllm-cmp170hx.git
ARG VLLM_BRANCH=ampere
ARG VLLM_COMMIT
ARG VLLM_WHEEL_COMMIT=4e0681ed5654c1d8afb3c104d4381afd08529de8
# Full transport clone: version construction and wheel lookup need history.
RUN git clone --no-local --no-tags --single-branch --branch "${VLLM_BRANCH}" "${VLLM_REPO}" /opt/vllm-src \
 && git -C /opt/vllm-src checkout -B "${VLLM_BRANCH}" "${VLLM_COMMIT}" \
 && git -C /opt/vllm-src update-ref "refs/remotes/origin/${VLLM_BRANCH}" "${VLLM_COMMIT}" \
 && git -C /opt/vllm-src fetch --no-tags https://github.com/vllm-project/vllm.git \
            refs/tags/v0.32.0rc0:refs/tags/v0.32.0rc0 \
 && git -C /opt/vllm-src merge-base --is-ancestor v0.32.0rc0 HEAD \
 && git -C /opt/vllm-src submodule update --init --recursive --depth 1 \
 && git -C /opt/vllm-src reflog expire --expire=now --all \
 && git -C /opt/vllm-src repack -ad \
 && git -C /opt/vllm-src prune --expire=now
RUN VLLM_BUILD_AMPERE_MARLIN=0 VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_WHEEL_COMMIT}" \
    VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
    uv pip install --no-deps -e /opt/vllm-src \
 && uv pip install --no-deps /opt/vllm-src/plugins/sm80 \
 && uv pip freeze > /opt/venv/requirements.lock.txt
COPY docker/runtime_tree.py docker/payload.py docker/cache_seed.py docker/rootfs_smoke.py docker/bwrap_targets.py docker/p2p_build.py docker/native_build.py docker/THIRD_PARTY_NOTICES p2p_probe.cu LICENSE /tmp/
COPY docker/native/ /tmp/native/
ARG NATIVE_MANIFEST_SHA256
RUN /opt/venv/bin/python /tmp/native_build.py pack --native /tmp/native --venv-check \
    --manifest-sha256 "${NATIVE_MANIFEST_SHA256}" --commit "${VLLM_COMMIT}" --opt /opt \
 && rm -rf /tmp/native
# The EXL3 extension is required (the routed experts); Marlin is checked when the manifest ships it.
RUN CUDA_VISIBLE_DEVICES= /opt/venv/bin/python -c "import json, importlib; artefacts = json.load(open('/opt/venv/.native-manifest.json'))['artefacts']; assert 'vllm/_exl3_C.abi3.so' in artefacts, 'no EXL3 extension shipped'; importlib.import_module('vllm._exl3_C'); shipped = 'vllm/_ampere_marlin_C.abi3.so' in artefacts; from vllm.sm80.kernels.marlin_moe.extension import require_extension; require_extension() if shipped else print('optional Marlin: not shipped')"
RUN CUDA_VISIBLE_DEVICES= /opt/venv/bin/python -c "import torch, vllm, vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch, vllm._custom_ops, triton, tilelang, flashinfer, humming; from vllm.sm80.attention import triton_mla_sparse; from vllm.sm80.kernels.attention import triton_mqa_logits, triton_e4m3; from vllm.sm80.distributed import host_shm_all_reduce; import vllm.plugins; vllm.plugins.load_general_plugins(); from vllm.sm80 import plugin; from vllm.sm80.guard import check_plugin_loaded; assert plugin.is_registered(), 'sm80 plugin not registered'; check_plugin_loaded(); print(vllm.__version__, torch.__version__, torch.version.cuda)" \
 && find /opt/python /opt/venv /opt/vllm-src -name __pycache__ -type d -prune -exec rm -rf {} + \
 && /opt/venv/bin/python /tmp/runtime_tree.py finalize /opt "${VLLM_COMMIT}" \
 && /opt/venv/bin/python /tmp/runtime_tree.py check /opt "${VLLM_COMMIT}"

ARG IMAGE_P2P_PREBUILD=1
RUN IMAGE_P2P_PREBUILD=${IMAGE_P2P_PREBUILD} /opt/venv/bin/python /tmp/p2p_build.py \
    /tmp/p2p_probe.cu /tmp/image-p2p
# The release's launch, generated from its settings.
ARG SERVE_SH=serve.sh
COPY ${SERVE_SH} /tmp/serve.sh
RUN /opt/venv/bin/python /tmp/payload.py /opt /payload /tmp/CONSTRAINTS \
 && mkdir -p /payload/engine/opt/image-tools /payload/engine/opt/recipe \
 && cp -a /tmp/image-p2p /payload/engine/opt/image-tools/p2p \
 && cp /tmp/cache_seed.py /tmp/rootfs_smoke.py /tmp/bwrap_targets.py /tmp/LICENSE /tmp/THIRD_PARTY_NOTICES /payload/engine/opt/image-tools/ \
 && head -1 /tmp/serve.sh | grep -q '^#!/bin/bash' \
 && install -m 0755 /tmp/serve.sh /payload/engine/opt/recipe/serve.sh \
 && mkdir -p /payload/engine/opt/cache-seed /payload/engine/recipe /payload/engine/cache \
    /payload/engine/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw \
    /payload/engine/models/DeepSeek-V4.1-Flash-1017-engram \
    /payload/engine/models/DeepSeek-V4.1-Flash-1017-hybrid-exl3-fp8

FROM ${BASE} AS runtime-1
COPY --from=build /payload/python/ /
COPY --from=build /payload/dependencies/ /
COPY --from=build /payload/native/ /
COPY --from=build /payload/engine/ /

FROM ${BASE} AS runtime-0
COPY --from=build /opt/ /opt/
COPY --from=build /payload/engine/opt/image-tools/ /opt/image-tools/
COPY --from=build /payload/engine/opt/recipe/ /opt/recipe/

FROM runtime-${IMAGE_LAYERED}
ARG VLLM_COMMIT
ARG IMAGE_TAG=engine-{{PIN}}
ENV PATH=/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    VIRTUAL_ENV=/opt/venv \
    CUDA_HOME=/usr/local/cuda \
    NVIDIA_REQUIRE_CUDA=cuda>=13.0
LABEL org.opencontainers.image.title="vllm-cmp170hx" \
      org.opencontainers.image.description="vLLM fork for DeepSeek-V4.1-Flash on 4x NVIDIA CMP 170HX (sm_80) with the sm_80 EXL3 and Marlin extensions; weights not included" \
      org.opencontainers.image.source="https://github.com/Morrowmake/vllm-cmp170hx" \
      org.opencontainers.image.revision="${VLLM_COMMIT}" \
      org.opencontainers.image.url="https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe" \
      org.opencontainers.image.version="${IMAGE_TAG}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.base.name="docker.io/nvidia/cuda:13.3.1-devel-ubuntu24.04" \
      org.opencontainers.image.base.digest="sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87"
RUN mkdir -p /opt/cache-seed /recipe /cache /dev/shm /proc /tmp \
    /models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw /models/DeepSeek-V4.1-Flash-1017-engram \
    /models/DeepSeek-V4.1-Flash-1017-hybrid-exl3-fp8 \
 && chown 0:0 /tmp && chmod 1777 /tmp
EXPOSE 8000
ENTRYPOINT ["/opt/venv/bin/vllm"]
