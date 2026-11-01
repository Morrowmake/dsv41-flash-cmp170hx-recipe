# Container image

The engine in a container: the only way this recipe runs. `./start.sh` and
`docker compose up` pull the image, start it and manage it for you (see the
[How to use this repo](../docs/how-to-use.md)); this page describes the
image, how to build it, and how to run it by hand with `docker run`.

## What the image contains

- **Engine:** an assembled engine in `/opt/assembly`, three pinned trees and
  nothing compiled in the image:
  - `engine/`: the [vLLM fork](https://github.com/Morrowmake/vllm-cmp170hx)'s patch
    queue at `e05403fdc3750880da4a5f4cd8c55c5fae3b3ae0` (tag
    `dsv41-flash-recipe-1.0.1-queue`), on upstream `330c5bffee`, with the compiled
    extensions of `330c5bffee`'s own nightly wheel, except `vllm._C_stable_libtorch`,
    rebuilt from the queue (it changes `csrc/quantization/marlin/marlin.cu`) the way
    upstream builds that wheel;
  - `plugin/`: the sm_80 plugin `vllm_sm80` at
    `0b34823c39ba0fd5c0d4690887882b4cfa69034e` (tag `dsv41-flash-recipe-1.0.1-plugin`)
    with its compiled extensions: the sm_80 EXL3 library (`vllm_sm80._exl3_C`,
    required: the routed experts run on it) and the Marlin library;
  - `assembly.json`: the pins, the wheel, the rebuilt engine library and the natives'
    sha256.

  Python 3.12 is `/opt/python`; `/opt/venv` holds the dependency lock (torch
  2.14.0, CUDA 13.0 build) and the engine installed from `/opt/assembly` without
  building it: `/opt/venv/bin/vllm`, vLLM's version and entry points, and
  `/opt/assembly/engine` and `/opt/assembly/plugin` on the venv's path, so
  `/opt/venv/bin/python` imports vLLM with no `PYTHONPATH` ([`venv_entry.py`](venv_entry.py)).
  The image's `PYTHONPATH` names the same two directories. The installed package
  list is in `/opt/venv/requirements.lock.txt`.
- **The release's launch:** `/opt/recipe/serve.sh`, generated from the
  settings this release was validated with: the engine command and every
  serving default, for the one layout (`LAYOUT=tp4`). The recipe runs it through
  [`docker/entry.sh`](entry.sh), which decides where the Engram tables go,
  checks peer access and runs the boot check first. Settings you give at run time override its defaults.
- **Base:** `nvidia/cuda:13.3.1-devel-ubuntu24.04`, pinned by digest. It is the
  devel image because Triton, TileLang, FlashInfer and the host-memory
  all-reduce compile kernels at run time and need `nvcc` and `g++`.
- **Entry point:** the `vllm` CLI (`/opt/venv/bin/python3 -m
  vllm.entrypoints.cli.main`). The recipe replaces it with `docker/entry.sh`.
- **Peer check:** a prebuilt sm_80 content probe, selected only when its source
  hash matches the mounted recipe. `P2P_PREBUILT=0` uses runtime compilation.
  The startup banner reports build, identity and check time, with transport,
  ordered-pair and size breakdowns on an uncached check. Content verification
  remains cached by driver, ordered GPU UUIDs, source hash and host boot.
- **No compiled kernel cache.** The first start compiles its kernels into the
  cache mount.
- **No weights.** The checkpoints are mounted from the host. The plugin tree
  includes the builder the recipe uses to make the hybrid checkpoint
  (`/opt/assembly/plugin/tools/sm80/build_hybrid_checkpoint.py`).

Engine image tag:

```
ghcr.io/morrowmake/vllm-cmp170hx:engine-330c5bffee-e05403fdc3-0b34823c39
```

The tag names all three pins (`engine-<upstream 10>-<queue 10>-<plugin 10>`), so
a release that changes only the plugin gets a new tag. The OCI revision is the
queue commit `e05403fdc3750880da4a5f4cd8c55c5fae3b3ae0`; labels name all three
pins and the native build. Image tags identify engine source, independently of
recipe release numbers. Publication requires validation and acceptance of the
exact image digest.

## Build it

[`build.sh`](build.sh) builds the image rootless and without a GPU or a Docker
daemon: it prepares `/opt` on the host, splits it into deterministic layers and
writes an OCI layout on the pinned base. Nothing of the engine is compiled into
the image; it assembles pinned inputs:

- **The queue and the plugin** are fetched from the public fork by their release
  tags (`dsv41-flash-recipe-1.0.1-queue`, `dsv41-flash-recipe-1.0.1-plugin`); each
  tag must name its pinned commit (`RELEASE_QUEUE_COMMIT`, `RELEASE_PLUGIN_COMMIT`).
  The trees are the commits' files exactly, without Git data.
- **The upstream wheel** of `RELEASE_UPSTREAM_COMMIT`, pinned by URL and sha256:
  its members under `vllm/` that neither the queue nor the upstream commit
  tracks (the compiled extensions and vendored trees) go into `engine/`, and its
  `vllm/_version.py` names the queue commit.
- **The engine library the queue makes stale.** [`native_inputs.py`](native_inputs.py)
  judges the queue's changes to the wheel's native inputs by the plugin's
  native-input rules (`vllm_sm80/launch/data/native_inputs.toml`): this queue
  changes an input of `vllm._C_stable_libtorch` alone. `docker/native_build.py
  build --no-marlin --stable` rebuilds that library from the queue the way
  upstream builds its x86_64 CUDA wheel: in upstream's wheel builder image
  (`pytorch/manylinux2_28-builder:cuda13.0-78e737ad…`, pinned by manifest digest;
  `BUILDER_OCI` names a local OCI layout of it, else `crane` pulls it), with its
  compilers, upstream's architecture list and the source at `/workspace`; the
  CMake FetchContent sources are cloned at their declared tags (`FETCHCONTENT`
  names a directory of them). The rebuilt library replaces the wheel's copy. The
  build must be the validated one in every recorded field (source trees,
  toolchain, flags, FetchContent commits, torch): its `manifest.json` ships as
  [`engine-native-manifest.json`](engine-native-manifest.json), pinned by
  `RELEASE_ENGINE_NATIVE_MANIFEST_SHA256` and named by `RELEASE_ENGINE_NATIVE_BUILD`.
  Its bytes differ from build to build (CUDA 13 numbers anonymous-namespace
  symbols per build); its kernels do not. A queue that makes any other wheel
  library stale is refused. The build takes about half an hour on 64 threads
  (`ENGINE_NATIVE_JOBS`, about 1 GiB of RAM each).
- **The plugin's compiled extensions** are built by `docker/native_build.py build
  --plugin` inside the pinned CUDA base image (rootless, with bubblewrap; fixed
  paths), from the plugin's `csrc/` alone. They must be the bytes of the validated
  build, whose `manifest.json` ships as [`native-manifest.json`](native-manifest.json)
  and is pinned by `RELEASE_NATIVE_MANIFEST_SHA256`: the build checks every file's
  sha256 against it and reports any other manifest field that differs (where and
  when it was built aside). `NATIVE_DIR` uses an earlier `native_build.py build
  --plugin` output instead.

[`assembly.json`](assembly.json), the release's record of the assembly, is checked
field by field against what was built and goes into the image unchanged but for
`engine_natives`, which names the rebuilt library's sha256. The
venv is the dependency lock [`CONSTRAINTS`](CONSTRAINTS), checked after install.
The build ends by checking that the OCI image's `/opt/assembly` holds exactly
the assembled files (paths, sha256, modes, link targets), the pinned plugin natives
(`native-check.json`) and the rebuilt engine library (`engine-native-check.json`).

**The launch** comes from `SERVE_SH`, this repository's `serve.sh` by default. It
is generated from the release settings, never written by hand; the builder
refuses an empty file or one that is not a bash launcher, and records its sha256
in `$OUT/serve-sh.sha256`.

```bash
crane pull --platform linux/amd64 --format oci \
  nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87 /path/to/base-oci
crane pull --format oci \
  pytorch/manylinux2_28-builder:cuda13.0-78e737ad29420ffc4800e677c51e2a852caf8359@sha256:c8d8dba2124931732d1b073dec1d8999cd4d6c5ff7c5e77232137e77d9f00f6a \
  /path/to/wheel-builder-oci
BASE_OCI=/path/to/base-oci BUILDER_OCI=/path/to/wheel-builder-oci ENGINE_NATIVE_JOBS=64 \
  STAGE=/path/to/new-owned-staging-directory OUT=/path/to/new-output ./docker/build.sh
```

Without `BASE_OCI` or `BUILDER_OCI` it pulls the image with `crane`. `CLONE_FROM=/path/to/repo`
fetches the queue and the plugin from another Git repository (by the tags, or by
the pinned commits where it has no such tags); `WHEEL=/path/to/wheel` uses a local
copy of the wheel. Both are checked against the same pins.
`STEP=relocate` and `STEP=assemble` resume a `STAGE` it built; choose a new `OUT`
each time.

The builder precompiles the peer probe without using a GPU; `IMAGE_P2P_PREBUILD=0`
omits it. `IMAGE_LAYERED=0` keeps one runtime layer and
`IMAGE_PARALLEL_COMPRESSION=0` uses gzip. It prints the total filesystem layer
count, including the base, and fails above 123 before compressing; the
troubleshooting kill switch is `IMAGE_LAYER_COUNT_CHECK=0` (default 1).

Before layering, nothing in `/opt` may still name the staging path, a home
directory or (reported for review in `hostname-hits.txt`) the build host.

The build needs no GPU. It downloads Python, the wheel, torch, the runtime
extras and the engine library's CMake FetchContent sources, and compiles that
library, so it takes a while and needs network access. The Docker builder
(`Dockerfile`, `build-docker.sh`) builds a single-tree engine and does not build
this release's assembled engine. See [release procedure](RELEASE.md) for the
compression switches and publication. A rebuild produces a separate artifact; it
does not revalidate the frozen digest: its rebuilt engine library, and so its
native and engine layers and its digest, differ from the release image's. Validate
the resulting provenance, layers and runtime before distributing any rebuilt image.

## Run it by hand

You need:

- the four cards, NVIDIA driver **580 or newer** and about 240 GiB of host RAM,
  or less with the Engram tables read from the SSD, slower
  ([What you need](../docs/how-to-use.md#what-you-need));
- Docker with the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  so that `docker run --rm --gpus all nvidia/cuda:13.3.1-base-ubuntu24.04 nvidia-smi`
  lists all four cards;
- the checkpoints and the hybrid built from them: `./download.sh` puts them in
  `./models` ([Where models are stored](../docs/how-to-use.md#where-models-are-stored)).

From the root of this repository:

```bash
IMAGE=ghcr.io/morrowmake/vllm-cmp170hx:engine-330c5bffee-e05403fdc3-0b34823c39
MODELS=$PWD/models                  # the three directories ./download.sh made
CACHE=$PWD/cache                    # kernel compile caches, kept between starts
MEM=$(awk '$1 == "MemTotal:" {print $2 "k"}' /proc/meminfo)   # limit, swap included: never swap
mkdir -p "$CACHE"

docker run -d --name dsv41-flash --user "$(id -u):$(id -g)" --workdir /cache \
  --gpus all --shm-size 16g --memory "$MEM" --memory-swap "$MEM" \
  -p 127.0.0.1:8000:8000 \
  --env-file docker/container.env \
  -v "$PWD/docker/entry.sh:/recipe/entry.sh:ro" \
  -v "$PWD/boot_check.py:/recipe/boot_check.py:ro" \
  -v "$PWD/p2p_check.py:/recipe/p2p_check.py:ro" \
  -v "$PWD/p2p_check.sh:/recipe/p2p_check.sh:ro" \
  -v "$PWD/p2p_probe.cu:/recipe/p2p_probe.cu:ro" \
  -v "$PWD/engram_placement.sh:/recipe/engram_placement.sh:ro" \
  -v "$MODELS/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw:/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw:ro" \
  -v "$MODELS/DeepSeek-V4.1-Flash-1017:/models/DeepSeek-V4.1-Flash-1017-engram:ro" \
  -v "$MODELS/DeepSeek-V4.1-Flash-1017-hybrid-exl3-fp8:/models/DeepSeek-V4.1-Flash-1017-hybrid-exl3-fp8:ro" \
  -v "$CACHE:/cache" \
  --entrypoint /bin/bash "$IMAGE" /recipe/entry.sh

docker logs -f dsv41-flash          # wait for "Application startup complete"
curl http://127.0.0.1:8000/health
```

If you built the image yourself, use your tag in place of `$IMAGE`. The same
launch, with the checkpoint download and storage ownership handled for you, is
[`docker-compose.yml`](../docker-compose.yml) at the root of this repository
([Docker Compose](../docs/how-to-use.md#docker-compose)). The official
checkpoint is mounted at `/models/DeepSeek-V4.1-Flash-1017-engram`: the
release's launch reads the Engram tables there, and the hybrid's links point
there.

Settings from [`.env.example`](../.env.example) or
[`.env.advanced.example`](../.env.advanced.example) that you add to
[`container.env`](container.env) override the launch's defaults the same way
as in `./start.sh`'s container, so the server is the same either way.

Inside the container the server listens on every interface;
`-p 127.0.0.1:8000:8000` keeps it reachable from this machine only, as
`./start.sh` does. It has **no API key**: to serve other machines, publish the
port on another address and add `-e API_KEY=<a long random string>`.

The server runs as you (`--user`), not root: Triton, FlashInfer, TileLang,
torch and vLLM write their compile caches under the cache mount (`HOME=/cache`
and the cache paths in `container.env`), so `./cache` stays yours. The first
start compiles kernels there and takes a while; later starts reuse them. Stop
it with `docker stop -t 120 dsv41-flash` and wait until `nvidia-smi` shows the
cards empty before starting again. The API is the same as always
([Talk to it](../docs/how-to-use.md#talk-to-it)).

**PCIe peer-to-peer is automatic** (`P2P=auto`): the same startup
[content check](../docs/how-to-use.md#pcie-peer-to-peer). Set `P2P=off` in
`container.env` to disable all peer transports, or `P2P=force` to bypass
verification.

The [engine switches](../docs/engine-switches.md) work the
same way: set the variable in `container.env` or add `-e NAME=0`.

## Component licences

The OCI Apache-2.0 label describes the vLLM engine, not a blanket licence for
every file in the image. CUDA/NVIDIA components, Python dependencies and
upstream third-party code retain their respective licences and notices.

The image carries the engine's Apache-2.0 licence (the queue tree's `LICENSE`)
at `/opt/image-tools/LICENSE` and the DeepGEMM derivative-reference notice at
`/opt/image-tools/THIRD_PARTY_NOTICES`; image smoke checks the complete bytes.
The plugin's licence files, the bundled FLA MIT grant and the other upstream
notices remain in `/opt/assembly`. This recipe's own MIT licence covers the
recipe's files, which are mounted from the checkout.
Model weights are downloaded separately and have
[their own licences](../docs/model-licences.md).
