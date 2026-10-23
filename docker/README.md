# Container image

The engine in a container: the only way this recipe runs. `./start.sh` and
`docker compose up` pull the image, start it and manage it for you (see the
[How to use this repo](../docs/how-to-use.md)); this page describes the
image, how to build it, and how to run it by hand with `docker run`.

## What the image contains

- **Engine:** the [vLLM fork](https://github.com/Morrowmake/vllm-cmp170hx/tree/{{PIN}})
  pinned at `{{PIN}}`, on upstream `4e0681ed56`, installed
  editable in `/opt/venv` (Python 3.12, torch 2.13.0, CUDA 13.0 build) with the
  upstream compiled extensions from `4e0681ed56`'s own nightly wheel, then the
  runtime extras the engine pins. The installed package list is in
  `/opt/venv/requirements.lock.txt`.
- **The fork's compiled extensions:** the sm_80 EXL3 library (`vllm._exl3_C`,
  required: the routed experts run on it) and the Marlin library, distinct
  from the upstream base extensions.
- **The release's launch:** `/opt/recipe/serve.sh`, generated from the
  settings this release was validated with: the engine command and every
  serving default, for the one layout (`LAYOUT=tp4`). The recipe runs it through
  [`docker/entry.sh`](entry.sh), which decides where the Engram tables go,
  checks peer access and runs the boot check first. Settings you give at run time override its defaults.
- **Base:** `nvidia/cuda:13.3.1-devel-ubuntu24.04`, pinned by digest. It is the
  devel image because Triton, TileLang, FlashInfer and the host-memory
  all-reduce compile kernels at run time and need `nvcc` and `g++`.
- **Entry point:** the `vllm` CLI. The recipe replaces it with
  `docker/entry.sh`.
- **Peer check:** a prebuilt sm_80 content probe, selected only when its source
  hash matches the mounted recipe. `P2P_PREBUILT=0` uses runtime compilation.
  The startup banner reports build, identity and check time, with transport,
  ordered-pair and size breakdowns on an uncached check. Content verification
  remains cached by driver, ordered GPU UUIDs, source hash and host boot.
- **No compilation-cache seed** unless the release says so (seed:
  {{SEED_YES_NO}}); without one, the first start compiles its kernels into the
  cache mount.
- **No weights.** The checkpoints are mounted from the host. The engine source
  in `/opt/vllm-src` includes the builder the recipe uses to make the hybrid
  checkpoint (`tools/sm80/build_hybrid_checkpoint.py`).

Engine image tag:

```
ghcr.io/morrowmake/vllm-cmp170hx:engine-{{PIN}}
```

The OCI revision is `{{PIN}}`.
Image tags identify engine source, independently of recipe release numbers.
Publication requires validation and acceptance of the exact image digest.

## Build it

**The fork's compiled extensions** are built once, before the image, by
`docker/native_build.py` inside the pinned CUDA base image (rootless, with
bubblewrap; fixed paths). Native validation runs those exact files, and both
builders pack them by hash; neither compiles them.
`RELEASE_NATIVE_MANIFEST_SHA256` in `build.sh` (and `NATIVE_MANIFEST_SHA256` in
the `Dockerfile`) pins the validated `manifest.json`; a different build, commit
or toolchain fails the image build.

**The launch** comes from `SERVE_SH` (rootless builder) or the `SERVE_SH` build
argument (Docker), this repository's `serve.sh` by default. It is generated from
the release settings, never written by hand; the builders refuse an empty file
or one that is not a bash launcher, and `build.sh` records its sha256 in
`$OUT/serve-sh.sha256`.

```bash
crane pull --platform linux/amd64 --format oci \
  nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87 /path/to/base-oci
python3 docker/native_build.py build --repo /path/to/fork-clone --commit <full-engine-commit> \
  --base-oci /path/to/base-oci --work /path/to/new-scratch --out /path/to/native --uv "$(command -v uv)"
NATIVE_DIR=/path/to/native VLLM_COMMIT=<full-engine-commit> CLONE_FROM=/path/to/source \
  CLONE_BRANCH=<source-branch> STAGE=/path/to/new-owned-staging-directory ./docker/build.sh
```

`manifest.json` records the source commit and input trees, the base image and
compiler digests, torch and its library hashes, the flags and each file's
sha256; it names no host path. For the Docker path, copy the output directory
to `docker/native/`. `build.sh` ends by checking that the assembled OCI image
holds exactly the manifest's files (`native-check.json`).

Both builders precompile the peer probe without using a GPU. To omit it, set
`IMAGE_P2P_PREBUILD=0` for the rootless builder or pass
`--build-arg IMAGE_P2P_PREBUILD=0` to `docker/build-docker.sh`. The Docker wrapper
uses the repository root as its build context, restricted by `.dockerignore`.

The builder replaces `STAGE`: choose a new dedicated path. It produces a local
OCI image in `docker/out/`, without publishing it. Set `OUT=/path/to/new-output`
to preserve earlier image evidence. Local `CLONE_FROM` sources use Git transport
cloning (`--no-local`), never object-directory copying or hardlinks. Only the
selected history and explicit upstream version tag are retained during the build;
objects newer than an overridden pin are pruned. Before removal, the actual
source HEAD, self-contained Git stores and absence of unreachable objects are
checked, including submodule stores. After version construction, both builders
remove all Git metadata **before** copying or archiving the application payload
into any final runtime layer. No contaminated application layer is reused.

The final source tree remains editable Python source, but is not a Git checkout.
Its root-owned, read-only `/opt/vllm-src/provenance.json` records `schema` (1),
`engine_commit` (full hash), `engine_repository` (public fork URL) and
`engine_version` (installed version). The OCI revision label retains the same
commit.
Layer assembly rejects retained Git metadata/object stores and forbidden build
paths, including standard nested gzip, bzip2, xz, tar and zip payloads. Set
`IMAGE_FORBIDDEN_STRINGS` to a JSON array of additional identifying strings for
the rootless build. Do not pass private identifier lists through Docker build
arguments or labels; use a separate private audit of the resulting image. Broad
name matches still need attribution review; upstream copyright/author notices
are retained.
For Docker, add build arguments `VLLM_REPO`, `VLLM_BRANCH`, `VLLM_COMMIT` naming
a source reachable inside the build.

Build the release source with Docker:

```bash
bash docker/build-docker.sh vllm-cmp170hx:engine-{{PIN}}
```

Both build paths print the total filesystem layer count, including the base,
and fail above 123. OCI assembly checks before compression; the Docker wrapper
checks the final image's `RootFS.Layers` after building. Use the wrapper for
Docker builds; a bare `docker build` bypasses this check. The troubleshooting
kill switch is `IMAGE_LAYER_COUNT_CHECK=0` (default 1).

The build needs no GPU. It clones the fork at the pinned commit and downloads
torch, the precompiled extensions and the runtime extras, so it takes a while
and needs network access.

The release uses the rootless application-layer assembly
approach in [`build.sh`](build.sh). The current builder prepares the runtime
tree on the host, creates separate deterministic layers and writes an OCI
layout without a Docker daemon. [crane](https://github.com/google/go-containerregistry)
fetches the pinned base and publishes the retained OCI without recompression.
See [release procedure](RELEASE.md) for the dependency lock, compression
switches, cache-seed gates and publication. Rebuilding with either
builder produces a separate artifact; it does not reproduce or revalidate the
frozen digest automatically. Validate the resulting provenance, layers and
runtime before distributing any rebuilt image.

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
IMAGE=ghcr.io/morrowmake/vllm-cmp170hx:engine-{{PIN}}
MODELS=$PWD/models                  # the three directories ./download.sh made
CACHE=$PWD/cache                    # kernel compile caches, kept between starts
mkdir -p "$CACHE"

docker run -d --name dsv41-flash --user "$(id -u):$(id -g)" --workdir /cache \
  --gpus all --shm-size 16g \
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

The image build retains the recipe's MIT grant at `/opt/image-tools/LICENSE`
and the DeepGEMM derivative-reference notice at
`/opt/image-tools/THIRD_PARTY_NOTICES`; image smoke checks the complete bytes.
The bundled FLA MIT grant remains in the engine source.
Model weights are downloaded separately and have
[their own licences](../docs/model-licences.md).
