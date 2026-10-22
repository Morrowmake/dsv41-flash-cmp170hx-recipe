# Building and publishing an image

`build.sh` is rootless and CPU-only (`CUDA_VISIBLE_DEVICES=""`); Docker is not
needed. Choose new, dedicated `STAGE` and `OUT` directories. Builds never
publish. `CONSTRAINTS` is the complete installed dependency lock, excluding
editable vLLM. Every installation uses it, and packaging rejects missing, extra
or changed distributions (including python-dotenv). Python is pinned to
3.12.14; the CUDA base and extension wheel remain pinned by digest and commit.
Version constraints reproduce the installed dependency set; wheel-download
hashes and a bit-for-bit compiler result are separate checks.

The layers are Python, dependencies, native engine libraries, editable engine
source/metadata with the release's launch (`/opt/recipe/serve.sh`), and
optionally a compilation-cache seed. Engine metadata, editable finders and the
native manifest (`/opt/venv/.native-manifest.json`) belong to the engine layer.
Our compiled extensions are not compiled by either builder:
`native_build.py build` makes them once in the pinned base image, native
validation installs exactly those files (`native_build.py install`), and the
release pins the resulting manifest's sha256 in `build.sh`
(`RELEASE_NATIVE_MANIFEST_SHA256`) and the `Dockerfile` (`NATIVE_MANIFEST_SHA256`).
`build.sh` packs them from `NATIVE_DIR` and finishes with `native_build.py
check-oci` against the assembled image.
Native libraries reuse a layer only when their bytes are identical; a Python
release never reuses native bytes solely on the basis of a branch name.
`layers.json` records every compressed size and digest. `IMAGE_LAYERED=1` and
`IMAGE_PARALLEL_COMPRESSION=1` are the defaults; set either to `0` to disable.
GNU tar normalizes metadata, then pigz compresses independent layers in parallel
(`IMAGE_COMPRESSION_THREADS=16` threads per layer). OCI assembly writes these
compressed blobs directly, so crane does not recompress them at publication.
Keep the same compression settings to reuse layer digests.

```bash
CUDA_VISIBLE_DEVICES="" VLLM_COMMIT=<full-commit> NATIVE_DIR=/path/to/native \
  CLONE_FROM=/path/to/source CLONE_BRANCH=<branch> \
  STAGE=/path/to/new-stage OUT=/path/to/new-output ./docker/build.sh
```

`STEP=assemble` packages an already finalized staging tree; `BASE_OCI` can name
an audited retained base for offline assembly. Neither changes the immutable
engine pin. Source Git stores are checked, then removed before any runtime
layer is created. `Dockerfile` uses the same constraints and final partition,
though its intermediate stages are build-only. Pass Docker build argument
`IMAGE_LAYERED=0` for its single-runtime-layer path. The Docker path has no optional
seed input; use the rootless builder for seeded release images.

## The launch

`serve.sh` is generated from the release settings, never edited by hand:
`release/checklist.py generate-serve` writes it, and the release checklist
regenerates it and refuses a recipe whose `serve.sh` differs. The image
carries the file `SERVE_SH` names (default `../serve.sh`) as
`/opt/recipe/serve.sh`; `$OUT/serve-sh.sha256` records which.

A reference image for the release's quality comparison is the same engine and
native files with the reference launch generated from the enablement-only
settings (only what the model needs to run on these cards):

```bash
CUDA_VISIBLE_DEVICES="" VLLM_COMMIT=<full-commit> NATIVE_DIR=/path/to/native \
  SERVE_SH=/path/to/enablement-only/serve.sh \
  STAGE=/path/to/new-stage OUT=/path/to/new-output ./docker/build.sh
```

It boots through this recipe like the release image. It is a validation
artifact and is not published.

## Compilation-cache seeds

`VLLM_IMAGE_CACHE_SEED=0` is the default. Enabling it emits one
`[image-cache]` banner, verifies that every visible GPU is CMP 170HX sm_80, and
selects `/cache/compiled/<identity-hash>`. Identity includes the full engine
commit, GPU architecture/name, dependency lock hash,
Torch/Triton/FlashInfer/TileLang versions and CUDA driver version. A mismatch
or damaged bundle falls back to a fresh namespace. Cache copying is locked and
never merges into an existing namespace; normal compiler cache updates remain
writable. Disable with `VLLM_IMAGE_CACHE_SEED=0` to keep the original cache
locations. `DRY=1` and `--help` never initialize CUDA for seeds.

Generate caches on the actual cards, with the release runtime mounted at
`/opt`, models at fixed container paths and the cache mounted at `/cache`.
Use the identity command's exported paths before launching, then stop only
that owned server before packing. Warm TP4 with the release launch and smoke
requests; only compilation caches are included, never weights, downloaded
credentials or logs. Seeds include Triton, FlashInfer, TileLang, TorchInductor,
torch C++ extensions, CUDA driver and vLLM compilation caches. Record health,
smoke, versions, KV, startup banners and cold/seeded startup durations.
Require smoke to pass and seeded startup to improve by at least 20 percent.

```bash
# Inside the isolated runtime, with its writable /cache mount:
eval "$(/opt/venv/bin/python /opt/image-tools/cache_seed.py identity /cache/identity.json)"
# Launch the release recipe; run smoke; stop that server.
# Pack from the host after the runtime exits:
python3 docker/cache_seed.py pack /path/to/cache/compiled/<identity-hash> \
  /path/to/cache/identity.json /path/to/new-seed
# Append the verified seed as its own release layer:
CUDA_VISIBLE_DEVICES="" STEP=assemble IMAGE_CACHE_SEED_DIR=/path/to/new-seed \
  STAGE=/path/to/finalized-stage OUT=/path/to/new-output ./docker/build.sh
```

The coordinator's reserved GPU leg must generate and compare these caches.
No cache bundle is produced by a CPU build. Do not reuse a cache from a native
installation with different absolute paths, or claim a startup improvement
before the cold/seeded comparison. Driver differences deliberately reject a
seed; users can continue with a cold cache. Seeded files are executable compiler
artifacts: distribute only trusted, audited release bundles.

## Artifact review and direct publication

1. Check every new commit's author and committer (`Morrowmake <noreply@github.com>`),
   messages, added lines and pre-push identity guard. No recipe or fork push is
   part of image assembly. Keep the Git push clones and their credentials
   separate from image publication.
2. Inspect the exact manifest/config and every distributed layer, including
   standard nested archives, with `audit_oci.py`. Supply private identifiers
   through local `IMAGE_FORBIDDEN_STRINGS` and `IMAGE_FORBIDDEN_PATTERNS` JSON
   environments only, never
   image arguments, labels or attestations. The audit verifies compressed
   digests and uncompressed diff IDs, rejects Git stores, screens registry-token
   shapes without printing values, and reports format coverage gaps. Inspect
   ELF fatbins/nonstandard compressed objects separately. A reviewed identifier
   substring in immutable third-party data may be recorded in local
   `IMAGE_REVIEWED_BYTE_MATCHES` JSON as `{file_sha256: [byte_offsets]}`;
   default `{}` disables this feature. Only those exact file bytes and offsets
   are allowed, and the audit records them with a banner. This never allows
   Git metadata, forbidden archive paths or credential candidates.
   Verify that provenance,
   OCI revision and actual imports agree. Retained native bytes must match
   the previously reviewed artifacts. Review the seed archive by the same gate.
3. Run real-image GPU acceptance and a fresh install (and, from the second
   release on, the previous-release update), including `.env` preservation,
   health, smoke, banners and KV. Record the exact final manifest digest. Keep
   the previous public image until public replacement and update checks pass.
4. After the user's explicit publication approval, supply a short-lived GitHub
   token for `Morrowmake` with `write:packages` scope at release time. Use a
   private stdin source (permissions 0600 if a file); never put it in argv,
   shell history, persistent Docker auth, source or build inputs. Revoke it
   after release. Do not create a token as part of the builder.
5. Write the reviewed acceptance JSON with `image_digest`, `passed: true`, and
   `artifact_audit_passed: true`, then publish directly on the build machine:

```bash
CUDA_VISIBLE_DEVICES="" IMAGE_PUBLISH=1 CRANE=/path/to/crane ./docker/push.sh \
  /path/to/output/oci ghcr.io/morrowmake/vllm-cmp170hx:<version>-<short-commit> \
  sha256:<exact-final-manifest> /path/to/reviewed-acceptance.json < /path/to/private-token
```

The script defaults publication off, verifies OCI blob integrity and matching
acceptance before consuming stdin, uses a fresh 0700 auth directory and removes
it on exit, then checks the published digest. It uses crane's direct OCI push,
which uploads missing layers without a staging registry or Docker daemon.
A failed or interrupted upload can leave blobs; retain the OCI and rerun the
same approved digest. An acceptance JSON is a review record, not a substitute
for the tests. Verify anonymous digest access before changing the recipe image
pin, then run the public fresh/update checks and token revocation procedure.


Before GPU acceptance, extract the built OCI and run the CPU inspection gate:

```bash
CUDA_VISIBLE_DEVICES="" python3 docker/rootfs_smoke.py \
  --rootfs /path/to/extracted-rootfs --config /path/to/build/config.json \
  --model /path/to/hybrid-checkpoint --result /path/to/new-cpu-results
```

This uses the image's `/opt/venv/bin/python`, captures the registry subprocess
stderr for the checkpoint, imports the serve CLI and validates its model
configuration. It stops at the unavailable-device boundary without starting an
engine or server. Bubblewrap creates a private `/dev` with no GPU nodes. Host
driver libraries are mounted read-only at `/usr/local/nvidia/lib64`, matching
the OCI `LD_LIBRARY_PATH`; no CUDA initialization is needed for the driver
search/load checks. `IMAGE_ROOTFS_CPU_SMOKE=0` disables this diagnostic (default
1), with a banner. Run the image regression in `docker/tests` by setting
`TEST_IMAGE_ROOTFS`, `TEST_IMAGE_CONFIG` and `TEST_TARGET_MODEL`.
For extracted-rootfs GPU legs, retain the OCI library search path and use
`rootfs_smoke.py --driver-options` for the same driver mounts. A cleared
environment with only host-library binds does not reproduce NVIDIA container
runtime setup and can fail Triton's library discovery before inspection.
