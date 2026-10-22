# How to use this repo

## What you need

| | |
|---|---|
| GPUs | 4× NVIDIA CMP 170HX, **each exposing 64 GiB** (`nvidia-smi` shows 65,536 MiB), on **PCIe Gen2 x16 links**. Stock cards expose less memory and run at x4; getting to 64 GiB and x16 is outside this repository. The preflight stops before any download if a card reports under 60 GiB, and warns below x16 ([Link width](#link-width)) |
| Host memory | **about 240 GiB**: the Engram tables (~203 GiB) are held in host RAM. With less, read them from the SSD instead ([Host memory and the Engram tables](#host-memory-and-the-engram-tables)) |
| OS and driver | Linux (the commands below are for Ubuntu) with NVIDIA driver **580 or newer**; `nvidia-smi` must list all four cards |
| Container runtime | Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), usable by your user, so that `docker run --rm --gpus all nvidia/cuda:13.3.1-base-ubuntu24.04 nvidia-smi` lists all four cards. This and the driver are a one-time host setup; the engine and its Python environment come in the image |
| Tools | for `./start.sh`: `git`, `curl`, `flock` and `setsid` (util-linux, on most systems already), Python 3 for the boot check, `jq` for the smoke test. Docker Compose needs only Docker |
| Disk | about 680 GiB (730 GB) for the two checkpoints, plus the engine image and the kernel compile caches. An SSD if the Engram tables are read from disk |

## Step by step

`./start.sh` on its own does steps 1–3 in order and skips anything already
done, so running it twice is safe. The same steps one at a time:

```bash
./install.sh       # 1. pull the engine image, pinned by digest
./download.sh      # 2. fetch the two checkpoints at the pinned revisions and build the hybrid (see "Where models are stored")
./start.sh         # 3. start the container, wait for /health, print the KV pool size
./start.sh smoke   # 4. one chat request and one tool call against the running server
./start.sh stop    # stop it; the image, weights and caches stay, so the next start is quick
```

The container runs the engine image, as your user rather than root, with the
checkpoints mounted read-only and the kernel compile caches in `./cache`
(owned by you). Its entry point, [`docker/entry.sh`](../docker/entry.sh) from
this repository, checks peer access and then runs the release's launch, which
the image carries as `/opt/recipe/serve.sh`: the engine command and every
serving default, generated from the settings this release was validated with.
The API is published on `127.0.0.1:8000` only. Its output goes to
`logs/serve.log`, and `./start.sh stop` stops only the container this checkout
started. The download runs in the same image as you, so the checkpoints are
yours too; an interrupted download resumes on the next run, and a checkpoint
counts as present only when every file its index names is there.

Both checkpoints are pinned to the Hugging Face revisions this release was
validated with: `EXPORT_REVISION` (`c178a01284fc5c0e859d8668cffaac81ca8f4042`)
and `OFFICIAL_REVISION` (`1bae0e1088993b9b82de08f59de5be5a34e42170`), for
`./start.sh` and Docker Compose alike. Checkpoints you already have are kept.

**The first boot after an install or an update is slower.** The image ships
no warm-cache seed (seed: {{SEED_YES_NO}}): it compiles on first boot (cold).
The kernels go into `./cache`, and later boots reuse them. Loading the Engram
tables into host memory also takes a while on every start. While a build runs,
the log can show lines like
`No available shared memory broadcast block found in 60 seconds`; they are
harmless and stop once the build finishes.

The smoke test checks the chat reply, tool call and KV cache; it prints
`smoke: ok` when those checks pass.

## Talk to it

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "dsv41-flash",
       "messages": [{"role": "user", "content": "Write a haiku about PCIe."}],
       "max_tokens": 4096}'
```

Any OpenAI-compatible client works with base URL `http://127.0.0.1:8000/v1`
and model `dsv41-flash`. DeepSeek-V4.1 reasons before it answers unless told
not to. The reasoning comes back in the message's `reasoning` field (streamed
as `delta.reasoning`) and the answer in `content`.

- `"chat_template_kwargs": {"thinking": false}` turns the reasoning off for a
  request.
- `"reasoning_effort"` of `low`, `high`, `xhigh` or `max` (or a number from 1
  to 100 in `chat_template_kwargs`) sets how much it reasons; `high` is the
  default.

**Set `max_tokens` for the reasoning plus the answer.** If the limit is
reached while the model is still reasoning, `content` is empty and
`finish_reason` is `length`; a client that reads only `content` then shows an
empty reply.

## Serving other machines

By default the API listens on `127.0.0.1` only, so nothing outside this
machine can reach it, and it has **no API key**. To serve other machines, set
both in `.env` and restart:

```bash
HOST=0.0.0.0          # every interface; or one of this machine's addresses
API_KEY=<a long random string>
```

With `API_KEY` set, every `/v1` request must send
`Authorization: Bearer <key>` (OpenAI clients: pass it as the API key);
`./start.sh smoke` and `status` send it for you. Without `API_KEY`, anyone who
can reach the port can use the model. The key is handed to vLLM through its
`VLLM_API_KEY` variable, so it does not appear in the process list.

## Settings

Everything lives in `.env`, copied from [`.env.example`](../.env.example) on first
run; `./start.sh` reads only `.env`. `.env.example` holds the settings people
actually change. Everything else is in
[`.env.advanced.example`](../.env.advanced.example), which you almost certainly
don't need: to use one of its settings, copy the line into your `.env`. The
engine's switches are described in [Engine switches](engine-switches.md).
A value on the command line beats `.env` for that run:

```bash
PORT=8001 ./start.sh restart
```

The serving defaults themselves (context, concurrency, speculative decoding,
every engine switch) are the release's launch, carried by the image; a setting
in `.env` overrides one of them for your server only.

In `.env.example`:

| Key | Default | What it does |
|---|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `8000` | where it listens; see [Serving other machines](#serving-other-machines) |
| `API_KEY` | unset | bearer key required on `/v1`; unset means no key |
| `SERVED_MODEL_NAME` | `dsv41-flash` | the model id clients send |
| `MODELS_DIR` | `./models` | where the checkpoints go; see [Where models are stored](#where-models-are-stored) |
| `VLLM_ENGRAM_PLACEMENT` | `ram` | the Engram tables in host RAM, or `ssd` to read them from the SSD ([Host memory](#host-memory-and-the-engram-tables)) |
| `P2P` | `auto` | [content-verified peer access](#pcie-peer-to-peer); `off` disables, `force` bypasses the check |

Commonly used from `.env.advanced.example`:

| Key | Default | What it does |
|---|---|---|
| `MAX_SEQS` | `8` | concurrent requests |
| `MAX_BATCHED` | `2048` | the scheduler's token budget per step (the prefill chunk) |
| `EXTRA_ARGS` | unset | appended to the `vllm serve` command line, e.g. `--max-model-len 131072` |
| `IMAGE` | this release's image, by digest | the engine image (container) |
| `CONTAINER_NAME` | `dsv41-flash` | the container's name |
| `READY_TIMEOUT` | `3600` | seconds `./start.sh` waits for `/health` |
| `NEED_RAM_GIB` | `212` | host memory the preflight asks for when the Engram tables are in RAM |
| `HF_TOKEN` | unset | a Hugging Face token makes the download faster |

Every setting in both example files is commented out and shows its default;
uncommenting one in `.env` overrides that default. A `.env` you have not edited
therefore picks up a later release's new defaults without changes. To move to
a later release, run `./start.sh update`.

Write each setting as `KEY=value` on its own line. In an unquoted value,
anything after a space and `#` is a note and is ignored; quote the value
(`"..."`) to keep it exactly as written.

### Boot check

Startup runs a boot check after `/health`; `BOOT_CHECK=0` disables it.
The temperature-0 prompt is `Reply with exactly OK, with no punctuation or other text.`
The reply's `content`, with surrounding whitespace stripped, must equal `OK`.
A second temperature-0 request streams 768 tokens of arithmetic-function code
with `reasoning_effort=low`, `min_tokens=768` and `ignore_eos=true`.
Model-scoped Prometheus `/metrics` deltas must show at least 64 draft steps,
64 proposed draft tokens and more than zero accepted tokens. Startup fails
and stops its own launch on an incorrect reply, incomplete stream, missing
counters or insufficient drafting. These are the only two boot generation
requests; starting an already-running checkout sends none. Keep user requests
out of startup so the model-wide counters describe the check. Compose launches
use the same check. `smoke.sh` retains its two independent chat/tool requests;
it does not repeat the boot check.

## Host memory and the Engram tables

DeepSeek-V4.1 looks up two Engram tables for every token: about 203 GiB, the
last two shards of the official checkpoint. This release holds them in host
RAM (`VLLM_ENGRAM_PLACEMENT=ram`), split across the four cards' processes, and
the cards fetch their rows from there. Before the start, the preflight checks
that the host has about 212 GiB available (`NEED_RAM_GIB`); the engine checks
again, with the container's memory limits, before it allocates, and refuses
with the shortfall rather than running out of memory.

On a host with less memory, read the tables from the SSD instead:

```bash
VLLM_ENGRAM_PLACEMENT=ssd ./start.sh restart     # this run
# or set VLLM_ENGRAM_PLACEMENT=ssd in .env (Docker Compose: in docker/container.env)
```

That needs only a small row cache in RAM, but every step waits on SSD reads,
so it is slower, and the published numbers do not apply. Keep `MODELS_DIR` on
an SSD then.

## Where models are stored

The checkpoints live in `MODELS_DIR`, `./models` in this checkout by default
(a relative `MODELS_DIR` is taken from this checkout), one plain directory
each:

| Directory | What it is |
|---|---|
| `DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/` | [`LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw`](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw): the EXL3 export, ~196 GiB |
| `DeepSeek-V4.1-Flash-1017/` | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017): the official FP8 checkpoint and the Engram tables, ~475 GiB |
| `DeepSeek-V4.1-Flash-1017-hybrid-exl3-fp8/` | the served checkpoint: links into the two above (the routed experts from the export, every other weight from the official checkpoint), plus its config, index and quantization config. No weight is copied |

The hybrid is built in the engine image with the engine's own builder, and its
config, index and quantization config must have exactly the bytes this
release was validated with; anything else stops the start. It holds only
links and three small files: if a start reports it as different or partial,
delete that directory's contents and run again.

Only the files the server reads are downloaded (the configs, the tokenizer and
the weight shards). The checkpoints are mounted into the container at fixed
paths, which is why they live in plain directories rather than in your Hugging
Face cache.

## Day to day

| Command | |
|---|---|
| `./start.sh` | preflight → install → download → launch → wait for `/health` |
| `./start.sh restart` | stop, then start (picks up `.env` changes) |
| `./start.sh status` | container, `/health`, KV line, install and checkpoint state |
| `./start.sh logs` | follow `logs/serve.log` |
| `./start.sh smoke` | one chat request and one tool call |
| `./start.sh stop` | stop the server this checkout started |
| `./start.sh restart` after a crash | `./start.sh` does not restart a crashed server; Docker Compose does, up to 3 times |
| `./start.sh update` | `git pull`, pull the new engine if the pin moved, restart |
| `DRY=1 ./start.sh` | print what would be pulled, downloaded and launched, without doing it |

`stop` only stops the container recorded in `logs/container.id`, after
confirming it carries this checkout's label; it never matches by name alone,
so another container or vLLM on the same machine is never touched.

The engine pin lives in `start.sh`, so a `git pull` can move it, and `start.sh`
pulls the matching image whenever it changes.

**Updating from an earlier release.** Run `./start.sh update`: it pulls this
release, restarts into its `start.sh`, pulls the new image if the pin moved and
restarts the server. Your `.env` is kept and its explicit values stay in force;
each release's [CHANGELOG](../CHANGELOG.md) entry names the lines to delete to
follow its new defaults. With Docker Compose: `git pull`, then
`docker compose up -d`.

Older releases are git tags; their images may be retired.

## Advanced

Advanced settings: peer-to-peer policy, the engine's switches, link width,
running the container yourself and Docker Compose.

### PCIe peer-to-peer

`P2P=auto` is the default for every start. Before the server starts, the
recipe checks every ordered pair of visible GPUs. It requires peer access to be
advertised, then copies random bytes with `cudaMemcpyPeer`, writes and reads
through peer access with an SM kernel, and verifies a separate-process CUDA IPC
write on the allocation's owner. Every byte is compared on its owner. The read
test uses fresh data written by the owner, so matching writes and reads at a
wrong address cannot pass together.

Nine sizes cover 128 KiB and 512 KiB, including one byte below and above each,
plus 1 MiB, 8 MiB and 32 MiB. Each pair also gets a 32 MiB IPC write. Compilation,
identity lookup, lock waits and the content check share a 120-second budget;
a timed-out probe and its IPC children are killed together, with at most one
additional second allowed for cleanup. The startup banner reports the outcome,
reason, elapsed time and whether the result came from this boot's cache.

A result is cached by driver version/build, visible GPU UUIDs, boot ID and probe
version. Restarts reuse it; a new boot, driver, GPU set or probe invalidates it.
The cache lives at `$HOME/.cache/recipe-p2p` (inside the persistent `/cache`
mount for containers); `P2P_CHECK_CACHE` selects another persistent directory.
A failed content check is cached too. After correcting a configuration within
the same boot, remove the check cache before retrying.

**Topology is not a data-integrity test.** `nvidia-smi topo -p2p r` reporting
`OK` and `cudaDeviceCanAccessPeer` returning true are not enough. A driver can
advertise peer access while writes land in the wrong memory; NCCL may then
hang. The startup content check decides whether this recipe uses P2P.

After a pass the release's peer transports apply: the custom all-reduce over
PCIe peer-to-peer (`VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=1`, flags-in-data), and
NCCL over peer-to-peer (`NCCL_P2P_LEVEL=SYS`).

If any pair does not advertise access, any byte differs, or the check cannot
complete, the engine starts with P2P disabled:
`VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=0`, `NCCL_P2P_LEVEL=LOC` and
`NCCL_P2P_DISABLE=1`. The `[p2p]` banner says `not advertised`,
`advertised but data check failed`, or `check unavailable` and gives the
reason. The all-reduces then go through host memory instead.

**Working P2P on CMP 170HX requires a P2P-capable driver build.**
This release uses mainline [cmpunlocker](https://github.com/amoghmunikote/cmpunlocker)
master, including its 74-SM override, plus the maintainer's P2P changes and
our minimal TRAP31 Booter PLM patch, gated on `ForceP2P=0x11`. The patch makes
peer mappings usable on these cards; advertising peer access alone is insufficient.
The recipe still requires the content check on every new driver/boot identity.
The ready-to-install build is [Morrowmake/cmpunlocker](https://github.com/Morrowmake/cmpunlocker): mainline master with these changes, installed the same way as mainline.

Choose the policy in `.env` or for a single start:

```bash
P2P=auto ./start.sh restart   # default: enable only after verification
P2P=off ./start.sh restart    # disable all peer transports
P2P=force ./start.sh restart  # bypass verification; may hang or corrupt data
```

Use `force` only when deliberately bypassing the check. A dry run initializes
no CUDA devices and leaves auto P2P off until a real start can verify it.

Read the last `[p2p]` line in `logs/serve.log`, then run `./start.sh smoke`.

### Engine switches

Every performance feature this release turns on can be switched off on its
own with one variable, without a rebuild. The list, and how to use the
switches for troubleshooting, is in [Engine switches](engine-switches.md).

```bash
VLLM_DEEPSEEK_V41_ATTN_PIPELINE=0 ./start.sh restart
```

### Link width

Tensor parallelism runs several small collectives between the cards for
every decode step and moves activations between them during prefill. It
assumes PCIe Gen2 x16 links; on x4 links it is bus-bound and far slower than
the [published numbers](results.md). `./start.sh` prints each card's link
width in its preflight and warns if any is narrower than x16.

### Run the container by hand

[`docker/`](../docker/README.md) has the Dockerfile and the build script for the
image, and a `docker run` command with [`container.env`](../docker/container.env)
for running it without `./start.sh`.

### Docker Compose

[`docker-compose.yml`](../docker-compose.yml) at the root of this repository
is the other one-command start. From a fresh clone, after the one-time host
setup ([What you need](#what-you-need)):

```bash
docker compose up                  # download, then serve in the foreground
docker compose up -d               # the same in the background
docker compose ps                  # the server shows (healthy) once /health answers
docker compose logs -f             # follow it
docker compose down                # stop (allows 120 s for a clean shutdown)
```

Two services run in order:

- **`prepare`** makes `./cache` and the three model directories (in
  `MODELS_DIR`, default `./models`) writable by the serving user, downloads
  either checkpoint that is missing as that user, at the pinned revisions
  (~672 GiB), and builds the hybrid ([Where models are stored](#where-models-are-stored)).
  An interrupted download resumes; complete checkpoints are skipped without
  contacting Hugging Face, so later starts go straight to the server. Docker
  creates a missing directory as root; `prepare` hands such a directory to the
  serving user when it is empty, but never changes the owner of existing
  checkpoints or of anything else, and never deletes anything. It must finish
  successfully before the server starts.
- **The server** is the same container `./start.sh` launches: the release image
  by digest, all four cards, the same shared-memory size, the API on
  `127.0.0.1:8000` only, the checkpoints mounted read-only, `./cache` for the
  kernel compile caches and the settings in
  [`container.env`](../docker/container.env), with
  [`docker/entry.sh`](../docker/entry.sh) as its entry point. It switches to the
  serving user before the server starts, so it never serves as root. It
  reports healthy once `/health` answers (`docker compose ps`).
- **Restarts.** If the server exits with an error, Compose starts it again, up
  to 3 times (`restart: on-failure:3`); `docker compose stop` and `down` stop it
  for good. `./start.sh` does not restart a crashed server: run
  `./start.sh restart`.

The serving user is the owner of this checkout, so it works for any user with
no settings; set `HOST_UID` and `HOST_GID` (in the shell or in `.env`) to use
another. `MODELS_DIR` chooses where the checkpoints are and `HF_TOKEN` speeds
up the download; any other setting goes in
[`container.env`](../docker/container.env), as for the `docker run` command.
`./start.sh` does not use this file; use one or the other (they share the
container name, so a second start fails rather than doubling up on the cards).
