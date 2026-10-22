<h1 align="center">DeepSeek-V4.1-Flash on 4× NVIDIA CMP 170HX</h1>

<p align="center">
  <strong>by <a href="https://x.com/Morrowmake">Morrowmake</a></strong>
  <br><br>
  <a href="https://x.com/Morrowmake"><img alt="Follow on X" src="https://img.shields.io/badge/Follow-%40Morrowmake-000000?style=flat&logo=x&logoColor=white"></a>
  &nbsp;
  <a href="https://github.com/Morrowmake/vllm-cmp170hx/tree/{{PIN}}"><img alt="engine" src="https://img.shields.io/badge/engine-vLLM%20fork%20%40%20{{PIN}}-4b32c3?style=flat"></a>
  &nbsp;
  <img alt="release" src="https://img.shields.io/badge/release-{{VERSION}}-2ea44f?style=flat">
  &nbsp;
  <img alt="licence" src="https://img.shields.io/badge/recipe-MIT-blue?style=flat">
</p>

**DeepSeek-V4.1-Flash with a 262,144-token context, served on four CMP 170HX
cards: {{C1_TOK_S}} tok/s for one user and {{C8_TOK_S}} tok/s across eight,
{{PREFILL_TOK_S}} tok/s cold prefill and a {{KV_TOKENS}}-token KV pool. The routed
experts are EXL3 at 2.9 bits per weight; every other weight is DeepSeek's own
FP8 or BF16, unchanged, and the KV cache is full precision.**

This repository installs and runs **DeepSeek-V4.1-Flash-1017** on **four NVIDIA
CMP 170HX cards** behind an OpenAI-compatible API, with tool calls, reasoning
and images, and the checkpoint's own **DSpark** speculative drafter. The CMP
170HX is Ampere (sm_80). Our
[vLLM fork](https://github.com/Morrowmake/vllm-cmp170hx/tree/ampere) adds the
Ampere kernels that make the model run at all, then spends the rest of its
patches on making it fast and making it repeatable. It runs only as a container
image, so the engine and its Python environment stay out of your system.
P2P is enabled automatically after a startup [content check](docs/how-to-use.md#pcie-peer-to-peer) passes; otherwise the server runs with P2P disabled.

## Quick start

One command from a fresh clone, after the [one-time host setup](#one-time-host-setup):

```bash
git clone https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe.git
cd dsv41-flash-cmp170hx-recipe
./start.sh          # pull the engine image, fetch the model, start, wait for /health
```

Then try it:

```bash
./start.sh smoke    # one chat request and one tool call
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "dsv41-flash",
       "messages": [{"role": "user", "content": "Write a haiku about PCIe."}],
       "max_tokens": 4096}'
```

> **DeepSeek-V4.1 reasons before it answers by default.** The reasoning comes
> back in the message's `reasoning` field and the answer in `content`. **Set
> `max_tokens` for the reasoning plus the answer:** if the limit is reached
> while the model is still reasoning, `content` comes back empty with
> `finish_reason: "length"`. `"chat_template_kwargs": {"thinking": false}`
> turns the reasoning off; `"reasoning_effort"` (`low`, `high`, `xhigh`,
> `max`) sets how much it reasons ([Talk to it](docs/how-to-use.md#talk-to-it)).

`./start.sh` skips anything already done, so running it twice is safe; an
interrupted download resumes. The first start compiles the kernels it needs
(cold), so it is slower; later starts reuse them from `./cache`.
Any OpenAI-compatible client works with base URL `http://127.0.0.1:8000/v1`
and model `dsv41-flash`. The API listens on this machine only and has no
API key unless you [change that](docs/how-to-use.md#serving-other-machines).
Python, streaming and tool-call clients are in [examples/](examples/README.md).

**Prefer Docker Compose?** The same container, also one command from a fresh
clone:

```bash
docker compose up              # or in the background: docker compose up -d
```

It downloads the checkpoints, then starts the server as the owner of the
checkout; `docker compose ps` shows it healthy once `/health` answers
([Docker Compose](docs/how-to-use.md#docker-compose)). Compose restarts a
crashed server up to 3 times; `./start.sh` does not (run `./start.sh restart`).
Use one or the other.

Both download the checkpoints at the exact Hugging Face revisions this release
was validated with.

> **What this release is for.** One layout: **tensor-parallel 4** with whole
> routed experts on each card, optimised for **PCIe x16 links** (cards with
> the x16 capacitor modification). On cards limited to x4 links it runs, but
> bus-bound and much slower; `./start.sh` warns when a card is narrower than
> x16.

## One-time host setup

- **4× NVIDIA CMP 170HX, each exposing 64 GiB**, on **PCIe Gen2 x16 links**
- **About 240 GiB of host RAM**: the model's Engram lookup tables (~189 GiB)
  are held in memory, and the server takes 202–204 GiB of host RAM in all.
  With less, set `VLLM_ENGRAM_PLACEMENT=ssd` to read the tables from the SSD
  instead; that is slower and wants the checkpoints on an SSD
  ([Host memory](docs/how-to-use.md#host-memory-and-the-engram-tables))
- Linux with NVIDIA driver **580 or newer**
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  usable by your user: `docker run --rm --gpus all nvidia/cuda:13.3.1-base-ubuntu24.04 nvidia-smi`
  lists all four cards
- optional: a P2P-capable driver build for verified peer-to-peer
  ([PCIe peer-to-peer](docs/how-to-use.md#pcie-peer-to-peer)); without it the
  server runs with P2P disabled
- for `./start.sh`: `git`, `curl`, `flock`, `setsid`, Python 3 for the boot
  check, and `jq` for the smoke test
- about 680 GiB (730 GB) of disk for the two checkpoints, in `./models` by
  default ([Where models are stored](docs/how-to-use.md#where-models-are-stored)),
  plus the engine image and the kernel compile caches ([details](docs/how-to-use.md#what-you-need))

## Headline numbers

Measured {{RESULTS_DATE}} on 4× CMP 170HX, {{SM_COUNT}} SMs per card, mainline
cmpunlocker with P2P, recipe defaults, 262,144-token context, 180 W per card
and PCIe x16 links, by this release's own release check. Decode uses the
MiaAI-Lab protocol: 400 tokens, temperature 0, requests sent with thinking off;
one user is the median of five runs, and two, four and eight users the median
aggregate rate of three runs, all on one boot.

| | Tensor-parallel 4<br>+ [peer-to-peer](docs/how-to-use.md#pcie-peer-to-peer) (default after verification) |
|---|---:|
| **1 user** · structured | **{{C1_STRUCTURED}}** |
| **1 user** · code | **{{C1_CODE}}** |
| **1 user** · prose | **{{C1_PROSE}}** |
| **8 users, total** · structured | {{C8_STRUCTURED}} |
| **8 users, total** · code | {{C8_CODE}} |
| **8 users, total** · prose | {{C8_PROSE}} |
| **Prompt reading** (cold prefill) | {{PREFILL_TOK_S}} |
| **KV pool** (tokens) | {{KV_TOKENS}} |

One user is streaming speed per request; eight users is the combined rate of
eight simultaneous requests. Structured output and code are drafted well, so
they decode fastest; prose least. DSpark drafts up to 7 tokens ahead for one
user and 5 for more; from three users on, each step verifies only as many
drafts as each request's recent acceptance makes worthwhile. Method, quality
and exact figures: [Results](docs/results.md).

## Settings

Settings live in `.env`; see [Settings](docs/how-to-use.md#settings).
Every serving default (context, concurrency, speculative decoding, engine
switches) is part of the release's launch, which the image carries; `.env`
only overrides.

## Documentation

- [Results](docs/results.md): throughput, quality, decode and prefill in detail
- [How to use this repo](docs/how-to-use.md): install, settings, host memory,
  peer-to-peer, day to day, updates, Docker Compose, containers
- [How it works](docs/how-it-works.md): the hybrid checkpoint, what makes it
  fast and correct, and what runs
- [Engine switches](docs/engine-switches.md): the engine's switches this release sets, for troubleshooting
- [Status and known limits](docs/known-limits.md)
- [Model licences](docs/model-licences.md): the checkpoints' MIT notice
- [Examples](examples/README.md): curl, Python, streaming, tool calls
- [Container image](docker/README.md) and [CHANGELOG.md](CHANGELOG.md)
- Found a problem? [Open an issue](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe/issues/new/choose);
  the bug template asks for the version, peer-to-peer setting and log lines we need

## Update

- `./start.sh update` pulls the new release and engine image and restarts; your `.env` is preserved.
- With Compose: `git pull`, then `docker compose up -d`.
- Older releases are git tags; their images may be retired.

---

## Licences

**This recipe** — the scripts and the documentation — is MIT, © 2026 Morrowmake.
See [LICENSE](LICENSE).

**The vLLM fork** is [Apache-2.0](https://github.com/Morrowmake/vllm-cmp170hx/blob/{{PIN}}/LICENSE).

The downloaded models have separate licences, both MIT:

- **Base and FP8 weights:** [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
  (MIT, © DeepSeek). Every weight but the routed experts, and the Engram tables,
  come from it.
- **EXL3 routed experts:** [`LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw`](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)
  declares `license: mit`, inherited from the base model; its LICENSE is the
  base model's MIT notice.

The [model-licence reference](docs/model-licences.md) preserves the complete
MIT notice.

**The benchmark prompts** come from MiaAI-Lab's repository (AGPL-3.0) and
their sparkDash prompt constants (MIT), used as data with attribution. No code
from their repositories is included here.

## Credits

If this recipe helps you, please credit Morrowmake and link this repository.
Citation metadata: [`CITATION.cff`](CITATION.cff).

- **[DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)**
  for DeepSeek-V4.1-Flash, and the **[vLLM](https://github.com/vllm-project/vllm)**
  project for the engine these patches sit on top of.
- **[LibertAIDAI](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)**
  for the EXL3 2.9 bpw quantisation of the routed experts, and
  [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (turboderp) for the
  EXL3 format.
- **[MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks)**
  for publishing the benchmark prompts and protocol used in the decode tables.

## Source

The fork contains our maintained changes, inherited upstream code and the
adaptations credited in its notices —
[Morrowmake/vllm-cmp170hx @ `ampere`](https://github.com/Morrowmake/vllm-cmp170hx/tree/ampere).

## Acknowledgements

- Mainline [cmpunlocker](https://github.com/amoghmunikote/cmpunlocker) and its maintainer's P2P work, plus our minimal TRAP31 patch, form the basis of the P2P driver build. Ready to install: [Morrowmake/cmpunlocker](https://github.com/Morrowmake/cmpunlocker).
- The engine builds on [Marlin](https://github.com/IST-DASLab/marlin)
  (Elias Frantar and subsequent Neural Magic contributors) and DeepSeek's
  [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) reference implementation.
  Original source notices and component licences apply separately.
