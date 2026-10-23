# How it works

## The checkpoint

DeepSeek-V4.1-Flash is a mixture-of-experts model: 384 routed experts per
layer, six of them active for each token, plus shared parts every token goes
through (attention, the shared expert, the mHC mixing, the router, the vision
tower) and two Engram lookup tables.

The routed experts are most of the weights and are read a few at a time, so
they are where quantisation pays: they come from the
[EXL3 2.9 bpw export](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw).
Everything else is read for every token, and comes from DeepSeek's official
checkpoint in its own precision (FP8 where DeepSeek ships FP8, BF16 or FP32
elsewhere), not re-quantised. The served checkpoint is a directory of links
into the two downloads, with a config, an index and a quantization config that
route each module to its format; the engine refuses a module no rule names.
The build is deterministic, and the recipe checks its three written files
against the bytes this release was validated with.

The Engram tables (about 189 GiB, in the official checkpoint's last two
shards) are not loaded onto the cards. Each card's process holds its share of
the rows in host RAM and the cards fetch the rows a step needs. With the
tables in RAM the server takes 202–204 GiB of host memory in all; on a host
with less, the entry point reads them from the checkpoint files on the SSD
instead, through a row cache in host RAM per card
(`VLLM_ENGRAM_PLACEMENT=auto`, the default).

## What makes it fast, and correct

- **Whole experts per card.** Tensor-parallel 4 with expert parallelism: each
  card holds 96 of the 384 routed experts whole, so an expert's work never
  needs a reduction of its own.
- **EXL3 experts on the tensor cores.** The routed experts run on Ampere
  tensor-core kernels for both decode-sized and prefill-sized calls, with the
  weights decoded on the fly from the EXL3 trellis.
- **Fewer launches per step.** The attention pipeline, the router, the gate,
  the mHC mixing and the drafter's prologue run as fused kernels; each
  produces the same bytes as the path it replaces, or is checked against an
  FP64 reference where it changes the summation order.
- **Cheaper communication.** A custom all-reduce over verified PCIe
  peer-to-peer, BF16 partial sums where they are exact enough, prefill
  all-reduces on a copy-engine ring and split in two so communication overlaps
  compute.
- **Speculative decoding.** The checkpoint's DSpark drafter proposes up to seven
  tokens per step while one request runs and up to five with more; from three
  concurrent requests on, each step's verify width is chosen from each
  request's recent acceptance, and a request keeps only the drafts worth
  verifying. The model verifies them in the same step, so the output is the
  model's own.
- **CUDA graphs** for decode steps up to 48 tokens wide, with room left for the
  Engram row staging.
- **Fair prefill.** While answers are streaming, a long prompt prefills in
  512-token pieces beside them; a short prompt is scheduled whole, so a burst
  of short requests starts at once.
- **Full-precision KV cache.** The KV cache is BF16; no 8-bit KV storage.

Every feature has a switch ([Engine switches](engine-switches.md)).

## What runs

| | |
|---|---|
| API | OpenAI-compatible, `http://127.0.0.1:8000/v1`; this machine only, no API key, unless you [change that](how-to-use.md#serving-other-machines) |
| Model id | `dsv41-flash` |
| Weights | the hybrid of [`LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw`](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) (routed experts, EXL3 2.9 bpw) and [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (everything else) |
| Drafter | DSpark, inside the checkpoint: up to 7 draft tokens per step at one request, 5 at two to eight; from three requests the verify width is chosen per step |
| Engine | [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx) `ampere` @ [`{{PIN}}`](https://github.com/Morrowmake/vllm-cmp170hx/commit/{{PIN}}), on upstream vLLM `4e0681ed56` |
| Container image | `ghcr.io/morrowmake/vllm-cmp170hx:engine-{{PIN}}`, pinned by digest in `start.sh` — the engine at that pin and the release's launch, no weights ([docker/](../docker/README.md)) |
| Layout | tensor-parallel 4 with expert parallelism; assumes PCIe Gen2 x16 |
| Context | 262,144 tokens |
| Concurrency | up to 8 requests; the scheduler's budget is 2,048 tokens per step |
| KV cache | BF16, **not quantised**: {{KV_TOKENS}} tokens at 262,144 context with verified peer-to-peer |
| Prefix caching | on |
| Tools and reasoning | `--enable-auto-tool-choice`, the deepseek_v41 tool-call and reasoning parsers |
| Vision | images on; each card runs the vision tower on its own share of the images |
| Engram tables | host RAM when 212 GiB is available, else read from the SSD (`VLLM_ENGRAM_PLACEMENT=auto`) |
| Memory target | `--gpu-memory-utilization 0.95` |

## Comparing launch configuration

The image's `/opt/recipe/serve.sh` is generated from the release settings and
is the same file as `serve.sh` in this repository. `DRY=1 ./start.sh` prints
the environment and the command it would run.
`tools/check-env-match.py --effective <saved-environment.json> --layout tp4`
compares a saved complete server environment, and the `non-default args` line
of the `serve.log` beside it, with that launch, on CPU.
