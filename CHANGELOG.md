# Changelog

## 1.0.2 — 2026-11-02

Fixes the startup crash on hosts where the GPUs can't reach each other over
PCIe (issue #3). Speed and quality are unchanged from 1.0.1.

- **Fix.** The sm_80 plugin `vllm_sm80`
  (`e11b375ab98ce84a3ee5792364e21bc25dcc78fa`, tag
  `dsv41-flash-recipe-1.0.2-plugin`) builds the copy-engine all-reduce ring
  only when PCIe peer access is available, and falls back to NCCL cleanly when
  the cards can't share buffers. On those hosts the server now starts without
  `VLLM_SM80_ALLREDUCE=0`. Image
  `ghcr.io/morrowmake/vllm-cmp170hx@sha256:73e79648b5e2a32633885c65ea2e4a83ffcd7f543899ffb890eeef7edbac709d`.
- **Image notices.** The image's third-party notices now include GNmac-lab,
  ExLlamaV3 and Curtis White; licence label `Apache-2.0 AND MIT`.

## 1.0.1 — 2026-11-01

Faster decode with more users: each routed expert is now split across the
four cards.

- **Engine.** Upstream vLLM `330c5bffee` with our patch queue
  (`e05403fdc3750880da4a5f4cd8c55c5fae3b3ae0`, tag
  `dsv41-flash-recipe-1.0.1-queue`) and the sm_80 plugin `vllm_sm80`
  (`0b34823c39ba0fd5c0d4690887882b4cfa69034e`, tag
  `dsv41-flash-recipe-1.0.1-plugin`); image
  `ghcr.io/morrowmake/vllm-cmp170hx@sha256:32ae8b0e91a6264ba5cd20978d1a44460a8a5a2464641f377987414aa35071e2`.
  Torch 2.14. The image's `vllm._C_stable_libtorch` is rebuilt from the patch
  queue's Marlin source, so `docker/build.sh` from this repository rebuilds
  the published image from public sources: every file the same except that one
  library, whose machine code is identical.
- **Layout.** Tensor-parallel 4 with every routed expert split across the
  four cards instead of whole experts per card: each card holds a column
  chunk of every expert, rotated per expert and layer so all four cards carry
  the same routed work (`VLLM_EXL3_MOE_EXPERT_ROTATION`).
- **Compressed all-reduce** for mid-sized messages
  (`VLLM_SM80_ALLREDUCE_COMPRESS`), and one rounding for the attention
  output all-reduce (`VLLM_SM80_ALLREDUCE_ROUND_ONCE`). The all-reduce
  switches are renamed (`VLLM_SM80_ALLREDUCE`, `VLLM_SM80_PREFILL_REDUCE_OVERLAP`,
  `VLLM_SM80_SHARD_ARGMAX`); see [Engine switches](docs/engine-switches.md).
- **Faster start:** the routed experts load faster (`VLLM_EXL3_FAST_LOAD`).
- **Sampling defaults.** Requests that set neither get the model card's
  temperature 1.0 and top_p 1.0.
- **`EXTRA_ARGS` is checked:** it may not set an argument the release fixes
  (checkpoint, parallel layout, quantisation, memory budget, KV cache); the
  start stops and names it. `--speculative-config None` still turns drafting
  off.
- **Download fixes** ([#2](https://github.com/Morrowmake/dsv41-flash-cmp170hx-recipe/issues/2)):
  the checkpoint download fetches every file (1.0.0 could skip
  `config.json`), and the start's disk check counts what an interrupted
  download already left in `MODELS_DIR`.
- **`.env`:** no default changed.
- Measured on the release image beside the 1.0.0 image on the same machine
  (tokens per second over each run, time to the first token included): one
  user 297 structured, 297 code, 170 prose (1.0.0: 295, 283, 163); eight users
  together 868, 795 and 464 (1.0.0: 760, 712, 426); cold prefill 3,776 (2K
  prompt) and 3,824 (32K) (1.0.0: 3,810, 3,795); KV pool 1,295,894 tokens
  (1.0.0: 1,247,406). HumanEval 158/164 and GSM8K 1,285/1,319, level with
  1.0.0 ([Results](docs/results.md)).

Update with `./start.sh update` (keeps `.env`).

## 1.0.0 — 2026-10-24

First release: DeepSeek-V4.1-Flash-1017 on four CMP 170HX cards.

- Engine: Morrowmake/vllm-cmp170hx at `a0c57b52376efd67c57735bc949d4bafa22ae29e`, on upstream vLLM
  `4e0681ed56`; image `ghcr.io/morrowmake/vllm-cmp170hx@sha256:f67d9be47ffbd4a8b84c77fe76bdb3f842187d9b2786fff573e390ae5361bfb0`.
- **Model.** A hybrid checkpoint built on your machine as links into two
  downloads: the routed experts from
  `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` (EXL3, 2.9 bits per
  weight) at `c178a01284fc5c0e859d8668cffaac81ca8f4042`, every other weight
  from `deepseek-ai/DeepSeek-V4.1-Flash-1017` at
  `1bae0e1088993b9b82de08f59de5be5a34e42170` in its own precision. Its config,
  index and quantization config must have the validated bytes. About 680 GiB
  of disk.
- **Layout.** Tensor-parallel 4 with whole routed experts on each card;
  262,144-token context, up to 8 concurrent requests, BF16 KV cache.
- **Engram tables in host RAM, or from the SSD on smaller hosts**
  (`VLLM_ENGRAM_PLACEMENT=auto`). The container's entry point holds the tables
  (about 189 GiB; the server takes 202–204 GiB of host RAM in all) in host RAM
  when 212 GiB is available to it (`NEED_RAM_GIB`), and otherwise reads them
  from the checkpoint files, with a row cache per card sized from the host
  memory, filled at start when it is 12 GiB or more, and direct reads. One log
  line says which, for `./start.sh` and Docker Compose alike. The SSD path is
  slower: with a 38 GiB cache per card a cold 2K prompt's first token comes
  6 % later and decode is within 1 %; smaller caches are slower. Keep
  `MODELS_DIR` on an SSD. `ram` or `ssd` chooses directly; an explicit `ram`
  on a host without the memory stops before the start.
- **No swap.** The server runs with swap disabled: the container's memory
  limit, swap included, is the host's memory (`MEM_LIMIT` overrides);
  swapping the Engram tables costs ~8 % of prefill speed.
- **Speculative decoding** with the checkpoint's DSpark drafter: up to seven
  tokens per step for one request and five for more; from three requests on,
  each step's verify width is chosen from each request's recent acceptance.
  `EXTRA_ARGS="--speculative-config None"` turns it off; the start then
  clears the adaptive verify depth and checks the boot without draft counters.
- **The launch is part of the image.** `/opt/recipe/serve.sh` holds the engine
  command and every serving default, generated from the release settings;
  `docker/entry.sh` places the Engram tables, checks peer access and runs the
  boot check before it.
  Settings in `.env` override the defaults.
- Measured on the release image (tokens per second over each run, time to the
  first token included): one user up to 319 structured, 287 code, 162 prose;
  eight users together up to 765, 729 and 432; cold prefill up to 3,818 (2K
  prompt) and 3,808 (32K); a 1,247,406-token KV pool. HumanEval 157/164 and GSM8K 1,280/1,319,
  level with the reference ([Results](docs/results.md)).
