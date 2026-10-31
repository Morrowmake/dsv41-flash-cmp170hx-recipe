# Changelog

## 1.0.1 — 2026-11-01

Faster decode with more users: each routed expert is now split across the
four cards.

- **Engine.** Upstream vLLM `330c5bffee` with our patch queue
  (`0af5991c009e0c3ff4181f026a8e7dffe1370b45`, tag
  `dsv41-flash-recipe-1.0.1-queue`) and the sm_80 plugin `vllm_sm80`
  (`0b34823c39ba0fd5c0d4690887882b4cfa69034e`, tag
  `dsv41-flash-recipe-1.0.1-plugin`); image
  `ghcr.io/morrowmake/vllm-cmp170hx@sha256:6e02917e575153f2f0d9415c8f528c7b895979db89ad1e5520065f8e0c53d35e`.
  Torch 2.14.
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
- Measured on the release image (tokens per second over each run, time to the
  first token included): one user up to 290 structured, 292 code, 166 prose;
  eight users together up to 852, 794 and 457; cold prefill up to 3,758 (2K
  prompt) and 3,790 (32K); KV pool 1,295,894 tokens. HumanEval 158/164 and
  GSM8K 1,279/1,319, level with 1.0.0 ([Results](docs/results.md)).

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
