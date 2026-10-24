# Changelog

## 1.0.0 — 2026-10-24

First release: DeepSeek-V4.1-Flash-1017 on four CMP 170HX cards.

- Engine: Morrowmake/vllm-cmp170hx at `a0c57b52376efd67c57735bc949d4bafa22ae29e`, on upstream vLLM
  `4e0681ed56`; image `ghcr.io/morrowmake/vllm-cmp170hx@sha256:f67d9be47ffbd4a8b84c77fe76bdb3f842187d9b2786fff573e390ae5361bfb0`
  (seed: no).
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
  first token included): one user 319 structured, 287 code, 162 prose; eight
  users together 765, 729 and 432; cold prefill 3,818 (2K prompt) and 3,808
  (32K); a 1,247,406-token KV pool. HumanEval 157/164 and GSM8K 1,280/1,319,
  level with the reference ([Results](docs/results.md)).
