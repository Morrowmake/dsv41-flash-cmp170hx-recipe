# Changelog

## {{VERSION}} — {{RELEASE_DATE}}

First release: DeepSeek-V4.1-Flash-1017 on four CMP 170HX cards.

- Engine: Morrowmake/vllm-cmp170hx at `{{PIN}}`, on upstream vLLM
  `4e0681ed56`; image `ghcr.io/morrowmake/vllm-cmp170hx@sha256:{{IMAGE_DIGEST}}`
  (seed: {{SEED_YES_NO}}).
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
- **Engram tables in host RAM** (about 189 GiB; the server takes 202–204 GiB
  of host RAM in all; `VLLM_ENGRAM_PLACEMENT=ram`). The preflight checks the
  host has room; `VLLM_ENGRAM_PLACEMENT=ssd` reads them from the SSD on
  smaller hosts, slower.
- **Speculative decoding** with the checkpoint's DSpark drafter: up to seven
  tokens per step for one request and five for more; from three requests on,
  each step's verify width is chosen from each request's recent acceptance.
- **The launch is part of the image.** `/opt/recipe/serve.sh` holds the engine
  command and every serving default, generated from the release settings;
  `docker/entry.sh` checks peer access and runs the boot check before it.
  Settings in `.env` override the defaults.
- Measured on the release image: {{C1_TOK_S}} tok/s for one user,
  {{C8_TOK_S}} tok/s across eight, {{PREFILL_TOK_S}} tok/s cold prefill, a
  {{KV_TOKENS}}-token KV pool ([Results](docs/results.md)).
