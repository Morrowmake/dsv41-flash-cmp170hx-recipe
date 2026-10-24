# Engine switches

The release's launch sets every engine switch below explicitly; the image
carries it as `/opt/recipe/serve.sh`, generated from the settings this release
was validated with. Each performance feature can be switched off on its own
with one variable, without a rebuild:

```bash
VLLM_DEEPSEEK_V41_ATTN_PIPELINE=0 ./start.sh restart
```

Put it in `.env` to keep it (under Docker Compose, in
[`container.env`](../docker/container.env)). A value you set wins over the
release's; `./start.sh status` and the startup log show what the server got.
These switches exist for troubleshooting: with all of them at the release
values the server runs the configuration the published numbers and quality
checks were measured on, and with one off it runs a configuration nobody
measured.

## Troubleshooting with switches

If the server fails at startup or a reply looks wrong, the useful order is:

1. `P2P=off`: rules out peer-to-peer transfers ([PCIe peer-to-peer](how-to-use.md#pcie-peer-to-peer)).
2. `VLLM_ENGRAM_PLACEMENT=ssd` (or `ram`, when `auto` chose `ssd`): rules out
   the other place for the Engram tables
   ([Host memory](how-to-use.md#host-memory-and-the-engram-tables)).
3. Then the switches below one at a time, a family at a time when a switch
   needs another (the table says which). The engine refuses at startup a
   switch whose prerequisite is off, and says which.

Report what you found with the bug template; it asks for the startup banner
lines that show the switches.

## What the release turns on

Kernels and fusions. Each produces the same bytes as the path it replaces
unless the row says otherwise.

| Switch | Release | What it does |
|---|---|---|
| `VLLM_DEEPSEEK_V41_FUSED_ATTN_DECODE` | `1` | the attention of a decode step in fewer kernel launches |
| `VLLM_DEEPSEEK_V41_ATTN_PIPELINE` | `1` | the attention pipeline for decode-sized steps |
| `VLLM_DEEPSEEK_V41_SELECT_ONE_LAUNCH` | `1` | the attention pipeline's index selection in one launch instead of fifteen |
| `VLLM_DEEPSEEK_V41_SWA_META_BATCH` | `1` | the sliding-window metadata of every sliding-window KV cache group in one launch |
| `VLLM_DEEPSEEK_V41_FUSED_MHC` / `_FINISH` / `_FINISH_V2` | `1` | the mHC (manifold hyper-connection) mixing and its finish as fused kernels; `_FINISH` and `_FINISH_V2` need `FUSED_MHC` |
| `VLLM_DEEPSEEK_V41_MHC_SIDE_STREAM` | `1` | the mHC finish's coefficient work beside the layer's all-reduce, on a second stream |
| `VLLM_DEEPSEEK_V41_FUSED_ROUTE` | `1` | the MoE routing after the gate as one kernel |
| `VLLM_DEEPSEEK_V41_GATE_GEMV` | `1` | the decode-sized MoE gate as one kernel, FP64 sums rounded once |
| `VLLM_DEEPSEEK_V41_ROUTER_MMA` | `1` | the MoE gate and routing on the BF16 tensor cores; needs `FUSED_ROUTE` and `GATE_GEMV` |
| `VLLM_DEEPSEEK_V41_WO_A_GROUPED_GEMM` | `1` | the attention output's grouped FP8 projection as one grouped GEMM |
| `VLLM_DEEPSEEK_V41_DRAFT_PROLOGUE_FUSED` | `1` | the DSpark drafter's context prologue as one kernel |
| `VLLM_EXL3_MOE_V2` | `1` | EXL3 routed experts on the tensor cores for calls of 1 to 64 rows; changes the summation order |
| `VLLM_EXL3_MOE_V2_FOLD` / `_RING` / `_DRAFT` | `1` | that kernel in three launches per call, its weight reads through a shared ring, and for the drafter's experts too; each needs `VLLM_EXL3_MOE_V2` |
| `VLLM_DEEPSEEK_V41_MOE_PREFILL_TC` | `1` | EXL3 routed experts on the tensor cores for prefill calls of more than 64 rows |
| `VLLM_EXL3_MOE_PREFILL_UNCHUNKED` | `1` | such a call of up to 2,048 rows as one launch sequence; needs `MOE_PREFILL_TC` |
| `VLLM_EXL3_MOE_CHUNK_ROWS` | `2048` | routed-expert calls processed in chunks of up to 2,048 rows; the workspace is sized for it at startup |
| `VLLM_EXL3_EXPERTS_CONTIGUOUS` | `1` | each card's routed experts stored contiguously, fixed at loading |
| `VLLM_SM80_TARGET_SHARD_ARGMAX` / `VLLM_DSPARK_SHARD_ARGMAX` | `1` | greedy sampling of the model and of the drafter by an argmax over each card's share of the vocabulary |
| `VLLM_DEEPSEEK_V41_ENGRAM_EARLY_GATHER` | `1` | Engram lookups started early on a second stream; needs full CUDA graphs for decode (the release's mode) |
| `VLLM_DEEPSEEK_V41_ENGRAM_WKV_REPLICATED_PREFILL` | `1` | every card holds the whole Engram projection and projects prefill steps without gathering it |

Speculative decoding:

| Switch | Release | What it does |
|---|---|---|
| `VLLM_DSPARK_PER_K_WIDTH` | `1` | the drafter and the verify step built for the step's true width, so a narrower step costs less |
| `VLLM_DSPARK_DEPTH_HINT` | `full=7,5;base=5,5` | seven drafts only while a single request runs, five otherwise |
| `VLLM_DSPARK_ADAPTIVE_DEPTH` | the policy file shipped in the engine | from three requests on, each step's verify width chosen from each request's acceptance two rounds back; a request keeps only the drafts worth verifying, so greedy output is unchanged (cleared at start when `EXTRA_ARGS` turns speculative decoding off) |

Communication between the cards:

| Switch | Release | What it does |
|---|---|---|
| `VLLM_CUSTOM_ALLREDUCE_PCIE_P2P` | `1` after a P2P pass, else `0` | the custom all-reduce over PCIe peer-to-peer |
| `VLLM_CUSTOM_ALLREDUCE_FLAGS` / `_ALGO` / `_FLAGS_MAX_BYTES` | `1` / `2stage` / `524288` | its flags-in-data variant, two-stage, for messages up to 512 KiB |
| `VLLM_HOST_ALLREDUCE` | `1` | a host-memory all-reduce for decode messages; stands aside while the peer-to-peer all-reduce serves |
| `VLLM_DEEPSEEK_V41_WO_B_BF16_ALLREDUCE` / `_PREFILL_ALLREDUCE` | `1` | the attention output's partial sums all-reduced in BF16 instead of FP32, for decode and for prefill |
| `VLLM_DEEPSEEK_V41_PREFILL_SPLIT_REDUCE` | `1` | prefill steps of 1,024 rows or more in two row slices, one slice's all-reduce beside the other's compute |
| `VLLM_DEEPSEEK_V41_PREFILL_CE_ALLREDUCE` | `1` | prefill-sized all-reduces on a copy-engine ring instead of NCCL |
| `NCCL_P2P_LEVEL` | `SYS` after a P2P pass, else `LOC` | NCCL over peer-to-peer |

Memory and placement:

| Switch | Release | What it does |
|---|---|---|
| `VLLM_ENGRAM_PLACEMENT` | `auto`: `ram` with 212 GiB of host memory available | the Engram tables in host RAM; `ssd` reads them from the checkpoint files; `auto` (the recipe's entry point) picks one at each start |
| `VLLM_ENGRAM_ROW_CACHE_GIB` | under `ssd`: from the host memory, 1–48 | the host row cache for `ssd` placement, in GiB per card; unused by `ram` |
| `VLLM_ENGRAM_ROW_CACHE_PRELOAD` | under `ssd`: `1` for a cache of 12 GiB or more | fills the row cache at start with one sequential read of each card's share |
| `VLLM_ENGRAM_DIRECT_READ` | under `ssd`: `1` | reads cache misses past the page cache; `0` on a file system that refuses direct reads |
| `VLLM_EXL3_HYBRID` | `1` | the hybrid checkpoint: EXL3 routed experts, the official FP8 weights for the rest |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | `1` | CUDA graphs that leave room for the Engram row staging |
| `VLLM_CUDAGRAPH_ESTIMATE_SPLIT` | `1` | the CUDA graph memory estimate the KV pool is sized with |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | `128` | the prefill indexer's score buffer, filled in more passes than at the default 512; the memory goes to the KV cache |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:False` | the allocator mode the custom all-reduce needs |

Scheduling:

| Setting | Release | What it does |
|---|---|---|
| `--prefill-chunk-with-decodes` | `512` | while other requests decode, a long prompt prefills in 512-token pieces, so running answers keep streaming |
| `VLLM_FAIR_PREFILL_SHORT_BYPASS` | `1` | a prompt no longer than that piece is scheduled whole beside decoding requests, so a burst of short prompts starts in one step |
| `--exclude-tools-when-tool-choice-none` | on | with `tool_choice: "none"` the tool definitions are left out of the prompt, so the model answers in text |

## What the release leaves off

These exist in the engine and stay at their defaults, which are off; they were
not part of the measured configuration: `VLLM_DEEPSEEK_V41_MOE_ROUND_ONCE_ALLREDUCE`,
`VLLM_ROUND_ONCE_ALLREDUCE_GROUPED`, an expert placement
(`VLLM_EXL3_EXPERT_PLACEMENT`), and the diagnostic switches `VLLM_DEV_AB_SWITCH`
and `VLLM_SM80_CAPTURE_DIR`. The launch does not set them.
