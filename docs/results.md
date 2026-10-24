# Results

## Release 1.0.0

Measured {{RESULTS_DATE}} on 4× CMP 170HX by this release's own release check:
the engine `a0c57b52376efd67c57735bc949d4bafa22ae29e`, on upstream vLLM `4e0681ed56`, the image this release
pins, and the release defaults, on 74 SMs per card, mainline
cmpunlocker plus the minimal P2P patch, verified P2P, PCIe x16 links,
262,144-token context, the Engram tables in host RAM and 180 W per card.
The fallback rows at the end are measured once each on the same image and
reported for reference; the published numbers are the defaults above.

## Throughput

Decode uses the MiaAI-Lab protocol: 400 tokens, temperature 0, requests sent
with thinking off; one user is the median of five runs, and two, four and eight
users the median aggregate rate of three runs, all on one boot. Prompt reading
is a cold 2,048- and 32,768-token prefill.

| Users | Structured | Code | Prose |
|---|---:|---:|---:|
| 1 | {{C1_STRUCTURED}} | {{C1_CODE}} | {{C1_PROSE}} |
| 2, total | {{C2_STRUCTURED}} | {{C2_CODE}} | {{C2_PROSE}} |
| 4, total | {{C4_STRUCTURED}} | {{C4_CODE}} | {{C4_PROSE}} |
| 8, total | {{C8_STRUCTURED}} | {{C8_CODE}} | {{C8_PROSE}} |

| | |
|---|---:|
| Prompt reading (cold prefill, 2,048 tokens), tok/s | {{PREFILL_2K_TOK_S}} |
| Prompt reading (cold prefill, 32,768 tokens), tok/s | {{PREFILL_32K_TOK_S}} |
| KV pool, tokens | 1,247,406 |

## Quality

Scored on the answer after the reasoning (`content`), fixed-order batches of up
to eight; length-capped outputs remain in the scores. The reference is the same
checkpoint served with only what the model needs to run on these cards: every
speed switch off and the prefill budget at the engine default.

| | Release | Reference |
|---|---:|---:|
| HumanEval | {{HUMANEVAL}} | {{HUMANEVAL_REF}} |
| GSM8K | {{GSM8K}} | {{GSM8K_REF}} |

{{QUALITY_VERDICT}}

## Repeatability

A request on its own is repeatable, and so is the same request served from the
prefix cache, but a prompt whose start is cached and the same prompt read cold
can diverge at a near-tie ([Known limits](known-limits.md)).

## Fallbacks (reference)

Each measured on one boot of the release image, prose prompts, same protocol:

| | 1 user | 8 users, total | Cold prefill 2,048 tokens, tok/s |
|---|---:|---:|---:|
| Engram tables from the SSD (`VLLM_ENGRAM_PLACEMENT=ssd`), {{SSD_CACHE_GIB}} GiB row cache per card | {{SSD_C1}} | {{SSD_C8}} | {{SSD_PREFILL_2K}} |
| Without peer-to-peer (`P2P=off`) | {{P2POFF_C1}} | {{P2POFF_C8}} | {{P2POFF_PREFILL_2K}} |
