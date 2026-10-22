# Results

## Release {{VERSION}}

Measured {{RESULTS_DATE}} on 4× CMP 170HX by this release's own release check:
the engine `{{PIN}}`, on upstream vLLM `4e0681ed56`, the image this release
pins, and the release defaults, on {{SM_COUNT}} SMs per card, mainline
cmpunlocker plus the minimal P2P patch, verified P2P, PCIe x16 links,
262,144-token context, the Engram tables in host RAM and 180 W per card.
P2P-off results and `VLLM_ENGRAM_PLACEMENT=ssd` results are not published.

## Throughput

Decode uses the MiaAI-Lab protocol: 400 tokens, temperature 0, requests sent
with thinking off; one user is the median of five runs, and two, four and eight
users the median aggregate rate of three runs, all on one boot. Prompt reading
is a cold 32,768-token prefill.

| Users | Structured | Code | Prose |
|---|---:|---:|---:|
| 1 | {{C1_STRUCTURED}} | {{C1_CODE}} | {{C1_PROSE}} |
| 2, total | {{C2_STRUCTURED}} | {{C2_CODE}} | {{C2_PROSE}} |
| 4, total | {{C4_STRUCTURED}} | {{C4_CODE}} | {{C4_PROSE}} |
| 8, total | {{C8_STRUCTURED}} | {{C8_CODE}} | {{C8_PROSE}} |

| | |
|---|---:|
| Prompt reading (cold prefill), tok/s | {{PREFILL_TOK_S}} |
| KV pool, tokens | {{KV_TOKENS}} |

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
