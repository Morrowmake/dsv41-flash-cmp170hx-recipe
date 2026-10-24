# Results

## Release 1.0.0

Measured 2026-10-24 on 4× CMP 170HX by this release's own release check:
the engine `a0c57b52376efd67c57735bc949d4bafa22ae29e`, on upstream vLLM `4e0681ed56`, the image this release
pins, and the release defaults, on 74 SMs per card, mainline
cmpunlocker plus the minimal P2P patch, verified P2P, PCIe x16 links,
262,144-token context, the Engram tables in host RAM and 180 W per card.
The fallback rows at the end are measured once each on the same image and
reported for reference; the published numbers are the defaults above.

## Throughput

Decode uses the MiaAI-Lab protocol: 400 tokens, temperature 0, requests sent
with thinking off; structured and code from its first prompt set, prose from
its second, each set on its own boot. Every figure is the median of five runs:
a run's completion tokens over its whole duration, from the first request sent
to the last answer finished, so the time to the first token is included (for
two to eight users, the rate of all requests together). Prompt reading is a
cold 2,048- and 32,768-token prefill: prompt tokens over the time to the first
token, the median of two requests.

| Users | Structured | Code | Prose |
|---|---:|---:|---:|
| 1 | 319 | 287 | 162 |
| 2, total | 360 | 431 | 230 |
| 4, total | 545 | 602 | 324 |
| 8, total | 765 | 729 | 432 |

| | |
|---|---:|
| Prompt reading (cold prefill, 2,048 tokens), tok/s | 3,818 |
| Prompt reading (cold prefill, 32,768 tokens), tok/s | 3,808 |
| KV pool, tokens | 1,247,406 |

## Quality

Scored on the answer after the reasoning (`content`), fixed-order batches of up
to eight; length-capped outputs remain in the scores. The reference is the same
checkpoint served with only what the model needs to run on these cards: every
speed switch off and the prefill budget at the engine default.

| | Release | Reference |
|---|---:|---:|
| HumanEval | 157 / 164 | 157 / 164 |
| GSM8K | 1,280 / 1,319 | 1,279 / 1,319 |

The release and the reference agree within the test's noise on both: HumanEval
differs on no problem, GSM8K on 15 (8 more right, 7 fewer; McNemar exact
two-sided p = 1).

## Repeatability

A request on its own is repeatable, and so is the same request served from the
prefix cache, but a prompt whose start is cached and the same prompt read cold
can diverge at a near-tie ([Known limits](known-limits.md)).

## Fallbacks (reference)

Each measured on one boot of the release image, prose prompts, same protocol:

| | 1 user | 8 users, total | Cold prefill 2,048 tokens, tok/s |
|---|---:|---:|---:|
| Engram tables from the SSD (`VLLM_ENGRAM_PLACEMENT=ssd`), 48 GiB row cache per card | 151 | 428 | 3,607 |
| Without peer-to-peer (`P2P=off`) | 133 | 344 | 3,804 |

Without peer-to-peer, two and four users reach 192 and 271 tok/s.
