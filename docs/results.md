# Results

## Release 1.0.1

Measured 2026-11-01 on 4× CMP 170HX by this release's own release check:
upstream vLLM `330c5bffee` with the patch queue `e05403fdc3` and the plugin
`vllm_sm80` `0b34823c39`, the image this release pins
(`sha256:32ae8b0e91a6264ba5cd20978d1a44460a8a5a2464641f377987414aa35071e2`),
and the release defaults, on 74 SMs per card, mainline cmpunlocker plus the
minimal P2P patch, verified P2P, PCIe x16 links, 262,144-token context, the
Engram tables in host RAM and 180 W per card. The 1.0.0 image was measured on
the same machine in the same check, two boots of each, alternating. Method as
for 1.0.0 below; each figure is the median of five runs on a boot, averaged
over the two boots; prompt reading is the median of four cold requests.

| Users | Structured | Code | Prose | 1.0.0 (structured / code / prose) |
|---|---:|---:|---:|---|
| 1 | 297 | 297 | 170 | 295 / 283 / 163 |
| 2, total | 392 | 458 | 260 | 359 / 423 / 235 |
| 4, total | 569 | 642 | 352 | 532 / 600 / 323 |
| 8, total | 868 | 795 | 464 | 760 / 712 / 426 |

| | 1.0.1 | 1.0.0 |
|---|---:|---:|
| Prompt reading (cold prefill, 2,048 tokens), tok/s | 3,776 | 3,810 |
| Prompt reading (cold prefill, 32,768 tokens), tok/s | 3,824 | 3,795 |
| KV pool, tokens | 1,295,894 | 1,247,406 |

All speeds in tok/s.

**Quality**, scored as for 1.0.0, against the 1.0.0 release: HumanEval
158 / 164 (1.0.0: 157), GSM8K 1,285 / 1,319 (1.0.0: 1,280); McNemar exact
two-sided p = 1 and 0.23. The release also reproduced the exact-model anchor,
found every needle up to 262,000 tokens (30 / 30) and held memory flat over
the varying-tail requests.

**Fallbacks**, each measured on one boot of the release image, prose prompts,
same protocol:

| | 1 user | 2 users, total | 4 users, total | 8 users, total | Cold prefill 2,048 tokens, tok/s |
|---|---:|---:|---:|---:|---:|
| Engram tables from the SSD (`VLLM_ENGRAM_PLACEMENT=ssd`), 48 GiB row cache per card | 168 | — | — | 468 | 3,658 |
| Without peer-to-peer (`P2P=off`) | 141 | 202 | 280 | 357 | 3,739 |

All speeds in tok/s.

**Reproducing the image.** `docker/build.sh` from this repository rebuilds the
published image from public sources: every file is the same except
`vllm._C_stable_libtorch`, rebuilt from the patch queue, whose machine code is
identical (its bytes differ only in compiler-generated symbol names).

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

All speeds are up to, in tok/s.

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

All speeds are up to, in tok/s.

Without peer-to-peer, two and four users reach up to 192 and 271 tok/s.
