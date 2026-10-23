# Status and known limits

## Status and roadmap

- **First release: one layout.** Tensor-parallel 4 with whole routed experts
  on each card, for x16 links.
- **Optimisation continues**; every change ships with a switch and measured
  numbers.

## Known limits

- **Ampere only.** Every kernel in the fork is written for sm_80. On newer GPUs
  upstream vLLM's own kernels are better, and forcing these features on
  elsewhere is untested.
- **Wide links for tensor-parallel.** See [Link width](how-to-use.md#link-width).
- **Host memory.** The Engram tables (~189 GiB) are held in host RAM, and the
  server takes 202–204 GiB of it in all. On a host with less, they are read
  from the SSD instead, automatically; that runs on smaller hosts but is
  slower, more so the less memory there is for the row cache, and the
  published numbers were measured with the tables in RAM
  ([Host memory](how-to-use.md#host-memory-and-the-engram-tables)).
- **Disk.** The served checkpoint is built from the full official FP8
  checkpoint (~475 GiB, which also holds the Engram tables) and the EXL3
  export (~196 GiB): about 680 GiB in all.
- **A cached prompt can give a different reply from a fresh one.** A request on
  its own is repeatable, and so is the same request served from the prefix
  cache. But a prompt whose start is already cached is processed in different
  pieces than the same prompt read cold, so at a near-tie the two can pick
  different tokens and the replies diverge from there. When requests share a
  batch, the result can also differ slightly from the same request alone.
- **Context and KV are a trade.** A longer context per request leaves room for
  fewer full-length requests at once.
- **Host-memory all-reduce is only for cards without peer access.** Where
  peer-to-peer works, it stands aside for the device-memory path.
