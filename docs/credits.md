# Third-party credits (1.0.2)

The engine for this release is upstream vLLM with our patch queue and the
`vllm_sm80` plugin, at tags `dsv41-flash-recipe-1.0.2-queue` and
[`dsv41-flash-recipe-1.0.2-plugin`](https://github.com/Morrowmake/vllm-cmp170hx/tree/dsv41-flash-recipe-1.0.2-plugin). The table lists every outside source
whose code or design the 1.0.2 engine uses, and where. Paths are in the plugin
tree. Original per-file notices apply as well; this page does not replace them.

| Source | What is used | Where | Licence |
|---|---|---|---|
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3/tree/d3739fd393337b1ff4d6c2a342b12f0c87a9592f) (Turboderp), `d3739fd` | EXL3 trellis packing, procedural codebooks, tensor-core lane ordering, Hadamard conventions and the n-gram codec mapping, adapted | `csrc/exl3/` (`codec.cuh` and the files that include it), `vllm_sm80/kernels/exl3/` | MIT; notice in `licenses/exl3/` and below |
| [DeepSeekv41-Flash-EXL3-TC-MoE](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE/tree/88eb6e1) (Curtis White), `88eb6e1` | The hidden-5120 tensor-core MoE lane shape, adapted; the idea of splitting each routed expert across the cards | `csrc/exl3/moe_v2.cu`, `csrc/exl3/moe_v2_try.cu`; the expert split (idea only) | Apache-2.0; notice in `licenses/exl3/NOTICE`, licence text in the plugin's `LICENSE` |
| [deepseek-v41-flash-1017-cmp170hx](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx/tree/bbaf367) (GNmac-lab), `bbaf367` | The E4M3 decode and grouped tile loop of the FP8 `wo_a` GEMM, adapted; the ideas of the greedy target argmax, the side-stream mHC and the per-layer rotated tensor split of the routed experts | `vllm_sm80/models/deepseek_v41/ops/sm80_wo_a_grouped.py` (code); `vllm_sm80/worker/target_argmax.py`, `vllm_sm80/models/deepseek_v41/ops/sm80_mhc_side.py` (ideas) | MIT; notice in the source header and below |
| [b12x](https://github.com/local-inference-lab/b12x) (local-inference-lab) | The design of the copy-engine ring all-reduce in `b12x/comm/pcie/pcie_dma.py`; no code is copied | `vllm_sm80/distributed/ce_ring_all_reduce.py` | Apache-2.0 (design credit only) |
| [Marlin](https://github.com/IST-DASLab/marlin) (Elias Frantar and Neural Magic contributors) | The Marlin MoE kernels, through upstream vLLM | `csrc/ampere_marlin/` | Apache-2.0; original notices kept in the files |
| [DeepGEMM](https://github.com/vllm-project/DeepGEMM/tree/e1f418c2a4f20818221f6b0e578b4c2f634d4c3f) (DeepSeek) | The `fp8_paged_mqa_logits_torch` attention reference, through upstream vLLM | `vllm_sm80/kernels/attention/triton_mqa_logits.py` | MIT; notice in the engine's `docs/THIRD_PARTY.md` and the image's notices file |
| [vLLM](https://github.com/vllm-project/vllm) contributors | The engine; the sparse-MLA kernel adapts upstream's XPU implementation | the patch queue; `vllm_sm80/kernels/attention/triton_mla_sparse.py` | Apache-2.0 |

The checkpoints' licences are in [model-licences.md](model-licences.md); the
benchmark prompts are covered in the README's Licences section.

The engine image's notices file (`docker/THIRD_PARTY_NOTICES`, also at
`/opt/image-tools/THIRD_PARTY_NOTICES` in the image, named by the image label
`org.morrowmake.notices`) carries the ExLlamaV3, GNmac-lab, Curtis White and
DeepGEMM notices in full; the plugin's `licenses/` directory carries the
ExLlamaV3 and Curtis White notices as well.

## ExLlamaV3 MIT notice

MIT License

Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## GNmac-lab MIT notice

MIT License

Copyright (c) 2026 GNmac-lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
