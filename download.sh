#!/usr/bin/env bash
# download.sh — fetch the two checkpoints into models/ (or MODELS_DIR) and
# build the served hybrid checkpoint from them (see "Where models are stored"
# in docs/how-to-use.md).
#
#   LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw   ~196 GiB (211 GB), 48 files
#   deepseek-ai/DeepSeek-V4.1-Flash-1017               ~475 GiB (510 GB), 50 files
#   DeepSeek-V4.1-Flash-1017-hybrid-exl3-fp8           links into the two, built here
#
# Budget ~680 GiB of free disk. With VLLM_ENGRAM_PLACEMENT=ssd the server reads
# the Engram tables from the official checkpoint while it runs: keep MODELS_DIR
# on an SSD then. Uses the hf client and the hybrid builder in the engine
# image, run as you, so run ./install.sh first. An interrupted download
# resumes. Runs the preflight first, so it stops before downloading if a card
# reports under 60 GiB or the disk is too small.
#
# Already complete? Exits 0. Re-fetch:  REFRESH_WEIGHTS=1 ./download.sh
#
# Equivalent to: ./start.sh download
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start.sh" download
