#!/usr/bin/env python3
"""Compile the boot probe without initializing CUDA."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def build(source, destination, nvcc):
    enabled = os.environ.get("IMAGE_P2P_PREBUILD", "1")
    if enabled not in ("0", "1"):
        raise ValueError("IMAGE_P2P_PREBUILD must be 0 or 1")
    destination.mkdir(parents=True, exist_ok=True)
    print(f"[image-p2p] prebuild={enabled}", flush=True)
    if enabled == "0":
        return
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    binary = destination / ("probe-" + digest)
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        shutil.copyfile(source, work / "probe.cu")
        subprocess.run([str(nvcc), "-O2", "-std=c++17", "-arch=sm_80",
                        "-o", "probe", "probe.cu"], check=True, cwd=work,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), timeout=120)
        shutil.copyfile(work / "probe", binary)
        binary.chmod(0o755)
    print(f"[image-p2p] compile_s={time.monotonic()-started:.3f} source_sha256={digest}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--nvcc", type=Path, default=Path(os.environ.get(
        "CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc")
    args = parser.parse_args()
    build(args.source, args.destination, args.nvcc)
