#!/usr/bin/env python3
"""Partition a finalized runtime into independently reusable image layers."""
import argparse
from email.parser import Parser
import os
from pathlib import Path
import re
import shutil

PARTS = ("python", "dependencies", "native", "engine")

def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()
# Distributions built from the engine source tree; they ship in the engine
# layer and are versioned by the engine commit, not by the dependency lock.
ENGINE_DISTRIBUTIONS = ("vllm", "vllm-sm80-plugin")

def check_lock(opt, constraints):
    expected = {}
    for line in constraints.read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            expected[normalized(name)] = version
    actual = {}
    for path in opt.glob("venv/lib/python*/site-packages/*.dist-info/METADATA"):
        metadata = Parser().parsestr(path.read_text())
        name = normalized(metadata["Name"])
        if name not in ENGINE_DISTRIBUTIONS:
            actual[name] = metadata["Version"]
    if actual != expected:
        differences = sorted(n for n in actual.keys() | expected.keys()
                             if actual.get(n) != expected.get(n))
        raise ValueError("Dependency lock mismatch: " + ", ".join(differences))

def partition(path):
    parts = path.parts
    if parts[0] == "python":
        return "python"
    if parts[0] == "vllm-src":
        return "native" if path.suffix == ".so" else "engine"
    if parts[0] != "venv":
        raise ValueError("Unexpected runtime payload root")
    if (any(p.startswith(("vllm-", "vllm_sm80_plugin-", "__editable__.vllm", "__editable___vllm"))
            for p in parts)
            or path.name in ("vllm", ".native-manifest.json")):
        return "engine"
    return "dependencies"

def split(opt, out, constraints):
    check_lock(opt, constraints)
    out.mkdir(parents=True, exist_ok=False)
    for part in PARTS:
        (out / part).mkdir()
    for root, dirs, files in os.walk(opt, followlinks=False):
        for name in sorted(dirs + files):
            source = Path(root) / name
            if source.is_dir() and not source.is_symlink():
                continue
            rel = source.relative_to(opt)
            target = out / partition(rel) / "opt" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink():
                target.symlink_to(os.readlink(source))
            else:
                shutil.copyfile(source, target)
                target.chmod(source.stat().st_mode & 0o777)
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("opt", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("constraints", type=Path)
    a = p.parse_args()
    split(a.opt, a.out, a.constraints)
