#!/usr/bin/env python3
"""Partition a finalized runtime into independently reusable image layers."""
import argparse
import base64
import csv
from email.parser import Parser
import hashlib
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

# A file two locked distributions both install (site-packages-relative path -> the distribution whose bytes the
# image keeps). The installer writes such files in no fixed order, so the dependencies layer differed between builds
# of one lock: the build reinstalls the winner last (winner_reinstalls) and check_shared refuses any other overlap and
# any shared file that is not the winner's bytes.
SHARED_FILES = {
    # Both wheels ship a top-level PEP 517 backend module of this name; nothing imports it at run time. The released
    # 1.0.1 image (sha256:32ae8b0e) holds flashinfer-python's, so a rebuild keeps those bytes.
    "build_backend.py": "flashinfer-python",
}


def site_packages(opt):
    found = sorted(opt.glob("venv/lib/python*/site-packages"))
    if len(found) != 1:
        raise ValueError("expected one venv site-packages under %s, found %d" % (opt, len(found)))
    return found[0]


def claims(opt):
    """{site-packages-relative path: [(distribution, RECORD sha256 field)]} over every installed RECORD."""
    out = {}
    for record in sorted(site_packages(opt).glob("*.dist-info/RECORD")):
        name = normalized(Parser().parsestr((record.parent / "METADATA").read_text())["Name"])
        for row in csv.reader(record.read_text().splitlines()):
            if row and row[0]:
                out.setdefault(os.path.normpath(row[0]), []).append((name, row[1] if len(row) > 1 else ""))
    return out


def check_shared(opt):
    """Refuse a file two distributions claim unless SHARED_FILES names it and its bytes are the winner's."""
    sp, problems = site_packages(opt), []
    for path, owners in sorted(claims(opt).items()):
        if len(owners) < 2:
            continue
        names = sorted(n for n, _ in owners)
        winner = SHARED_FILES.get(path)
        if winner not in names:
            problems.append("%s is installed by %s (no SHARED_FILES winner)" % (path, ", ".join(names)))
            continue
        want = dict(owners)[winner]
        digest = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256((sp / path).read_bytes()).digest()).rstrip(b"=").decode()
        if digest != want:
            problems.append("%s holds other bytes than %s's (%s, want %s)" % (path, winner, digest, want))
    if problems:
        raise ValueError("Files claimed by two distributions: " + "; ".join(problems))


def winner_reinstalls(constraints):
    """The locked `name==version` of each SHARED_FILES winner the lock holds, reinstalled after the dependencies so
    its bytes win (a winner outside the lock shares no file in this venv)."""
    locked = {}
    for line in Path(constraints).read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            locked[normalized(name)] = "%s==%s" % (name, version)
    return [locked[n] for n in sorted(set(SHARED_FILES.values()) & set(locked))]


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
    check_shared(opt)

def partition(path):
    parts = path.parts
    if parts[0] == "python":
        return "python"
    # /opt/vllm-src: a fork tree; /opt/assembly: an assembled engine (build.sh)
    if parts[0] in ("vllm-src", "assembly"):
        return "native" if path.suffix == ".so" else "engine"
    # /opt/build holds the engine tree's build record: it belongs to the engine
    if parts[0] == "build":
        return "engine"
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
