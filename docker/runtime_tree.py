#!/usr/bin/env python3
"""Construct and check the Git-free runtime payload before layer creation."""
import argparse
import bz2
import gzip
import hashlib
import lzma
import tarfile
import tempfile
import zipfile
import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def git_integrity(root, pin):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if git("rev-parse", "HEAD") != pin:
        raise ValueError("Source HEAD differs from the image pin")
    stores = [root / ".git"]
    modules = root / ".git/modules"
    if modules.exists():
        stores.extend(p.parent for p in modules.rglob("HEAD") if (p.parent / "objects").is_dir())
    for store in stores:
        if not store.is_dir() or (store / "objects/info/alternates").exists():
            raise ValueError("Git object store must be self-contained")
        result = subprocess.run(["git", "--git-dir", str(store), "fsck", "--full",
                                 "--unreachable", "--no-reflogs"], capture_output=True, text=True)
        if result.returncode or result.stdout.strip() or result.stderr.strip():
            raise ValueError("Git object integrity/reachability check failed")


def version(root):
    # The installed version was generated while Git was available. Do not
    # recompute it after removing history, or import the editable installation.
    tree = ast.parse((root / "vllm/_version.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Installed engine version is missing")



def scan_stream(stream, needles, label, depth=0, credential_patterns=()):
    """Scan bytes and standard nested archives, without extracting to the tree."""
    if depth > 12:
        raise ValueError("Archive nesting limit exceeded")
    stream.seek(0)
    header = stream.read(512)
    stream.seek(0)
    compressed = None
    if header.startswith(b"\x1f\x8b"):
        compressed = gzip.GzipFile(fileobj=stream)
    elif header.startswith(b"BZh"):
        compressed = bz2.BZ2File(stream)
    elif header.startswith(b"\xfd7zXZ\x00"):
        compressed = lzma.LZMAFile(stream)
    if compressed is not None:
        with compressed, tempfile.TemporaryFile() as expanded:
            shutil.copyfileobj(compressed, expanded, 4 * 1024 * 1024)
            expanded.seek(0)
            scan_stream(expanded, needles, label + "!compressed", depth + 1, credential_patterns)
    elif header.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(stream) as archive:
            check_stores(m.filename for m in archive.infolist())
            for member in archive.infolist():
                check_member(member.filename, needles)
                if not member.is_dir():
                    with archive.open(member) as source, tempfile.TemporaryFile() as expanded:
                        shutil.copyfileobj(source, expanded, 4 * 1024 * 1024)
                        expanded.seek(0)
                        scan_stream(expanded, needles, label + "!" + member.filename, depth + 1, credential_patterns)
    elif header[257:262] == b"ustar":
        with tarfile.open(fileobj=stream, mode="r:") as archive:
            members = set()
            for member in archive:
                check_member(member.name, needles)
                check_member(member.linkname, needles)
                members.add(member.name.rstrip("/"))
                if member.isfile():
                    with archive.extractfile(member) as source, tempfile.TemporaryFile() as expanded:
                        shutil.copyfileobj(source, expanded, 4 * 1024 * 1024)
                        expanded.seek(0)
                        scan_stream(expanded, needles, label + "!" + member.name, depth + 1, credential_patterns)
            check_stores(members)
    else:
        overlap = max(max(map(len, needles), default=1) - 1, 256 if credential_patterns else 0)
        previous = b""
        position = 0
        reviewed_hash = None
        reviews = json.loads(os.environ.get("IMAGE_REVIEWED_BYTE_MATCHES", "{}"))

        def reviewed(offset):
            nonlocal reviewed_hash
            if not reviews:
                return False
            if reviewed_hash is None:
                current = stream.tell()
                stream.seek(0)
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    h.update(block)
                reviewed_hash = h.hexdigest()
                stream.seek(current)
            return offset in reviews.get(reviewed_hash, [])

        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            data = previous + chunk.lower()
            for needle in needles:
                start = 0
                while True:
                    match = data.find(needle, start)
                    if match < 0:
                        break
                    if not reviewed(position - len(previous) + match):
                        raise ValueError("Forbidden identifier in runtime file: " + label)
                    start = match + 1
            # Credential candidates always fail, including in reviewed files.
            if any(p.search(data) for p in credential_patterns):
                raise ValueError("Credential candidate in runtime file: " + label)
            position += len(chunk)
            previous = data[-overlap:] if overlap else b""


def check_stores(names):
    paths = {Path(name) for name in names}
    heads = {p.parent for p in paths if p.name == "HEAD"}
    roots = {parent.parent for p in paths for parent in (p, *p.parents)
             if parent.name == "objects"}
    if heads & roots:
        raise ValueError("Retained archived Git object store")


def check_member(name, needles):
    parts = Path(name).parts
    if ".git" in parts or any(p.startswith("pack-") and p.endswith((".pack", ".idx")) for p in parts):
        raise ValueError("Retained archived Git metadata")
    if any(n in name.lower().encode() for n in needles):
        raise ValueError("Forbidden identifier in archive path")

def check(opt, pin, forbidden):
    root = opt / "vllm-src"
    record = json.loads((root / "provenance.json").read_text())
    expected = dict(schema=1, engine_commit=pin,
                    engine_repository="https://github.com/Morrowmake/vllm-cmp170hx.git",
                    engine_version=version(root))
    if record != expected or (root / "provenance.json").stat().st_mode & 0o222:
        raise ValueError("Runtime provenance mismatch or writable provenance")
    needles = [s.encode().lower() for s in forbidden if s]
    for directory, dirs, files in os.walk(opt, followlinks=False):
        parent = Path(directory)
        if {"HEAD", "objects"} <= set(dirs + files) or any(name.endswith((".pack", ".idx")) and name.startswith("pack-") for name in files):
            raise ValueError("Retained Git object store")
        for name in dirs + files:
            path = parent / name
            if name == ".git":
                raise ValueError("Retained Git metadata")
            metadata = str(path.relative_to(opt))
            if path.is_symlink():
                metadata += "\n" + os.readlink(path)
            if any(n in metadata.lower().encode() for n in needles):
                raise ValueError("Forbidden identifier in runtime path")
            if path.is_file() and not path.is_symlink():
                with path.open("rb") as stream:
                    scan_stream(stream, needles, str(path.relative_to(opt)))
    return record


def finalize(opt, pin, forbidden):
    root = opt / "vllm-src"
    git_integrity(root, pin)
    record = dict(schema=1, engine_commit=pin,
                  engine_repository="https://github.com/Morrowmake/vllm-cmp170hx.git",
                  engine_version=version(root))
    for directory, dirs, files in os.walk(opt, topdown=True, followlinks=False):
        if ".git" in dirs:
            path = Path(directory) / ".git"
            if path.is_symlink():
                path.unlink()
            else:
                shutil.rmtree(path)
            dirs.remove(".git")
        if ".git" in files:
            (Path(directory) / ".git").unlink()
    target = root / "provenance.json"
    target.write_text(json.dumps(record, sort_keys=True) + "\n")
    target.chmod(0o444)
    return check(opt, pin, forbidden)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("finalize", "check"))
    parser.add_argument("opt", type=Path)
    parser.add_argument("pin")
    parser.add_argument("--forbid", action="append", default=[])
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.pin):
        parser.error("pin must be a full lowercase commit hash")
    forbidden = args.forbid + json.loads(os.environ.get("IMAGE_FORBIDDEN_STRINGS", "[]"))
    action = finalize if args.mode == "finalize" else check
    print(json.dumps(action(args.opt, args.pin, forbidden), sort_keys=True))


if __name__ == "__main__":
    main()
