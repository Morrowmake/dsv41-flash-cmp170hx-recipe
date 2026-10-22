#!/usr/bin/env python3
"""Check local OCI integrity and scan every distributed layer and config."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import io
import json
import os
import re
from pathlib import Path
import sys
import tarfile

from oci_layers import digest_file, image
from runtime_tree import scan_stream

# Candidates are reported without including their value.
TOKEN_PATTERNS = [re.compile(rb"(?<![a-z0-9_-])github_pat_[a-z0-9_]{40,}"), re.compile(rb"(?<![a-z0-9_-])gh[pousr]_[a-z0-9]{30,}")]

def layer_check(job):
    root, desc, diff, forbidden, patterns = job
    path = root / "blobs/sha256" / desc["digest"].split(":")[1]
    if digest_file(path) != desc["digest"] or path.stat().st_size != desc["size"]:
        raise ValueError("Layer descriptor mismatch")
    h = hashlib.sha256()
    with gzip.open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    if "sha256:" + h.hexdigest() != diff:
        raise ValueError("Layer diff ID mismatch")
    with path.open("rb") as f:
        scan_stream(f, forbidden, desc["digest"], credential_patterns=patterns)
    tmp = None
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            name = member.name.removeprefix("./").rstrip("/")
            if name == "tmp":
                tmp = dict(mode=member.mode, uid=member.uid, gid=member.gid,
                           directory=member.isdir())
            elif name in (".wh.tmp", ".wh..wh..opq"):
                tmp = dict(directory=False)
    return dict(digest=desc["digest"], passed=True, tmp=tmp)

def checked_layer(job):
    try:
        return layer_check(job)
    except ValueError as error:
        return dict(digest=job[1]["digest"], passed=False, reason=str(error))

def audit(root, workers=8):
    reviews = json.loads(os.environ.get("IMAGE_REVIEWED_BYTE_MATCHES", "{}"))
    print("[image-audit] reviewed_byte_matches=" + str(sum(len(v) for v in reviews.values())), file=sys.stderr)
    manifest, config = image(root)
    forbidden = [s.lower().encode() for s in json.loads(os.environ.get("IMAGE_FORBIDDEN_STRINGS", "[]"))]
    patterns = TOKEN_PATTERNS + [re.compile(p.encode()) for p in
                  json.loads(os.environ.get("IMAGE_FORBIDDEN_PATTERNS", "[]"))]
    metadata = json.dumps(dict(manifest=manifest, config=config)).encode()
    scan_stream(io.BytesIO(metadata), forbidden, "OCI metadata", credential_patterns=patterns)
    if len(manifest["layers"]) != len(config["rootfs"]["diff_ids"]):
        raise ValueError("Layer count mismatch")
    jobs = [(root, d, diff, forbidden, patterns) for d, diff in
            zip(manifest["layers"], config["rootfs"]["diff_ids"])]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        checked = list(pool.map(checked_layer, jobs))
    tmp = None
    for row in checked:
        if row.get("tmp") is not None:
            tmp = row["tmp"]
    tmp_passed = tmp == dict(mode=0o1777, uid=0, gid=0, directory=True)
    record = dict(passed=all(r["passed"] for r in checked) and tmp_passed,
                  tmp=dict(passed=tmp_passed, entry=tmp), layers=len(checked), checked=checked,
                  reviewed_byte_matches=reviews,
                  formats=["OCI JSON", "gzip", "bzip2", "xz", "tar", "zip"],
                  gaps=["Embedded ELF fatbins and nonstandard compression (including zstd) require separate inspection",
                        "Credential candidates use known registry-token shapes"])
    print(json.dumps(record, sort_keys=True))
    return record

if __name__ == "__main__":
    result = audit(Path(sys.argv[1]), int(os.environ.get("IMAGE_AUDIT_WORKERS", "8")))
    sys.exit(0 if result["passed"] else 1)
