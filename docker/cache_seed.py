#!/usr/bin/env python3
"""Create and import an integrity-checked, device-specific compilation cache."""
import argparse
import ctypes
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import sys
import tarfile
import tempfile

ROOTS = (".triton/cache", ".cache/flashinfer", ".cache/vllm", ".tilelang/cache", "torchinductor",
         ".cache/torch_extensions", ".nv/ComputeCache")

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

def pci_device_id(uuid):
    import pynvml
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return None
    try:
        # UUID identifies the CUDA-visible device even when ordinals are remapped.
        value = str(uuid)
        if not value.startswith("GPU-"):
            value = "GPU-" + value
        handle = pynvml.nvmlDeviceGetHandleByUUID(value)
        return pynvml.nvmlDeviceGetPciInfo(handle).pciDeviceId
    except pynvml.NVMLError:
        return None
    finally:
        pynvml.nvmlShutdown()

def is_cmp170hx(torch, index):
    if torch.cuda.get_device_capability(index) != (8, 0):
        return False
    if torch.cuda.get_device_name(index).startswith("NVIDIA CMP 170HX"):
        return True
    try:
        uuid = torch.cuda.get_device_properties(index).uuid
        # NVML combines the 16-bit device ID and NVIDIA vendor ID.
        return pci_device_id(uuid) == 0x20C210DE
    except (ImportError, AttributeError):
        return False

def runtime_identity(opt=Path("/opt")):
    import torch
    if not torch.cuda.is_available():
        raise ValueError("Cache seed requires CUDA devices")
    count = torch.cuda.device_count()
    if not count or not all(is_cmp170hx(torch, i) for i in range(count)):
        raise ValueError("Cache seed is restricted to CMP 170HX sm_80")
    driver = ctypes.c_int()
    cuda = ctypes.CDLL("libcuda.so.1")
    if cuda.cuDriverGetVersion(ctypes.byref(driver)) != 0:
        raise ValueError("Cannot identify CUDA driver")
    provenance = json.loads((opt / "vllm-src/provenance.json").read_text())
    return dict(schema=1, engine_commit=provenance["engine_commit"], gpu_arch="sm_80",
                gpu_name="NVIDIA CMP 170HX", driver=driver.value,
                dependencies=sha(opt / "venv/requirements.lock.txt"),
                versions={n: importlib.metadata.version(n) for n in
                          ("torch", "triton", "flashinfer-python", "tilelang")})

def exports(identity):
    root = "/cache/compiled/" + key(identity)
    values = dict(HOME=root, XDG_CACHE_HOME=root + "/.cache",
                  FLASHINFER_WORKSPACE_BASE=root,
                  TORCHINDUCTOR_CACHE_DIR=root + "/torchinductor",
                  TRITON_CACHE_DIR=root + "/.triton/cache",
                  TORCH_EXTENSIONS_DIR=root + "/.cache/torch_extensions",
                  VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR=root + "/.cache/vllm/custom_all_reduce_flags",
                  VLLM_CACHE_ROOT=root + "/.cache/vllm",
                  TILELANG_CACHE_DIR=root + "/.tilelang/cache",
                  TILELANG_TMP_DIR=root + "/.tilelang/cache/tmp",
                  CUDA_CACHE_PATH=root + "/.nv/ComputeCache")
    return "\n".join("export " + n + "=" + shlex.quote(v) for n, v in values.items())

def allowed(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        return False
    return any(p == PurePosixPath(r) or PurePosixPath(r) in p.parents for r in ROOTS)

def check(bundle, pin, lock):
    m = json.loads((bundle / "manifest.json").read_text())
    i = m["identity"]
    if (i["schema"] != 1 or i["engine_commit"] != pin or i["gpu_arch"] != "sm_80"
            or i["dependencies"] != sha(lock) or m["key"] != key(i)):
        raise ValueError("Cache seed identity mismatch")
    archive = bundle / "cache.tar.gz"
    if sha(archive) != m["archive_sha256"]:
        raise ValueError("Cache seed archive hash mismatch")
    with tarfile.open(archive, "r:gz") as t:
        names = set()
        for member in t:
            if not allowed(member.name) or not member.isfile() or member.name in names:
                raise ValueError("Unsafe cache archive member")
            names.add(member.name)
            h = hashlib.sha256()
            with t.extractfile(member) as f:
                for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
                    h.update(block)
            if m["files"].get(member.name) != h.hexdigest():
                raise ValueError("Cache member hash mismatch")
        if names != set(m["files"]) or not names:
            raise ValueError("Cache inventory mismatch or empty cache")
    return m

def pack(cache, identity_path, bundle):
    identity = json.loads(identity_path.read_text())
    bundle.mkdir(parents=True, exist_ok=False)
    files = {}
    with tarfile.open(bundle / "cache.tar.gz", "w:gz", format=tarfile.PAX_FORMAT) as t:
        for root in ROOTS:
            for path in sorted((cache / root).rglob("*")):
                if path.is_symlink():
                    raise ValueError("Cache symlink is not distributable")
                if not path.is_file() or path.suffix in (".lock", ".log") or "tmp" in path.relative_to(cache).parts:
                    continue
                name = path.relative_to(cache).as_posix()
                files[name] = sha(path)
                info = t.gettarinfo(str(path), arcname=name)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                info.mode = 0o644
                with path.open("rb") as f:
                    t.addfile(info, f)
    if not files:
        raise ValueError("No compilation cache files produced")
    m = dict(identity=identity, key=key(identity), files=files,
             archive_sha256=sha(bundle / "cache.tar.gz"))
    (bundle / "manifest.json").write_text(json.dumps(m, sort_keys=True) + "\n")
    return m

def seed_locked(bundle, opt, cache, identity):
    root = cache / "compiled" / key(identity)
    root.mkdir(parents=True, exist_ok=True)
    outcome = "cold (seed unavailable)"
    if bundle.exists():
        try:
            m = check(bundle, identity["engine_commit"], opt / "venv/requirements.lock.txt")
            if m["identity"] != identity:
                raise ValueError("Runtime differs from seed")
            # Never merge a seed into existing caches. Compilers own their updates.
            marker = root / ".seed-complete"
            if marker.exists():
                outcome = "already seeded"
            elif any(root.iterdir()):
                outcome = "cold (existing namespace)"
            else:
                with tempfile.TemporaryDirectory(dir=root.parent) as temp:
                    with tarfile.open(bundle / "cache.tar.gz", "r:gz") as t:
                        for member in t:
                            target = Path(temp) / member.name
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with t.extractfile(member) as f, target.open("wb") as out:
                                shutil.copyfileobj(f, out)
                    for child in Path(temp).iterdir():
                        shutil.move(str(child), root / child.name)
                marker.write_text(m["archive_sha256"] + "\n")
                outcome = "seeded"
        except (ValueError, KeyError, OSError, tarfile.TarError, json.JSONDecodeError):
            outcome = "cold (seed rejected)"
    print("[image-cache] " + outcome, file=sys.stderr)
    return exports(identity)

def seed(bundle, opt=Path("/opt"), cache=Path("/cache")):
    identity = runtime_identity(opt)
    parent = cache / "compiled"
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / (key(identity) + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return seed_locked(bundle, opt, cache, identity)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("seed")
    s.add_argument("bundle", type=Path)
    s = sub.add_parser("identity")
    s.add_argument("output", type=Path)
    s = sub.add_parser("pack")
    for n in ("cache", "identity", "bundle"):
        s.add_argument(n, type=Path)
    s = sub.add_parser("check")
    s.add_argument("bundle", type=Path)
    s.add_argument("pin")
    s.add_argument("lock", type=Path)
    a = p.parse_args()
    if a.mode == "seed":
        print(seed(a.bundle))
    elif a.mode == "identity":
        identity = runtime_identity()
        a.output.write_text(json.dumps(identity, sort_keys=True))
        print(exports(identity))
    elif a.mode == "pack":
        pack(a.cache, a.identity, a.bundle)
    else:
        check(a.bundle, a.pin, a.lock)
