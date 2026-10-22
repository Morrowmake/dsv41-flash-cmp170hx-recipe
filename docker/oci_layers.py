#!/usr/bin/env python3
"""Append deterministic, parallel-compressed layers to a local OCI base."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from image_identity import engine_tag
from layer_count import check as check_layer_count

OCI = "application/vnd.oci.image."

def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()

def put(out, data, media):
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    (out / "blobs/sha256" / digest.split(":")[1]).write_bytes(data)
    return dict(mediaType=media, digest=digest, size=len(data))

def read_blob(root, desc):
    path = root / "blobs/sha256" / desc["digest"].split(":")[1]
    if digest_file(path) != desc["digest"] or path.stat().st_size != desc["size"]:
        raise ValueError("OCI blob integrity mismatch")
    return json.loads(path.read_bytes())

def image(root):
    index = json.loads((root / "index.json").read_text())
    desc = index["manifests"][0]
    manifest = read_blob(root, desc)
    if "manifests" in manifest:
        desc = next(d for d in manifest["manifests"] if d.get("platform", {}).get("os") == "linux"
                    and d["platform"].get("architecture") == "amd64")
        manifest = read_blob(root, desc)
    return manifest, read_blob(root, manifest["config"])

def pack(root, name, work, parallel, threads):
    tar = work / (name + ".tar")
    zipped = work / (name + ".tar.gz")
    subprocess.run(["tar", "--sort=name", "--owner=0", "--group=0", "--numeric-owner",
                    "--mtime=2026-09-26 00:00:00Z", "--format=posix",
                    "--pax-option=exthdr.name=%d/PaxHeaders/%f,delete=atime,delete=ctime",
                    "-C", str(root), "-cf", str(tar), "."], check=True)
    diff = digest_file(tar)
    compressor = ["pigz", "-n", "-p", str(threads), "-6", "-c"] if parallel else ["gzip", "-n", "-6", "-c"]
    with zipped.open("wb") as f:
        subprocess.run(compressor + [str(tar)], stdout=f, check=True)
    descriptor = dict(mediaType=OCI + "layer.v1.tar+gzip", digest=digest_file(zipped), size=zipped.stat().st_size)
    size = tar.stat().st_size
    tar.unlink()
    return name, zipped, descriptor, diff, size

def assemble(base, payload, out, pin, tag, layered=True, parallel=True, threads=16):
    if tag != engine_tag(pin):
        raise ValueError("Image tag differs from the full engine revision")
    manifest, config = image(base)
    if config["os"] != "linux" or config["architecture"] != "amd64":
        raise ValueError("Base must be linux/amd64")
    names = ["python", "dependencies", "native", "engine"]
    if (payload / "cache-seed").exists():
        names.append("cache-seed")
    if len(manifest["layers"]) != len(config["rootfs"]["diff_ids"]):
        raise ValueError("Base layer/config count mismatch")
    check_layer_count(len(manifest["layers"]) + (len(names) if layered else 1))
    if out.exists():
        raise ValueError("Output OCI path already exists")
    (out / "blobs/sha256").mkdir(parents=True)
    work = out.parent / (out.name + "-layers")
    work.mkdir(exist_ok=False)
    for desc in [manifest["config"], *manifest["layers"]]:
        source = base / "blobs/sha256" / desc["digest"].split(":")[1]
        if digest_file(source) != desc["digest"]:
            raise ValueError("Base layer integrity mismatch")
        shutil.copyfile(source, out / "blobs/sha256" / source.name)
    if not layered:
        merged = work / "runtime"
        merged.mkdir()
        for name in names:
            shutil.copytree(payload / name, merged, dirs_exist_ok=True, symlinks=True)
        jobs = [(merged, "runtime")]
    else:
        jobs = [(payload / name, name) for name in names]
    with ThreadPoolExecutor(max_workers=len(jobs) if parallel else 1) as pool:
        results = list(pool.map(lambda job: pack(*job, work, parallel, threads), jobs))
    report = [dict(name="base-" + str(i), compressed=d["size"], digest=d["digest"])
              for i, d in enumerate(manifest["layers"])]
    for name, zipped, desc, diff, size in results:
        shutil.move(zipped, out / "blobs/sha256" / desc["digest"].split(":")[1])
        manifest["layers"].append(desc)
        config["rootfs"]["diff_ids"].append(diff)
        config.setdefault("history", []).append(dict(created="2026-09-26T00:00:00Z", created_by="runtime " + name))
        report.append(dict(name=name, compressed=desc["size"], uncompressed=size, digest=desc["digest"]))
    env = dict(e.split("=", 1) for e in config["config"].get("Env", []))
    env.update(PATH="/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
               VIRTUAL_ENV="/opt/venv", CUDA_HOME="/usr/local/cuda", NVIDIA_REQUIRE_CUDA="cuda>=13.0")
    config["config"].update(Env=[k + "=" + v for k, v in sorted(env.items())],
                            Entrypoint=["/opt/venv/bin/vllm"], ExposedPorts={"8000/tcp": {}})
    labels = config["config"].setdefault("Labels", {})
    labels.update({"org.opencontainers.image.revision": pin, "org.opencontainers.image.version": tag,
                   "org.opencontainers.image.title": "vllm-cmp170hx",
                   "org.opencontainers.image.base.name": "docker.io/nvidia/cuda:13.3.1-devel-ubuntu24.04",
                   "org.opencontainers.image.base.digest": "sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87",
                   "org.opencontainers.image.source": "https://github.com/Morrowmake/vllm-cmp170hx",
                   "org.opencontainers.image.licenses": "Apache-2.0"})
    manifest["config"] = put(out, encoded(config), OCI + "config.v1+json")
    manifest["mediaType"] = OCI + "manifest.v1+json"
    desc = put(out, encoded(manifest), manifest["mediaType"])
    (out / "index.json").write_bytes(encoded(dict(schemaVersion=2, manifests=[desc])))
    (out / "oci-layout").write_text('{"imageLayoutVersion":"1.0.0"}')
    (out.parent / "layers.json").write_bytes(encoded(report))
    (out.parent / "digest.txt").write_text(desc["digest"] + "\n")
    (out.parent / "manifest.json").write_bytes(encoded(manifest))
    (out.parent / "config.json").write_bytes(encoded(config))
    for row in report:
        print(row["name"], row["compressed"], "bytes", row["digest"], flush=True)
    return desc["digest"]

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "payload", "out"):
        p.add_argument(name, type=Path)
    p.add_argument("pin")
    p.add_argument("tag")
    a = p.parse_args()
    print(assemble(a.base, a.payload, a.out, a.pin, a.tag,
                   layered=os.environ.get("IMAGE_LAYERED", "1") == "1",
                   parallel=os.environ.get("IMAGE_PARALLEL_COMPRESSION", "1") == "1",
                   threads=int(os.environ.get("IMAGE_COMPRESSION_THREADS", "16"))))
