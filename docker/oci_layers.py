#!/usr/bin/env python3
"""Append deterministic, parallel-compressed layers to a local OCI image."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from image_identity import engine_tag
from layer_count import check as check_layer_count

OCI = "application/vnd.oci.image."
# One fixed time for every tar member and history entry, so equal content
# gives equal layer digests.
EPOCH = "2026-09-26 00:00:00Z"
CREATED = "2026-09-26T00:00:00Z"
PARTS = ("python", "dependencies", "native", "engine")
# The pinned base: the image index and its linux/amd64 manifest (native_build.py builds the
# natives in that manifest).
BASE_NAME = "docker.io/nvidia/cuda:13.3.1-devel-ubuntu24.04"
BASE_INDEX = "sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87"
BASE_MANIFEST = "sha256:03c372fd9c65fe7739279f8c65473b315dc61efaaffab03e1e65bc7be7aee61e"


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


def verify_base(root, index_digest, manifest_digest):
    """The OCI layout is the pinned base: its top descriptor is the pinned image index,
    whose single linux/amd64 entry is the pinned manifest, or (a layout pulled for one
    platform) the pinned manifest itself. Every blob read is verified; returns the identity
    the image's labels and keys use. A layout of the manifest alone proves no index: its
    index is recorded only when the pin is this code's own (index, manifest) pair, whose
    association is fixed here, and is None otherwise."""
    index = json.loads((root / "index.json").read_text())
    tops = index.get("manifests") or []
    if len(tops) != 1:
        raise ValueError(f"base layout must name one image, not {len(tops)}")
    top = tops[0]
    blob = read_blob(root, top)
    form = "manifest"
    if "manifests" in blob:
        if top["digest"] != index_digest:
            raise ValueError(f"base index {top['digest']} is not the pinned {index_digest}")
        amd64 = [d for d in blob["manifests"] if d.get("platform", {}).get("os") == "linux"
                 and d["platform"].get("architecture") == "amd64"]
        if len(amd64) != 1:
            raise ValueError("base index must hold exactly one linux/amd64 manifest")
        top, form = amd64[0], "index"
        blob = read_blob(root, top)
    if top["digest"] != manifest_digest:
        raise ValueError(f"base manifest {top['digest']} is not the pinned {manifest_digest}")
    config = read_blob(root, blob["config"])
    if config.get("os") != "linux" or config.get("architecture") != "amd64":
        raise ValueError("base must be linux/amd64")
    pinned = (index_digest, manifest_digest) == (BASE_INDEX, BASE_MANIFEST)
    index = index_digest if form == "index" or pinned else None
    return {"index": index, "manifest": manifest_digest, "config": blob["config"]["digest"],
            "form": form, "name": BASE_NAME if pinned else None}


def pack(root, name, work, parallel, threads):
    tar = work / (name + ".tar")
    zipped = work / (name + ".tar.gz")
    subprocess.run(["tar", "--sort=name", "--owner=0", "--group=0", "--numeric-owner",
                    "--mtime=" + EPOCH, "--format=posix",
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


def write_layout(out, manifest, config):
    """Config, manifest and index of a complete OCI layout whose layer blobs are in place;
    the digest, manifest and config are also kept beside the layout."""
    manifest["config"] = put(out, encoded(config), OCI + "config.v1+json")
    manifest["mediaType"] = OCI + "manifest.v1+json"
    desc = put(out, encoded(manifest), manifest["mediaType"])
    (out / "index.json").write_bytes(encoded(dict(schemaVersion=2, manifests=[desc])))
    (out / "oci-layout").write_text('{"imageLayoutVersion":"1.0.0"}')
    (out.parent / "digest.txt").write_text(desc["digest"] + "\n")
    (out.parent / "manifest.json").write_bytes(encoded(manifest))
    (out.parent / "config.json").write_bytes(encoded(config))
    return desc["digest"]


def image_env(base_env, env):
    """The image's Env is exactly `env` (the engine's declared container environment);
    a name the base sets must be declared, so nothing of the base is lost unseen."""
    undeclared = sorted(set(base_env) - set(env))
    if undeclared:
        raise ValueError("Base image sets undeclared environment names: " + ", ".join(undeclared))
    return [k + "=" + v for k, v in sorted(env.items())]


def assemble(base, payload, out, pin, tag, env, base_pin, layered=True, parallel=True, threads=16,
             entrypoint=("/opt/venv/bin/vllm",), extra_labels=None):
    """base_pin = (index digest, manifest digest): the base layout must be that base
    (verify_base), and the labels name it from that verified identity. An assembled engine's image
    (build.sh) passes its own entry point and its pins as extra labels."""
    labels = extra_labels or {}
    if tag != engine_tag(pin, labels.get("org.morrowmake.upstream-commit"), labels.get("org.morrowmake.plugin-commit")):
        raise ValueError("Image tag differs from the full engine revision")
    identity = verify_base(base, *base_pin)
    manifest, config = image(base)
    if config["os"] != "linux" or config["architecture"] != "amd64":
        raise ValueError("Base must be linux/amd64")
    names = list(PARTS)
    if len(manifest["layers"]) != len(config["rootfs"]["diff_ids"]):
        raise ValueError("Base layer/config count mismatch")
    check_layer_count(len(manifest["layers"]) + (len(names) if layered else 1))
    base_env = dict(e.split("=", 1) for e in config["config"].get("Env", []))
    config["config"]["Env"] = image_env(base_env, env)
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
        config.setdefault("history", []).append(dict(created=CREATED, created_by="runtime " + name))
        report.append(dict(name=name, compressed=desc["size"], uncompressed=size, digest=desc["digest"]))
    config["config"].update(Entrypoint=list(entrypoint), ExposedPorts={"8000/tcp": {}})
    config["config"].pop("Cmd", None)
    labels = config["config"].setdefault("Labels", {})
    labels.update({"org.opencontainers.image.revision": pin, "org.opencontainers.image.version": tag,
                   "org.opencontainers.image.title": "vllm-cmp170hx",
                   "org.opencontainers.image.base.digest": identity["index"] or identity["manifest"],
                   "org.opencontainers.image.source": "https://github.com/Morrowmake/vllm-cmp170hx",
                   "org.opencontainers.image.licenses": "Apache-2.0"})
    labels.update(extra_labels or {})
    labels.pop("org.opencontainers.image.base.name", None)
    if identity["name"]:
        labels["org.opencontainers.image.base.name"] = identity["name"]
    digest = write_layout(out, manifest, config)
    (out.parent / "layers.json").write_bytes(encoded(report))
    for row in report:
        print(row["name"], row["compressed"], "bytes", row["digest"], flush=True)
    return digest


def link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def append_layer(engine, layer_root, out, *, entrypoint, labels, created_by, threads=16):
    """A new OCI layout: the engine image's exact blobs (hard links where possible), one
    more layer packed from layer_root, and the engine config with this entrypoint and these
    labels added. The engine's Env is kept unchanged."""
    manifest, config = image(engine)
    engine_digest = json.loads((engine / "index.json").read_text())["manifests"][0]["digest"]
    check_layer_count(len(manifest["layers"]) + 1)
    if out.exists():
        raise ValueError("Output OCI path already exists")
    (out / "blobs/sha256").mkdir(parents=True)
    for desc in manifest["layers"]:
        source = engine / "blobs/sha256" / desc["digest"].split(":")[1]
        if source.stat().st_size != desc["size"]:
            raise ValueError("Engine layer size mismatch")
        link_or_copy(source, out / "blobs/sha256" / source.name)
    work = out.parent / (out.name + "-layers")
    work.mkdir(exist_ok=False)
    name, zipped, desc, diff, size = pack(layer_root, "model", work, True, threads)
    shutil.move(zipped, out / "blobs/sha256" / desc["digest"].split(":")[1])
    manifest, config = copy.deepcopy(manifest), copy.deepcopy(config)
    manifest["layers"].append(desc)
    config["rootfs"]["diff_ids"].append(diff)
    config.setdefault("history", []).append(dict(created=CREATED, created_by=created_by))
    config["config"]["Entrypoint"] = list(entrypoint)
    config["config"].pop("Cmd", None)
    config["config"].setdefault("Labels", {}).update(labels)
    digest = write_layout(out, manifest, config)
    report = [dict(name="engine-" + str(i), compressed=d["size"], digest=d["digest"])
              for i, d in enumerate(manifest["layers"][:-1])]
    report.append(dict(name="model", compressed=desc["size"], uncompressed=size, digest=desc["digest"]))
    (out.parent / "layers.json").write_bytes(encoded(report))
    return digest, engine_digest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "payload", "out"):
        p.add_argument(name, type=Path)
    p.add_argument("pin")
    p.add_argument("tag")
    p.add_argument("env", type=Path, help="JSON object: the image's complete Env")
    p.add_argument("--base-index", default=BASE_INDEX)
    p.add_argument("--base-manifest", default=BASE_MANIFEST)
    a = p.parse_args()
    print(assemble(a.base, a.payload, a.out, a.pin, a.tag, json.loads(a.env.read_text()),
                   (a.base_index, a.base_manifest),
                   layered=os.environ.get("IMAGE_LAYERED", "1") == "1",
                   parallel=os.environ.get("IMAGE_PARALLEL_COMPRESSION", "1") == "1",
                   threads=int(os.environ.get("IMAGE_COMPRESSION_THREADS", "16"))))
