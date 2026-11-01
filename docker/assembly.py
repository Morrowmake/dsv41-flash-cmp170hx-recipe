#!/usr/bin/env python3
"""The assembled engine of build.sh: three pinned trees, nothing compiled.

  wheel     the upstream wheel's build products into the queue tree (engine/): every
            member under vllm/ that neither the queue commit nor the wheel's own commit
            tracks (compiled modules, vendored trees), and vllm/_version.py naming the
            queue commit
  engine    the wheel library the queue makes stale (native_inputs.py), rebuilt from the
            queue by native_build.py build --no-marlin --stable as upstream builds its
            wheel, over the wheel's copy in engine/ (checked against the pinned build)
  plugin    the plugin tree's natives (NATIVE_DIR, checked against the pinned native
            build) and its metadata-only vllm_sm80-<version>.dist-info, derived from the
            plugin commit's pyproject.toml alone
  record    the release's assembly.json, checked field by field against what was built,
            written into the image with the rebuilt engine library's sha256
  files     record every file of /opt/assembly (sha256, mode, link target) as built
  check     the stage (--opt) or an OCI image (--oci) holds exactly those files
  env       the image's Env: the plugin launcher's [runtime.container.image_env]
  oci       lay out the image: the pinned base, the four runtime layers, the Env,
            the entry point and the pin labels

The trees are bytes of Git: engine/ is the queue commit's files, plugin/ the plugin
commit's, each with only the products named above added.
"""
import argparse
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = "/opt/assembly"
PYTHONPATH = ROOT + "/engine:" + ROOT + "/plugin"
LAUNCH = "vllm_sm80/launch/defaults.toml"
ENTRYPOINT = ["/opt/venv/bin/python3", "-m", "vllm.entrypoints.cli.main"]
PLUGIN_DISTRIBUTION = "vllm-sm80"
PLUGIN_PACKAGE = "vllm_sm80"
PINS = ("upstream_commit", "queue_commit", "plugin_commit")
# What a rebuild of the natives may differ in from the pinned build: where and when it
# was built from (a commit with the same csrc tree) and the builder's own bytes.
PROVENANCE = {("source", "commit"), ("source", "commit_time"), ("environment", "SOURCE_DATE_EPOCH"),
              ("builder_sha256",)}
# The engine's source inputs a wheel-library build records (native_build.py SOURCE_INPUTS).
ENGINE_INPUTS = ("CMakeLists.txt", "cmake", "csrc")


class Refusal(Exception):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def git(src, *args):
    return subprocess.run(["git", "--git-dir", str(src), *args], check=True, capture_output=True,
                          text=True).stdout


def tracked(src, commit, *paths):
    return set(filter(None, git(src, "ls-tree", "-r", "-z", "--name-only", commit, "--", *paths).split("\0")))


def write(path, data, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(mode)


def public_version(version):
    return version.split("+", 1)[0]


def lock_torch(constraints):
    for line in Path(constraints).read_text().splitlines():
        m = re.match(r"^\s*torch\s*==\s*([^\s;#,]+)", line)
        if m:
            return public_version(m.group(1))
    raise Refusal(f"{constraints} pins no torch")


# ---------------------------------------------------------------- engine/
def wheel(a):
    """The wheel's build products into the queue tree (module doc)."""
    actual = sha256_file(a.wheel)
    if actual != a.sha256:
        raise Refusal(f"{a.wheel} is {actual}, not the pinned wheel {a.sha256}")
    tree = Path(a.tree)
    known = tracked(a.src, a.queue, "vllm") | tracked(a.src, a.upstream, "vllm")
    copied, version_text = 0, None
    with zipfile.ZipFile(a.wheel) as archive:
        for info in archive.infolist():
            name = info.filename
            if name.startswith("/") or ".." in Path(name).parts or "\\" in name:
                raise Refusal(f"unsafe path in the wheel: {name}")
            if not name.startswith("vllm/") or name.endswith("/") or name in known:
                continue
            if name == "vllm/_version.py":
                version_text = archive.read(info).decode()
                continue
            write(tree / name, archive.read(info), ((info.external_attr >> 16) & 0o777) or 0o644)
            copied += 1
    tag = re.search(r"\+g([0-9a-f]{7,})", Path(a.wheel).name)
    if version_text and tag:
        version_text = version_text.replace("g" + tag.group(1), "g" + a.queue[:9])
    write(tree / "vllm/_version.py",
          (version_text or f'__version__ = version = "0+g{a.queue[:9]}"\n').encode(), 0o644)
    print(f"[assembly] {copied} wheel build products and vllm/_version.py into {tree}")


# ---------------------------------------------------------------- plugin/
def leaves(value, prefix=()):
    if isinstance(value, dict) and value:
        for key, item in value.items():
            yield from leaves(item, prefix + (key,))
    else:
        yield prefix, value


def native_differences(built, pinned, allowed=frozenset()):
    """The manifest fields a rebuild differs in from the pinned build, provenance and ALLOWED aside."""
    a, b = dict(leaves(built)), dict(leaves(pinned))
    return sorted(".".join(k) for k in a.keys() | b.keys()
                  if a.get(k) != b.get(k) and not any(k[:len(p)] == p for p in PROVENANCE | allowed))


def engine(a):
    """The engine's rebuilt wheel library (--native: native_build.py build --no-marlin --stable of the queue)
    over the wheel's copy in the engine tree. Its manifest must be the pinned build's in every field but where and
    when it was built and the library's bytes: CUDA 13 numbers anonymous-namespace symbols per build, so two builds
    of one source are never byte-identical (their kernels are)."""
    sys.path.insert(0, str(HERE))
    from native_build import STABLE, variants  # noqa: PLC0415
    tree, native = Path(a.tree), Path(a.native)
    pinned_path = Path(a.pinned_manifest)
    if sha256_file(pinned_path) != a.manifest_sha256:
        raise Refusal(f"{pinned_path} is not the pinned engine native manifest {a.manifest_sha256}")
    pinned = json.loads(pinned_path.read_text())
    built = json.loads((native / "manifest.json").read_text())
    inputs = {name: git(a.src, "rev-parse", f"{a.queue}:{name}").strip() for name in ENGINE_INPUTS}
    for doc, where in ((built, native / "manifest.json"), (pinned, pinned_path)):
        if doc.get("package") or "wheel_toolchain" not in doc or set(doc["artefacts"]) != {STABLE}:
            raise Refusal(f"{where}: not a build of {STABLE} alone (native_build.py build --no-marlin --stable)")
        if doc["source"]["inputs"] != inputs:
            raise Refusal(f"{where}: built from {doc['source']['inputs']}, not the queue's {inputs}")
    differences = native_differences(built, pinned, {("artefacts", STABLE, "sha256"), ("artefacts", STABLE, "bytes")})
    if differences:
        raise Refusal(f"{STABLE} was not built as the pinned build; manifest fields that differ: "
                      + ", ".join(differences))
    data = (native / STABLE).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != built["artefacts"][STABLE]["sha256"]:
        raise Refusal(f"{native / STABLE} differs from its manifest")
    target = tree / STABLE
    if not target.is_file() or target.is_symlink():
        raise Refusal(f"{target}: the wheel holds no {STABLE} to replace")
    shadows = [str(p) for p in variants(target) if p.exists() or p.is_symlink()]
    if shadows:
        raise Refusal(f"other builds of {STABLE} in the engine tree: {shadows}")
    write(target, data, 0o755)
    validated = pinned["artefacts"][STABLE]["sha256"]
    print(f"[assembly] {STABLE} rebuilt from the queue into {tree}: sha256 {actual} ("
          + ("the validated build's bytes)" if actual == validated else f"the validated build's is {validated}; "
             "the manifests agree in every other field)"))


def plugin_metadata(plugin):
    """(dist-info name, {path inside it: bytes}): METADATA, entry_points.txt, the licence
    files under licenses/ and RECORD, from pyproject.toml alone, the same bytes every run."""
    project = tomllib.loads((plugin / "pyproject.toml").read_text()).get("project") or {}
    name, version = project.get("name"), project.get("version")
    if name != PLUGIN_DISTRIBUTION or not isinstance(version, str) or project.get("dynamic"):
        raise Refusal(f"{plugin / 'pyproject.toml'}: [project] must name {PLUGIN_DISTRIBUTION} with a static "
                      f"version and no dynamic fields")
    dist = f"{name.replace('-', '_')}-{version}.dist-info"
    lines = ["Metadata-Version: 2.4", f"Name: {name}", f"Version: {version}"]
    if project.get("description"):
        lines.append(f"Summary: {project['description']}")
    if isinstance(project.get("license"), str):
        lines.append(f"License-Expression: {project['license']}")
    if project.get("requires-python"):
        lines.append(f"Requires-Python: {project['requires-python']}")
    files = {}
    for rel in project.get("license-files") or ():
        source = plugin / rel
        if not source.is_file() or any(c in rel for c in "*?["):
            raise Refusal(f"{plugin / 'pyproject.toml'}: license file {rel!r} is not one file of the plugin")
        lines.append(f"License-File: {rel}")
        files[f"licenses/{rel}"] = source.read_bytes()
    files["METADATA"] = ("\n".join(lines) + "\n").encode()
    groups = project.get("entry-points") or {}
    files["entry_points.txt"] = "".join(
        f"[{group}]\n" + "".join(f"{key} = {value}\n" for key, value in sorted(groups[group].items())) + "\n"
        for group in sorted(groups)).encode()

    def record_hash(data):
        return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    rows = [f"{dist}/{rel},{record_hash(data)},{len(data)}" for rel, data in sorted(files.items())]
    files["RECORD"] = ("\n".join(rows + [f"{dist}/RECORD,,"]) + "\n").encode()
    return dist, files


def plugin(a):
    """NATIVE_DIR's natives (the pinned build's files) and the metadata into the plugin tree."""
    tree, native = Path(a.tree), Path(a.native)
    pinned_path = Path(a.pinned_manifest)
    if sha256_file(pinned_path) != a.manifest_sha256:
        raise Refusal(f"{pinned_path} is not the pinned native manifest {a.manifest_sha256}")
    pinned = json.loads(pinned_path.read_text())
    built = json.loads((native / "manifest.json").read_text())
    if built.get("package") != PLUGIN_PACKAGE or pinned.get("package") != PLUGIN_PACKAGE:
        raise Refusal(f"{native}: not a plugin native build (native_build.py build --plugin)")
    csrc = git(a.src, "rev-parse", f"{a.plugin}:csrc").strip()
    for doc, where in ((built, native / "manifest.json"), (pinned, pinned_path)):
        if doc["source"]["inputs"] != {"csrc": csrc}:
            raise Refusal(f"{where}: built from csrc {doc['source']['inputs']}, not the plugin's {csrc}")
    if built["artefacts"] != pinned["artefacts"]:
        raise Refusal("the natives are not the pinned build's bytes:\n" + json.dumps(
            {"built": built["artefacts"], "pinned": pinned["artefacts"]}, indent=1, sort_keys=True))
    for rel, entry in built["artefacts"].items():
        if not rel.startswith(PLUGIN_PACKAGE + "/") or ".." in Path(rel).parts:
            raise Refusal(f"artefact {rel} is not under {PLUGIN_PACKAGE}/")
        if sha256_file(native / rel) != entry["sha256"]:
            raise Refusal(f"{native / rel} differs from its manifest")
    differences = native_differences(built, pinned)
    if differences:
        print("[assembly] native manifest fields that differ from the pinned build (its files are the same): "
              + ", ".join(differences))
    for rel in sorted(built["artefacts"]):
        write(tree / rel, (native / rel).read_bytes(), 0o755)
    dist, files = plugin_metadata(tree)
    for rel, data in files.items():
        write(tree / dist / rel, data, 0o644)
    print(f"[assembly] {len(built['artefacts'])} native(s) and {dist} into {tree}")


# ---------------------------------------------------------------- assembly.json
def engine_native_build(src, queue, rebuild, constraints, recipe, torch):
    """The engine native build's name: engine-<16 hex>-t<torch>, the hash of what it is built from (the queue's native
    source trees, the rebuilt native globs REBUILD, the dependency lock) and how (RECIPE: the queue's wheel recipe)."""
    inputs = {name: git(src, "rev-parse", f"{queue}:{name}").strip() for name in ENGINE_INPUTS}
    key = json.dumps(dict(inputs=inputs, rebuild=sorted(rebuild), constraints=sha256_file(constraints),
                          wheel_recipe=recipe), sort_keys=True)
    return f"engine-{hashlib.sha256(key.encode()).hexdigest()[:16]}-t{torch}"


def record(a):
    """The release's assembly.json against the build: every field but its creation time. It goes into the image
    (--out) with engine_natives naming the rebuilt engine library this build holds."""
    doc = json.loads(Path(a.record).read_text())
    plugin_tree = Path(a.opt) / "assembly/plugin"
    engine_tree = Path(a.opt) / "assembly/engine"
    pinned = json.loads(Path(a.pinned_manifest).read_text())
    torch = lock_torch(a.constraints)
    csrc = git(a.src, "rev-parse", f"{a.plugin}:csrc").strip()
    dist, _ = plugin_metadata(plugin_tree)
    version = dist[len(PLUGIN_DISTRIBUTION) + 1:-len(".dist-info")]
    expected = {
        "schema": "engine-assembly/1", "upstream_commit": a.upstream, "queue_commit": a.queue,
        "plugin_commit": a.plugin, "native_build": f"native-{csrc[:10]}-t{torch}", "torch": torch,
        "native_manifest_sha256": a.manifest_sha256,
        "natives": {rel: sha256_file(plugin_tree / rel) for rel in sorted(pinned["artefacts"])},
        "plugin_distribution": {"name": PLUGIN_DISTRIBUTION, "version": version, "dist_info": dist},
        "wheel": {"commit": a.upstream, "file": a.wheel_file, "url": a.wheel_url, "sha256": a.wheel_sha256},
        "engine_native_build": None, "engine_native_manifest_sha256": None, "engine_natives": {},
        "engine_native_toolchain": None,
    }
    if a.native_build != expected["native_build"]:
        raise Refusal(f"RELEASE_NATIVE_BUILD {a.native_build} is not {expected['native_build']} "
                      f"(the plugin's csrc tree and the lock's torch)")
    if a.engine_manifest_sha256:
        sys.path.insert(0, str(HERE))
        import native_inputs  # noqa: PLC0415
        if sha256_file(a.engine_pinned_manifest) != a.engine_manifest_sha256:
            raise Refusal(f"{a.engine_pinned_manifest} is not the pinned engine native manifest "
                          f"{a.engine_manifest_sha256}")
        engine_pinned = json.loads(Path(a.engine_pinned_manifest).read_text())
        built = engine_pinned["wheel_toolchain"]
        try:
            options = native_inputs.rebuild_options(native_inputs.load_rules(a.src, a.plugin))
        except native_inputs.Refusal as error:
            raise Refusal(str(error)) from None
        rebuild = [glob for glob, option in options.items() if option == "--stable"]
        expected.update(
            engine_native_build=engine_native_build(a.src, a.queue, rebuild, a.constraints, built["recipe"], torch),
            engine_native_manifest_sha256=a.engine_manifest_sha256,
            engine_natives={rel: e["sha256"] for rel, e in sorted(engine_pinned["artefacts"].items())},
            engine_native_toolchain=dict(built["recipe"], config=built["config"], nvcc=built["nvcc"],
                                         cxx=built["cxx"]))
    if (a.engine_native_build or None) != expected["engine_native_build"]:
        raise Refusal(f"RELEASE_ENGINE_NATIVE_BUILD {a.engine_native_build!r} is not "
                      f"{expected['engine_native_build']!r} (the queue's native trees, the rebuilt library, the lock "
                      f"and the queue's wheel recipe)")
    problems = [f"{k}: the record has {doc.get(k)!r}, the build {v!r}" for k, v in expected.items()
                if doc.get(k) != v]
    problems += [f"{k}: not an assembly field" for k in set(doc) - set(expected) - {"created"}]
    if problems:
        raise Refusal(f"{a.record} does not describe this build:\n  " + "\n  ".join(problems))
    held = {rel: sha256_file(engine_tree / rel) for rel in doc["engine_natives"]}
    if held == doc["engine_natives"]:
        write(Path(a.out), Path(a.record).read_bytes(), 0o664)
        print(f"[assembly] {a.record} describes this build; into {a.out} unchanged")
    else:
        write(Path(a.out), (json.dumps(dict(doc, engine_natives=held), indent=2, sort_keys=True) + "\n").encode(),
              0o664)
        print(f"[assembly] {a.record} describes this build; into {a.out} with the rebuilt engine natives "
              f"{json.dumps(held, sort_keys=True)}")


# ---------------------------------------------------------------- files and checks
def identity(path):
    path = Path(path)
    if path.is_symlink():
        return ["symlink", os.readlink(path), None]
    return ["file", sha256_file(path), path.stat().st_mode & 0o777]


def stage_tree(opt):
    tree = Path(opt) / "assembly"
    return {str(p.relative_to(tree)): identity(p) for p in sorted(tree.rglob("*"))
            if p.is_symlink() or not p.is_dir()}


def oci_tree(oci, prefix=ROOT):
    """{path below PREFIX: identity} of the image's final state across its layers (whiteouts
    and opaque directories applied; a hard link takes its target's identity)."""
    index = json.loads((Path(oci) / "index.json").read_text())
    blob = lambda d: Path(oci) / "blobs/sha256" / d["digest"].split(":", 1)[1]  # noqa: E731
    manifest = json.loads(blob(index["manifests"][0]).read_text())
    prefix = prefix.strip("/") + "/"
    state = {}
    for desc in manifest["layers"]:
        opener = gzip.open if desc["mediaType"].endswith("gzip") else open
        written = set()
        with opener(blob(desc), "rb") as stream, tarfile.open(fileobj=stream, mode="r|") as layer:
            for member in layer:
                name = PurePosixPath(member.name.lstrip("/").removeprefix("./")).as_posix()
                base = PurePosixPath(name)
                if base.name == ".wh..wh..opq":
                    gone = base.parent.as_posix() + "/"
                    state = {k: v for k, v in state.items() if not k.startswith(gone) or k in written}
                    continue
                if base.name.startswith(".wh."):
                    gone = (base.parent / base.name[4:]).as_posix()
                    state = {k: v for k, v in state.items() if k != gone and not k.startswith(gone + "/")}
                    continue
                if member.isdir():
                    continue
                written.add(name)
                if member.isreg():
                    h = hashlib.sha256()
                    data = layer.extractfile(member)
                    for block in iter(lambda: data.read(1 << 22), b""):
                        h.update(block)
                    state[name] = ["file", h.hexdigest(), member.mode & 0o777]
                elif member.issym():
                    state[name] = ["symlink", member.linkname, None]
                elif member.islnk():
                    target = PurePosixPath(member.linkname.lstrip("/").removeprefix("./")).as_posix()
                    state[name] = state.get(target, ["hardlink", member.linkname, None])
                else:
                    state[name] = ["other", member.type.decode(errors="replace"), None]
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def files(a):
    """Record /opt/assembly as built (before relocation, which must leave it alone)."""
    found = stage_tree(a.opt)
    expected = {"assembly.json"}
    for tree, commit in (("engine", a.queue), ("plugin", a.plugin)):
        expected |= {f"{tree}/{rel}" for rel in tracked(a.src, commit)}
    missing = sorted(expected - set(found))
    if missing:
        raise Refusal("files of the commits missing from /opt/assembly: " + ", ".join(missing[:20]))
    Path(a.out).write_text(json.dumps(found, indent=1, sort_keys=True) + "\n")
    print(f"[assembly] {len(found)} files of /opt/assembly recorded in {a.out} "
          f"({len(found) - len(expected)} build products)")


def check(a):
    expected = json.loads(Path(a.files).read_text())
    found = stage_tree(a.opt) if a.opt else oci_tree(a.oci)
    where = "stage" if a.opt else "image"
    problems = [f"{rel}: the {where} holds {found.get(rel, '<absent>')}, the assembly {want}"
                for rel, want in sorted(expected.items()) if found.get(rel) != want]
    problems += [f"{rel}: in the {where}, not in the assembly" for rel in sorted(set(found) - set(expected))]
    if problems:
        raise Refusal(f"the {where} does not hold the assembly's files:\n  " + "\n  ".join(problems[:40]))
    print(f"[assembly] the {where} holds exactly the assembly's {len(expected)} files")


# ---------------------------------------------------------------- image
def declared_env(opt):
    path = Path(opt) / "assembly/plugin" / LAUNCH
    table = tomllib.loads(path.read_text())["runtime"]["container"]["image_env"]
    if not isinstance(table, dict) or not table or not all(isinstance(v, str) for v in table.values()):
        raise Refusal(f"{path}: [runtime.container.image_env] must map names to strings")
    if table.setdefault("PYTHONPATH", PYTHONPATH) != PYTHONPATH:
        raise Refusal(f"the plugin launcher declares PYTHONPATH={table['PYTHONPATH']}, not {PYTHONPATH}")
    return dict(sorted(table.items()))


def env(a):
    print(json.dumps(declared_env(a.opt), indent=1, sort_keys=True))


def oci(a):
    sys.path.insert(0, str(HERE))
    import oci_layers  # noqa: PLC0415
    doc = json.loads((Path(a.opt) / "assembly/assembly.json").read_text())
    name = "assembly-" + "-".join(doc[k][:10] for k in PINS)
    labels = {"org.morrowmake.assembly": name, "org.morrowmake.native-build": doc["native_build"],
              **{"org.morrowmake." + k.replace("_", "-"): doc[k] for k in PINS}}
    digest = oci_layers.assemble(Path(a.base), Path(a.payload), Path(a.out), doc["queue_commit"], a.tag,
                                 declared_env(a.opt), (oci_layers.BASE_INDEX, oci_layers.BASE_MANIFEST),
                                 layered=a.layered, parallel=a.parallel, threads=a.threads,
                                 entrypoint=ENTRYPOINT, extra_labels=labels)
    print(digest)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    w = sub.add_parser("wheel")
    for name in ("wheel", "sha256", "tree", "src", "queue", "upstream"):
        w.add_argument("--" + name, required=True)
    g = sub.add_parser("plugin")
    for name in ("tree", "native", "pinned-manifest", "manifest-sha256", "src", "plugin"):
        g.add_argument("--" + name, required=True)
    n = sub.add_parser("engine")
    for name in ("tree", "native", "pinned-manifest", "manifest-sha256", "src", "queue"):
        n.add_argument("--" + name, required=True)
    r = sub.add_parser("record")
    for name in ("record", "opt", "src", "constraints", "pinned-manifest", "manifest-sha256", "upstream", "queue",
                 "plugin", "native-build", "wheel-file", "wheel-url", "wheel-sha256", "out"):
        r.add_argument("--" + name, required=True)
    for name in ("engine-native-build", "engine-pinned-manifest", "engine-manifest-sha256"):
        r.add_argument("--" + name, default="")
    f = sub.add_parser("files")
    for name in ("opt", "src", "queue", "plugin", "out"):
        f.add_argument("--" + name, required=True)
    c = sub.add_parser("check")
    c.add_argument("--files", required=True)
    where = c.add_mutually_exclusive_group(required=True)
    where.add_argument("--opt")
    where.add_argument("--oci")
    e = sub.add_parser("env")
    e.add_argument("--opt", required=True)
    o = sub.add_parser("oci")
    for name in ("opt", "base", "payload", "out", "tag"):
        o.add_argument("--" + name, required=True)
    o.add_argument("--threads", type=int, default=16)
    o.add_argument("--no-layers", dest="layered", action="store_false")
    o.add_argument("--no-parallel", dest="parallel", action="store_false")
    a = p.parse_args(argv)
    try:
        globals()[a.command](a)
    except Refusal as error:
        print(f"[assembly] FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
