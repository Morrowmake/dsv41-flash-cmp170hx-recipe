import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import native_build as nb
import oci_layers
import payload

MARLIN = nb.MARLIN


def sha(data):
    return hashlib.sha256(data).hexdigest()


def layer(entries):
    """entries: (name, kind, payload) -> uncompressed tar bytes."""
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as tar:
        for name, kind, value in entries:
            info = tarfile.TarInfo(name)
            if kind == "dir":
                info.type, info.mode = tarfile.DIRTYPE, 0o755
                tar.addfile(info)
            elif kind == "file":
                info.size, info.mode = len(value), 0o644
                tar.addfile(info, io.BytesIO(value))
            elif kind == "sym":
                info.type, info.linkname = tarfile.SYMTYPE, value
                tar.addfile(info)
            elif kind == "link":
                info.type, info.linkname = tarfile.LNKTYPE, value
                tar.addfile(info)
            elif kind == "chr":
                info.type, info.devmajor, info.devminor = tarfile.CHRTYPE, 1, 3
                tar.addfile(info)
    return data.getvalue()


def oci_layout(tmp_path, layers, platform_digest=None):
    root = tmp_path / "oci"
    blobs = root / "blobs/sha256"
    blobs.mkdir(parents=True)

    def put(data):
        digest = sha(data)
        (blobs / digest).write_bytes(data)
        return {"digest": "sha256:" + digest, "size": len(data)}

    config = dict(put(json.dumps({"architecture": "amd64", "os": "linux"}).encode()),
                  mediaType="application/vnd.oci.image.config.v1+json")
    descs = [dict(put(gzip.compress(raw)), mediaType="application/vnd.oci.image.layer.v1.tar+gzip")
             for raw in layers]
    manifest = json.dumps({"schemaVersion": 2, "config": config, "layers": descs}).encode()
    desc = dict(put(manifest), mediaType="application/vnd.oci.image.manifest.v1+json")
    if platform_digest:
        desc["digest"] = platform_digest
    (root / "index.json").write_text(json.dumps({"schemaVersion": 2, "manifests": [desc]}))
    return root, desc["digest"]


def test_base_pins_match_the_image_labels():
    assert (oci_layers.BASE_INDEX, oci_layers.BASE_MANIFEST) == (nb.BASE_INDEX, nb.BASE_MANIFEST)


def test_build_sh_pins_the_base_and_the_native_builds():
    """build.sh builds on the base the natives were built in; its native pins are the shipped manifests' sha256 and
    the ones the release's assembly.json records; the engine library's pinned build is a --stable build of the queue
    alone, in a wheel builder native_build.py pins, and the record names its library and toolchain."""
    build = (ROOT / "build.sh").read_text()

    def pin(name):
        return re.search(rf"^RELEASE_{name}='([^']*)'$", build, re.M).group(1)
    assert re.search(r'^BASE=".*@' + nb.BASE_INDEX + '"$', build, re.M)
    record = json.loads((ROOT / "assembly.json").read_text())
    assert sha((ROOT / "native-manifest.json").read_bytes()) == pin("NATIVE_MANIFEST_SHA256") \
        == record["native_manifest_sha256"]
    data = (ROOT / "engine-native-manifest.json").read_bytes()
    assert sha(data) == pin("ENGINE_NATIVE_MANIFEST_SHA256") == record["engine_native_manifest_sha256"]
    assert pin("ENGINE_NATIVE_BUILD") == record["engine_native_build"]
    manifest = json.loads(data)
    assert manifest["source"]["commit"] == pin("QUEUE_COMMIT") == record["queue_commit"]
    assert set(manifest["artefacts"]) == {nb.STABLE} and "package" not in manifest
    assert record["engine_natives"] == {nb.STABLE: manifest["artefacts"][nb.STABLE]["sha256"]}
    built = manifest["wheel_toolchain"]
    assert nb.WHEEL_BUILDERS[built["recipe"]["image"]] == built["recipe"]["manifest"]
    assert record["engine_native_toolchain"] == dict(built["recipe"], config=built["config"], nvcc=built["nvcc"],
                                                     cxx=built["cxx"])



def test_layers_apply_whiteouts_links_and_skip_devices(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    lower = layer([("a", "dir", None), ("a/x", "file", b"x"), ("a/y", "file", b"y"),
                   ("b", "dir", None), ("b/old", "file", b"o"), ("dev", "dir", None),
                   ("dev/null", "chr", None), ("l", "sym", "a/y"), ("h", "link", "a/y")])
    upper = layer([("a/.wh.x", "file", b""), ("b/new", "file", b"n"),
                   ("b/.wh..wh..opq", "file", b""), ("a/y", "file", b"Y")])
    assert nb.apply_layer(root, io.BytesIO(lower)) == 1
    assert nb.apply_layer(root, io.BytesIO(upper)) == 0
    assert not (root / "a/x").exists() and (root / "a/y").read_bytes() == b"Y"
    assert sorted(p.name for p in (root / "b").iterdir()) == ["new"]
    assert os.readlink(root / "l") == "a/y" and (root / "h").read_bytes() == b"y"
    assert not (root / "dev/null").exists()


def test_layers_with_a_root_entry_apply(tmp_path):
    """Layers packed with `tar -C <dir> .` (the builder's) name the root as "./"."""
    root = tmp_path / "root"
    root.mkdir()
    data = layer([("./", "dir", None), ("./opt", "dir", None), ("./opt/x", "file", b"x")])
    assert nb.apply_layer(root, io.BytesIO(data)) == 0
    assert (root / "opt/x").read_bytes() == b"x"


@pytest.mark.parametrize("name", ["../escape", "/abs/escape"])
def test_layers_refuse_paths_outside_the_root(tmp_path, name):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(nb.NativeError, match="escapes|outside"):
        nb.apply_layer(root, io.BytesIO(layer([(name, "file", b"x")])))


def test_layers_refuse_writes_through_symlinks_out_of_the_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    raw = layer([("out", "sym", str(tmp_path)), ("out/evil", "file", b"x")])
    with pytest.raises(nb.NativeError, match="outside"):
        nb.apply_layer(root, io.BytesIO(raw))
    assert not (tmp_path / "evil").exists()


def test_extract_base_checks_pin_and_layer_digests(tmp_path, monkeypatch):
    raw = layer([("etc", "dir", None), ("etc/os-release", "file", b"ubuntu")])
    oci, digest = oci_layout(tmp_path, [raw])
    with pytest.raises(nb.NativeError, match="not the pinned base"):
        nb.extract_base(oci, tmp_path / "rootfs")
    monkeypatch.setattr(nb, "BASE_MANIFEST", digest)
    base = nb.extract_base(oci, tmp_path / "rootfs")
    assert (tmp_path / "rootfs/etc/os-release").read_bytes() == b"ubuntu"
    assert (tmp_path / "rootfs/etc/resolv.conf").read_bytes() == b""
    assert base["manifest"] == digest
    manifest = json.loads((oci / "blobs/sha256" / digest.split(":")[1]).read_text())
    blob = oci / "blobs/sha256" / manifest["layers"][0]["digest"].split(":")[1]
    data = bytearray(blob.read_bytes())
    data[-5] ^= 1  # gzip trailer: the layer digest must catch it
    blob.write_bytes(bytes(data))
    with pytest.raises(Exception):
        nb.extract_base(oci, tmp_path / "rootfs2")


def test_extract_base_takes_the_multi_platform_pull_of_the_pinned_base(tmp_path, monkeypatch):
    """A `crane pull` of the base tag keeps the image index (4ff85952...) on top; the pinned linux/amd64
    manifest is its entry. Native builds refused that layout (10-18) while the image build took it."""
    raw = layer([("etc", "dir", None), ("etc/os-release", "file", b"ubuntu")])
    oci, digest = oci_layout(tmp_path, [raw])
    top = json.loads((oci / "index.json").read_text())["manifests"][0]
    arm = dict(top, digest="sha256:" + "a" * 64, platform={"os": "linux", "architecture": "arm64"})
    entry = dict(top, platform={"os": "linux", "architecture": "amd64"})
    data = json.dumps({"schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json",
                       "manifests": [arm, entry]}).encode()
    (oci / "blobs/sha256" / sha(data)).write_bytes(data)
    index = {"digest": "sha256:" + sha(data), "size": len(data),
             "mediaType": "application/vnd.oci.image.index.v1+json"}
    (oci / "index.json").write_text(json.dumps({"schemaVersion": 2, "manifests": [index]}))
    monkeypatch.setattr(nb, "BASE_MANIFEST", digest)
    with pytest.raises(nb.NativeError, match="not the pinned"):
        nb.extract_base(oci, tmp_path / "rootfs0")      # an index that is not the pinned one
    monkeypatch.setattr(nb, "BASE_INDEX", index["digest"])
    base = nb.extract_base(oci, tmp_path / "rootfs")
    assert (tmp_path / "rootfs/etc/os-release").read_bytes() == b"ubuntu"
    assert base["manifest"] == digest and base["index"] == index["digest"]


@pytest.mark.parametrize("option", ["VLLM_AMPERE_MARLIN_BUILD", "VLLM_BUILD_AMPERE_MARLIN"])
def test_marlin_build_env_passes_the_builder_gate_of_each_tree(tmp_path, option):
    # The optional Marlin builder exits unless its tree's build option is 1 (current trees and
    # trees from before the option was renamed); the build step's command prefix opens both.
    builder = tmp_path / "build_standalone.py"
    builder.write_text("import os, sys\n"
                       f"sys.exit(0 if os.environ.get({option!r}, '0') == '1' else 3)\n")
    run = subprocess.run(["env", "-i", "PATH=/usr/bin:/bin", "sh", "-c",
                          f"{nb.MARLIN_BUILD_ENV} {sys.executable} {builder}"])
    assert run.returncode == 0
    refused = subprocess.run(["env", "-i", "PATH=/usr/bin:/bin", sys.executable, str(builder)])
    assert refused.returncode == 3


def test_namespace_is_clean_and_offline_unless_asked(tmp_path):
    env = nb.environment(1700000000, 4)
    assert not any(str(Path.home()) in v for v in env.values())
    assert env["SOURCE_DATE_EPOCH"] == "1700000000" and env["NVCC_APPEND_FLAGS"] == "-objtemp"
    argv = nb.namespace_argv(tmp_path / "rootfs", tmp_path / "opt", tmp_path / "build", env,
                             ["true"])
    assert "--clearenv" in argv and "--share-net" not in argv
    assert argv[argv.index("--hostname") + 1] == "native-build"
    i = argv.index(str(tmp_path / "opt/vllm-src"))
    assert argv[i - 1] == "--ro-bind" and argv[i + 1] == nb.SRC
    online = nb.namespace_argv(tmp_path / "rootfs", tmp_path / "opt", tmp_path / "build", env,
                               ["true"], network=True, source_readonly=False)
    assert "--share-net" in online and str(tmp_path / "opt/vllm-src") not in online


def test_artefact_scan_rejects_host_values(tmp_path):
    good = tmp_path / "good.so"
    good.write_bytes(b"\x7fELF vllm/csrc/x.cu")
    nb.scan_artefact(good, [str(tmp_path / "work")])
    for data in (b"tmpxft_001e3744_00000000-6_decode", str(tmp_path / "work/csrc").encode()):
        bad = tmp_path / "bad.so"
        bad.write_bytes(b"\x7fELF " + data)
        with pytest.raises(nb.NativeError, match="embeds"):
            nb.scan_artefact(bad, [str(tmp_path / "work")])


def test_build_flag_records(tmp_path):
    ninja = tmp_path / "build.ninja"
    ninja.write_text("ninja_required_version = 1.3\ncxx = c++\nnvcc = /usr/local/cuda/bin/nvcc\n"
                     "cuda_cflags = -O3 -Xcompiler=-ffile-prefix-map=/opt/vllm-src=vllm\n"
                     "rule link\n  command = $cxx $in $ldflags -o $out\n")
    assert nb.ninja_rules(ninja) == {"cxx": "c++", "nvcc": "/usr/local/cuda/bin/nvcc",
                                     "cuda_cflags": "-O3 -Xcompiler=-ffile-prefix-map="
                                                    "/opt/vllm-src=vllm"}
    cmake = tmp_path / "cmake.ninja"
    cmake.write_text("build CMakeFiles/_exl3_C.dir/csrc/exl3/gemm.cu.o: CUDA_COMPILER x\n"
                     "  FLAGS = -O3\n  OBJECT_DIR = y\n"
                     "build CMakeFiles/_C.dir/csrc/a.cu.o: CUDA_COMPILER z\n  FLAGS = -O2\n"
                     "build _exl3_C.abi3.so: CUDA_SHARED_LIBRARY_LINKER\n  LINK_FLAGS = -s\n")
    assert nb.cmake_target_flags(cmake, "_exl3_C") == {
        "CMakeFiles/_exl3_C.dir/csrc/exl3/gemm.cu.o": {"FLAGS": "-O3"},
        "_exl3_C.abi3.so": {"LINK_FLAGS": "-s"}}


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True, env=dict(os.environ, GIT_AUTHOR_NAME="t",
                                              GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                                              GIT_COMMITTER_EMAIL="t@t",
                                              GIT_CONFIG_GLOBAL="/dev/null")).stdout.strip()


@pytest.fixture
def fork(tmp_path):
    tree = tmp_path / "tree"
    for rel, text in (("csrc/a.cu", "a"), ("cmake/u.cmake", "u"), ("CMakeLists.txt", "c"),
                      ("vllm/__init__.py", "")):
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text(text)
    git(tree.parent, "init", "-q", str(tree))
    git(tree, "add", ".")
    git(tree, "commit", "-qm", "base")
    commit = git(tree, "rev-parse", "HEAD")
    native = tmp_path / "native"
    (native / "vllm").mkdir(parents=True)
    (native / MARLIN).write_bytes(b"marlin bytes")
    manifest = {"schema": 1,
                "source": {"commit": commit, "commit_time": 0,
                           "inputs": {n: git(tree, "rev-parse", "HEAD:" + n)
                                      for n in nb.SOURCE_INPUTS}},
                "toolchain": {"base_image": {"manifest": nb.BASE_MANIFEST}},
                "python": {"torch": "2.13.0+cu130", "torch_cuda": "13.0", "cxx11_abi": True,
                           "torch_include_sha256": "i", "torch_libs": {"libc10.so": "c"}},
                "artefacts": {MARLIN: {"sha256": sha(b"marlin bytes"), "bytes": 12}}}
    (native / "manifest.json").write_text(nb.dump(manifest))
    return tree, native, commit


def test_install_puts_the_exact_files_and_removes_other_builds(fork):
    tree, native, commit = fork
    shadow = tree / "vllm/_ampere_marlin_C.cpython-312-x86_64-linux-gnu.so"
    shadow.write_bytes(b"old")
    assert nb.main(["install", "--native", str(native), "--tree", str(tree)]) == 0
    assert (tree / MARLIN).read_bytes() == b"marlin bytes" and not shadow.exists()
    assert os.stat(tree / MARLIN).st_mode & 0o777 == 0o755
    assert nb.main(["check-tree", "--manifest", str(native / "manifest.json"),
                    "--tree", str(tree), "--commit", commit]) == 0
    shadow.write_bytes(b"old")
    assert nb.main(["check-tree", "--manifest", str(native / "manifest.json"),
                    "--tree", str(tree)]) == 1


def test_install_refuses_changed_files_or_sources(fork, capsys):
    tree, native, _ = fork
    (native / MARLIN).write_bytes(b"other bytes")
    assert nb.main(["install", "--native", str(native), "--tree", str(tree)]) == 1
    assert "differs from the manifest" in capsys.readouterr().err
    (native / MARLIN).write_bytes(b"marlin bytes")
    (tree / "csrc/a.cu").write_text("changed")
    assert nb.main(["install", "--native", str(native), "--tree", str(tree)]) == 1
    assert "uncommitted changes" in capsys.readouterr().err
    git(tree, "commit", "-qam", "change")
    assert nb.main(["install", "--native", str(native), "--tree", str(tree)]) == 1
    assert "the build used" in capsys.readouterr().err
    assert not (tree / MARLIN).exists()


def fake_venv(tmp_path, version):
    site = tmp_path / ("site-" + version)
    torch = site / "torch"
    (torch / "include/c10").mkdir(parents=True)
    (torch / "include/c10/a.h").write_text("h")
    (torch / "lib").mkdir()
    (torch / "lib/libc10.so").write_bytes(b"c10")
    (torch / "__init__.py").write_text(
        f"__version__ = {version!r}\nclass version:\n    cuda = '13.0'\n"
        "class _C:\n    _GLIBCXX_USE_CXX11_ABI = True\n")
    venv = tmp_path / ("venv-" + version)
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").write_text(f"#!/bin/sh\nPYTHONPATH={site} exec {sys.executable} \"$@\"\n")
    (venv / "bin/python").chmod(0o755)
    return venv


def test_install_checks_the_validation_venv_torch(fork, tmp_path, capsys):
    tree, native, _ = fork
    venv = fake_venv(tmp_path, "2.13.0+cu130")
    probe = json.loads(subprocess.run([str(venv / "bin/python"), "-c", nb.PROBE, *nb.TORCH_LIBS],
                                      check=True, capture_output=True, text=True).stdout)
    manifest = json.loads((native / "manifest.json").read_text())
    manifest["python"] = probe
    (native / "manifest.json").write_text(nb.dump(manifest))
    assert nb.main(["install", "--native", str(native), "--tree", str(tree),
                    "--venv", str(venv)]) == 0
    other = fake_venv(tmp_path, "2.14.0+cu130")
    assert nb.main(["install", "--native", str(native), "--tree", str(tree),
                    "--venv", str(other)]) == 1
    assert "torch differs" in capsys.readouterr().err


def test_pack_requires_the_pinned_manifest_and_commit(fork, tmp_path, capsys):
    tree, native, commit = fork
    opt = tmp_path / "opt"
    (opt / "venv").mkdir(parents=True)
    tree.rename(opt / "vllm-src")
    pin = nb.sha256(native / "manifest.json")
    base = ["pack", "--native", str(native), "--opt", str(opt)]
    assert nb.main(base + ["--manifest-sha256", "0" * 64, "--commit", commit]) == 1
    assert "not the pinned" in capsys.readouterr().err
    assert nb.main(base + ["--manifest-sha256", pin, "--commit", "b" * 40]) == 1
    assert "not the engine" in capsys.readouterr().err
    assert nb.main(base + ["--manifest-sha256", pin, "--commit", commit]) == 0
    assert (opt / "vllm-src" / MARLIN).read_bytes() == b"marlin bytes"
    assert (opt / nb.SHIPPED_MANIFEST).read_bytes() == (native / "manifest.json").read_bytes()


def test_pack_takes_a_build_of_a_commit_with_the_same_inputs(fork, tmp_path, capsys):
    tree, native, built = fork
    (tree / "vllm/__init__.py").write_text("# Python only\n")
    git(tree, "commit", "-qam", "python change")
    engine = git(tree, "rev-parse", "HEAD")
    opt = tmp_path / "opt"
    (opt / "venv").mkdir(parents=True)
    tree.rename(opt / "vllm-src")
    pin = nb.sha256(native / "manifest.json")
    argv = ["pack", "--native", str(native), "--opt", str(opt), "--manifest-sha256", pin]
    assert nb.main(argv + ["--commit", engine]) == 0
    assert f"built from {built}" in capsys.readouterr().out
    assert nb.main(["check-tree", "--manifest", str(native / "manifest.json"),
                    "--tree", str(opt / "vllm-src"), "--commit", engine]) == 0
    (opt / "vllm-src/csrc/a.cu").write_text("changed kernel")
    git(opt / "vllm-src", "commit", "-qam", "kernel change")
    other = git(opt / "vllm-src", "rev-parse", "HEAD")
    assert nb.main(argv + ["--commit", other]) == 1
    assert "the build used" in capsys.readouterr().err


def test_check_tree_on_a_finalized_tree_reads_its_provenance(fork, capsys):
    tree, native, commit = fork
    assert nb.main(["install", "--native", str(native), "--tree", str(tree)]) == 0
    shutil.rmtree(tree / ".git")
    check = ["check-tree", "--manifest", str(native / "manifest.json"), "--tree", str(tree)]
    (tree / "provenance.json").write_text(json.dumps({"engine_commit": commit}))
    assert nb.main(check + ["--commit", commit]) == 0
    assert nb.main(check + ["--commit", "c" * 40]) == 1
    assert "records engine" in capsys.readouterr().err


def test_shipped_manifest_belongs_to_the_engine_layer():
    assert payload.partition(Path(nb.SHIPPED_MANIFEST)) == "engine"
    assert payload.partition(Path("vllm-src") / MARLIN) == "native"


def test_check_oci_reads_the_final_state_of_every_layer(fork, tmp_path, capsys):
    _, native, _ = fork
    manifest = native / "manifest.json"
    good = (native / MARLIN).read_bytes()
    shipped = ("opt/" + nb.SHIPPED_MANIFEST, "file", manifest.read_bytes())
    path = "opt/vllm-src/" + MARLIN
    cases = {
        "pass": ([[(path, "file", b"old")], [(path, "file", good), shipped]], True),
        "older layer only": ([[(path, "file", good), shipped], [(path, "file", b"new")]], False),
        "whited out": ([[(path, "file", good), shipped],
                        [("opt/vllm-src/vllm/.wh._ampere_marlin_C.abi3.so", "file", b"")]],
                       False),
        "other build": ([[(path, "file", good), shipped,
                          ("opt/vllm-src/vllm/_ampere_marlin_C.so", "file", b"x")]], False),
        "no manifest": ([[(path, "file", good)]], False),
    }
    for name, (layers, ok) in cases.items():
        oci, _ = oci_layout(tmp_path / name.replace(" ", "-"), [layer(e) for e in layers])
        code = nb.main(["check-oci", "--oci", str(oci), "--manifest", str(manifest),
                        "--require-shipped-manifest"])
        assert (code == 0) == ok, (name, capsys.readouterr())


def exl3_fork(tmp_path, files):
    repo = tmp_path / "exl3fork"
    for rel in ("csrc/a.cu", "cmake/u.cmake", "CMakeLists.txt", nb.MARLIN_BUILDER, *files):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("x")
    git(tmp_path, "init", "-q", str(repo))
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    (tmp_path / "rootfs.base.json").write_text(json.dumps({"manifest": nb.BASE_MANIFEST}))
    uv = tmp_path / "uv"
    uv.write_text(f"#!/bin/sh\necho {tmp_path / 'uv-cache'}\n")
    uv.chmod(0o755)
    constraints = tmp_path / "CONSTRAINTS"
    constraints.write_text("# dependency lock\n")
    return ["build", "--repo", str(repo), "--commit", "HEAD", "--rootfs", str(rootfs), "--constraints", str(constraints),
            "--work", str(tmp_path / "work"), "--out", str(tmp_path / "out"),
            "--uv", str(uv), "--jobs", "3"]


class Recorder:
    """Records the namespace steps; stops the build at the toolchain probe."""
    steps = []

    def __init__(self, rootfs, opt, build, env, log):
        self.build = build

    def __call__(self, script, **kw):
        Recorder.steps.append((script, kw))
        for name in ("ampere_marlin", "exl3", "stable"):
            (self.build / name).mkdir(exist_ok=True)
            (self.build / name / "build.ninja").write_text("cxx = c++\n")

    def output(self, script):
        raise nb.NativeError("stop after the compile steps")


@pytest.mark.parametrize("files, mode", [
    ((nb.EXL3_BUILDER, nb.EXL3_SOURCE), "standalone"),
    ((nb.EXL3_SOURCE,), "cmake"), ((), None)])
def test_exl3_build_follows_the_commit(tmp_path, monkeypatch, capsys, files, mode):
    argv = exl3_fork(tmp_path, files)
    monkeypatch.setattr(nb, "Namespace", Recorder)
    monkeypatch.setattr(Recorder, "steps", [])
    assert nb.main(argv) == 1
    err = capsys.readouterr().err
    scripts = [script for script, _ in Recorder.steps]
    if mode == "cmake":
        # The CMake target needs its FetchContent sources and compiles nothing without them.
        assert "pass --fetchcontent" in err and not scripts
        return
    assert "stop after the compile steps" in err
    venv, marlin = scripts[:2]
    assert "native-tools" not in venv and nb.MARLIN_BUILDER in marlin
    if mode is None:
        assert len(scripts) == 2
        return
    (exl3, kw), = Recorder.steps[2:]
    assert nb.EXL3_BUILDER in exl3 and kw == {"source_readonly": True} and "cmake" not in exl3


def test_install_and_check_tree_take_the_standalone_exl3(fork):
    tree, native, commit = fork
    manifest = json.loads((native / "manifest.json").read_text())
    (native / nb.EXL3).write_bytes(b"exl3 bytes")
    manifest["artefacts"][nb.EXL3] = {"sha256": sha(b"exl3 bytes"), "bytes": 10}
    manifest["extensions"] = {nb.EXL3: {"builder": nb.EXL3_BUILDER}}
    (native / "manifest.json").write_text(nb.dump(manifest))
    shadow = tree / "vllm/_exl3_C.so"
    shadow.write_bytes(b"host build")
    assert nb.main(["install", "--native", str(native), "--tree", str(tree)]) == 0
    assert (tree / nb.EXL3).read_bytes() == b"exl3 bytes" and not shadow.exists()
    assert nb.main(["check-tree", "--manifest", str(native / "manifest.json"),
                    "--tree", str(tree), "--commit", commit]) == 0


def test_build_without_marlin_compiles_only_exl3(tmp_path, monkeypatch, capsys):
    argv = exl3_fork(tmp_path, (nb.EXL3_BUILDER, nb.EXL3_SOURCE)) + ["--no-marlin"]
    monkeypatch.setattr(nb, "Namespace", Recorder)
    monkeypatch.setattr(Recorder, "steps", [])
    assert nb.main(argv) == 1
    assert "stop after the compile steps" in capsys.readouterr().err
    scripts = [script for script, _ in Recorder.steps]
    assert len(scripts) == 2 and not any(nb.MARLIN_BUILDER in s for s in scripts)
    assert nb.EXL3_BUILDER in scripts[1]


def test_build_without_marlin_needs_exl3(tmp_path, monkeypatch, capsys):
    argv = exl3_fork(tmp_path, ()) + ["--no-marlin"]
    monkeypatch.setattr(nb, "Namespace", Recorder)
    monkeypatch.setattr(Recorder, "steps", [])
    assert nb.main(argv) == 1
    assert "leaves nothing to build" in capsys.readouterr().err and not Recorder.steps


def test_check_tree_refuses_an_extension_the_manifest_leaves_out(fork):
    tree, native, commit = fork
    manifest = json.loads((native / "manifest.json").read_text())
    (native / MARLIN).unlink()
    (native / nb.EXL3).write_bytes(b"exl3 bytes")
    manifest["artefacts"] = {nb.EXL3: {"sha256": sha(b"exl3 bytes"), "bytes": 10}}
    (native / "manifest.json").write_text(nb.dump(manifest))
    assert nb.main(["install", "--native", str(native), "--tree", str(tree)]) == 0
    check = ["check-tree", "--manifest", str(native / "manifest.json"), "--tree", str(tree),
             "--commit", commit]
    assert nb.main(check) == 0
    for name in (MARLIN, "vllm/_ampere_marlin_C.so"):
        (tree / name).write_bytes(b"unlisted build")
        assert nb.main(check) == 1
        (tree / name).unlink()
    assert nb.main(check) == 0


# ------------------------------------------------------------------ --stable, FetchContent, --cache
def fetchcontent_dir(tmp_path, names=("cutlass",)):
    fc = tmp_path / "fetchcontent"
    for name in names:
        (fc / f"{name}-src").mkdir(parents=True, exist_ok=True)
        (fc / f"{name}-src" / "CMakeLists.txt").write_text(name)
        (fc / f"{name}-src" / nb.FETCH_MARKER).write_text(nb.dump({"repo": "r", "tag": "t", "submodules": None,
                                                                   "head": "h-" + name}))
    return fc


IMAGE = "pytorch/manylinux2_28-builder:cuda13.0-78e737ad29420ffc4800e677c51e2a852caf8359"
ARCHS = "7.5 8.0 8.6 8.9 9.0 10.0 12.0"
# upstream 330c5bffee's .buildkite/release-pipeline.yaml, cut to its env and the CUDA wheel and image steps
PIPELINE = f"""env:
  CUDA_ARCH_X86: "{ARCHS}"
  CUDA_ARCH_AARCH64: "8.0 8.7 8.9 9.0 10.0 11.0 12.0"

steps:
  - group: "Build CUDA 13.0 Python wheels"
    steps:
      - label: "Build wheel - aarch64 - CUDA 13.0"
        commands:
          - "DOCKER_BUILDKIT=1 docker build --build-arg max_jobs=16 --build-arg USE_SCCACHE=1 --build-arg CUDA_VERSION=13.0.2 --build-arg torch_cuda_arch_list=\\"${{CUDA_ARCH_AARCH64}}\\" --build-arg BUILD_BASE_IMAGE=pytorch/manylinuxaarch64-builder:cuda13.0-b8b5 --tag vllm-ci:build-image --target build --progress plain -f docker/Dockerfile ."
      - label: "Build wheel - x86_64 - CUDA 13.0"
        depends_on: ~
        commands:
          - "DOCKER_BUILDKIT=1 docker build --build-arg max_jobs=16 --build-arg USE_SCCACHE=1 --build-arg GIT_REPO_CHECK=1 --build-arg CUDA_VERSION=13.0.2 --build-arg torch_cuda_arch_list=\\"${{CUDA_ARCH_X86}}\\" --build-arg BUILD_BASE_IMAGE={IMAGE} --tag vllm-ci:build-image --target build --progress plain -f docker/Dockerfile ."
          - "mkdir artifacts"
  - label: "Build release image - x86_64 - CUDA 13.0"
    commands:
      - |
        DOCKER_BUILDKIT=1 docker build \\
          --build-arg BUILD_BASE_IMAGE={IMAGE} \\
          --target vllm-openai -f docker/Dockerfile .
"""
WHEEL_ENV = ["PATH=/usr/local/cuda/bin:/opt/rh/gcc-toolset-13/root/usr/bin:/usr/bin:/bin", "LC_ALL=en_US.UTF-8"]


def recipe_repo(tmp_path, pipeline=PIPELINE):
    repo = tmp_path / "recipe"
    (repo / ".buildkite").mkdir(parents=True)
    (repo / nb.WHEEL_PIPELINE).write_text(pipeline)
    git(tmp_path, "init", "-q", str(repo))
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "pipeline")
    return repo


def test_the_wheel_recipe_is_the_pipeline_s_x86_64_cuda_wheel_build(tmp_path):
    """MatchedRebuild 10-31: a --stable rebuild in our CUDA 13.3 builder at TORCH_CUDA_ARCH_LIST 8.0 changed the SASS of
    1387 of the wheel's 3537 sm_80 kernels for a host-only source change. The rebuild follows the wheel's own build: the
    x86_64 CUDA wheel step's image (pinned by digest) and arch list (its env variable), never the aarch64 or image
    steps'."""
    repo = recipe_repo(tmp_path)
    assert nb.wheel_recipe(repo, git(repo, "rev-parse", "HEAD")) == {
        "image": IMAGE, "manifest": nb.WHEEL_BUILDERS[IMAGE], "torch_cuda_arch_list": ARCHS, "build_type": "Release",
        "nvcc_threads": 8, "cmake": nb.CMAKE_VERSION, "source": "/workspace",
        "build": "/workspace/build/temp.linux-x86_64-cpython-312", "fetchcontent": "/workspace/.deps"}
    assert nb.main(["wheel-recipe", "--repo", str(repo), "--commit", "HEAD"]) == 0


@pytest.mark.parametrize("old, new, why", [
    (IMAGE + " --tag", "pytorch/manylinux2_28-builder:cuda13.0-other --tag", "which WHEEL_BUILDERS does not pin"),
    ("--build-arg USE_SCCACHE=1 --build-arg GIT", "--build-arg GIT", "no USE_SCCACHE=1"),
    ('  CUDA_ARCH_X86: "', '  CUDA_ARCH_OTHER: "', "is defined 0 times"),
    ("Build wheel - x86_64 - CUDA 13.0", "Build wheel - x86_64 - CPU", "not one 'Build wheel - x86_64 - CUDA' step"),
])
def test_a_wheel_build_the_recipe_cannot_follow_is_refused(tmp_path, capsys, old, new, why):
    assert old in PIPELINE
    repo = recipe_repo(tmp_path, PIPELINE.replace(old, new, 1))
    assert nb.main(["wheel-recipe", "--repo", str(repo), "--commit", "HEAD"]) == 1
    assert why in capsys.readouterr().err


def wheel_fork(tmp_path, monkeypatch, files=(nb.EXL3_BUILDER, nb.EXL3_SOURCE)):
    """exl3_fork with upstream's pipeline, the wheel builder's extraction stubbed (its Env WHEEL_ENV), and the build
    argv's --wheel-oci."""
    argv = exl3_fork(tmp_path, files)
    repo = Path(argv[argv.index("--repo") + 1])
    (repo / ".buildkite").mkdir()
    (repo / nb.WHEEL_PIPELINE).write_text(PIPELINE)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "pipeline")
    extracted = []

    def extract(oci, rootfs, recipe):
        rootfs.mkdir(parents=True)
        extracted.append(oci)
        return {"image": recipe["image"], "manifest": recipe["manifest"], "config": "sha256:wc", "env": WHEEL_ENV}
    monkeypatch.setattr(nb, "extract_wheel_builder", extract)
    monkeypatch.setattr(nb, "wheel_builder_oci", lambda directory, recipe: directory / recipe["manifest"].split(":")[1])
    monkeypatch.setattr(nb, "fetch_sources", lambda *a: {})
    return argv + ["--wheel-oci", str(tmp_path / "wheel-oci")], extracted


class WheelRecorder(Recorder):
    """Records the wheel builder's steps (with its rootfs and environment) and plays them: the CMake tree and the
    installed libraries in the source it binds at /workspace."""
    def __init__(self, rootfs, opt, build, env, log):
        super().__init__(rootfs, opt, build, env, log)
        self.rootfs, self.opt, self.env = rootfs, opt, env

    def __call__(self, script, **kw):
        Recorder.steps.append((script, dict(kw, rootfs=self.rootfs, env=self.env)))
        src = self.opt / "vllm-src"
        tree = nb.wheel_path(src, nb.WHEEL_BUILD)
        tree.mkdir(parents=True, exist_ok=True)
        (tree / "build.ninja").write_text("cxx = c++\n")
        (tree / "CMakeCache.txt").write_text("kept\n")
        for rel, target in nb.WHEEL_TARGETS.values():
            if f"--component {target}\n" in script:
                (nb.wheel_path(src, nb.WHEEL_LIB) / rel).parent.mkdir(parents=True, exist_ok=True)
                (nb.wheel_path(src, nb.WHEEL_LIB) / rel).write_bytes(b"built " + rel.encode())


def test_stable_builds_the_cmake_target_in_the_wheel_builder_as_upstream(tmp_path, monkeypatch, capsys):
    """NatUpstream 10-22: --stable adds the CMake target _C_stable_libtorch (configure once, build only it, writable
    source for the generated Marlin kernels) after the fork natives, with the FetchContent sources it fetched.
    MatchedRebuild 10-31: it builds the way upstream's wheel is built (test_the_wheel_recipe_...): in the pinned wheel
    builder's root and environment, at the wheel's arch list, with setup.py's configure of /workspace and its install."""
    fc = fetchcontent_dir(tmp_path)
    argv, extracted = wheel_fork(tmp_path, monkeypatch)
    monkeypatch.setattr(nb, "Namespace", Recorder)
    monkeypatch.setattr(nb, "WheelNamespace", WheelRecorder)
    monkeypatch.setattr(Recorder, "steps", [])
    assert nb.main(argv + ["--stable", "--fetchcontent", str(fc)]) == 1
    assert "stop after the compile steps" in capsys.readouterr().err
    assert extracted == [tmp_path / "wheel-oci" / nb.WHEEL_BUILDERS[IMAGE].split(":")[1]]
    (venv, _), (marlin, _), (exl3, _), (stable, kw) = Recorder.steps
    assert f"pip install cmake=={nb.CMAKE_VERSION}" in venv and nb.MARLIN_BUILDER in marlin
    assert kw["rootfs"] == tmp_path / "work/wheel-rootfs"
    assert kw["env"]["TORCH_CUDA_ARCH_LIST"] == ARCHS and kw["env"]["NVCC_APPEND_FLAGS"] == "-objtemp"
    assert kw["env"]["PATH"] == "/opt/venv/bin:/usr/local/cuda/bin:/opt/rh/gcc-toolset-13/root/usr/bin:/usr/bin:/bin"
    assert kw["env"]["LC_ALL"] == "en_US.UTF-8" and "CUDA_HOME" not in kw["env"]
    assert f"cd {nb.WHEEL_BUILD}\n[ -f CMakeCache.txt ] || /opt/native-tools/bin/cmake /workspace -G Ninja " \
           "-DCMAKE_BUILD_TYPE=Release" in stable
    for flag in ("-DFETCHCONTENT_BASE_DIR=/workspace/.deps", "-DNVCC_THREADS=8", "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
                 "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc", "--target _C_stable_libtorch -j 1"):
        assert flag in stable
    assert f"--install . --prefix {nb.WHEEL_LIB} --component _C_stable_libtorch\n" in stable
    # --jobs threads as setup.py spends MAX_JOBS: jobs // NVCC_THREADS compile jobs of up to 8 nvcc threads each
    assert "-j 4\n" in nb.wheel_script(32, [(nb.STABLE, "_C_stable_libtorch")])
    assert (tmp_path / "work/opt/vllm-src/.deps/cutlass-src/CMakeLists.txt").read_text() == "cutlass"


def test_moe_stable_builds_the_moe_target_in_the_same_cmake_tree(tmp_path, monkeypatch, capsys):
    """Main 10-26: --moe-stable adds the CMake target _moe_C_stable_libtorch; with --stable both build in one
    `cmake --build` of the same tree, each installed to its wheel path."""
    fc = fetchcontent_dir(tmp_path)
    argv, _ = wheel_fork(tmp_path, monkeypatch)
    monkeypatch.setattr(nb, "Namespace", Recorder)
    monkeypatch.setattr(nb, "WheelNamespace", WheelRecorder)
    monkeypatch.setattr(Recorder, "steps", [])
    assert nb.main(argv + ["--stable", "--moe-stable", "--fetchcontent", str(fc)]) == 1
    stable = Recorder.steps[-1][0]
    assert "--target _C_stable_libtorch _moe_C_stable_libtorch -j 1" in stable
    assert "--component _moe_C_stable_libtorch\n" in stable and "--component _C_stable_libtorch\n" in stable
    Recorder.steps.clear()
    (tmp_path / "alone").mkdir()
    alone, _ = wheel_fork(tmp_path / "alone", monkeypatch)
    assert nb.main(alone + ["--moe-stable", "--fetchcontent", str(fc)]) == 1
    stable = Recorder.steps[-1][0]
    assert "--target _moe_C_stable_libtorch -j 1" in stable and "--component _C_stable_libtorch\n" not in stable


def test_stable_needs_fetchcontent_and_the_wheel_builder(tmp_path, monkeypatch, capsys):
    for flags in (["--stable"], ["--moe-stable"]):
        (tmp_path / flags[0].strip("-")).mkdir()
        argv = exl3_fork(tmp_path / flags[0].strip("-"), (nb.EXL3_BUILDER, nb.EXL3_SOURCE)) + flags
        monkeypatch.setattr(nb, "Namespace", Recorder)
        monkeypatch.setattr(Recorder, "steps", [])
        assert nb.main(argv) == 1
        target = {"--stable": "_C_stable_libtorch", "--moe-stable": "_moe_C_stable_libtorch"}[flags[0]]
        assert (f"--stable/--moe-stable build the CMake targets {target}: pass --fetchcontent"
                in capsys.readouterr().err)
        assert not Recorder.steps
    (tmp_path / "oci").mkdir()
    argv, _ = wheel_fork(tmp_path / "oci", monkeypatch)
    argv = argv[:argv.index("--wheel-oci")]
    assert nb.main(argv + ["--stable", "--fetchcontent", str(fetchcontent_dir(tmp_path))]) == 1
    assert "pass --wheel-oci DIR" in capsys.readouterr().err and not Recorder.steps


def test_the_wheel_builder_namespace_is_offline_with_our_venv_beside_its_compilers(tmp_path):
    env = nb.wheel_environment(1700000000, 4, WHEEL_ENV, ARCHS)
    assert not any(str(Path.home()) in v for v in env.values()) and env["SOURCE_DATE_EPOCH"] == "1700000000"
    argv = nb.wheel_namespace_argv(tmp_path / "rootfs", tmp_path / "opt", env, ["true"])
    assert "--clearenv" in argv and "--share-net" not in argv and nb.BUILD not in argv
    binds = {argv[i + 2]: (argv[i], argv[i + 1]) for i, a in enumerate(argv) if a in ("--bind", "--ro-bind")}
    assert binds == {"/": ("--ro-bind", str(tmp_path / "rootfs")),
                     **{f"/opt/{n}": ("--ro-bind", str(tmp_path / "opt" / n)) for n in ("python", "venv", "native-tools")},
                     "/workspace": ("--bind", str(tmp_path / "opt/vllm-src"))}


def test_fetch_declarations_read_the_commit_s_git_sources(tmp_path):
    repo = tmp_path / "decl"
    files = {
        "CMakeLists.txt": 'set(CUTLASS_REVISION "v4.7.1")\nif(X)\n  FetchContent_Declare(cutlass SOURCE_DIR ${D})\n'
                          'else()\n  FetchContent_Declare(\n    cutlass\n    GIT_REPOSITORY https://h/cutlass.git\n'
                          '    # keep in sync\n    GIT_TAG ${CUTLASS_REVISION}\n    GIT_SHALLOW TRUE\n  )\nendif()\n',
        "cmake/external_projects/t.cmake": 'if(ROCM)\n  set(T_GIT "https://h/rocm.git")\nelse()\n'
                                           '  set(T_GIT "https://h/t.git")\nendif()\nFetchContent_Populate(\n'
                                           '  DeepGEMM SOURCE_DIR "${S}" GIT_REPOSITORY "${T_GIT}" GIT_TAG "abc"\n'
                                           '  GIT_SUBMODULES "third-party/cutlass" "third-party/jit"\n'
                                           '  GIT_PROGRESS TRUE)\n',
        "cmake/cpu_extension.cmake": "FetchContent_Declare(oneDNN GIT_REPOSITORY https://h/o.git GIT_TAG v1)\n",
    }
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    git(tmp_path, "init", "-q", str(repo))
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "c")
    assert nb.fetch_declarations(repo, "HEAD") == {
        "cutlass": {"repo": "https://h/cutlass.git", "tag": "v4.7.1", "submodules": None},
        "deepgemm": {"repo": "https://h/t.git", "tag": "abc", "submodules": ["third-party/cutlass", "third-party/jit"]}}


def test_fetch_sources_clone_the_declared_tag_once(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    git(tmp_path, "init", "-q", str(upstream))
    (upstream / "include").mkdir()
    (upstream / "include/x.h").write_text("v1\n")
    git(upstream, "add", ".")
    git(upstream, "commit", "-qm", "v1")
    git(upstream, "tag", "v1")
    head = git(upstream, "rev-parse", "HEAD")
    dep = {"repo": f"file://{upstream}", "tag": "v1", "submodules": None}
    monkeypatch.setattr(nb, "fetch_declarations", lambda repo, commit: {"lib": dep})
    fc = tmp_path / "fc"
    assert nb.fetch_sources(tmp_path, "HEAD", fc) == {"lib": head}
    assert (fc / "lib-src/include/x.h").read_text() == "v1\n"
    assert json.loads((fc / "lib-src" / nb.FETCH_MARKER).read_text()) == {**dep, "head": head}
    calls = []
    monkeypatch.setattr(nb, "run", lambda argv, **kw: calls.append(argv))
    assert nb.fetch_sources(tmp_path, "HEAD", fc) == {"lib": head} and not calls   # kept: same declaration
    deps = tmp_path / "deps"
    assert nb.sync_sources(fc, deps) == {"lib": head} and (deps / "lib-src/include/x.h").is_file()
    assert not (deps / "lib-src/.git").exists()


class Completer(Recorder):
    """Records the namespace steps and plays the build: writes each extension's output (and the venv), answers the
    probes."""
    def __init__(self, rootfs, opt, build, env, log):
        super().__init__(rootfs, opt, build, env, log)
        self.opt = opt

    def __call__(self, script, **kw):
        if "uv venv" in script:
            (self.opt / "venv").mkdir(exist_ok=True)
        super().__call__(script, **kw)
        for package in ("vllm", nb.PLUGIN_PACKAGE):
            (self.build / "out" / package).mkdir(parents=True, exist_ok=True)
        for rel in (nb.MARLIN, nb.EXL3, nb.PLUGIN_MARLIN, nb.PLUGIN_EXL3):
            (self.build / "out" / rel).write_bytes(b"built " + rel.encode())

    def output(self, script):
        return "{}"


class WheelCompleter(WheelRecorder):
    def output(self, script):
        return "{}"


def test_cache_build_rewrites_only_changed_sources_and_keeps_venv_and_cmake_tree(tmp_path, monkeypatch, capsys):
    """--cache: the second build reuses the base, the wheel builder, the venv and the CMake tree, rewrites only the
    files the commit changes (fresh mtime, so Ninja rebuilds what they reach) and names the commit the CMake tree was
    built at. A commit of another wheel recipe configures a fresh CMake tree (a kept one keeps its configure's arch
    list)."""
    fc = fetchcontent_dir(tmp_path)
    argv, extracted = wheel_fork(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    i = argv.index("--work")
    argv[i:i + 2] = ["--cache", str(cache)]
    (tmp_path / "rootfs.base.json").write_text(json.dumps({"index": nb.BASE_INDEX, "manifest": nb.BASE_MANIFEST,
                                                           "config": "sha256:c"}))
    uv = tmp_path / "uv"
    uv.write_text(f"#!/bin/sh\n[ \"$1\" = --version ] && echo uv 0.9.0 || echo {tmp_path / 'uv-cache'}\n")
    repo = Path(argv[argv.index("--repo") + 1])
    monkeypatch.setattr(nb, "Namespace", Completer)
    monkeypatch.setattr(nb, "WheelNamespace", WheelCompleter)
    monkeypatch.setattr(Recorder, "steps", [])
    stable = argv + ["--stable", "--moe-stable", "--fetchcontent", str(fc)]
    first = git(repo, "rev-parse", "HEAD")
    assert nb.main(stable) == 0
    record = json.loads((tmp_path / "out/build-record.json").read_text())
    assert record["cache"] == {"fresh": True, "changed_paths": None} and "python_venv_s" in record["timings"]
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert set(manifest["artefacts"]) == {nb.MARLIN, nb.EXL3, nb.STABLE, nb.MOE_STABLE}
    assert manifest["extensions"][nb.STABLE]["incremental_from"] is None
    assert manifest["extensions"][nb.STABLE]["fetchcontent"] == {"cutlass": "h-cutlass"}
    assert manifest["extensions"][nb.MOE_STABLE]["builder"] == "CMakeLists.txt target _moe_C_stable_libtorch"
    assert manifest["wheel_toolchain"]["recipe"]["image"] == IMAGE and manifest["wheel_toolchain"]["config"] == "sha256:wc"
    assert manifest["wheel_toolchain"]["environment"]["TORCH_CUDA_ARCH_LIST"] == ARCHS
    assert (tmp_path / "out" / nb.STABLE).read_bytes() == b"built " + nb.STABLE.encode()
    src = cache / "opt/vllm-src"
    os.utime(src / "cmake/u.cmake", (1, 1))
    os.utime(src / "csrc/a.cu", (1, 1))
    (repo / "csrc/a.cu").write_text("dynamic smem\n")
    git(repo, "commit", "-qam", "change")
    shutil.rmtree(tmp_path / "out")
    Recorder.steps.clear()
    assert nb.main(stable) == 0
    record = json.loads((tmp_path / "out/build-record.json").read_text())
    assert record["cache"] == {"fresh": False, "changed_paths": 1} and "python_venv_s" not in record["timings"]
    assert (src / "csrc/a.cu").read_text() == "dynamic smem\n" and (src / "csrc/a.cu").stat().st_mtime > 1
    assert (src / "cmake/u.cmake").stat().st_mtime == 1                     # untouched: not rebuilt
    assert not any("uv pip install" in s for s, _ in Recorder.steps) and len(extracted) == 1
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert manifest["extensions"][nb.STABLE]["incremental_from"] == first
    assert manifest["extensions"][nb.MOE_STABLE]["incremental_from"] == first
    assert json.loads((cache / "state.json").read_text())["stable_commit"] == git(repo, "rev-parse", "HEAD")
    shutil.rmtree(tmp_path / "out")
    (repo / nb.WHEEL_PIPELINE).write_text(PIPELINE.replace(f'CUDA_ARCH_X86: "{ARCHS}"', 'CUDA_ARCH_X86: "8.0 9.0"'))
    git(repo, "commit", "-qam", "another arch list")
    assert nb.main(stable) == 0
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert manifest["extensions"][nb.STABLE]["incremental_from"] is None
    assert manifest["wheel_toolchain"]["recipe"]["torch_cuda_arch_list"] == "8.0 9.0"
    shutil.rmtree(tmp_path / "out")
    (tmp_path / "CONSTRAINTS").write_text("# another lock\n")                 # another venv: start afresh
    assert nb.main(argv) == 0
    record = json.loads((tmp_path / "out/build-record.json").read_text())
    assert record["cache"]["fresh"] is True and nb.STABLE not in record["artefacts"]


def plugin_fork(tmp_path, builders=(nb.PLUGIN_MARLIN_BUILDER, nb.EXL3_BUILDER)):
    """A plugin commit (the plugin branch's layout) and the build argv for it (exl3_fork's rootfs, uv, lock)."""
    argv = exl3_fork(tmp_path, ())
    repo = tmp_path / "pluginrepo"
    for rel in ("csrc/exl3/q.cu", "csrc/ampere_marlin/m.cu", "vllm_sm80/__init__.py", "README.md", *builders):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("x")
    git(tmp_path, "init", "-q", str(repo))
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "plugin")
    argv[argv.index("--repo") + 1] = str(repo)
    return argv + ["--plugin"], repo


def test_plugin_build_builds_the_plugin_natives_from_csrc(tmp_path, monkeypatch, capsys):
    argv, repo = plugin_fork(tmp_path)
    monkeypatch.setattr(nb, "Namespace", Completer)
    monkeypatch.setattr(Recorder, "steps", [])
    (tmp_path / "rootfs.base.json").write_text(json.dumps({"index": nb.BASE_INDEX, "manifest": nb.BASE_MANIFEST,
                                                           "config": "sha256:c"}))
    uv = tmp_path / "uv"
    uv.write_text(f"#!/bin/sh\n[ \"$1\" = --version ] && echo uv 0.9.0 || echo {tmp_path / 'uv-cache'}\n")
    assert nb.main(argv) == 0, capsys.readouterr().err
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert manifest["package"] == "vllm_sm80" and set(manifest["artefacts"]) == {nb.PLUGIN_MARLIN, nb.PLUGIN_EXL3}
    assert manifest["source"]["inputs"] == {"csrc": git(repo, "rev-parse", "HEAD:csrc")}
    assert manifest["extensions"][nb.PLUGIN_MARLIN]["builder"] == nb.PLUGIN_MARLIN_BUILDER
    venv, marlin, exl3 = [script for script, _ in Recorder.steps]
    assert "native-tools" not in venv
    assert nb.PLUGIN_MARLIN_BUILDER in marlin and "--out /tmp/native-build/out/vllm_sm80 " in marlin
    assert nb.EXL3_BUILDER in exl3 and "--out /tmp/native-build/out/vllm_sm80 " in exl3
    for rel in (nb.PLUGIN_MARLIN, nb.PLUGIN_EXL3):
        assert (tmp_path / "out" / rel).read_bytes() == b"built " + rel.encode()


@pytest.mark.parametrize("missing", [nb.PLUGIN_MARLIN_BUILDER, nb.EXL3_BUILDER])
def test_plugin_build_needs_both_plugin_builders(tmp_path, monkeypatch, capsys, missing):
    argv, _ = plugin_fork(tmp_path, [b for b in (nb.PLUGIN_MARLIN_BUILDER, nb.EXL3_BUILDER) if b != missing])
    monkeypatch.setattr(nb, "Namespace", Recorder)
    monkeypatch.setattr(Recorder, "steps", [])
    assert nb.main(argv) == 1
    assert missing in capsys.readouterr().err and not Recorder.steps


@pytest.mark.parametrize("option", [["--stable"], ["--moe-stable"], ["--no-marlin"], ["--fetchcontent", "/x"],
                                    ["--wheel-oci", "/x"]])
def test_plugin_build_takes_no_engine_options(tmp_path, option):
    argv, _ = plugin_fork(tmp_path)
    with pytest.raises(SystemExit) as stop:
        nb.main(argv + option)
    assert stop.value.code == 2


def test_install_and_check_tree_of_plugin_natives(tmp_path):
    _, repo = plugin_fork(tmp_path)
    native = tmp_path / "pnative"
    (native / "vllm_sm80").mkdir(parents=True)
    (native / nb.PLUGIN_EXL3).write_bytes(b"exl3 bytes")
    manifest = {"schema": 1, "package": "vllm_sm80",
                "source": {"commit": git(repo, "rev-parse", "HEAD"), "commit_time": 0,
                           "inputs": {"csrc": git(repo, "rev-parse", "HEAD:csrc")}},
                "artefacts": {nb.PLUGIN_EXL3: {"sha256": sha(b"exl3 bytes"), "bytes": 10}}}
    (native / "manifest.json").write_text(nb.dump(manifest))
    assert nb.main(["install", "--native", str(native), "--tree", str(repo)]) == 0
    assert (repo / nb.PLUGIN_EXL3).read_bytes() == b"exl3 bytes"
    check = ["check-tree", "--manifest", str(native / "manifest.json"), "--tree", str(repo)]
    assert nb.main(check) == 0
    (repo / nb.PLUGIN_MARLIN).write_bytes(b"unlisted build")        # a plugin native the manifest leaves out
    assert nb.main(check) == 1
    (repo / nb.PLUGIN_MARLIN).unlink()
    (repo / "csrc/exl3/q.cu").write_text("edited")                   # the plugin's one source input changed
    assert nb.main(["install", "--native", str(native), "--tree", str(repo)]) == 1
