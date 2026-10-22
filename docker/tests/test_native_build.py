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


def test_base_pins_match_both_builders():
    build = (ROOT / "build.sh").read_text()
    docker = (ROOT / "Dockerfile").read_text()
    assert re.search(r'^BASE=".*@' + nb.BASE_INDEX + '"$', build, re.M)
    assert re.search(r"^ARG BASE=.*@" + nb.BASE_INDEX + "$", docker, re.M)
    pin = re.search(r"^RELEASE_NATIVE_MANIFEST_SHA256='(.*)'$", build, re.M).group(1)
    assert re.search(r"^ARG NATIVE_MANIFEST_SHA256=" + re.escape(pin) + "$", docker, re.M)


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
    return ["build", "--repo", str(repo), "--commit", "HEAD", "--rootfs", str(rootfs),
            "--work", str(tmp_path / "work"), "--out", str(tmp_path / "out"),
            "--uv", str(uv), "--jobs", "3"]


class Recorder:
    """Records the namespace steps; stops the build at the toolchain probe."""
    steps = []

    def __init__(self, rootfs, opt, build, env, log):
        self.build = build

    def __call__(self, script, **kw):
        Recorder.steps.append((script, kw))
        for name in ("ampere_marlin", "exl3"):
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
        assert "cmake --build" in nb.exl3_script("cmake", 3)
        return
    assert "stop after the compile steps" in err
    venv, marlin = scripts[:2]
    assert "native-tools" not in venv and nb.MARLIN_BUILDER in marlin
    if mode is None:
        assert len(scripts) == 2
        return
    (exl3, kw), = Recorder.steps[2:]
    assert exl3 == (f"cd {nb.SRC} && /opt/venv/bin/python {nb.EXL3_BUILDER} "
                    f"--out {nb.BUILD}/out/vllm --build-dir {nb.BUILD}/exl3 --verbose")
    assert kw == {"source_readonly": True} and "cmake" not in exl3


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
