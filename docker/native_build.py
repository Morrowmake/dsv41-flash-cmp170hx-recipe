#!/usr/bin/env python3
"""Build the engine's own compiled extensions once, in the image's pinned toolchain.

  build       compile them inside the digest-pinned CUDA base image (rootless,
              bubblewrap) from one fork commit; write the files and manifest.json
  install     put a manifest's exact files into a fork tree for native validation
  check-tree  a fork tree holds exactly the manifest's files
  pack        the image builders' step: check the pinned manifest, then install
              the files into the image stage (the image never compiles them)
  check-oci   an assembled OCI image holds exactly the manifest's files

Our extensions are vllm._ampere_marlin_C (the fork's
csrc/libtorch_stable/moe/ampere_marlin/build_standalone.py) and, on commits that
have csrc/exl3, vllm._exl3_C: csrc/exl3/build_standalone.py where the commit has it,
else the fork's CMake target (module.cpp only). Flags stay the fork's.

build --plugin: the natives of a plugin commit (the fork repo's `plugin` branch, package
vllm_sm80): vllm_sm80/_ampere_marlin_C.abi3.so (csrc/ampere_marlin/build_standalone.py) and
vllm_sm80/_exl3_C.abi3.so (csrc/exl3/build_standalone.py), from its csrc/ alone (the only
source input the manifest records; the manifest names "package": "vllm_sm80"). A release
names the build native-<csrc tree hash10>-t<torch>: one build per (csrc tree, torch of the
dependency lock). Not with --cache, --stable, --moe-stable, --no-marlin or --fetchcontent.

build --stable also rebuilds the wheel's core library vllm._C_stable_libtorch (the
CMake target _C_stable_libtorch, TORCH_CUDA_ARCH_LIST 8.0), and build --moe-stable the
wheel's MoE library vllm._moe_C_stable_libtorch (the CMake target _moe_C_stable_libtorch,
same CMake tree), for a commit that changes an upstream-owned input they read.
Its CMake configure reads every FetchContent source the commit's CMakeLists.txt and
cmake/external_projects/*.cmake declare: --fetchcontent DIR holds them as <name>-src,
cloned on the host at the declared GIT_TAG (and submodules) when missing or declared
otherwise (the base image has no git; the namespace has no network then).

build --cache DIR (instead of --work): a persistent work directory, locked for the
build: the extracted base, the Python and venv (kept while the base, the constraints and
the pinned versions are the same), the FetchContent copies and the CMake tree of the
wheel targets. The source is moved from the cached commit to --commit by rewriting only the
files that differ (fresh mtimes), so Ninja recompiles only what they reach; the fork
natives are built in fresh directories each time. The manifest names the commit the
CMake tree was last built at (extensions[...].incremental_from; null: a full build).

Reproducibility: the build runs in a private mount namespace with one fixed
layout (/opt/python, /opt/venv, /opt/vllm-src, /tmp/native-build), the CUDA
13.3.1 devel base image as its root (nvcc and g++ by image digest), the source
from `git archive` (file list and order from the commit, mtimes at the commit
time), SOURCE_DATE_EPOCH at the commit time, nvcc intermediates named after
their object instead of the process id (-objtemp), a fixed host name, user and
HOME, and a cleared environment. CUDA's front end names internal-linkage
symbols after the absolute source path, so a fixed layout, not only
-ffile-prefix-map, is what makes the bytes independent of the build directory.
The manifest names no host path, so two builds of one commit give the same
manifest bytes (a --cache build names the commit its CMake tree was built from).
"""
import argparse
import contextlib
import fcntl
import gzip
import hashlib
from importlib.machinery import EXTENSION_SUFFIXES
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

HERE = Path(__file__).resolve().parent

# build.sh/Dockerfile BASE (multi-platform index) and its linux/amd64 manifest.
BASE_INDEX = "sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87"
BASE_MANIFEST = "sha256:03c372fd9c65fe7739279f8c65473b315dc61efaaffab03e1e65bc7be7aee61e"
PYTHON_INSTALL_VERSION = "3.12.14"
CMAKE_VERSION = "3.31.10"
EXTRA_INDEX = "https://flashinfer.ai/whl/"
SOURCE_INPUTS = ("csrc", "cmake", "CMakeLists.txt")
SRC = "/opt/vllm-src"
BUILD = "/tmp/native-build"
MARLIN = "vllm/_ampere_marlin_C.abi3.so"
MARLIN_BUILDER = "csrc/libtorch_stable/moe/ampere_marlin/build_standalone.py"
# The builder refuses to run unless its build option is 1: VLLM_AMPERE_MARLIN_BUILD on current engine
# trees, VLLM_BUILD_AMPERE_MARLIN on trees from before that option was renamed.
MARLIN_BUILD_ENV = "VLLM_AMPERE_MARLIN_BUILD=1 VLLM_BUILD_AMPERE_MARLIN=1"
EXL3 = "vllm/_exl3_C.abi3.so"
EXL3_SOURCE = "csrc/exl3/module.cpp"
EXL3_BUILDER = "csrc/exl3/build_standalone.py"
# build --plugin: the plugin's package, its one source input and its Marlin builder (EXL3_BUILDER is the same path).
PLUGIN_PACKAGE = "vllm_sm80"
PLUGIN_INPUTS = ("csrc",)
PLUGIN_MARLIN = PLUGIN_PACKAGE + "/_ampere_marlin_C.abi3.so"
PLUGIN_MARLIN_BUILDER = "csrc/ampere_marlin/build_standalone.py"
PLUGIN_EXL3 = PLUGIN_PACKAGE + "/_exl3_C.abi3.so"
SHIPPED_MANIFEST = "venv/.native-manifest.json"
STABLE = "vllm/_C_stable_libtorch.abi3.so"
MOE_STABLE = "vllm/_moe_C_stable_libtorch.abi3.so"
# The wheel's CMake targets build rebuilds, by option (argparse dest): its artefact and target.
WHEEL_TARGETS = {"stable": (STABLE, "_C_stable_libtorch"), "moe_stable": (MOE_STABLE, "_moe_C_stable_libtorch")}
# The CMake files of a CUDA configure that declare FetchContent sources (cmake/cpu_extension.cmake: CPU builds only).
FETCH_FILES = ("CMakeLists.txt", "cmake/external_projects")
FETCH_MARKER = ".native-fetch.json"
# --cache: what the kept rootfs, Python, venv and native-tools depend on; another value starts the cache afresh.
CACHE_LAYOUT = 1
TOOLS = ("/usr/local/cuda/bin/nvcc", "/usr/local/cuda/bin/cudafe++",
         "/usr/local/cuda/nvvm/bin/cicc", "/usr/local/cuda/bin/ptxas",
         "/usr/local/cuda/bin/fatbinary", "/usr/local/cuda/bin/nvlink")
# Torch files the extensions compile and link against (ABI identity).
TORCH_LIBS = ("libc10.so", "libc10_cuda.so", "libtorch.so", "libtorch_cpu.so",
              "libtorch_cuda.so", "libtorch_python.so")
PROBE = r"""
import hashlib, json, os, sys, sysconfig
import torch
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()
def tree(root):
    h = hashlib.sha256()
    for d, dirs, files in os.walk(root):
        dirs.sort()
        for n in sorted(files):
            p = os.path.join(d, n)
            h.update(os.path.relpath(p, root).encode() + b"\0")
            h.update((os.readlink(p) if os.path.islink(p) else sha(p)).encode() + b"\0")
    return h.hexdigest()
t = os.path.dirname(torch.__file__)
print(json.dumps({"python": sys.version.split()[0], "torch": torch.__version__,
                  "torch_cuda": torch.version.cuda,
                  "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                  "torch_include_sha256": tree(os.path.join(t, "include")),
                  "python_include_sha256": tree(sysconfig.get_paths()["include"]),
                  "torch_libs": {n: sha(os.path.join(t, "lib", n))
                                 for n in sys.argv[1:]
                                 if os.path.exists(os.path.join(t, "lib", n))}},
                 sort_keys=True))
"""


class NativeError(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def oci_image(root):
    sys.path.insert(0, str(HERE))
    from oci_layers import image
    return image(Path(root))


def run(argv, **kw):
    return subprocess.run(argv, check=True, **kw)


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def dump(value):
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


# --------------------------------------------------------------------- rootfs
def _inside(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise NativeError(f"layer member escapes the root: {name!r}")
    parts = [p for p in path.parts if p not in ("", ".")]
    if not parts:
        # the layer's own root ("./", as tar -C <dir> . writes it)
        return root
    target = root.joinpath(*parts)
    parent = os.path.realpath(target.parent)
    if parent != str(root) and not parent.startswith(str(root) + os.sep):
        raise NativeError(f"layer member resolves outside the root: {name!r}")
    return target


def _remove(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _forget(modes, path, tree):
    """Drop the recorded modes of a removed path (and, for a directory, of all below it)."""
    if modes is None:
        return
    modes.pop(path, None)
    if tree:
        for key in [k for k in modes if path in k.parents]:
            del modes[key]


def apply_layer(root, stream, modes=None):
    """Apply one uncompressed layer tar stream (OCI whiteouts) to root, rootless.

    Whiteouts hide lower layers only, wherever they sit in the layer's tar. Files and
    directories stay writable by the owner while layers apply; `modes`, when given,
    collects the image's own mode of every file and directory (set-id and sticky bits
    included) for the caller to set once every layer is applied."""
    skipped, created = 0, set()
    with tarfile.open(fileobj=stream, mode="r|") as layer:
        for member in layer:
            target = _inside(root, member.name)
            name = target.name
            if name == ".wh..wh..opq":
                for child in list(target.parent.iterdir()) if target.parent.is_dir() else []:
                    if child not in created:
                        tree = child.is_dir() and not child.is_symlink()
                        _remove(child)
                        _forget(modes, child, tree)
                continue
            if name.startswith(".wh."):
                victim = target.with_name(name[4:])
                tree = victim.is_dir() and not victim.is_symlink()
                if victim.is_symlink() or victim.exists():
                    _remove(victim)
                _forget(modes, victim, tree)
                continue
            created.add(target)
            if member.isdev():
                skipped += 1
                continue
            if target.is_symlink() or (target.exists() and not (member.isdir() and target.is_dir())):
                tree = target.is_dir() and not target.is_symlink()
                _remove(target)
                _forget(modes, target, tree)
            target.parent.mkdir(parents=True, exist_ok=True)
            kept = member.isdir() or member.isreg() or member.islnk()
            if member.isdir():
                target.mkdir(exist_ok=True)
                target.chmod((member.mode & 0o777) | 0o700)
            elif member.issym():
                os.symlink(member.linkname, target)
            elif member.islnk():
                os.link(_inside(root, member.linkname), target)
            elif member.isreg():
                with open(target, "wb") as out:
                    shutil.copyfileobj(layer.extractfile(member), out, 1 << 22)
                target.chmod((member.mode & 0o777) | 0o600)
            else:
                skipped += 1
            if modes is not None and kept:
                modes[target] = member.mode & 0o7777
    return skipped


def extract_base(oci, rootfs):
    """Verify the pinned base OCI layout and extract its root filesystem. Either pull form is the
    base (oci_layers.verify_base, as the image build checks it): the pinned image index whose one
    linux/amd64 entry is the pinned manifest, or the pinned manifest itself."""
    sys.path.insert(0, str(HERE))
    from oci_layers import verify_base  # noqa: PLC0415
    try:
        verify_base(Path(oci), BASE_INDEX, BASE_MANIFEST)
    except (OSError, KeyError, ValueError) as exc:
        raise NativeError(f"{oci} is not the pinned base (linux/amd64 {BASE_MANIFEST}): {exc}") from None
    manifest, config = oci_image(oci)
    if config.get("architecture") != "amd64" or config.get("os") != "linux":
        raise NativeError("base image is not linux/amd64")
    rootfs.mkdir(parents=True)
    root = rootfs.resolve()
    skipped = 0
    for desc in manifest["layers"]:
        blob = oci / "blobs/sha256" / desc["digest"].split(":", 1)[1]
        if blob.stat().st_size != desc["size"]:
            raise NativeError(f"layer size mismatch: {desc['digest']}")
        h = hashlib.sha256()
        with blob.open("rb") as raw:
            class Hashing:
                def read(self, n=-1):
                    data = raw.read(n)
                    h.update(data)
                    return data
            with gzip.GzipFile(fileobj=Hashing(), mode="rb") as stream:
                skipped += apply_layer(root, stream)
            for block in iter(lambda: raw.read(1 << 22), b""):
                h.update(block)
        if "sha256:" + h.hexdigest() != desc["digest"]:
            raise NativeError(f"layer digest mismatch: {desc['digest']}")
    # Mount points the namespace binds over; created empty, never shipped.
    (root / "etc/resolv.conf").unlink(missing_ok=True)
    (root / "etc/resolv.conf").touch()
    for name in ("opt", "tmp"):
        (root / name).mkdir(exist_ok=True)
    return {"index": BASE_INDEX, "manifest": BASE_MANIFEST,
            "config": manifest["config"]["digest"], "device_nodes_skipped": skipped,
            "mount_points": ["/etc/resolv.conf", "/opt", "/tmp"]}


# ------------------------------------------------------------------ namespace
def environment(epoch, jobs, extra=None):
    env = {
        "PATH": "/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:"
                "/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/root", "USER": "root", "LOGNAME": "root",
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC", "TMPDIR": "/tmp",
        "SOURCE_DATE_EPOCH": str(epoch),
        "CUDA_HOME": "/usr/local/cuda", "CUDA_VISIBLE_DEVICES": "",
        "TORCH_CUDA_ARCH_LIST": "8.0",
        # Name nvcc's intermediates after the object, not the process id.
        "NVCC_APPEND_FLAGS": "-objtemp",
        "MAX_JOBS": str(jobs),
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
        "UV_CACHE_DIR": "/tmp/uv-cache", "UV_LINK_MODE": "copy", "UV_NO_CONFIG": "1",
        "UV_PYTHON_INSTALL_DIR": "/opt/python-dl",
    }
    env.update(extra or {})
    return env


def namespace_argv(rootfs, opt, build, env, cmd, *, network=False, uv=None,
                   uv_cache=None, source_readonly=True):
    argv = ["bwrap", "--unshare-all", "--unshare-user"]
    if network:
        argv.append("--share-net")
    argv += ["--die-with-parent", "--hostname", "native-build", "--uid", "0", "--gid", "0",
             "--ro-bind", str(rootfs), "/", "--dev", "/dev", "--proc", "/proc",
             "--tmpfs", "/tmp", "--bind", str(opt), "/opt", "--bind", str(build), BUILD]
    if source_readonly:
        argv += ["--ro-bind", str(Path(opt) / "vllm-src"), SRC]
    if network:
        argv += ["--ro-bind", "/etc/resolv.conf", "/etc/resolv.conf"]
    if uv:
        argv += ["--ro-bind", str(uv), "/tmp/uv/uv", "--bind", str(uv_cache), "/tmp/uv-cache"]
    argv += ["--chdir", "/", "--clearenv"]
    for key in sorted(env):
        argv += ["--setenv", key, env[key]]
    return argv + ["--", *cmd]


class Namespace:
    def __init__(self, rootfs, opt, build, env, log):
        self.rootfs, self.opt, self.build, self.env, self.log = rootfs, opt, build, env, log

    def __call__(self, script, **kw):
        argv = namespace_argv(self.rootfs, self.opt, self.build, self.env,
                              ["/bin/sh", "-euc", script], **kw)
        with self.log.open("a") as log:
            log.write(f"\n$ {script}\n")
            log.flush()
            result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise NativeError(f"namespace step failed ({result.returncode}); see {self.log}")

    def output(self, script):
        argv = namespace_argv(self.rootfs, self.opt, self.build, self.env,
                              ["/bin/sh", "-euc", script])
        return subprocess.run(argv, check=True, capture_output=True, text=True).stdout


# ---------------------------------------------------------------------- build
def ninja_rules(path):
    keep = ("cxx", "nvcc", "cflags", "post_cflags", "cuda_cflags", "cuda_post_cflags", "ldflags")
    rules = {}
    for line in path.read_text().splitlines():
        if " = " in line and not line.startswith(" "):
            key, value = line.split(" = ", 1)
            if key in keep:
                rules[key] = value
    return rules


def cmake_target_flags(path, target):
    flags, current = {}, None
    for line in path.read_text().splitlines():
        if line.startswith("build "):
            output = line[6:].split(":", 1)[0]
            current = output if f"CMakeFiles/{target}.dir/" in output or \
                output.endswith(f"{target}.abi3.so") else None
        elif current and line.startswith("  ") and " = " in line:
            key, value = line.strip().split(" = ", 1)
            if key in ("DEFINES", "FLAGS", "INCLUDES", "LINK_FLAGS", "LINK_LIBRARIES"):
                flags.setdefault(current, {})[key] = value
    return flags


def scan_artefact(path, forbidden):
    data = Path(path).read_bytes()
    hits = [needle for needle in forbidden if needle and needle.encode() in data]
    if b"tmpxft_" in data:
        hits.append("tmpxft_ (process-id names)")
    if hits:
        raise NativeError(f"{path} embeds build-host values: {hits}")


def exl3_mode(src):
    """How a source tree builds vllm._exl3_C: its standalone builder, the CMake target, or not."""
    if (src / EXL3_BUILDER).exists():
        return "standalone"
    if (src / EXL3_SOURCE).exists():
        return "cmake"
    return None


def exl3_script(mode, jobs, package="vllm"):
    if mode == "standalone":
        return (f"cd {SRC} && /opt/venv/bin/python {EXL3_BUILDER} "
                f"--out {BUILD}/out/{package} --build-dir {BUILD}/exl3 --verbose")
    return f"""
/opt/native-tools/bin/cmake -S {SRC} -B {BUILD}/exl3 -G Ninja \\
    -DCMAKE_MAKE_PROGRAM=/opt/venv/bin/ninja -DVLLM_PYTHON_EXECUTABLE=/opt/venv/bin/python \\
    -DVLLM_TARGET_DEVICE=cuda -DCMAKE_BUILD_TYPE=Release \\
    -DFETCHCONTENT_BASE_DIR={BUILD}/_deps -DFETCHCONTENT_FULLY_DISCONNECTED=ON \\
    -DCUTLASS_DIR={BUILD}/_deps/cutlass-src
/opt/native-tools/bin/cmake --build {BUILD}/exl3 --target _exl3_C -j {jobs}
cp {BUILD}/exl3/_exl3_C.abi3.so {BUILD}/out/vllm/_exl3_C.abi3.so
"""


def stable_script(jobs, targets):
    """The wheel's CMake TARGETS ([(artefact, target)]): configure once per build tree (a kept tree re-runs CMake
    itself when its CMake inputs change), build only those targets, copy them out."""
    return f"""
[ -f {BUILD}/stable/CMakeCache.txt ] || /opt/native-tools/bin/cmake -S {SRC} -B {BUILD}/stable -G Ninja \\
    -DCMAKE_MAKE_PROGRAM=/opt/venv/bin/ninja -DVLLM_PYTHON_EXECUTABLE=/opt/venv/bin/python \\
    -DVLLM_TARGET_DEVICE=cuda -DCMAKE_BUILD_TYPE=Release \\
    -DFETCHCONTENT_BASE_DIR={BUILD}/_deps -DFETCHCONTENT_FULLY_DISCONNECTED=ON
/opt/native-tools/bin/cmake --build {BUILD}/stable --target {' '.join(t for _, t in targets)} -j {jobs}
mkdir -p {BUILD}/out/vllm
""" + "".join(f"cp {BUILD}/stable/{target}.abi3.so {BUILD}/out/{rel}\n" for rel, target in targets)


# ------------------------------------------------------------ FetchContent sources
def _cmake_calls(text, command):
    """(offset, argument text) of each `command(...)` call of a comment-stripped CMake text."""
    calls = []
    for match in re.finditer(r"\b" + command + r"\s*\(", text):
        depth, i = 1, match.end()
        while i < len(text) and depth:
            depth += {"(": 1, ")": -1}.get(text[i], 0)
            i += 1
        calls.append((match.start(), text[match.end():i - 1]))
    return calls


def fetch_declarations(repo, commit):
    """{name: {repo, tag, submodules}} of the git FetchContent sources COMMIT's CUDA configure declares
    (FetchContent_Declare / FetchContent_Populate with GIT_REPOSITORY in FETCH_FILES). ${VAR} is the last
    set(VAR value) above the call in the same file (the else branch of an if/else set pair); submodules is None
    for all (CMake's default), else the GIT_SUBMODULES paths."""
    found = {}
    for path in git(repo, "ls-tree", "-r", "--name-only", commit, "--", *FETCH_FILES).splitlines():
        if path != "CMakeLists.txt" and not path.endswith(".cmake"):
            continue
        text = re.sub(r"#[^\n]*", "", git(repo, "show", f"{commit}:{path}"))
        sets = [(m.start(), m.group(1), m.group(2).strip('"'))
                for m in re.finditer(r"\bset\s*\(\s*([A-Za-z0-9_]+)\s+(\"[^\"]*\"|[^\s)]+)", text)]

        def value(word, at):
            def var(m):
                known = [v for pos, name, v in sets if name == m.group(1) and pos < at]
                if not known:
                    raise NativeError(f"{path}: ${{{m.group(1)}}} has no set() above its FetchContent call")
                return known[-1]
            return re.sub(r"\$\{([A-Za-z0-9_]+)\}", var, word)

        for command in ("FetchContent_Declare", "FetchContent_Populate"):
            for at, args in _cmake_calls(text, command):
                words = [w.strip('"') for w in re.findall(r'"[^"]*"|[^\s"]+', args)]
                if "GIT_REPOSITORY" not in words or "GIT_TAG" not in words:
                    continue
                submodules = None
                if "GIT_SUBMODULES" in words:
                    submodules = []
                    for w in words[words.index("GIT_SUBMODULES") + 1:]:
                        if re.fullmatch(r"[A-Z][A-Z0-9_]*", w):
                            break
                        submodules.append(value(w, at))
                dep = {"repo": value(words[words.index("GIT_REPOSITORY") + 1], at),
                       "tag": value(words[words.index("GIT_TAG") + 1], at), "submodules": submodules}
                name = value(words[0], at).lower()
                if found.setdefault(name, dep) != dep:
                    raise NativeError(f"FetchContent source {name} is declared twice, differently: "
                                      f"{found[name]} and {dep}")
    return found


def fetch_sources(repo, commit, directory):
    """Clone (host git) each FetchContent source COMMIT declares into DIRECTORY/<name>-src, unless the clone there
    is of the same declaration (its FETCH_MARKER); {name: HEAD}."""
    directory.mkdir(parents=True, exist_ok=True)
    heads = {}
    for name, dep in sorted(fetch_declarations(repo, commit).items()):
        dest = directory / f"{name}-src"
        marker = dest / FETCH_MARKER
        if marker.is_file():
            recorded = json.loads(marker.read_text())
            if {k: recorded.get(k) for k in dep} == dep:
                heads[name] = recorded["head"]
                continue
        tmp = directory / f".{name}-src.tmp"
        if tmp.exists() or tmp.is_symlink():
            _remove(tmp)
        run(["git", "init", "-q", str(tmp)])
        run(["git", "-C", str(tmp), "remote", "add", "origin", dep["repo"]])
        run(["git", "-C", str(tmp), "fetch", "-q", "--depth", "1", "origin", dep["tag"]])
        run(["git", "-C", str(tmp), "checkout", "-q", "FETCH_HEAD"])
        if dep["submodules"] is None or dep["submodules"]:
            run(["git", "-C", str(tmp), "submodule", "update", "-q", "--init", "--recursive", "--depth", "1",
                 "--", *(dep["submodules"] or [])])
        head = git(tmp, "rev-parse", "HEAD")
        (tmp / FETCH_MARKER).write_text(dump({**dep, "head": head}))
        if dest.exists():
            _remove(dest)
        tmp.rename(dest)
        heads[name] = head
    return heads


def sync_sources(fetchcontent, deps):
    """Copy each <name>-src of FETCHCONTENT into DEPS (CMake's FETCHCONTENT_BASE_DIR) without Git data, unless the
    copy there is of the same fetch (FETCH_MARKER); {name: HEAD, None when unknown}."""
    deps.mkdir(parents=True, exist_ok=True)
    heads = {}
    for src in sorted(fetchcontent.glob("*-src")):
        marker, dst = src / FETCH_MARKER, deps / src.name
        if marker.is_file():
            ident = marker.read_text()
            heads[src.name[:-4]] = json.loads(ident)["head"]
            if (dst / FETCH_MARKER).is_file() and (dst / FETCH_MARKER).read_text() == ident:
                continue
        else:
            head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True)
            heads[src.name[:-4]] = head.stdout.strip() if head.returncode == 0 else None
        if dst.exists() or dst.is_symlink():
            _remove(dst)
        shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns(".git"))
    return heads


# ------------------------------------------------------------------ --cache
@contextlib.contextmanager
def locked(directory):
    """Hold DIRECTORY/lock (one build of a cache at a time); no-op without a cache."""
    if directory is None:
        yield
        return
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def has_commit(repo, commit):
    return subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
                          capture_output=True).returncode == 0


def update_source(repo, src, old, new):
    """Move the extracted source SRC from commit OLD to NEW: remove the paths NEW deletes, write the ones it adds or
    changes with the current time as mtime (newer than every object built before); the number of paths."""
    out = subprocess.run(["git", "-C", str(repo), "diff", "--name-status", "--no-renames", "-z", old, new],
                         check=True, capture_output=True).stdout.decode()
    fields = out.split("\0")[:-1]
    write, gone = [], []
    for status, path in zip(fields[::2], fields[1::2]):
        (gone if status == "D" else write).append(path)
    for path in gone:
        target = src / path
        if target.exists() or target.is_symlink():
            _remove(target)
    env = dict(os.environ, GIT_LITERAL_PATHSPECS="1")
    for i in range(0, len(write), 500):
        archive = subprocess.Popen(["git", "-C", str(repo), "archive", "--format=tar", new, "--",
                                    *write[i:i + 500]], stdout=subprocess.PIPE, env=env)
        run(["tar", "-x", "-m", "--no-same-owner", "-C", str(src)], stdin=archive.stdout)
        if archive.wait():
            raise NativeError("git archive failed")
    return len(write) + len(gone)


def build(args):
    repo, out = args.repo.resolve(), args.out.resolve()
    cache = args.cache.resolve() if args.cache else None
    work = cache or args.work.resolve()
    if cache is None and work.exists() and any(work.iterdir()):
        raise NativeError(f"work directory must be new or empty: {work}")
    if out.exists() and any(out.iterdir()):
        raise NativeError(f"output directory must be new or empty: {out}")
    with locked(cache):
        _build(args, repo, work, out, cache)


def _build(args, repo, work, out, cache):
    t0 = time.monotonic()
    timings = {}
    builder = sha256(__file__)
    commit = git(repo, "rev-parse", "--verify", args.commit + "^{commit}")
    epoch = int(git(repo, "show", "-s", "--format=%ct", commit))
    inputs = {name: git(repo, "rev-parse", f"{commit}:{name}")
              for name in (PLUGIN_INPUTS if args.plugin else SOURCE_INPUTS)}
    work.mkdir(parents=True, exist_ok=True)
    opt, bld, log = work / "opt", work / "build", work / "build.log"
    src = opt / "vllm-src"
    log.write_text("")
    # --cache: what the kept rootfs, Python, venv and native-tools were made for; any difference starts afresh.
    key = {"layout": CACHE_LAYOUT, "base_manifest": BASE_MANIFEST, "python": PYTHON_INSTALL_VERSION,
           "cmake": CMAKE_VERSION, "constraints_sha256": sha256(args.constraints)}
    state_file = work / "state.json"
    state = json.loads(state_file.read_text()) if cache and state_file.is_file() else {}
    fresh = not (state.get("key") == key and state.get("commit") and has_commit(repo, state["commit"])
                 and src.is_dir() and (opt / "venv").is_dir())
    if cache:
        state_file.unlink(missing_ok=True)  # until the source and venv are whole again
    rootfs = args.rootfs.resolve() if args.rootfs else work / "rootfs"
    marker = rootfs.parent / (rootfs.name + ".base.json")

    step = time.monotonic()
    if args.rootfs:
        base = json.loads(marker.read_text())
        if base.get("manifest") != BASE_MANIFEST:
            raise NativeError(f"{rootfs} was not extracted from the pinned base")
    elif cache and marker.is_file() and json.loads(marker.read_text()).get("manifest") == BASE_MANIFEST \
            and rootfs.is_dir():
        base = json.loads(marker.read_text())
    else:
        marker.unlink(missing_ok=True)
        if rootfs.exists():
            _remove(rootfs)
        base = extract_base(args.base_oci.resolve(), rootfs)
        marker.write_text(dump(base))
    timings["base_rootfs_s"] = round(time.monotonic() - step, 3)

    step = time.monotonic()
    changed_paths = None
    if fresh:
        for d in (opt, bld):
            if d.exists():
                _remove(d)
        src.mkdir(parents=True)
        bld.mkdir()
        archive = subprocess.Popen(["git", "-C", str(repo), "archive", "--format=tar", commit],
                                   stdout=subprocess.PIPE)
        run(["tar", "-x", "--no-same-owner", "-C", str(src)], stdin=archive.stdout)
        if archive.wait():
            raise NativeError("git archive failed")
    else:
        changed_paths = update_source(repo, src, state["commit"], commit)
        for name in ("out", "ampere_marlin", "exl3"):
            if (bld / name).exists():
                _remove(bld / name)
    exl3 = exl3_mode(src)
    # (marlin artefact, its builder, exl3 artefact, the output package) of this build
    marlin_rel, marlin_builder, exl3_rel, package = (
        (PLUGIN_MARLIN, PLUGIN_MARLIN_BUILDER, PLUGIN_EXL3, PLUGIN_PACKAGE) if args.plugin
        else (MARLIN, MARLIN_BUILDER, EXL3, "vllm"))
    if args.plugin and exl3 != "standalone":
        raise NativeError(f"{commit} is not a plugin commit: it has no {EXL3_BUILDER}")
    if not args.no_marlin and not (src / marlin_builder).exists():
        raise NativeError(f"{commit} has no {marlin_builder}")
    wheel = [WHEEL_TARGETS[dest] for dest in WHEEL_TARGETS if getattr(args, dest)]
    if args.no_marlin and not exl3 and not wheel:
        raise NativeError(f"--no-marlin leaves nothing to build: {commit} has no EXL3 extension")
    shutil.copyfile(args.constraints, bld / "CONSTRAINTS")
    deps = {}
    if (exl3 == "cmake" or wheel) and not args.fetchcontent:
        raise NativeError(("--stable/--moe-stable build the CMake targets " + ", ".join(t for _, t in wheel)
                           if wheel
                           else "this commit builds vllm._exl3_C") +
                          ": pass --fetchcontent DIR (CMake FetchContent sources, <name>-src)")
    timings["source_s"] = round(time.monotonic() - step, 3)
    if wheel:
        step = time.monotonic()
        fetch_sources(repo, commit, args.fetchcontent.resolve())
        timings["fetchcontent_s"] = round(time.monotonic() - step, 3)
    if exl3 == "cmake" or wheel:
        deps = sync_sources(args.fetchcontent.resolve(), bld / "_deps")

    env = environment(epoch, args.jobs)
    ns = Namespace(rootfs, opt, bld, env, log)
    uv = Path(shutil.which(args.uv) or args.uv).resolve()
    uv_cache = Path(subprocess.run([str(uv), "cache", "dir"], check=True, capture_output=True,
                                   text=True).stdout.strip())
    uv_cache.mkdir(parents=True, exist_ok=True)
    if fresh:
        step = time.monotonic()
        # A cache keeps CMake for a later --stable/--moe-stable build of the same venv.
        ns(f"""
/tmp/uv/uv python install --no-bin {PYTHON_INSTALL_VERSION}
mv /opt/python-dl/cpython-3.12.*-linux-x86_64-gnu /opt/python
rm -rf /opt/python-dl
grep -rlI /opt/python-dl/ /opt/python | xargs -r sed -i -E "s#/opt/python-dl/cpython-[^/\\"' ]+#/opt/python#g"
/tmp/uv/uv venv --relocatable --python /opt/python/bin/python3.12 /opt/venv
VIRTUAL_ENV=/opt/venv UV_CONSTRAINT={BUILD}/CONSTRAINTS /tmp/uv/uv pip install \\
    --extra-index-url {EXTRA_INDEX} -r {BUILD}/CONSTRAINTS
""" + (f"""/tmp/uv/uv venv --python /opt/python/bin/python3.12 /opt/native-tools
VIRTUAL_ENV=/opt/native-tools /tmp/uv/uv pip install cmake=={CMAKE_VERSION}
""" if exl3 == "cmake" or wheel or cache else ""), network=True, uv=uv, uv_cache=uv_cache,
           source_readonly=False)
        timings["python_venv_s"] = round(time.monotonic() - step, 3)
    if cache:
        state = {"key": key, "commit": commit, "stable_commit": None if fresh else state.get("stable_commit")}
        state_file.write_text(dump(state))

    extensions = {}
    if not args.no_marlin:
        step = time.monotonic()
        ns(f"cd {SRC} && {MARLIN_BUILD_ENV} /opt/venv/bin/python {marlin_builder} "
           f"--out {BUILD}/out/{package} --build-dir {BUILD}/ampere_marlin --verbose")
        timings["ampere_marlin_s"] = round(time.monotonic() - step, 3)
        extensions[marlin_rel] = {"builder": marlin_builder,
                                  "flags": ninja_rules(bld / "ampere_marlin/build.ninja")}
    if exl3:
        step = time.monotonic()
        # The source is read-only for the standalone builder; CMake's configure writes
        # generated upstream kernels into csrc.
        ns(exl3_script(exl3, args.jobs, package), source_readonly=exl3 == "standalone")
        timings["exl3_s"] = round(time.monotonic() - step, 3)
        if exl3 == "standalone":
            extensions[exl3_rel] = {"builder": EXL3_BUILDER, "command": exl3_script(exl3, args.jobs, package),
                                    "flags": ninja_rules(bld / "exl3/build.ninja")}
        else:
            extensions[EXL3] = {"builder": "CMakeLists.txt target _exl3_C",
                                "cmake": CMAKE_VERSION, "fetchcontent": deps,
                                "flags": cmake_target_flags(bld / "exl3/build.ninja", "_exl3_C")}
    if wheel:
        kept = (bld / "stable/CMakeCache.txt").is_file()
        step = time.monotonic()
        # Writable source: the configure generates the Marlin kernels into csrc.
        ns(stable_script(args.jobs, wheel), source_readonly=False)
        timings["stable_s"] = round(time.monotonic() - step, 3)
        for rel, target in wheel:
            extensions[rel] = {"builder": f"CMakeLists.txt target {target}", "cmake": CMAKE_VERSION,
                               "fetchcontent": deps,
                               "incremental_from": state.get("stable_commit") if kept else None,
                               "flags": cmake_target_flags(bld / "stable/build.ninja", target)}
        if cache:
            state["stable_commit"] = commit
            state_file.write_text(dump(state))

    tools = json.loads(ns.output(
        "python3=/opt/venv/bin/python; $python3 - <<'EOF'\n"
        "import hashlib, json, os, subprocess\n"
        "def v(*a): return subprocess.run(a, capture_output=True, text=True).stdout.strip()\n"
        "def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()\n"
        f"tools = {list(TOOLS)!r}\n"
        "cc1 = v('c++', '-print-prog-name=cc1plus')\n"
        "paths = tools + [os.path.realpath('/usr/bin/c++'), cc1, os.path.realpath('/usr/bin/as'),"
        " os.path.realpath('/usr/bin/ld')]\n"
        "print(json.dumps({'nvcc': v('nvcc', '--version').splitlines()[-2:],"
        " 'cxx': v('c++', '--version').splitlines()[0], 'ld': v('ld', '--version').splitlines()[0],"
        " 'ninja': v('/opt/venv/bin/ninja', '--version'),"
        " 'sha256': {os.path.realpath(p): sha(p) for p in paths}}))\n"
        "EOF\n"))
    tools["uv"] = subprocess.run([str(uv), "--version"], check=True, capture_output=True,
                                 text=True).stdout.split()[1]
    python = json.loads(ns.output("/opt/venv/bin/python -c " + _shell(PROBE) + " " +
                                  " ".join(TORCH_LIBS)))

    forbidden = sorted({str(work), str(repo), str(Path.home())})
    artefacts = {}
    out.mkdir(parents=True, exist_ok=True)
    for rel in extensions:
        built = bld / "out" / rel
        scan_artefact(built, forbidden)
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(built, out / rel)
        (out / rel).chmod(0o755)
        artefacts[rel] = {"sha256": sha256(out / rel), "bytes": (out / rel).stat().st_size}
    manifest = {
        "schema": 1,
        "source": {"commit": commit, "commit_time": epoch, "inputs": inputs},
        "toolchain": {"base_image": {k: base[k] for k in ("index", "manifest", "config")},
                      **tools},
        "python": python,
        "layout": {"source": SRC, "build": BUILD, "python": "/opt/python", "venv": "/opt/venv"},
        "environment": {k: v for k, v in env.items()
                        if not k.startswith("UV_") and k != "MAX_JOBS"},
        "constraints_sha256": sha256(args.constraints),
        "extensions": extensions,
        "artefacts": artefacts,
        "builder_sha256": builder,
    }
    if args.plugin:
        manifest["package"] = PLUGIN_PACKAGE
    (out / "manifest.json").write_text(dump(manifest))
    timings["total_s"] = round(time.monotonic() - t0, 3)
    record = {"manifest_sha256": sha256(out / "manifest.json"), "artefacts": artefacts,
              "timings": timings, "work": str(work), "log": str(log)}
    if cache:
        record["cache"] = {"fresh": fresh, "changed_paths": changed_paths}
    (out / "build-record.json").write_text(dump(record))
    print(dump(record), end="")
    if not args.keep_work and cache is None:
        shutil.rmtree(work, ignore_errors=True)


def _shell(text):
    return "'" + text.replace("'", "'\"'\"'") + "'"


# ------------------------------------------------------------ install / check
def load(native):
    native = Path(native)
    manifest = json.loads((native / "manifest.json").read_text())
    for rel, entry in manifest["artefacts"].items():
        if sha256(native / rel) != entry["sha256"]:
            raise NativeError(f"{native / rel} differs from the manifest")
    return manifest


def check_source(tree, manifest):
    tree = Path(tree)
    for name, expected in manifest["source"]["inputs"].items():
        actual = git(tree, "rev-parse", f"HEAD:{name}")
        if actual != expected:
            raise NativeError(f"{tree}: {name} at HEAD is {actual}, the build used {expected}")
    dirty = git(tree, "status", "--porcelain", "--untracked-files=no", "--", *manifest["source"]["inputs"])
    if dirty:
        raise NativeError(f"{tree}: uncommitted changes in build inputs:\n{dirty}")


def check_venv(venv, manifest):
    probe = subprocess.run([str(Path(venv) / "bin/python"), "-c", PROBE, *TORCH_LIBS],
                           check=True, capture_output=True, text=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    actual = json.loads(probe.stdout)
    expected = manifest["python"]
    keys = ("torch", "torch_cuda", "cxx11_abi", "torch_include_sha256", "torch_libs")
    diff = [k for k in keys if actual.get(k) != expected.get(k)]
    if diff:
        raise NativeError(f"{venv}: torch differs from the build's ({', '.join(diff)})")
    return actual


def variants(path):
    stem = path.name.split(".", 1)[0]
    return [path.with_name(stem + suffix) for suffix in EXTENSION_SUFFIXES
            if path.with_name(stem + suffix) != path]


def check_tree(tree, manifest):
    """The tree holds exactly the manifest's files and no other build of them, and
    no build of one of our extensions that the manifest leaves out."""
    tree = Path(tree)
    for rel, entry in manifest["artefacts"].items():
        path = tree / rel
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise NativeError(f"{path} is not the manifest's file")
        shadow = [str(p) for p in variants(path) if p.exists() or p.is_symlink()]
        if shadow:
            raise NativeError(f"other builds of {rel} present: {shadow}")
    ours = (PLUGIN_MARLIN, PLUGIN_EXL3) if manifest.get("package") == PLUGIN_PACKAGE else (MARLIN, EXL3)
    for rel in ours:
        if rel in manifest["artefacts"]:
            continue
        path = tree / rel
        stray = [str(p) for p in [path, *variants(path)] if p.exists() or p.is_symlink()]
        if stray:
            raise NativeError(f"builds of {rel} present that the manifest does not list: {stray}")


def install_files(native, tree, manifest):
    tree = Path(tree)
    for rel in manifest["artefacts"]:
        target = tree / rel
        for shadow in variants(target):
            shadow.unlink(missing_ok=True)
        fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=target.name, suffix=".tmp")
        os.close(fd)
        try:
            shutil.copyfile(Path(native) / rel, temporary)
            os.chmod(temporary, 0o755)
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    check_tree(tree, manifest)


def install(args):
    manifest = load(args.native)
    check_source(args.tree, manifest)
    venv = check_venv(args.venv, manifest) if args.venv else None
    install_files(args.native, args.tree, manifest)
    record = {"manifest_sha256": sha256(Path(args.native) / "manifest.json"),
              "source_commit": manifest["source"]["commit"],
              "tree_head": git(args.tree, "rev-parse", "HEAD"),
              "artefacts": manifest["artefacts"], "venv_torch": venv and venv["torch"]}
    print(dump(record), end="")


def pack(args):
    manifest = load(args.native)
    actual = sha256(Path(args.native) / "manifest.json")
    if actual != args.manifest_sha256:
        raise NativeError(f"native manifest {actual} is not the pinned {args.manifest_sha256}")
    same_source(opt_tree := Path(args.opt) / "vllm-src", manifest, args.commit)
    if manifest["toolchain"]["base_image"]["manifest"] != args.base_manifest:
        raise NativeError(f"native files were not built in the base image {args.base_manifest}")
    opt = Path(args.opt)
    check_source(opt_tree, manifest)
    if args.venv_check:
        check_venv(opt / "venv", manifest)
    install_files(args.native, opt / "vllm-src", manifest)
    shutil.copyfile(Path(args.native) / "manifest.json", opt / SHIPPED_MANIFEST)
    print(f"[native] packed {len(manifest['artefacts'])} file(s), manifest {actual}")


def same_source(tree, manifest, commit):
    """The manifest belongs to `commit`: built from it, or from a commit whose
    build inputs (csrc, cmake, CMakeLists.txt) are the same trees. A Git tree
    must be a checkout of `commit` with the manifest's inputs at HEAD. A
    finalized image tree has no Git data: its provenance.json must name
    `commit`, and the inputs were compared when pack installed the files."""
    tree = Path(tree)
    built = manifest["source"]["commit"]
    if not (tree / ".git").exists() and (tree / "provenance.json").is_file():
        recorded = json.loads((tree / "provenance.json").read_text()).get("engine_commit")
        if recorded != commit:
            raise NativeError(f"{tree} records engine {recorded}, not {commit}")
        if built != commit:
            print(f"[native] built from {built}; build inputs were compared with {commit} at pack")
        return
    head = git(tree, "rev-parse", "HEAD")
    if head != commit:
        raise NativeError(f"{tree} is at {head}, not the engine {commit}")
    if built == commit:
        return
    check_source(tree, manifest)
    print(f"[native] built from {built}, whose {', '.join(SOURCE_INPUTS)} equal the engine {commit}'s")


def layer_files(oci, wanted):
    """Final content sha256 of the wanted paths across the image's layers."""
    manifest, _ = oci_image(oci)
    state = {}
    for desc in manifest["layers"]:
        blob = Path(oci) / "blobs/sha256" / desc["digest"].split(":", 1)[1]
        opener = gzip.open if desc["mediaType"].endswith("gzip") else open
        layer_written = set()
        with opener(blob, "rb") as stream, tarfile.open(fileobj=stream, mode="r|") as layer:
            for member in layer:
                name = PurePosixPath(member.name.lstrip("/")).as_posix()
                base = PurePosixPath(name)
                if base.name == ".wh..wh..opq":
                    prefix = base.parent.as_posix() + "/"
                    state = {k: v for k, v in state.items()
                             if not k.startswith(prefix) or k in layer_written}
                    continue
                if base.name.startswith(".wh."):
                    gone = (base.parent / base.name[4:]).as_posix()
                    state = {k: v for k, v in state.items()
                             if k != gone and not k.startswith(gone + "/")}
                    continue
                if name not in wanted:
                    continue
                layer_written.add(name)
                if member.isreg():
                    h = hashlib.sha256()
                    data = layer.extractfile(member)
                    for block in iter(lambda: data.read(1 << 22), b""):
                        h.update(block)
                    state[name] = h.hexdigest()
                else:
                    state[name] = f"<{'symlink' if member.issym() else 'non-file'}>"
    return state


def check_oci(args):
    manifest = json.loads(Path(args.manifest).read_text())
    prefix = args.prefix.strip("/")
    expected = {f"{prefix}/{rel}": entry["sha256"] for rel, entry in manifest["artefacts"].items()}
    shadows = {f"{prefix}/{p.relative_to(Path('/x')).as_posix()}"
               for rel in manifest["artefacts"] for p in variants(Path("/x") / rel)}
    shipped = f"opt/{SHIPPED_MANIFEST}"
    found = layer_files(args.oci, set(expected) | shadows | {shipped})
    problems = [f"{p}: {found.get(p, '<absent>')} != {h}" for p, h in expected.items()
                if found.get(p) != h]
    problems += [f"{p}: other build present" for p in shadows if p in found]
    if args.require_shipped_manifest and found.get(shipped) != sha256(args.manifest):
        problems.append(f"{shipped}: {found.get(shipped, '<absent>')} is not this manifest")
    result = {"oci": str(args.oci), "manifest_sha256": sha256(args.manifest),
              "checked": sorted(expected), "pass": not problems, "problems": problems}
    print(dump(result), end="")
    if problems:
        raise NativeError("image does not hold the manifest's native files")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="compile our extensions in the pinned base image")
    b.add_argument("--repo", type=Path, required=True, help="fork git repository")
    b.add_argument("--commit", required=True)
    b.add_argument("--base-oci", type=Path, help="OCI layout of the pinned base (crane pull)")
    b.add_argument("--rootfs", type=Path, help="reuse a root filesystem extract_base wrote")
    b.add_argument("--constraints", type=Path, required=True,
                   help="the engine's dependency lock (the image builder's --constraints)")
    b.add_argument("--fetchcontent", type=Path,
                   help="CMake FetchContent <name>-src dirs (EXL3's CMake target; --stable/--moe-stable clone what "
                        "is missing)")
    w = b.add_mutually_exclusive_group(required=True)
    w.add_argument("--work", type=Path, help="new scratch directory")
    w.add_argument("--cache", type=Path, help="persistent work directory, built incrementally (module doc)")
    b.add_argument("--stable", action="store_true",
                   help="also rebuild the wheel's vllm._C_stable_libtorch (its CMake target; needs --fetchcontent)")
    b.add_argument("--moe-stable", action="store_true",
                   help="also rebuild the wheel's vllm._moe_C_stable_libtorch (its CMake target; needs --fetchcontent)")
    b.add_argument("--out", type=Path, required=True, help="new output directory")
    b.add_argument("--jobs", type=int, default=8)
    b.add_argument("--no-marlin", action="store_true",
                   help="leave out vllm._ampere_marlin_C (engines that ship without it)")
    b.add_argument("--uv", default="uv")
    b.add_argument("--keep-work", action="store_true")
    b.add_argument("--plugin", action="store_true",
                   help="the natives of a plugin commit, vllm_sm80/ (module doc); with --work only")
    i = sub.add_parser("install", help="install a manifest's files into a fork tree")
    i.add_argument("--native", type=Path, required=True, help="build output directory")
    i.add_argument("--tree", type=Path, required=True, help="fork worktree to validate")
    i.add_argument("--venv", type=Path, help="venv that will run it (torch must match)")
    c = sub.add_parser("check-tree", help="a fork tree holds the manifest's files")
    c.add_argument("--manifest", type=Path, required=True)
    c.add_argument("--tree", type=Path, required=True)
    c.add_argument("--manifest-sha256", help="also require this manifest")
    c.add_argument("--commit", help="also require this source commit")
    k = sub.add_parser("pack", help="image builders: pinned manifest into the stage")
    k.add_argument("--native", type=Path, required=True)
    k.add_argument("--manifest-sha256", required=True)
    k.add_argument("--commit", required=True)
    k.add_argument("--opt", type=Path, required=True, help="the image's /opt (stage)")
    k.add_argument("--venv-check", action="store_true", help="also compare the stage torch")
    k.add_argument("--base-manifest", default=BASE_MANIFEST,
                   help="the base image manifest the image is built on (the natives must be built in it)")
    o = sub.add_parser("check-oci", help="an OCI image holds the manifest's files")
    o.add_argument("--oci", type=Path, required=True)
    o.add_argument("--manifest", type=Path, required=True)
    o.add_argument("--prefix", default="/opt/vllm-src")
    o.add_argument("--require-shipped-manifest", action="store_true")
    a = p.parse_args(argv)
    try:
        if a.command == "build":
            if not (a.base_oci or a.rootfs):
                p.error("build needs --base-oci or --rootfs")
            if a.plugin and (a.cache or a.stable or a.moe_stable or a.no_marlin or a.fetchcontent):
                p.error("build --plugin takes neither --cache, --stable, --moe-stable, --no-marlin nor --fetchcontent")
            build(a)
        elif a.command == "install":
            install(a)
        elif a.command == "check-tree":
            manifest = json.loads(a.manifest.read_text())
            if a.manifest_sha256 and sha256(a.manifest) != a.manifest_sha256:
                raise NativeError(f"{a.manifest} is not the pinned {a.manifest_sha256}")
            if a.commit:
                same_source(Path(a.tree), manifest, a.commit)
            check_tree(a.tree, manifest)
            print(f"[native] {a.tree} holds the files of manifest {sha256(a.manifest)}")
        elif a.command == "pack":
            pack(a)
        else:
            check_oci(a)
    except (NativeError, subprocess.CalledProcessError) as error:
        print(f"[native] FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
