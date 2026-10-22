import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import cache_seed as cache
import oci_layers as oci
import payload

PIN = "a" * 40

def test_partition():
    cases = {"python/bin/python": "python", "venv/lib/site-packages/torch/x.py": "dependencies",
             "venv/lib/site-packages/vllm-1.dist-info/METADATA": "engine",
             "venv/lib/site-packages/vllm_sm80_plugin-0.1.0.dist-info/entry_points.txt": "engine",
             "venv/lib/site-packages/__editable__.vllm-1.pth": "engine",
             "venv/lib/site-packages/__editable___vllm_1_finder.py": "engine",
             "venv/.native-manifest.json": "engine", "venv/bin/vllm": "engine",
             "vllm-src/vllm/x.py": "engine", "vllm-src/vllm/x.abi3.so": "native"}
    for path, part in cases.items():
        assert payload.partition(Path(path)) == part

@pytest.mark.parametrize("version", ["1.2.3", "1.2.4"])
def test_lock(tmp_path, version):
    opt = tmp_path / "opt"
    path = opt / "venv/lib/python3.12/site-packages/python_dotenv-1.dist-info"
    path.mkdir(parents=True)
    (path / "METADATA").write_text("Name: python-dotenv\nVersion: " + version)
    constraints = tmp_path / "CONSTRAINTS"
    constraints.write_text("python-dotenv==1.2.3\n")
    if version == "1.2.3":
        payload.check_lock(opt, constraints)
    else:
        with pytest.raises(ValueError, match="lock mismatch"):
            payload.check_lock(opt, constraints)

def test_lock_ignores_engine_distributions(tmp_path):
    opt = tmp_path / "opt"
    for name, dist in (("vllm", "vllm-1"), ("vllm-sm80-plugin", "vllm_sm80_plugin-0.1.0")):
        path = opt / "venv/lib/python3.12/site-packages" / (dist + ".dist-info")
        path.mkdir(parents=True)
        (path / "METADATA").write_text("Name: " + name + "\nVersion: 1")
    constraints = tmp_path / "CONSTRAINTS"
    constraints.write_text("")
    payload.check_lock(opt, constraints)

def identity(tmp_path):
    lock = tmp_path / "lock"
    lock.write_text("python-dotenv==1.2.3")
    value = dict(schema=1, engine_commit=PIN, gpu_arch="sm_80", gpu_name="NVIDIA CMP 170HX",
                 driver=1, dependencies=cache.sha(lock), versions={"triton": "3.7.1"})
    path = tmp_path / "identity"
    path.write_text(json.dumps(value))
    return value, path, lock

def bundle(tmp_path):
    value, identity_path, lock = identity(tmp_path)
    source = tmp_path / "compiled"
    (source / ".triton/cache/key").mkdir(parents=True)
    (source / ".triton/cache/key/kernel.cubin").write_bytes(b"compiled")
    result = tmp_path / "bundle"
    cache.pack(source, identity_path, result)
    return value, lock, result

def test_bundle_integrity(tmp_path):
    value, lock, result = bundle(tmp_path)
    assert cache.check(result, PIN, lock)["identity"] == value
    with pytest.raises(ValueError, match="identity mismatch"):
        cache.check(result, "b" * 40, lock)
    with (result / "cache.tar.gz").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="archive hash mismatch"):
        cache.check(result, PIN, lock)

@pytest.mark.parametrize("name", ["/outside", "../outside", ".triton/cache/../../outside", "huggingface/token"])
def test_reject_path(name):
    assert not cache.allowed(name)

def test_pack_reject_symlink(tmp_path):
    value, identity_path, lock = identity(tmp_path)
    source = tmp_path / "compiled/.triton/cache"
    source.mkdir(parents=True)
    (source / "escape").symlink_to(lock)
    with pytest.raises(ValueError, match="symlink"):
        cache.pack(tmp_path / "compiled", identity_path, tmp_path / "bundle")

def test_seed_mismatch_and_existing(tmp_path, monkeypatch):
    value, lock, result = bundle(tmp_path)
    opt = tmp_path / "opt/venv"
    opt.mkdir(parents=True)
    (opt / "requirements.lock.txt").write_bytes(lock.read_bytes())
    monkeypatch.setattr(cache, "runtime_identity", lambda opt: value)
    dest = tmp_path / "volume"
    exports = cache.seed(result, opt.parent, dest)
    root = dest / "compiled" / cache.key(value)
    file = root / ".triton/cache/key/kernel.cubin"
    assert file.read_bytes() == b"compiled" and "/cache/compiled/" in exports
    file.write_bytes(b"newer")
    cache.seed(result, opt.parent, dest)
    assert file.read_bytes() == b"newer"
    value["gpu_arch"] = "sm_90"
    cache.seed(result, opt.parent, dest)
    assert not (dest / "compiled" / cache.key(value) / ".seed-complete").exists()

def test_keys(tmp_path):
    value, path, lock = identity(tmp_path)
    k = cache.key(value)
    for name, replacement in (("engine_commit", "b"*40), ("gpu_arch", "sm_90"),
                              ("driver", 2), ("dependencies", "different")):
        assert cache.key(dict(value, **{name: replacement})) != k

def base(tmp_path):
    b = tmp_path / "base"
    (b / "blobs/sha256").mkdir(parents=True)
    c = dict(architecture="amd64", os="linux", config={}, rootfs=dict(type="layers", diff_ids=[]))
    desc = oci.put(b, oci.encoded(c), oci.OCI+"config.v1+json")
    m = dict(schemaVersion=2, config=desc, layers=[], mediaType=oci.OCI+"manifest.v1+json")
    md = oci.put(b, oci.encoded(m), m["mediaType"])
    (b / "index.json").write_text(json.dumps(dict(manifests=[md])))
    return b


def publication_image(tmp_path, labels=None):
    root = base(tmp_path)
    manifest, config = oci.image(root)
    config["config"]["Labels"] = labels if labels is not None else {
        "org.opencontainers.image.revision": PIN,
        "org.opencontainers.image.version": "engine-" + PIN[:10]}
    manifest["config"] = oci.put(root, oci.encoded(config), oci.OCI + "config.v1+json")
    desc = oci.put(root, oci.encoded(manifest), manifest["mediaType"])
    (root / "index.json").write_text(json.dumps(dict(manifests=[desc])))
    return root

def test_layer_reuse_and_reproducibility(tmp_path):
    b = base(tmp_path)
    p = tmp_path / "payload"
    for name in payload.PARTS:
        d = p / name / "opt" / name
        d.mkdir(parents=True)
        (d / "file").write_text(name)
    reports = []
    for n in range(3):
        out = tmp_path / str(n) / "oci"
        if n == 2:
            (p / "engine/opt/engine/file").write_text("changed Python")
        oci.assemble(b, p, out, PIN if n < 2 else "b"*40, "engine-" + (PIN if n < 2 else "b"*40)[:10], threads=2)
        reports.append(json.loads((out.parent / "layers.json").read_text()))
    assert reports[0] == reports[1]
    assert reports[0][:3] == reports[2][:3]
    assert reports[0][3]["digest"] != reports[2][3]["digest"]
    for out in (tmp_path / "0/oci", tmp_path / "2/oci"):
        m, c = oci.image(out)
        assert len(m["layers"]) == len(c["rootfs"]["diff_ids"]) == 4
        for desc, diff in zip(m["layers"], c["rootfs"]["diff_ids"]):
            import gzip
            data = gzip.decompress((out / "blobs/sha256" / desc["digest"].split(":")[1]).read_bytes())
            assert "sha256:" + hashlib.sha256(data).hexdigest() == diff

def test_kill_switches(tmp_path):
    b = base(tmp_path)
    p = tmp_path / "payload"
    for name in payload.PARTS:
        d = p / name
        d.mkdir(parents=True)
        (d / name).write_text(name)
    oci.assemble(b, p, tmp_path / "out/oci", PIN, "engine-" + PIN[:10], layered=False, parallel=False)
    m, c = oci.image(tmp_path / "out/oci")
    assert len(m["layers"]) == 1


def test_push_guards_and_private_auth(tmp_path):
    import os
    import subprocess
    b = publication_image(tmp_path)
    digest = json.loads((b/"index.json").read_text())["manifests"][0]["digest"]
    record = tmp_path/"acceptance.json"
    record.write_text(json.dumps(dict(image_digest=digest, engine_commit=PIN, passed=True, artifact_audit_passed=True)))
    log = tmp_path/"calls"
    fake = tmp_path/"crane"
    fake.write_text("#!/bin/bash\nset -eu\nprintf '%s\\n' \"$*\" >> \"$CALL_LOG\"\ncase \"$1\" in\n auth) cat >/dev/null; echo \"$DOCKER_CONFIG\" > \"$CONFIG_LOG\" ;;\n push) ;;\n digest) echo \"$EXPECTED\" ;;\nesac\n")
    fake.chmod(0o755)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CRANE=str(fake), CALL_LOG=str(log),
               CONFIG_LOG=str(tmp_path/"config-path"), EXPECTED=digest)
    argv = ["bash", str(ROOT/"push.sh"), str(b), "ghcr.io/morrowmake/vllm-cmp170hx:engine-aaaaaaaaaa", digest, str(record)]
    assert subprocess.run(argv, env=env, input="", text=True, capture_output=True).returncode == 2
    assert not log.exists()
    env["IMAGE_PUBLISH"] = "1"
    secret = "fixture-private-auth-value"
    run = subprocess.run(argv, env=env, input=secret, text=True, capture_output=True)
    assert run.returncode == 0, run.stderr
    assert secret not in run.stdout + run.stderr + log.read_text()
    assert not Path((tmp_path/"config-path").read_text().strip()).exists()
    record.write_text(json.dumps(dict(image_digest="sha256:"+"b"*64, passed=True, artifact_audit_passed=True)))
    log.unlink()
    assert subprocess.run(argv, env=env, input=secret, text=True, capture_output=True).returncode != 0
    assert not log.exists()

@pytest.mark.parametrize("enabled", ["0", "1"])
def test_entry_dry_no_cuda(tmp_path, enabled):
    import os
    import subprocess
    launch = tmp_path / "serve.sh"
    launch.write_text("#!/bin/bash\necho launch\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", DRY="1", VLLM_IMAGE_CACHE_SEED=enabled, P2P="off",
               VENV=sys.prefix, RECIPE_DIR=str(ROOT.parent), SERVE_SH=str(launch))
    result = subprocess.run(["bash", str(ROOT/"entry.sh")], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert ("[image-cache] disabled" if enabled == "0" else "[image-cache] enabled (dry run)") in result.stdout
    assert result.stdout.rstrip().endswith("launch")


def test_audit_token_shapes_and_nested_archive():
    import gzip
    import io
    from audit_oci import TOKEN_PATTERNS
    from runtime_tree import scan_stream
    scan_stream(io.BytesIO(b"ghp_example"), [], "fixture", credential_patterns=TOKEN_PATTERNS)
    for data in (b"ghp_" + b"x"*36, gzip.compress(b"github_pat_" + b"x"*50)):
        with pytest.raises(ValueError, match="Credential candidate"):
            scan_stream(io.BytesIO(data), [], "fixture", credential_patterns=TOKEN_PATTERNS)


def test_audit_collects_integrity_failure(tmp_path):
    from audit_oci import checked_layer
    root = tmp_path/"oci"
    (root/"blobs/sha256").mkdir(parents=True)
    name = "a"*64
    (root/"blobs/sha256"/name).write_bytes(b"wrong bytes")
    desc = dict(digest="sha256:"+name, size=11)
    result = checked_layer((root, desc, "sha256:"+name, [], []))
    assert result["passed"] is False and "descriptor mismatch" in result["reason"]


@pytest.mark.parametrize("name", [".cache/torch_extensions/kernel/module.so", ".nv/ComputeCache/kernel"])
def test_cpp_and_driver_caches(name):
    assert cache.allowed(name)


def test_audit_ignores_embedded_symbol_and_hash_fragments():
    import io
    from audit_oci import TOKEN_PATTERNS
    from runtime_tree import scan_stream
    scan_stream(io.BytesIO(b"symbol_ghr_" + b"x"*50), [], "fixture", credential_patterns=TOKEN_PATTERNS)


def test_review_bound_to_exact_bytes_and_offset(monkeypatch):
    import io
    from runtime_tree import scan_stream
    value = b"prefix example suffix"
    digest = hashlib.sha256(value).hexdigest()
    monkeypatch.setenv("IMAGE_REVIEWED_BYTE_MATCHES", json.dumps({digest: [7]}))
    scan_stream(io.BytesIO(value), [b"example"], "fixture")
    with pytest.raises(ValueError, match="Forbidden identifier"):
        scan_stream(io.BytesIO(value + b"changed"), [b"example"], "fixture")
    monkeypatch.setenv("IMAGE_REVIEWED_BYTE_MATCHES", json.dumps({digest: [8]}))
    with pytest.raises(ValueError, match="Forbidden identifier"):
        scan_stream(io.BytesIO(value), [b"example"], "fixture")


def test_review_cannot_allow_credential_candidates(monkeypatch):
    import io
    from audit_oci import TOKEN_PATTERNS
    from runtime_tree import scan_stream
    value = b"ghp_" + b"x"*36
    digest = hashlib.sha256(value).hexdigest()
    monkeypatch.setenv("IMAGE_REVIEWED_BYTE_MATCHES", json.dumps({digest: [0]}))
    with pytest.raises(ValueError, match="Credential candidate"):
        scan_stream(io.BytesIO(value), [b"ghp_"], "fixture", credential_patterns=TOKEN_PATTERNS)


def test_rootfs_driver_discovery_mounts(tmp_path):
    from rootfs_smoke import DRIVER_DIR, driver_options
    library = tmp_path / "libcuda.so.999"
    library.write_bytes(b"fixture")
    (tmp_path / "libcuda.so.1").symlink_to(library.name)
    options = driver_options([tmp_path])
    mount = options.index(DRIVER_DIR + "/libcuda.so.1")
    assert options[mount - 2:mount] == ["--ro-bind", str(library)]
    assert options[:4] == ["--tmpfs", "/usr/local/nvidia", "--dir", DRIVER_DIR]


def test_rootfs_driver_missing_fails(tmp_path):
    from rootfs_smoke import driver_options
    with pytest.raises(ValueError, match="Host libcuda.so.1"):
        driver_options([tmp_path])


def test_built_rootfs_inspection(tmp_path):
    """Opt-in real-image regression: CUDA remains hidden and /dev is isolated."""
    import os
    import subprocess
    required = ("TEST_IMAGE_ROOTFS", "TEST_IMAGE_CONFIG", "TEST_TARGET_MODEL")
    if not all(os.environ.get(name) for name in required):
        pytest.skip("Set built-rootfs and checkpoint paths for image inspection")
    result = tmp_path / "inspection"
    command = [sys.executable, str(ROOT / "rootfs_smoke.py")]
    for flag, name in zip(("rootfs", "config", "model"), required):
        command += ["--" + flag, os.environ[name]]
    command += ["--result", str(result)]
    run = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", IMAGE_ROOTFS_CPU_SMOKE="1"),
                         capture_output=True, text=True, timeout=500)
    assert run.returncode == 0, run.stdout + run.stderr
    record = json.loads((result / "summary.json").read_text())
    assert record["passed"] and "target_registry" in record["checks"]
    assert "release_launch" in record["checks"] and "serve_arguments" in record["checks"]
    assert "engine_config_boundary" in record["checks"]


@pytest.mark.parametrize("library_path", ["", "/usr/local/cuda/lib64"])
def test_rootfs_rejects_cleared_driver_search_path(tmp_path, library_path):
    from rootfs_smoke import image_environment
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"config": {"Env": ["LD_LIBRARY_PATH=" + library_path]}}))
    with pytest.raises(ValueError, match="OCI LD_LIBRARY_PATH"):
        image_environment(config)


def test_seed_exports_all_compiler_caches(tmp_path):
    import shlex
    value, _, _ = identity(tmp_path)
    exports = dict(line.removeprefix('export ').split('=', 1) for line in cache.exports(value).splitlines())
    root = '/cache/compiled/' + cache.key(value)
    expected = {'HOME', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR',
                'VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR', 'VLLM_CACHE_ROOT',
                'TILELANG_CACHE_DIR', 'TILELANG_TMP_DIR', 'CUDA_CACHE_PATH',
                'FLASHINFER_WORKSPACE_BASE', 'TORCHINDUCTOR_CACHE_DIR'}
    assert set(exports) == expected
    for name, value in exports.items():
        assert Path(shlex.split(value)[0]).is_relative_to(root), name


@pytest.mark.parametrize('layered', [False, True])
@pytest.mark.parametrize('mode,uid,gid,passed', [(0o755, 0, 0, False), (0o1777, 0, 0, True),
                                               (0o777, 0, 0, False), (0o1777, 1000, 0, False),
                                               (0o1777, 0, 1000, False)])
def test_audit_tmp_tar_entry(tmp_path, layered, mode, uid, gid, passed):
    import gzip
    import io
    from audit_oci import audit
    b = base(tmp_path)
    p = tmp_path / 'payload'
    for name in payload.PARTS:
        (p / name).mkdir(parents=True)
    (p / 'engine/tmp').mkdir()
    (p / 'engine/tmp').chmod(0o1777)
    out = tmp_path / 'out/oci'
    oci.assemble(b, p, out, PIN, "engine-" + PIN[:10], layered=layered, parallel=False)
    manifest, config = oci.image(out)
    # Mutate real tar metadata and recompute OCI integrity to isolate the mode gate.
    desc = manifest['layers'][-1]
    path = out / 'blobs/sha256' / desc['digest'].split(':')[1]
    data = io.BytesIO()
    with tarfile.open(path) as source, tarfile.open(fileobj=data, mode='w') as target:
        for member in source:
            if member.name.removeprefix('./').rstrip('/') == 'tmp':
                assert member.mode == 0o1777 and member.uid == member.gid == 0
                member.mode, member.uid, member.gid = mode, uid, gid
            target.addfile(member)
    raw = data.getvalue()
    manifest['layers'][-1] = oci.put(out, gzip.compress(raw), oci.OCI + 'layer.v1.tar+gzip')
    config['rootfs']['diff_ids'][-1] = 'sha256:' + hashlib.sha256(raw).hexdigest()
    manifest['config'] = oci.put(out, oci.encoded(config), oci.OCI + 'config.v1+json')
    md = oci.put(out, oci.encoded(manifest), manifest['mediaType'])
    (out / 'index.json').write_text(json.dumps(dict(manifests=[md])))
    record = audit(out, workers=2)
    assert record['passed'] == passed and record['tmp']['passed'] == passed


@pytest.mark.parametrize('mode', [0o755, 0o777, 0o1777])
def test_smoke_checks_image_tmp_before_mount(tmp_path, mode):
    from rootfs_smoke import check_tmp
    (tmp_path / 'tmp').mkdir(mode=mode)
    (tmp_path / 'tmp').chmod(mode)
    if mode == 0o1777:
        assert check_tmp(tmp_path) == '1777'
    else:
        with pytest.raises(ValueError, match='mode 1777'):
            check_tmp(tmp_path)


@pytest.mark.parametrize('notice', ['complete', 'missing', 'truncated', 'changed'])
def test_smoke_checks_packaged_recipe_license(tmp_path, monkeypatch, notice):
    import rootfs_smoke
    (tmp_path / 'tmp').mkdir()
    (tmp_path / 'tmp').chmod(0o1777)
    packaged = tmp_path / 'opt/image-tools/LICENSE'
    packaged.parent.mkdir(parents=True)
    (packaged.parent / 'THIRD_PARTY_NOTICES').write_bytes((ROOT / 'THIRD_PARTY_NOTICES').read_bytes())
    source = (ROOT.parent / 'LICENSE').read_bytes()
    if notice != 'missing':
        packaged.write_bytes(source if notice == 'complete' else
                             source[:-1] if notice == 'truncated' else b'X' + source[1:])
    monkeypatch.setenv('IMAGE_ROOTFS_CPU_SMOKE', '0')
    command = ['rootfs_smoke.py']
    for flag in ('rootfs', 'config', 'model', 'draft', 'result'):
        command += ['--' + flag, str(tmp_path)]
    monkeypatch.setattr(sys, 'argv', command)
    if notice == 'complete':
        rootfs_smoke.main()
    else:
        with pytest.raises(ValueError, match='/opt/image-tools/LICENSE'):
            rootfs_smoke.main()


@pytest.mark.parametrize('seeded', [False, True])
def test_container_setup_creates_private_tmp_and_writable_caches(tmp_path, seeded):
    import os
    import shutil
    import subprocess
    if not shutil.which('bwrap'):
        pytest.skip('Requires rootless namespace support')
    environment = dict(line.split('=', 1) for line in (ROOT / 'container.env').read_text().splitlines()
                       if line and not line.startswith('#'))
    # docker/entry.sh up to the peer check: the private temporary directory and the caches.
    prelude = (ROOT / 'entry.sh').read_text().split('\n# 2. Peer access', 1)[0]
    # Seed integrity/identity is tested separately; apply its real exports on CPU.
    environment['VLLM_IMAGE_CACHE_SEED'] = '0'
    if seeded:
        value, _, _ = identity(tmp_path)
        exports = cache.exports(value)
        root = '/cache/compiled/' + cache.key(value)
        prelude = prelude.replace('# Explicit compiler destinations', exports + '\n# Explicit compiler destinations')
    else:
        root = '/cache'
    script = tmp_path / 'setup.sh'
    script.write_text(prelude + '\nenv -0\n')
    command = ['bwrap', '--clearenv', '--unshare-all', '--tmpfs', '/',
               '--dev', '/dev', '--tmpfs', '/tmp', '--bind', str(tmp_path), '/cache']
    for directory in ('/usr', '/bin', '/lib', '/lib64', '/etc'):
        if Path(directory).exists():
            command += ['--ro-bind', directory, directory]
    for key, value in dict(environment, PATH=os.defpath, CUDA_VISIBLE_DEVICES='',
                           SERVE_SH='/cache/setup.sh').items():
        command += ['--setenv', key, value]
    command += ['bash', '/cache/setup.sh']
    run = subprocess.run(command, capture_output=True)
    assert run.returncode == 0, run.stderr.decode()
    # The prelude emits one status line before its environment.
    data = run.stdout.split(b'\n', 1)[1]
    resolved = dict(row.decode().split('=', 1) for row in data.split(b'\0') if row)
    assert resolved['TMPDIR'].startswith('/cache/tmp/run.')
    assert resolved['TMP'] == resolved['TEMP'] == resolved['TMPDIR']
    temporary = tmp_path / Path(resolved['TMPDIR']).relative_to('/cache')
    assert temporary.is_dir() and temporary.stat().st_mode & 0o777 == 0o700
    assert resolved['HOME'] == root
    assert resolved['XDG_CACHE_HOME'] == root + '/.cache'
    assert resolved['HF_HOME'] == '/cache/huggingface'
    for name in ('HOME', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR',
                 'VLLM_CACHE_ROOT', 'TILELANG_CACHE_DIR',
                 'TILELANG_TMP_DIR', 'CUDA_CACHE_PATH', 'FLASHINFER_WORKSPACE_BASE',
                 'TORCHINDUCTOR_CACHE_DIR'):
        assert Path(resolved[name]).is_relative_to(root), name
        directory = tmp_path / Path(resolved[name]).relative_to('/cache')
        assert directory.is_dir() and os.access(directory, os.W_OK), name


def test_container_environment_matches_start(tmp_path):
    import os
    import subprocess
    rows = [line.split('=', 1) for line in (ROOT / 'container.env').read_text().splitlines()
            if line and not line.startswith('#')]
    expected = dict(rows)
    assert len(rows) == len(expected), 'Duplicate container environment key'
    source = (ROOT.parent / 'start.sh').read_text().rsplit('main "$@"', 1)[0]
    script = tmp_path / 'start.sh'
    (tmp_path / '.env.example').write_text('')
    script.write_text(source + '\ncontainer_env\n')
    run = subprocess.run(['bash', str(script)], capture_output=True, text=True,
                         env=dict(PATH=os.defpath, CUDA_VISIBLE_DEVICES='', HOME=str(tmp_path)))
    assert run.returncode == 0, run.stderr
    actual = dict(line.split('=', 1) for line in run.stdout.splitlines() if '=' in line)
    for key in ('HOME', 'XDG_CACHE_HOME', 'HF_HOME', 'TMPDIR', 'RECIPE_CACHE_ROOT',
                'PYTHONDONTWRITEBYTECODE', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR',
                'VLLM_CACHE_ROOT', 'LAYOUT', 'HOST', 'PORT',
                'TILELANG_CACHE_DIR', 'TILELANG_TMP_DIR', 'CUDA_CACHE_PATH',
                'FLASHINFER_WORKSPACE_BASE', 'TORCHINDUCTOR_CACHE_DIR'):
        assert actual[key] == expected[key], key
