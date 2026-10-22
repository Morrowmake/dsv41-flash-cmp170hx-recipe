import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
import p2p_check as p2p


class Probe:
    def __init__(self, advertised=True, count=4):
        self.info = dict(driver=13030, kernel_driver="driver-build", uuids=[f"GPU-{i}" for i in range(count)],
                         advertised=advertised)
        self.calls = 0
        self.checked = dict(passed=True, pairs=count*(count-1), sizes=9, max_bytes=33554432)
        self.error = None

    def identity(self, deadline):
        return self.info

    def check(self, deadline):
        self.calls += 1
        if self.error:
            raise self.error
        return self.checked


def decide(tmp_path, probe, **kwargs):
    values = dict(mode="auto", tp=4, pp=1, cache=tmp_path, backend=probe, boot="boot-a")
    values.update(kwargs)
    return p2p.decide(**values)


def test_success_and_boot_cache(tmp_path):
    probe = Probe()
    first = decide(tmp_path, probe)
    second = decide(tmp_path, probe)
    assert first["enabled"] and not first["cached"]
    assert second["enabled"] and second["cached"] and probe.calls == 1


@pytest.mark.parametrize("field", ["driver", "kernel_driver", "uuids", "boot"])
def test_cache_identity_invalidation(tmp_path, field):
    probe = Probe()
    decide(tmp_path, probe)
    kwargs = {}
    if field == "boot":
        kwargs["boot"] = "boot-b"
    elif field == "driver":
        probe.info[field] += 1
    elif field == "uuids":
        probe.info[field] = list(reversed(probe.info[field]))
    else:
        probe.info[field] += "-new"
    assert not decide(tmp_path, probe, **kwargs)["cached"] and probe.calls == 2


@pytest.mark.parametrize("tp,pp", [(4, 1), (1, 4)])
@pytest.mark.parametrize("error", [RuntimeError("owner mismatch"), TimeoutError("check timed out")])
def test_corruption_and_hang_fail_closed_cached(tmp_path, tp, pp, error):
    probe = Probe()
    probe.error = error
    first = decide(tmp_path, probe, tp=tp, pp=pp)
    assert not first["enabled"] and "advertised but data check failed" in first["reason"]
    assert decide(tmp_path, probe, tp=tp, pp=pp)["cached"] and probe.calls == 1


@pytest.mark.parametrize("advertised,count", [(False, 4), (False, 1)])
def test_unadvertised_does_not_copy(tmp_path, advertised, count):
    probe = Probe(advertised, count)
    actual = decide(tmp_path, probe)
    assert not actual["enabled"] and "not advertised" in actual["reason"] and probe.calls == 0


@pytest.mark.parametrize("mode,tp,pp,enabled", [("off", 4, 1, False), ("force", 4, 1, True),
                                               ("auto", 1, 1, False), ("force", 1, 1, False)])
def test_overrides_and_single_gpu_skip(tmp_path, mode, tp, pp, enabled):
    class Never:
        def identity(self, deadline):
            pytest.fail("CUDA must not be initialized")
    assert decide(tmp_path, Never(), mode=mode, tp=tp, pp=pp)["enabled"] is enabled
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("change", [dict(passed=False), dict(pairs=6), dict(sizes=8), dict(max_bytes=1024)])
def test_partial_check_rejected(tmp_path, change):
    probe = Probe()
    probe.checked.update(change)
    actual = decide(tmp_path, probe)
    assert not actual["enabled"] and "incomplete" in actual["reason"]


def test_corrupt_cache_rechecks(tmp_path):
    probe = Probe()
    decide(tmp_path, probe)
    next(tmp_path.glob("*.json")).write_text('{bad json')
    assert decide(tmp_path, probe)["enabled"] and probe.calls == 2


def test_unavailable_probe_and_unwritable_cache(tmp_path):
    probe = Probe()
    probe.identity = lambda deadline: (_ for _ in ()).throw(OSError("compiler unavailable"))
    assert "check unavailable" in decide(tmp_path, probe)["reason"]
    blocked = tmp_path / "file"
    blocked.write_text("file")
    assert not decide(blocked, Probe())["enabled"]


def test_subprocess_deadline(tmp_path):
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out"):
        p2p.bounded([sys.executable, "-c", "import time; time.sleep(30)"], started + 0.1)
    assert time.monotonic() - started < 2


def test_lock_deadline(tmp_path):
    import fcntl
    with (tmp_path / "lock").open("a") as first, (tmp_path / "lock").open("a") as second:
        fcntl.flock(first, fcntl.LOCK_EX)
        with pytest.raises(TimeoutError):
            p2p.acquire(second, time.monotonic() + 0.05)


@pytest.fixture(scope="module")
def cpu_probe(tmp_path_factory):
    root = tmp_path_factory.mktemp("cpu-p2p")
    text = (ROOT / "p2p_probe.cu").read_text()
    start = text.index("__global__ void peer_copy")
    end = text.index("static void exact", start)
    text = text[:start] + text[end:]
    text = text.replace("peer_copy<<<256, 256>>>", "mock_peer_copy")
    (root / "cuda_runtime.h").write_text((Path(__file__).with_name("cuda_mock.h")).read_text())
    source = root / "probe.cpp"
    source.write_text(text)
    binary = root / "probe"
    subprocess.run(["g++", "-O2", "-std=c++17", "-I", str(root), str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True, timeout=60)
    return binary


def run_cpu(binary, *args, **env):
    return subprocess.run([str(binary), *args], text=True, capture_output=True, timeout=120,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES="", **env))


def test_cpu_mock_all_pairs_content_and_ipc(cpu_probe):
    run = run_cpu(cpu_probe, "--check")
    assert run.returncode == 0, run.stderr
    value = json.loads(run.stdout)
    assert p2p.verified(value, 4)
    assert len(value["pairs_s"]) == 12 and len(value["sizes_s"]) == 9
    assert set(value["timings_s"]) == {"setup", "randomize", "memcpy", "write", "read", "ipc", "cleanup"}
    assert all(t >= 0 for t in value["timings_s"].values())


@pytest.mark.parametrize("method,message", [("memcpy", "cudaMemcpyPeer"), ("write", "SM peer write"),
                                            ("read", "SM peer read"), ("ipc", "IPC write")])
def test_cpu_mock_corruption_each_transport(cpu_probe, method, message):
    run = run_cpu(cpu_probe, "--check", MOCK_CORRUPT=method)
    assert run.returncode == 1 and message + " mismatch" in run.stderr


MOCK_MEMORY = Path("/tmp")   # where the CPU mock once named its 32 MiB allocations (p2p-cpu-*); RAM on many hosts


@pytest.mark.parametrize("corrupt", ["", "memcpy", "write", "read", "ipc"])
def test_cpu_mock_check_leaves_no_memory_files(cpu_probe, tmp_path, corrupt):
    """A passed check and a check failed by each transport leave no file behind, in TMPDIR or in /tmp."""
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    before = set(MOCK_MEMORY.glob("p2p-cpu-*"))
    run = run_cpu(cpu_probe, "--check", TMPDIR=str(scratch), **({"MOCK_CORRUPT": corrupt} if corrupt else {}))
    assert run.returncode == (1 if corrupt else 0), run.stderr
    assert list(scratch.iterdir()) == []
    assert sorted(set(MOCK_MEMORY.glob("p2p-cpu-*")) - before) == []


@pytest.mark.parametrize("env,count,advertised", [({}, 4, True), ({"MOCK_NO_PEER": "1"}, 4, False),
                                                ({"MOCK_COUNT": "1"}, 1, False)])
def test_cpu_mock_advertisement(cpu_probe, env, count, advertised):
    run = run_cpu(cpu_probe, "--identity", **env)
    assert run.returncode == 0, run.stderr
    info = json.loads(run.stdout)
    assert len(info["uuids"]) == count and info["advertised"] is advertised


def test_cache_pass_requires_complete_verification(tmp_path):
    probe = Probe()
    decide(tmp_path, probe)
    path = next(tmp_path.glob("*.json"))
    saved = json.loads(path.read_text())
    saved.pop("verification")
    path.write_text(json.dumps(saved))
    assert not decide(tmp_path, probe)["cached"] and probe.calls == 2


def test_invalid_mode_rejected_before_cuda():
    run = subprocess.run([sys.executable, str(ROOT / "p2p_check.py"), "--tp", "4", "--pp", "1", "--mode", "invalid"],
                         text=True, capture_output=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert run.returncode == 2 and "invalid choice" in run.stderr


def probe_source():
    text = (ROOT / "p2p_probe.cu").read_text()
    return re.sub(r"//[^\n]*", "", text)


def test_probe_every_upload_finishes_before_further_cuda_work():
    text = probe_source()
    uploads = list(re.finditer(r"ck\(cudaMemcpy\([^;]+cudaMemcpyHostToDevice\)\);", text))
    assert len(uploads) == 4
    for upload in uploads:
        assert re.match(r"\s*ck\(cudaDeviceSynchronize\(\)\);", text[upload.end():])


@pytest.mark.parametrize("out,source", [("remote", "local"), ("readback", "remote")])
def test_probe_peer_kernels_finish_before_read_or_process_exit(out, source):
    text = probe_source()
    launches = list(re.finditer(r"peer_copy<<<256, 256>>>\(" + out + ", " + source + r", n\);", text))
    assert len(launches) == (2 if out == "remote" else 1)
    for launch in launches:
        assert re.match(r"\s*ck\(cudaGetLastError\(\)\);\s*ck\(cudaDeviceSynchronize\(\)\);",
                        text[launch.end():])


def test_probe_comparison_waits_on_owner_before_download():
    text = probe_source().split("static void compare(", 1)[1].split("static void randomize", 1)[0]
    assert re.search(r"ck\(cudaSetDevice\(owner\)\);\s*ck\(cudaDeviceSynchronize\(\)\);", text)
    assert text.index("cudaDeviceSynchronize") < text.index("cudaMemcpyDeviceToHost")


def test_probe_peer_copy_finishes_before_comparison():
    assert re.search(r"ck\(cudaMemcpyPeer\(remote, owner, local, actor, n\)\);"
                     r"\s*ck\(cudaDeviceSynchronize\(\)\);\s*compare\(owner, remote", probe_source())


def test_probe_remote_clears_finish_before_peer_writers():
    text = probe_source()
    clears = list(re.finditer(r"ck\(cudaMemset\((?:remote|target), 0, (?:n|data.size\(\))\)\);", text))
    assert len(clears) == 3
    for clear in clears:
        assert re.match(r"\s*ck\(cudaDeviceSynchronize\(\)\);", text[clear.end():])


def test_probe_ipc_writer_exits_successfully_before_comparison():
    text = probe_source().split("static void ipc_write(", 1)[1].split("static void identity", 1)[0]
    assert text.index("waitpid(pid") < text.index("!WIFEXITED(status)") < text.index("compare(owner")


@pytest.mark.parametrize('enabled,matched,origin', [('1', True, 'image'), ('0', True, 'compiled'),
                                                   ('1', False, 'compiled')])
def test_prebuilt_source_hash_and_kill_switch(tmp_path, monkeypatch, enabled, matched, origin):
    import hashlib
    source = tmp_path/'probe.cu'
    source.write_text('fixture')
    image = tmp_path/'image'
    image.mkdir()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (image/('probe-'+(digest if matched else 'other'))).write_bytes(b'fixture')
    cache = tmp_path/'cache'
    cache.mkdir()
    monkeypatch.setattr(p2p, 'SOURCE', source)
    monkeypatch.setattr(p2p, 'PREBUILT', image)
    monkeypatch.setenv('P2P_PREBUILT', enabled)
    calls = []
    def bounded(argv, deadline):
        calls.append(argv)
        if argv[-1] == '--identity':
            return json.dumps(dict(driver=1, uuids=['GPU-0', 'GPU-1'], advertised=True))
        Path(argv[argv.index('-o')+1]).write_bytes(b'compiled')
        return ''
    monkeypatch.setattr(p2p, 'bounded', bounded)
    read = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda path, *a, **k: 'driver' if str(path) ==
                        '/proc/driver/nvidia/version' else read(path, *a, **k))
    probe = p2p.CudaProbe(cache)
    probe.identity(time.monotonic()+30)
    assert probe.binary_source == origin
    assert len(calls) == (1 if origin == 'image' else 2)
    assert set(probe.timings) == {'build_s', 'identity_s'}
