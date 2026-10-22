#!/usr/bin/env python3
"""Bounded, boot-scoped content verification for CUDA peer access."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

SOURCE = Path(__file__).with_name("p2p_probe.cu")
PREBUILT = Path("/opt/image-tools/p2p")
TIMEOUT = 120.0
SIZES = (128*1024-1, 128*1024, 128*1024+1, 512*1024-1,
         512*1024, 512*1024+1, 1024*1024, 8*1024*1024, 32*1024*1024)


def remaining(deadline):
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("check budget exhausted")
    return value


def bounded(argv, deadline):
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=remaining(deadline))
    except (subprocess.TimeoutExpired, TimeoutError):
        os.killpg(proc.pid, signal.SIGKILL)
        try:
            proc.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        raise TimeoutError("check timed out") from None
    if proc.returncode:
        detail = " ".join(err.split())[:240]
        raise RuntimeError(detail or f"probe exited {proc.returncode}")
    return out


def acquire(lock, deadline):
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            time.sleep(min(0.05, remaining(deadline)))


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as out:
        temp = Path(out.name)
        json.dump(value, out, sort_keys=True)
        out.write("\n")
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class CudaProbe:
    def __init__(self, cache):
        self.cache = cache
        self.binary = None
        self.timings = {}
        self.binary_source = "cache"

    def identity(self, deadline):
        digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        binary = self.cache / ("probe-" + digest)
        prebuilt = PREBUILT / ("probe-" + digest)
        started = time.monotonic()
        if os.environ.get("P2P_PREBUILT", "1") == "1" and prebuilt.is_file():
            binary = prebuilt
            self.binary_source = "image"
        with (self.cache / ("build-" + digest + ".lock")).open("a") as lock:
            acquire(lock, deadline)
            if not binary.exists():
                self.binary_source = "compiled"
                nvcc = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc"
                with tempfile.TemporaryDirectory(dir=self.cache) as temp:
                    output = Path(temp) / "probe"
                    bounded([str(nvcc), "-O2", "-std=c++17", "-arch=sm_80",
                             "-o", str(output), str(SOURCE)], deadline)
                    os.replace(output, binary)
        self.timings["build_s"] = round(time.monotonic() - started, 6)
        self.binary = binary
        started = time.monotonic()
        value = json.loads(bounded([str(binary), "--identity"], deadline))
        value["kernel_driver"] = Path("/proc/driver/nvidia/version").read_text().strip()
        self.timings["identity_s"] = round(time.monotonic() - started, 6)
        return value

    def check(self, deadline):
        started = time.monotonic()
        try:
            return json.loads(bounded([str(self.binary), "--check"], deadline))
        finally:
            self.timings["check_s"] = round(time.monotonic() - started, 6)


def cache_key(identity, boot):
    fields = dict(driver=identity["driver"], kernel_driver=identity["kernel_driver"],
                  uuids=identity["uuids"], boot=boot,
                  probe=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), schema=1)
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def verified(result, count):
    return (isinstance(result, dict) and result.get("passed") is True and result.get("pairs") == count*(count-1)
            and result.get("sizes") == len(SIZES) and result.get("max_bytes") == max(SIZES))


def decide(mode, tp, pp, cache, backend=None, boot=None, timeout=TIMEOUT):
    started = time.monotonic()
    def result(enabled, reason, cached=False):
        return dict(enabled=enabled, reason=reason, cached=cached,
                    duration_s=round(time.monotonic()-started, 6),
                    timings_s=dict(getattr(backend, "timings", {})),
                    binary_source=getattr(backend, "binary_source", "external"))
    if mode not in ("off", "auto", "force"):
        raise ValueError("P2P must be off, auto or force")
    if tp < 1 or pp < 1:
        raise ValueError("TP and PP must be positive")
    if mode == "off":
        return result(False, "P2P=off")
    if tp == 1 and pp == 1:
        return result(False, "single GPU layout; no peer check needed")
    if mode == "force":
        return result(True, "P2P=force; content check bypassed")
    deadline = started + timeout
    advertised = False
    try:
        cache.mkdir(parents=True, exist_ok=True)
        backend = backend or CudaProbe(cache)
        identity = backend.identity(deadline)
        if (not isinstance(identity["uuids"], list) or not identity["uuids"]
                or len(set(identity["uuids"])) != len(identity["uuids"])
                or type(identity["advertised"]) is not bool):
            raise ValueError("invalid GPU identity")
        advertised = identity["advertised"]
        boot = boot if boot is not None else Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if not boot:
            raise ValueError("boot identity unavailable")
        key = cache_key(identity, boot)
        path = cache / (key + ".json")
        with (cache / (key + ".lock")).open("a") as lock:
            acquire(lock, deadline)
            try:
                saved = json.loads(path.read_text())
                if (saved.get("key") == key and type(saved.get("enabled")) is bool
                        and isinstance(saved.get("reason"), str)
                        and (not saved["enabled"] or (advertised and saved["reason"] == "content check passed"
                             and verified(saved.get("verification", {}), len(identity["uuids"]))))):
                    return result(saved["enabled"], saved["reason"], True)
            except (OSError, ValueError, AttributeError):
                pass
            checked = {}
            if not advertised:
                outcome = result(False, "not advertised for every ordered GPU pair")
            else:
                try:
                    checked = backend.check(deadline)
                    remaining(deadline)
                    if not verified(checked, len(identity["uuids"])):
                        raise ValueError("incomplete content verification")
                    outcome = result(True, "content check passed")
                    outcome["verification"] = checked
                except (OSError, RuntimeError, ValueError, TimeoutError) as error:
                    outcome = result(False, "advertised but data check failed: " + str(error))
            atomic_json(path, dict(outcome, key=key, verification=checked))
            return outcome
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, TimeoutError) as error:
        reason = "advertised but data check failed" if advertised else "check unavailable"
        return result(False, reason + ": " + str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("off", "auto", "force"), default="auto")
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--cache", type=Path, default=Path(os.environ.get(
        "P2P_CHECK_CACHE", str(Path.home() / ".cache/recipe-p2p"))))
    args = parser.parse_args()
    try:
        value = decide(args.mode, args.tp, args.pp, args.cache)
    except ValueError as error:
        parser.error(str(error))
    state = "enabled" if value["enabled"] else "disabled"
    cache = "; cached this boot" if value["cached"] else ""
    timings = "; ".join(f"{key}={seconds:.3f}s" for key, seconds in value["timings_s"].items())
    detail = f"; probe={value['binary_source']}; {timings}" if timings else ""
    verification = value.get("verification", {})
    if verification:
        detail += f"; pairs={verification['pairs']} sizes={verification['sizes']} max_bytes={verification['max_bytes']}"
        detail += "; transports_s=" + json.dumps(verification.get("timings_s", {}), sort_keys=True)
        detail += "; pairs_s=" + json.dumps(verification.get("pairs_s", []))
        detail += "; sizes_s=" + json.dumps(verification.get("sizes_s", []))
    print(f"[p2p] P2P {state}: {value['reason']}{cache}; {value['duration_s']:.3f}s{detail}", file=sys.stderr)
    print("1" if value["enabled"] else "0")


if __name__ == "__main__":
    main()
