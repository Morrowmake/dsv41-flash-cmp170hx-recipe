#!/usr/bin/env python3
"""Inspect the extracted image with its Python and no GPU device nodes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

DRIVER_DIR = "/usr/local/nvidia/lib64"
# The release's speculative decoding: the checkpoint's DSpark drafter, five
# tokens at one to three requests and three at four to eight.
SPECULATIVE = {"method": "dspark", "num_speculative_tokens": 5,
               "num_speculative_tokens_per_batch_size": [[1, 3, 5], [4, 8, 3]]}


def driver_options(directories=None, rootfs=None):
    directories = directories or (Path("/usr/lib/x86_64-linux-gnu"),
                                  Path("/lib/x86_64-linux-gnu"))
    libraries = {}
    for directory in directories:
        for pattern in ("libcuda.so*", "libnvidia-*.so*"):
            for library in sorted(directory.glob(pattern)):
                if library.is_file():
                    libraries.setdefault(library.name, library.resolve())
    if "libcuda.so.1" not in libraries:
        raise ValueError("Host libcuda.so.1 is required for driver discovery")
    options = []
    if rootfs is not None:
        options += ["--tmpfs", "/usr/local"]
        for path in sorted((rootfs / "usr/local").iterdir()):
            if path.name != "nvidia":
                destination = "/usr/local/" + path.name
                if path.is_symlink():
                    options += ["--symlink", os.readlink(path), destination]
                else:
                    options += ["--ro-bind", str(path.resolve()), destination]
    options += ["--tmpfs", "/usr/local/nvidia", "--dir", DRIVER_DIR]
    for name, source in sorted(libraries.items()):
        options += ["--ro-bind", str(source), DRIVER_DIR + "/" + name]
    return options


def inside(model, draft=None):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert not [p for p in Path("/dev").glob("nvidia*") if p.is_char_device()]
    import ctypes
    import pickle
    import tempfile
    import cloudpickle
    import torch
    from triton.backends.nvidia.driver import libcuda_dirs
    from vllm.model_executor.models.registry import ModelRegistry, _ModelInfo

    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert not [p for p in Path("/dev").glob("nvidia*") if p.is_char_device()]
    assert torch.cuda.device_count() == 0
    # Discovery and loading exercise the original failure without CUDA init.
    assert DRIVER_DIR in libcuda_dirs()
    ctypes.CDLL(DRIVER_DIR + "/libcuda.so.1")
    checks = ["rootfs_tmp_1777", "no_gpu_nodes", "driver_discovery", "driver_load"]
    # The engine refuses to start unless the fork's plugin distribution
    # registered its entry points; prove the shipped dist-info survived packaging.
    import vllm.plugins
    from vllm.sm80 import plugin
    from vllm.sm80.guard import check_plugin_loaded
    vllm.plugins.load_general_plugins()
    if not plugin.is_registered():
        raise RuntimeError("the fork's plugin sm80 is not registered")
    check_plugin_loaded()
    print("PLUGIN PASS sm80", flush=True)
    checks.append("sm80_plugin")
    # The image carries the release's launch.
    launch = Path("/opt/recipe/serve.sh")
    if not launch.is_file() or not launch.read_text().startswith("#!/bin/bash"):
        raise RuntimeError("the image carries no release launch at /opt/recipe/serve.sh")
    checks.append("release_launch")
    pairs = (("target", model),) + ((("draft", draft),) if draft is not None else ())
    for label, path in pairs:
        config = json.loads((path / "config.json").read_text())
        for arch in config["architectures"]:
            with tempfile.TemporaryDirectory() as tmp:
                output = str(Path(tmp) / "result")
                fn = lambda: _ModelInfo.from_model_cls(
                    ModelRegistry.models[arch].load_model_cls())
                run = subprocess.run(
                    [sys.executable, "-m", "vllm.model_executor.models.registry"],
                    input=cloudpickle.dumps((fn, output)), capture_output=True,
                    timeout=180)
                Path("/tmp/results/" + label + ".registry.stderr").write_bytes(run.stderr)
                Path("/tmp/results/" + label + ".registry.stdout").write_bytes(run.stdout)
                if run.returncode:
                    raise RuntimeError(run.stderr.decode(errors="replace"))
                info = pickle.loads(Path(output).read_bytes())
                assert info.is_text_generation_model
                print("REGISTRY PASS", label, arch, flush=True)
                checks.append(label + "_registry")

    from vllm.entrypoints.cli.serve import ServeSubcommand
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    checks.append("serve_import")
    engine = AsyncEngineArgs(model=str(model), load_format="exl3",
                             max_model_len=262144, tensor_parallel_size=4,
                             enable_expert_parallel=True,
                             gpu_memory_utilization=0.95, kv_cache_dtype="bfloat16",
                             speculative_config=dict(SPECULATIVE))
    # EXL3 checks for the CUDA platform while the model config is built; on
    # CPU that check is the boundary (pydantic reports it as a ValueError).
    exl3_boundary = "EXL3 execution requires the NVIDIA CUDA platform"
    boundary = None
    try:
        model_config = engine.create_model_config()
    except ValueError as error:
        if exl3_boundary not in str(error):
            raise
        boundary = exl3_boundary
        print("CPU_BOUNDARY", boundary, flush=True)
    else:
        assert model_config.hf_config.architectures == json.loads(
            (model / "config.json").read_text())["architectures"]
        checks.append("model_config")
    # Never initialize an engine or bind a server port in the CPU gate.
    try:
        parser = FlexibleArgumentParser()
        ServeSubcommand().subparser_init(parser.add_subparsers(dest="command"))
        args = parser.parse_args([
            "serve", str(model), "--load-format", "exl3", "--tensor-parallel-size", "4",
            "--enable-expert-parallel", "--mm-encoder-tp-mode", "data",
            "--kv-cache-dtype", "bfloat16",
            "--max-model-len", "262144", "--gpu-memory-utilization", "0.95",
            "--enable-auto-tool-choice", "--tool-call-parser", "deepseek_v41",
            "--reasoning-parser", "deepseek_v41", "--speculative-config",
            json.dumps(SPECULATIVE)])
        args.model = args.model_tag
        ServeSubcommand().validate(args)
        checks.append("serve_arguments")
        if boundary is None:
            AsyncEngineArgs.from_cli_args(args).create_engine_config()
    except RuntimeError as error:
        if not str(error).startswith("Failed to infer device type,"):
            raise
        boundary = boundary + "; " + str(error) if boundary else str(error)
        print("CPU_BOUNDARY", boundary, flush=True)
    if boundary is None:
        boundary = "engine configuration completed; engine not started"
    checks.append("engine_config_boundary")
    record = dict(passed=True, checks=checks, boundary=boundary)
    Path("/tmp/results/summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print("ROOTFS_SMOKE PASS", len(checks), flush=True)


def check_tmp(rootfs):
    path = rootfs / "tmp"
    if path.is_symlink() or not path.is_dir() or path.stat().st_mode & 0o7777 != 0o1777:
        raise ValueError("Image /tmp must be a directory with mode 1777")
    return "1777"


def check_recipe_license(rootfs):
    recipe = Path(__file__).resolve().parents[1]
    for name, source in (("LICENSE", recipe / "LICENSE"),
                         ("THIRD_PARTY_NOTICES", recipe / "docker/THIRD_PARTY_NOTICES")):
        notice = rootfs / "opt/image-tools" / name
        if not notice.is_file() or notice.read_bytes() != source.read_bytes():
            raise ValueError(f"Image /opt/image-tools/{name} must match recipe notice bytes")


def image_environment(config):
    env = dict(e.split("=", 1) for e in
               json.loads(config.read_text())["config"]["Env"])
    if DRIVER_DIR not in env.get("LD_LIBRARY_PATH", "").split(":"):
        raise ValueError("OCI LD_LIBRARY_PATH omits the driver mount directory")
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver-options", action="store_true")
    parser.add_argument("--inside", action="store_true")
    # The DSpark drafter is inside the checkpoint; --draft is for a separate drafter only.
    for name in ("rootfs", "config", "model", "draft", "result"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.driver_options:
        options = driver_options(rootfs=args.rootfs)
        if args.config is not None:
            env = image_environment(args.config)
            for key, value in sorted(env.items()):
                if key != "CUDA_VISIBLE_DEVICES":
                    options += ["--setenv", key, value]
        sys.stdout.buffer.write(b"\0".join(s.encode() for s in options) + b"\0")
        return
    if args.inside:
        inside(args.model, args.draft)
        return
    if any(getattr(args, name) is None for name in
           ("rootfs", "config", "model", "result")):
        parser.error("rootfs, config, model and result are required")
    check_tmp(args.rootfs)
    check_recipe_license(args.rootfs)
    print("[image-rootfs] recipe LICENSE bytes match", flush=True)
    enabled = os.environ.get("IMAGE_ROOTFS_CPU_SMOKE", "1") == "1"
    print("[image-rootfs-cpu] enabled=" + str(int(enabled)), flush=True)
    if not enabled:
        return
    args.result.mkdir(parents=True, exist_ok=False)
    env = image_environment(args.config)
    env.update(CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="void",
               HOME="/tmp", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONDONTWRITEBYTECODE="1")
    command = ["bwrap", "--clearenv", "--unshare-all", "--ro-bind",
               str(args.rootfs.resolve()), "/", "--dev", "/dev", "--proc", "/proc",
               "--tmpfs", "/tmp", "--bind", str(args.result.resolve()), "/tmp/results",
               "--ro-bind", str(args.model.resolve()), "/tmp/target",
               "--ro-bind", str(Path(__file__).resolve()), "/tmp/rootfs_smoke.py"]
    inner = ["--model", "/tmp/target"]
    if args.draft is not None:
        command += ["--ro-bind", str(args.draft.resolve()), "/tmp/draft"]
        inner += ["--draft", "/tmp/draft"]
    command += driver_options(rootfs=args.rootfs)
    for key, value in sorted(env.items()):
        command += ["--setenv", key, value]
    command += ["/opt/venv/bin/python", "/tmp/rootfs_smoke.py", "--inside", *inner]
    subprocess.run(command, check=True, timeout=480,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))


if __name__ == "__main__":
    main()
