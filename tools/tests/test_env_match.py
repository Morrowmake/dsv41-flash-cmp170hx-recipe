"""tools/check-env-match.py against a launch in the generated serve.sh's form."""
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import env_effective as E  # noqa: E402

PIN = "c" * 40
# The generated launch's shape: it clears what it inherits, exports its settings
# (keeping an inherited engine switch), and under DRY=1 prints them and the command.
LAUNCH = r'''#!/bin/bash
set -euo pipefail
layout=${LAYOUT:?the session names the layout}
dry=${DRY:-${DRY_RUN:-0}}
case "$layout" in
  tp4)
    export HOME=/cache
    export PATH=/opt/venv/bin:/usr/bin:/bin
    export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL-SYS}
    export VLLM_ENGRAM_PLACEMENT=${VLLM_ENGRAM_PLACEMENT-ram}
    export VLLM_EXL3_MOE_V2=${VLLM_EXL3_MOE_V2-1}
    argv=(/opt/venv/bin/vllm serve /models/DeepSeek-V4.1-Flash-1017-hybrid-exl3-fp8 \
      --host "${HOST:-0.0.0.0}" --served-model-name dsv41-flash --max-num-seqs 8 \
      --max-num-batched-tokens 2048 --port "${PORT:-8001}" \
      --hf-overrides '{"engram_table_dir":"/models/DeepSeek-V4.1-Flash-1017-engram"}' \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8]}')
    if [ "$dry" = 1 ]; then
      env | sort
      printf -v _line '%q ' "${argv[@]}"
      echo "DRY_RUN_ARGV_SHELL=${_line% }"
      exit 0
    fi
    exec "${argv[@]}"
    ;;
  *) echo "layout $layout: not a layout of this recipe" >&2; exit 2 ;;
esac
'''
LOGGED = ("(APIServer pid=1) INFO [api_utils.py:357] non-default args: {'model_tag': '/srv/lab/hybrid', "
          "'host': '127.0.0.1', 'port': 8001, 'model': '/srv/lab/hybrid', 'served_model_name': ['dsv41-flash'], "
          "'hf_overrides': {'engram_table_dir': '/srv/lab/engram'}, 'max_num_batched_tokens': 2048, "
          "'max_num_seqs': 8, 'compilation_config': {'mode': None, 'cudagraph_mode': "
          "<CUDAGraphMode.FULL_DECODE_ONLY: (2, 0)>, 'cudagraph_capture_sizes': [4, 8], 'pass_config': {}}}")
UPSTREAM = 'environment_variables = {\n    "VLLM_UNUSED": lambda: os.getenv("VLLM_UNUSED", "x"),\n}'
FORK = ('FORK_ENV_VARS = {\n'
        '    "VLLM_EXL3_MOE_V2": lambda: bool(int(os.getenv("VLLM_EXL3_MOE_V2", "0"))),\n'
        '    "VLLM_ENGRAM_ROW_CACHE_GIB": lambda: float(os.getenv("VLLM_ENGRAM_ROW_CACHE_GIB", "0")),\n}')


@pytest.fixture
def recipe(tmp_path, monkeypatch):
    (tmp_path / "serve.sh").write_text(LAUNCH)
    (tmp_path / "start.sh").write_text(f"RELEASE_VLLM_COMMIT={PIN}\n")
    monkeypatch.setattr(E, "git_source", lambda fork, name, pin: {
        "vllm/envs.py": UPSTREAM, "vllm/sm80/envs.py": FORK}[name])
    return tmp_path


def load_checker():
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_env_match", ROOT / "tools/check-env-match.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_reads_the_environment_and_the_command(recipe):
    env, command, _ = E.dry_run(recipe, "tp4")
    assert env["VLLM_ENGRAM_PLACEMENT"] == "ram" and env["NCCL_P2P_LEVEL"] == "SYS"
    assert command["max_num_seqs"] == 8 and command["served_model_name"] == ["dsv41-flash"]
    assert command["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"


def test_dry_run_inherits_nothing_from_the_caller(recipe, monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_MOE_V2", "0")
    monkeypatch.setenv("NCCL_P2P_LEVEL", "LOC")
    env, _, _ = E.dry_run(recipe, "tp4")
    assert env["VLLM_EXL3_MOE_V2"] == "1" and env["NCCL_P2P_LEVEL"] == "SYS"


def test_reference_mode_against_this_checkout(tmp_path):
    """The checkout's own serve.sh, once generated, runs under the checker from a clean environment."""
    if not (ROOT / "serve.sh").exists():
        pytest.skip("serve.sh is generated from the release settings")
    path = tmp_path / "ref.json"
    path.write_text(json.dumps({"tp4": {}}))
    run = subprocess.run([sys.executable, str(ROOT / "tools/check-env-match.py"), str(path)],
                         text=True, capture_output=True)
    assert run.returncode == 0, run.stdout + run.stderr


def test_reference_values_and_arguments(recipe, capsys):
    module = load_checker()
    serve = recipe / "serve.sh"
    good = {"tp4": {"VLLM_ENGRAM_PLACEMENT": "ram", "MAX_SEQS": "8", "MAX_BATCHED": "2048",
                    "HF_HOME": "/somewhere/else"}}
    assert module.check(good, serve) == 0
    assert "ignored HF_HOME (path valued)" in capsys.readouterr().out
    assert module.check({"tp4": {"VLLM_ENGRAM_PLACEMENT": "ssd"}}, serve) == 1
    assert module.check({"tp4": {"MAX_SEQS": "4"}}, serve) == 1
    assert module.check({"tp4": {}, "pp4": {}}, serve) == 1


def write_leg(tmp_path, **changes):
    env, _, _ = E.dry_run(tmp_path, "tp4")
    env.update({"CUDA_VISIBLE_DEVICES": "0,1,2,3", "GPU_LOCK_TOKENS": "t", "HOME": "/srv/lab",
                "NV_LIBNCCL_PACKAGE": "libnccl2"})
    for key, value in changes.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    path = tmp_path / "leg" / "effective_env_private.json"
    path.parent.mkdir()
    path.write_text(json.dumps(env))
    (path.parent / "serve.log").write_text(LOGGED + "\n")
    return path


def test_effective_match_allows_only_reasoned_differences(recipe, capsys):
    leg = write_leg(recipe)
    assert E.check_effective(recipe, leg, "tp4", recipe) == 0
    out = capsys.readouterr().out
    assert "Full environment match: PASS" in out
    assert "allowed GPU_LOCK_TOKENS" in out and "allowed NV_LIBNCCL_PACKAGE" in out
    assert "MATCH compilation_config on the keys the launch sets" in out


def test_effective_default_equal_switch_is_proven_by_the_pinned_source(recipe, capsys):
    """A validation leg that set a switch to its engine default matches a launch that leaves it unset."""
    leg = write_leg(recipe, VLLM_ENGRAM_ROW_CACHE_GIB="0")
    assert E.check_effective(recipe, leg, "tp4", recipe) == 0
    assert "equals fork default" in capsys.readouterr().out


@pytest.mark.parametrize("key,value", [("VLLM_EXL3_MOE_V2", "0"), ("VLLM_ENGRAM_PLACEMENT", "ssd"),
                                       ("VLLM_EXL3_MOE_V2", None), ("VLLM_NEW_SWITCH", "1")])
def test_effective_changed_or_missing_switch_fails(recipe, key, value):
    leg = write_leg(recipe, **{key: value})
    assert E.check_effective(recipe, leg, "tp4", recipe) == 1


def test_effective_command_differences_fail(recipe):
    leg = write_leg(recipe)
    log = leg.parent / "serve.log"
    log.write_text(LOGGED.replace("'max_num_seqs': 8", "'max_num_seqs': 4") + "\n")
    assert E.check_effective(recipe, leg, "tp4", recipe) == 1
    log.write_text(LOGGED.replace("[4, 8]", "[4, 8, 16]") + "\n")
    assert E.check_effective(recipe, leg, "tp4", recipe) == 1
    log.write_text("")
    with pytest.raises(ValueError, match="non-default args"):
        E.check_effective(recipe, leg, "tp4", recipe)


def test_effective_needs_a_release_pin(recipe):
    (recipe / "start.sh").write_text("RELEASE_VLLM_COMMIT=" + "{{" + "PIN" + "}}\n")
    with pytest.raises(ValueError, match="no release pin"):
        E.check_effective(recipe, write_leg(recipe), "tp4", recipe)


def test_effective_image_tree_recipe_keys_and_engine_default_args(capsys):
    """The image's engine-tree file equals the leg's same file (tree from its PYTHONPATH); a recipe-only key
    with no serving behaviour and an argument at vLLM's default are allowed; anything else still fails."""
    policy = "/vllm/models/deepseek_v41/nvidia/dspark_depth_policy.json"
    leg = {"PYTHONPATH": "/srv/lab/tree", "VLLM_DSPARK_ADAPTIVE_DEPTH": "/srv/lab/tree" + policy}
    mine = {"VLLM_DSPARK_ADAPTIVE_DEPTH": E.IMAGE_TREE + policy, "PYTHONDONTWRITEBYTECODE": "1"}
    assert E.compare_environment(mine, leg, {}, PIN) == []
    out = capsys.readouterr().out
    assert "the same engine-tree file" in out and "allowed PYTHONDONTWRITEBYTECODE" in out
    other = dict(mine, VLLM_DSPARK_ADAPTIVE_DEPTH=E.IMAGE_TREE + "/other.json")
    assert len(E.compare_environment(other, leg, {}, PIN)) == 1
    assert len(E.compare_environment(mine, dict(leg, PYTHONDONTWRITEBYTECODE="0"), {}, PIN)) == 1
    assert E.compare_commands({"pipeline_parallel_size": 1}, {}) == []
    assert len(E.compare_commands({"pipeline_parallel_size": 2}, {})) == 1
