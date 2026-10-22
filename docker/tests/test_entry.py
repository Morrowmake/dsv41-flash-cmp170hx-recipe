"""docker/entry.sh: peer access, boot check and the hand-off to the image's serve.sh."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).parents[2]
TRANSPORTS = ("VLLM_CUSTOM_ALLREDUCE_PCIE_P2P", "NCCL_P2P_LEVEL", "NCCL_P2P_DISABLE")
# Stands in for the image's serve.sh: it reports what it inherited.
SERVE = '''#!/bin/bash
echo serve >> "$CALL_LOG"
printf 'RESULT:%s,%s,%s DRY=%s LAYOUT=%s\\n' "${VLLM_CUSTOM_ALLREDUCE_PCIE_P2P-unset}" \\
  "${NCCL_P2P_LEVEL-unset}" "${NCCL_P2P_DISABLE-unset}" "${DRY-unset}" "${LAYOUT-unset}"
'''
PYTHON = '''#!/bin/bash
case "$1" in
  */p2p_check.py) echo "check:$*" >> "$CALL_LOG"; echo "$CHECK_RESULT" ;;
  */boot_check.py) echo "boot:$*" >> "$CALL_LOG" ;;
  *) echo unexpected-interpreter-call >&2; exit 1 ;;
esac
'''


@pytest.fixture
def stage(tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").write_text(PYTHON)
    (venv / "bin/python").chmod(0o755)
    (tmp_path / "serve.sh").write_text(SERVE)
    return tmp_path


def entry(stage, **settings):
    env = {k: v for k, v in os.environ.items() if k not in TRANSPORTS + ("LAYOUT", "DRY", "P2P")}
    env.update(CUDA_VISIBLE_DEVICES="", RECIPE_DIR=str(ROOT), VENV=str(stage / "venv"),
               SERVE_SH=str(stage / "serve.sh"), CALL_LOG=str(stage / "calls"), VLLM_IMAGE_CACHE_SEED="0",
               CHECK_RESULT="0")
    env.update(settings)
    run = subprocess.run(["bash", str(ROOT / "docker/entry.sh")], text=True, capture_output=True, env=env)
    calls = (stage / "calls").read_text().splitlines() if (stage / "calls").exists() else []
    (stage / "calls").unlink(missing_ok=True)
    return run, calls


def started(calls):
    """What ran after the peer check: the server, or the boot check wrapping it."""
    return [call for call in calls if not call.startswith("check:")]


@pytest.mark.parametrize("mode,result", [("off", "RESULT:0,LOC,1"), ("auto", "RESULT:0,LOC,1"),
                                         ("force", "RESULT:unset,unset,unset")])
def test_dry_run_transports(stage, mode, result):
    """A dry run never checks the cards: auto stays off, force passes, and serve.sh gets DRY=1."""
    run, calls = entry(stage, DRY="1", P2P=mode, NCCL_P2P_DISABLE="1")
    assert run.returncode == 0, run.stderr
    assert result in run.stdout and "DRY=1 LAYOUT=tp4" in run.stdout
    assert calls == ["serve"]


@pytest.mark.parametrize("passed,result", [("1", "RESULT:1,SYS,unset"), ("0", "RESULT:0,LOC,1")])
def test_check_runs_before_the_server_and_sets_transports(stage, passed, result):
    """After a pass the inherited transports reach serve.sh; a failure turns each one off explicitly,
    since serve.sh would put its own value back for a name left unset."""
    run, calls = entry(stage, DRY="0", BOOT_CHECK="0", P2P="auto", CHECK_RESULT=passed,
                       VLLM_CUSTOM_ALLREDUCE_PCIE_P2P="1", NCCL_P2P_LEVEL="SYS", NCCL_P2P_DISABLE="1")
    assert run.returncode == 0, run.stderr
    assert len(calls) == 2 and calls[0].startswith("check:") and calls[1] == "serve"
    assert "--mode auto --tp 4 --pp 1" in calls[0]
    assert result in run.stdout


def test_boot_check_wraps_the_server_unless_start_owns_it(stage):
    run, calls = entry(stage, DRY="0", BOOT_CHECK="1", P2P="off", PORT="8000")
    assert run.returncode == 0, run.stderr
    calls = started(calls)
    assert len(calls) == 1 and calls[0].startswith("boot:") and calls[0].endswith(
        "--serve /bin/bash " + str(stage / "serve.sh"))
    assert "--model dsv41-flash" in calls[0]
    run, calls = entry(stage, DRY="0", BOOT_CHECK="1", P2P="off", RECIPE_BOOT_CHECK_OWNER="start")
    assert run.returncode == 0 and started(calls) == ["serve"]
    assert "[boot-check] delegated to start.sh after health" in run.stdout


def test_missing_launch_and_bad_settings_refuse(stage):
    run, calls = entry(stage, SERVE_SH=str(stage / "absent.sh"))
    assert run.returncode == 1 and "has no" in run.stderr and calls == []
    run, calls = entry(stage, DRY="0", P2P="off", BOOT_CHECK="2")
    assert run.returncode == 2 and started(calls) == []
    run, calls = entry(stage, P2P="sometimes")
    assert run.returncode == 2 and calls == []


def test_cache_root_must_be_the_cache_mount(stage):
    run, calls = entry(stage, DRY="0", P2P="off", BOOT_CHECK="0", RECIPE_CACHE_ROOT="/elsewhere")
    assert run.returncode == 1 and "cache root must be /cache" in run.stderr and calls == []


def test_start_and_compose_mount_the_entry_and_the_peer_check():
    start = (ROOT / "start.sh").read_text()
    compose = (ROOT / "docker-compose.yml").read_text()
    for name in ("p2p_check.py", "p2p_check.sh", "p2p_probe.cu", "boot_check.py"):
        assert f'"$SCRIPT_DIR/{name}:/recipe/{name}:ro"' in start
        assert f"./{name}:/recipe/{name}:ro" in compose
    assert '"$SCRIPT_DIR/docker/entry.sh:/recipe/entry.sh:ro"' in start
    assert "./docker/entry.sh:/recipe/entry.sh:ro" in compose
    # The launch is the image's own: nothing mounts a serve.sh over it.
    assert ":/recipe/serve.sh" not in start and ":/recipe/serve.sh" not in compose
    assert ":/opt/recipe" not in start and ":/opt/recipe" not in compose
