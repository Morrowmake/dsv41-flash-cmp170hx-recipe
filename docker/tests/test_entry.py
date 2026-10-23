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
printf 'ENGRAM:%s,%s,%s,%s\\n' "${VLLM_ENGRAM_PLACEMENT-unset}" "${VLLM_ENGRAM_ROW_CACHE_GIB-unset}" \\
  "${VLLM_ENGRAM_ROW_CACHE_PRELOAD-unset}" "${VLLM_ENGRAM_DIRECT_READ-unset}"
'''
ENGRAM_KEYS = ("VLLM_ENGRAM_PLACEMENT", "VLLM_ENGRAM_ROW_CACHE_GIB", "VLLM_ENGRAM_ROW_CACHE_PRELOAD",
               "VLLM_ENGRAM_DIRECT_READ", "NEED_RAM_GIB")
GiB = 1 << 30
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
    (tmp_path / "cgroup").mkdir()
    memory(tmp_path, 300)
    return tmp_path


def memory(stage, avail_gib, limit=None, current=0, v1=False):
    """The host's MemAvailable, and the container cgroup's limit and use in bytes (None: no limit files)."""
    (stage / "meminfo").write_text(f"MemTotal: {512 * 1048576} kB\nMemAvailable: {avail_gib * 1048576} kB\n")
    if limit is None:
        return
    if v1:
        (stage / "cgroup/memory").mkdir()
        (stage / "cgroup/memory/memory.limit_in_bytes").write_text(f"{limit}\n")
        (stage / "cgroup/memory/memory.usage_in_bytes").write_text(f"{current}\n")
    else:
        (stage / "cgroup/memory.max").write_text(f"{limit}\n")
        (stage / "cgroup/memory.current").write_text(f"{current}\n")


def entry(stage, **settings):
    env = {k: v for k, v in os.environ.items() if k not in TRANSPORTS + ENGRAM_KEYS + ("LAYOUT", "DRY", "P2P")}
    env.update(CUDA_VISIBLE_DEVICES="", RECIPE_DIR=str(ROOT), VENV=str(stage / "venv"),
               SERVE_SH=str(stage / "serve.sh"), CALL_LOG=str(stage / "calls"), VLLM_IMAGE_CACHE_SEED="0",
               CHECK_RESULT="0", RECIPE_MEMINFO=str(stage / "meminfo"), RECIPE_CGROUP=str(stage / "cgroup"))
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
    for name in ("p2p_check.py", "p2p_check.sh", "p2p_probe.cu", "boot_check.py", "engram_placement.sh"):
        assert f'"$SCRIPT_DIR/{name}:/recipe/{name}:ro"' in start
        assert f"./{name}:/recipe/{name}:ro" in compose
    assert '"$SCRIPT_DIR/docker/entry.sh:/recipe/entry.sh:ro"' in start
    assert "./docker/entry.sh:/recipe/entry.sh:ro" in compose
    # The launch is the image's own: nothing mounts a serve.sh over it.
    assert ":/recipe/serve.sh" not in start and ":/recipe/serve.sh" not in compose
    assert ":/opt/recipe" not in start and ":/opt/recipe" not in compose


def engram(stage, **settings):
    """The Engram line entry.sh prints and the four settings serve.sh inherits."""
    run, calls = entry(stage, DRY="0", BOOT_CHECK="0", P2P="off", **settings)
    assert run.returncode == 0, run.stderr
    assert started(calls) == ["serve"]
    lines = [line for line in run.stdout.splitlines() if line.startswith("Engram tables: ")]
    assert len(lines) == 1, run.stdout
    inherited = next(line for line in run.stdout.splitlines() if line.startswith("ENGRAM:"))
    return lines[0], inherited.split(":", 1)[1].split(",")


def test_auto_holds_the_tables_in_ram_when_the_host_has_the_memory(stage):
    line, env = engram(stage)
    assert line == "Engram tables: ram (auto: 300 GiB available, RAM placement needs ~212 GiB)"
    assert env == ["ram", "unset", "unset", "unset"]
    memory(stage, 212)
    assert engram(stage)[1][0] == "ram"


def test_auto_reads_the_tables_from_the_ssd_below_the_need(stage):
    memory(stage, 150)
    line, env = engram(stage)
    assert line == ("Engram tables: ssd (auto: 150 GiB available, RAM placement needs ~212 GiB; slower), "
                    "row cache 30 GiB per card, filled at start, direct reads")
    assert env == ["ssd", "30", "1", "1"]
    memory(stage, 211)
    assert engram(stage)[1][0] == "ssd"


def test_need_is_one_setting_for_entry_and_start(stage):
    memory(stage, 300)
    assert engram(stage, NEED_RAM_GIB="400")[1][0] == "ssd"
    memory(stage, 100)
    assert engram(stage, NEED_RAM_GIB="64")[1][0] == "ram"
    start = (ROOT / "start.sh").read_text()
    assert "212" not in start and 'NEED_RAM_GIB="${NEED_RAM_GIB:-' not in start
    assert "ENGRAM_NEED_RAM_GIB_DEFAULT=212\n" in (ROOT / "engram_placement.sh").read_text()


@pytest.mark.parametrize("v1", [False, True])
def test_a_container_memory_limit_counts_what_is_left_under_it(stage, v1):
    """MemAvailable 300 GiB, but the container may use only 160 GiB and already holds 10."""
    memory(stage, 300, limit=160 * GiB, current=10 * GiB, v1=v1)
    line, env = engram(stage)
    assert line.startswith("Engram tables: ssd (auto: 150 GiB available,") and env[:2] == ["ssd", "30"]


@pytest.mark.parametrize("limit", ["max", "9223372036854771712"])
def test_no_container_limit_leaves_meminfo(stage, limit):
    memory(stage, 300, limit=limit, current=10 * GiB, v1=limit != "max")
    assert engram(stage)[1][0] == "ram"


def test_a_lower_meminfo_wins_over_a_looser_limit(stage):
    memory(stage, 100, limit=400 * GiB, current=0)
    assert engram(stage)[0].startswith("Engram tables: ssd (auto: 100 GiB available,")


@pytest.mark.parametrize("avail,cache,preload", [
    (64, "8", "0"), (128, "24", "1"), (192, "40", "1"),   # (available - 24 - 5) / 4
    (76, "11", "0"), (77, "12", "1"),                     # filled at start from 12 GiB, a quarter of a share
    (20, "1", "0"),                                        # never below 1 GiB
])
def test_ssd_row_cache_is_sized_from_the_memory_available(stage, avail, cache, preload):
    memory(stage, avail)
    line, env = engram(stage)
    assert env == ["ssd", cache, preload, "1"]
    assert f"row cache {cache} GiB per card, " in line and line.endswith(", direct reads")
    if preload == "1":
        assert ", filled at start," in line
    else:
        assert ", not filled at start (under 12 GiB)," in line


def test_explicit_placements_are_honoured(stage):
    memory(stage, 100)
    line, env = engram(stage, VLLM_ENGRAM_PLACEMENT="ram")
    # The engine itself refuses a host that cannot hold the tables.
    assert line == "Engram tables: ram (VLLM_ENGRAM_PLACEMENT=ram: 100 GiB available, RAM placement needs ~212 GiB)"
    assert env == ["ram", "unset", "unset", "unset"]
    memory(stage, 400)
    line, env = engram(stage, VLLM_ENGRAM_PLACEMENT="ssd")
    assert line == ("Engram tables: ssd (VLLM_ENGRAM_PLACEMENT=ssd: 400 GiB available, RAM placement needs ~212 GiB; "
                    "slower), row cache 48 GiB per card, filled at start, direct reads")
    assert env == ["ssd", "48", "1", "1"]


def test_ssd_settings_you_set_are_kept(stage):
    memory(stage, 150)
    line, env = engram(stage, VLLM_ENGRAM_ROW_CACHE_GIB="4")
    assert env == ["ssd", "4", "0", "1"] and "row cache 4 GiB per card (VLLM_ENGRAM_ROW_CACHE_GIB)" in line
    line, env = engram(stage, VLLM_ENGRAM_ROW_CACHE_GIB="4", VLLM_ENGRAM_ROW_CACHE_PRELOAD="1",
                       VLLM_ENGRAM_DIRECT_READ="0")
    assert env == ["ssd", "4", "1", "0"]
    assert "preload 1 (VLLM_ENGRAM_ROW_CACHE_PRELOAD)" in line and "direct reads 0 (VLLM_ENGRAM_DIRECT_READ)" in line
    # Empty means unset, as in start.sh's .env.
    assert engram(stage, VLLM_ENGRAM_ROW_CACHE_GIB="")[1] == ["ssd", "30", "1", "1"]


def test_unreadable_host_memory_reads_from_the_ssd(stage):
    (stage / "meminfo").write_text("MemTotal: 1 kB\n")
    line, env = engram(stage)
    assert line.startswith("Engram tables: ssd (auto: host memory unreadable, RAM placement needs ~212 GiB; slower)")
    assert env == ["ssd", "1", "0", "1"]


def test_dry_run_shows_the_decision(stage):
    memory(stage, 150)
    run, calls = entry(stage, DRY="1", P2P="off")
    assert run.returncode == 0 and calls == ["serve"]
    assert "Engram tables: ssd (auto: 150 GiB available" in run.stdout and "ENGRAM:ssd,30,1,1" in run.stdout


@pytest.mark.parametrize("settings,message", [
    ({"VLLM_ENGRAM_PLACEMENT": "disk"}, "VLLM_ENGRAM_PLACEMENT must be auto, ram or ssd, not 'disk'"),
    ({"NEED_RAM_GIB": "lots"}, "NEED_RAM_GIB must be a whole number of GiB, not 'lots'"),
])
def test_bad_engram_settings_refuse_before_the_server(stage, settings, message):
    run, calls = entry(stage, DRY="0", BOOT_CHECK="0", P2P="off", **settings)
    assert run.returncode == 2 and message in run.stderr and calls == []
