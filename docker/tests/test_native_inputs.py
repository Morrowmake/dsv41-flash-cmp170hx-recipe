"""native_inputs.py: which wheel libraries a queue makes stale, by the plugin's native-input rules."""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import native_inputs as ni

RULES = """schema = "native-inputs/1"
include_dirs = ["csrc"]

[[extensions]]
artifact = "vllm/_C_stable_libtorch*.so"
paths = []
wheel = true

[[extensions]]
artifact = "vllm/_moe_C_stable_libtorch*.so"
paths = []
wheel = true

[[cmake_option]]
tokens = ["VLLM_AMPERE_MARLIN_BUILD"]
native = "vllm_sm80/_ampere_marlin_C*.so"

[[stale]]
paths = ["/CMakeLists.txt", "cmake/**", "csrc/spinloop.cpp"]
native = "wheel"

[[stale]]
paths = ["csrc/moe/**"]
native = "vllm/_moe_C_stable_libtorch*.so"

[[stale]]
paths = ["csrc/**"]
native = "vllm/_C_stable_libtorch*.so"
"""
CMAKE = "project(vllm)\nadd_library(_C_stable_libtorch csrc/a.cu)\n"


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def commit(repo, files, message):
    for rel, text in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    upstream = commit(repo, {"CMakeLists.txt": CMAKE, "csrc/a.cu": '#include "ops.h"\n', "csrc/ops.h": "",
                             "csrc/moe/m.cu": '#include "moe/moe.h"\n', "csrc/moe/moe.h": "",
                             "csrc/spinloop.cpp": "", "vllm/x.py": ""}, "upstream")
    plugin = commit(repo, {"vllm_sm80/launch/data/native_inputs.toml": RULES}, "plugin")
    git(repo, "checkout", "-q", upstream)
    return repo, upstream, plugin


def decide(repo, files):
    path, upstream, plugin = repo
    queue = commit(path, files, "queue")
    git(path, "checkout", "-q", upstream)
    return ni.decide(path / ".git", upstream, queue, plugin)


def test_a_csrc_source_rebuilds_the_core_library(repo):
    result = decide(repo, {"csrc/a.cu": "changed\n", "vllm/x.py": "python only\n"})
    assert result == {"changed": ["csrc/a.cu"], "stale": {"vllm/_C_stable_libtorch*.so": ["csrc/a.cu"]},
                      "rebuild": ["--stable"], "unbuilt": []}


def test_a_header_goes_to_the_natives_that_include_it(repo):
    assert decide(repo, {"csrc/ops.h": "x\n"})["rebuild"] == ["--stable"]
    assert decide(repo, {"csrc/moe/moe.h": "x\n"})["rebuild"] == ["--moe-stable"]


def test_inputs_of_libraries_no_option_builds_are_unbuilt(repo):
    for files in ({"csrc/spinloop.cpp": "x\n"}, {"CMakeLists.txt": CMAKE + "add_library(other x.cpp)\n"},
                  {"setup.py": "x\n"}):
        result = decide(repo, files)
        assert result["unbuilt"] == sorted(files) and result["rebuild"] == []
    assert ni.main(["--src", str(repo[0] / ".git"), "--upstream", repo[1], "--queue", "HEAD@{1}",
                    "--plugin", repo[2]]) == 1


def test_cmake_comments_and_fork_options_rebuild_no_wheel_library(repo):
    assert decide(repo, {"CMakeLists.txt": "# a comment\n" + CMAKE})["stale"] == {}
    option = CMAKE + "if(VLLM_AMPERE_MARLIN_BUILD)\n  add_library(m x.cu)\nendif()\n"
    result = decide(repo, {"CMakeLists.txt": option})
    assert result["unbuilt"] == ["CMakeLists.txt"] and result["rebuild"] == []


def test_other_platforms_and_no_change_rebuild_nothing(repo):
    assert decide(repo, {"csrc/cpu/x.cpp": "x\n"})["changed"] == []
    assert decide(repo, {"vllm/x.py": "y\n"}) == {"changed": [], "stale": {}, "rebuild": [], "unbuilt": []}
