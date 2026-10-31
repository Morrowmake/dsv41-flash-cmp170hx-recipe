"""CPU-only comparison of the recipe's launch with a validated server environment.

The recipe's serve.sh is generated from the release settings: under DRY=1 it
prints the environment it would give the server (`env | sort`) and the command
(`DRY_RUN_ARGV_SHELL=...`) and exits. These are compared with a validation
leg's complete server environment and the `non-default args` record of its
serve.log, so the release ships what was validated.
"""
import ast
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LAYOUTS = ("tp4",)
NAME = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
ARGV_LINE = "DRY_RUN_ARGV_SHELL="
# vLLM logs enum members of a resolved config as <Class.NAME: value>.
ENUM = re.compile(r"<\w+\.(\w+): [^<>]*>")
# Validation harness names: present in a lab leg's server environment, never a
# recipe setting. Each value is the reason it is not compared.
LAB_KEYS = {
    "GPU_LOCK_TOKENS": "validation GPU lock ownership",
    "GPU_LOCK_CARDS": "validation GPU lock ownership",
    "LANG": "validation shell locale",
    "LOGNAME": "validation login identity",
    "SHLVL": "validation shell nesting",
    "USER": "validation login identity",
    "VLLM_SERVER_DEV_MODE": "validation server diagnostics",
    "CUDA_VISIBLE_DEVICES": "validation GPU selection; the recipe serves on every visible card",
    "LAB_MODEL": "validation model-profile selection",
    "RAM_JOB_MARKER": "validation memory-lease bookkeeping",
    "RAM_LEASE_TOKEN": "validation memory-lease bookkeeping",
    "SESSION_JOB": "validation session bookkeeping",
    "GPU_QUEUE_JOB": "validation queue bookkeeping",
    "LAYOUT": "launcher input; the parallel sizes are compared in the command",
}
# Locations: installation, checkpoint, cache and search paths differ by design.
PATH_KEYS = {
    "CUDA_HOME", "FLASHINFER_WORKSPACE_BASE", "HOME", "HF_HOME", "PATH", "PWD", "PYTHONPATH",
    "TMPDIR", "TILELANG_CACHE_DIR", "TILELANG_TMP_DIR", "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "VLLM_CACHE_ROOT", "XDG_CACHE_HOME",
    "CUDA_CACHE_PATH", "VIRTUAL_ENV", "LD_LIBRARY_PATH", "LIBRARY_PATH",
}
# Base-image metadata the container runtime puts in every environment.
IMAGE_PREFIXES = ("NV_", "NVIDIA_", "CUDA_VERSION", "NCCL_VERSION", "NVARCH")
# Engine switch getters: upstream's table and the fork's own switches. A single-tree engine (start.sh
# RELEASE_VLLM_COMMIT) holds both at its commit; an assembled engine (the four pins RELEASE_UPSTREAM_COMMIT,
# RELEASE_QUEUE_COMMIT, RELEASE_PLUGIN_COMMIT, RELEASE_NATIVE_BUILD) holds upstream's in its queue commit and the
# fork's in its plugin commit (vllm_sm80/envs.py). Both commits are in the fork repository.
ENV_FILES = ("vllm/envs.py", "vllm/sm80/envs.py")
PINNED_ENV_FILES = (("vllm/envs.py", "RELEASE_QUEUE_COMMIT"), ("vllm_sm80/envs.py", "RELEASE_PLUGIN_COMMIT"))
ENGINE_PINS = ("RELEASE_UPSTREAM_COMMIT", "RELEASE_QUEUE_COMMIT", "RELEASE_PLUGIN_COMMIT", "RELEASE_NATIVE_BUILD")
# Command arguments that name a location, not a behaviour.
IGNORED_ARGS = {"model", "model_tag", "host", "port", "enable_prompt_tokens_details"}
PATH_OVERRIDES = {"engram_table_dir"}
# Arguments vLLM logs as the whole resolved config: compared on the keys the launch sets.
RESOLVED_CONFIGS = {"compilation_config"}
# Arguments the launch passes at vLLM's own default, so vLLM leaves them out of
# its non-default args record; a validation leg that omits them ran the same.
ENGINE_DEFAULT_ARGS = {"pipeline_parallel_size": 1}
# Set by the recipe's launch only, with no serving behaviour.
RECIPE_KEYS = {
    "PYTHONDONTWRITEBYTECODE": "the container writes no bytecode into its read-only engine tree",
}
# Where the image holds the engine tree; a validation leg runs it from its own
# checkout, named by its PYTHONPATH. An assembled engine's image holds engine and
# plugin side by side (imagebuild/sm80/build.py assembly), as the leg's PYTHONPATH
# names its engine:plugin.
IMAGE_TREE = "/opt/vllm-src"
IMAGE_ASSEMBLY = ("/opt/assembly/engine", "/opt/assembly/plugin")
COMMIT = re.compile(r"[0-9a-f]{40}")


def release_engine(root=ROOT):
    """The engine the recipe releases, from its start.sh pins: {label, sources: [(getter file, commit)], image: the
    image's code directories}. A single-tree engine names RELEASE_VLLM_COMMIT; an assembled one the four ENGINE_PINS
    and no RELEASE_VLLM_COMMIT. ValueError when start.sh names neither form completely."""
    text = (root / "start.sh").read_text()
    pins = {m[1]: m[2] for m in re.finditer(r"^(RELEASE_[A-Z_]+)=[\"']?([^\"'\n]*)", text, re.M)}
    named = [n for n in ENGINE_PINS if n in pins]
    if named:
        if len(named) != len(ENGINE_PINS) or "RELEASE_VLLM_COMMIT" in pins:
            raise ValueError("start.sh names %s: an assembled engine names all of %s and no RELEASE_VLLM_COMMIT"
                             % (", ".join(sorted(set(named) | {"RELEASE_VLLM_COMMIT"} & set(pins))),
                                ", ".join(ENGINE_PINS)))
        bad = [n for n in ENGINE_PINS[:3] if not COMMIT.fullmatch(pins[n])]
        if bad:
            raise ValueError("start.sh %s must be full commits" % ", ".join(bad))
        return {"label": "queue %s, plugin %s" % (pins["RELEASE_QUEUE_COMMIT"][:10],
                                                  pins["RELEASE_PLUGIN_COMMIT"][:10]),
                "sources": [(name, pins[pin]) for name, pin in PINNED_ENV_FILES], "image": IMAGE_ASSEMBLY}
    pin = pins.get("RELEASE_VLLM_COMMIT", "")
    if not COMMIT.fullmatch(pin):
        raise ValueError("start.sh carries no release pin yet (tools/set-pin.sh sets it)")
    return {"label": pin, "sources": [(name, pin) for name in ENV_FILES], "image": (IMAGE_TREE,)}


def engine_getters(fork, engine):
    """{name: (getter file, its source, getter)} of the released ENGINE (release_engine) in the FORK repository; a
    later file's getter wins (the fork's over upstream's)."""
    getters = {}
    for filename, commit in engine["sources"]:
        source = git_source(fork, filename, commit)
        for key, getter in read_getters(source).items():
            getters[key] = (filename, source, getter)
    return getters


def git_source(fork, name, pin):
    return subprocess.check_output(["git", "-C", str(fork), "show", f"{pin}:{name}"], text=True)


def read_getters(source):
    getters = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(value, ast.Lambda):
                    getters[key.value] = value
    return getters


def evaluate(node, key, values):
    """Interpret only literal getenv defaults and their simple conversions."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Call) and not node.keywords:
        func = node.func
        if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.value.id == "os" and func.attr == "getenv"):
            args = [evaluate(arg, key, values) for arg in node.args]
            if args[0] != key:
                raise ValueError("default depends on another environment key")
            return values.get(key, args[1] if len(args) == 2 else None)
        if isinstance(func, ast.Name) and func.id in {"bool", "int", "float", "str"}:
            cast = {"bool": bool, "int": int, "float": float, "str": str}[func.id]
            return cast(evaluate(node.args[0], key, values))
        if isinstance(func, ast.Attribute) and func.attr == "strip" and not node.args:
            return evaluate(func.value, key, values).strip()
    raise ValueError("unsupported default expression")


def default_equal(getter, key, value):
    try:
        default = evaluate(getter.body, key, {})
        return evaluate(getter.body, key, {key: value}) == default, default
    except (TypeError, ValueError):
        return False, None


def dry_run(root, layout, serve=None):
    """(environment, command, stdout) of the launch under DRY=1, from a clean
    environment: only PATH, the layout and the DRY switch go in."""
    serve = Path(serve) if serve else root / "serve.sh"
    result = subprocess.run(["bash", str(serve)], cwd=root, text=True, capture_output=True,
                            env={"PATH": os.defpath, "LAYOUT": layout, "DRY": "1"})
    if result.returncode:
        raise ValueError(f"dry run failed ({result.returncode}): {result.stderr.strip()}")
    env, argv = {}, None
    for line in result.stdout.splitlines():
        if line.startswith(ARGV_LINE):
            argv = shlex.split(line[len(ARGV_LINE):])
            break
        match = NAME.match(line)
        if match:
            env[match[1]] = match[2]
    if argv is None:
        raise ValueError("dry run printed no DRY_RUN_ARGV_SHELL line")
    for key in ("DRY", "LAYOUT", "_", "SHLVL", "PWD", "OLDPWD"):
        env.pop(key, None)
    return env, parse_command(argv), result.stdout


def parse_command(command):
    if len(command) < 3 or command[1] != "serve":
        raise ValueError("expected vllm serve command")
    values = {"model": command[2]}
    index = 3
    while index < len(command):
        token = command[index]
        if not token.startswith("--"):
            raise ValueError(f"unexpected command token {token!r}")
        key = token[2:].replace("-", "_")
        if key in values:
            raise ValueError(f"duplicate command argument {key}")
        if index + 1 == len(command) or command[index + 1].startswith("--"):
            value = True
        else:
            index += 1
            value = command[index]
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        values[key] = [value] if key == "served_model_name" else value
        index += 1
    return values


def log_command(path):
    commands = []
    for line in path.read_text().splitlines():
        if "non-default args:" not in line:
            continue
        text = ENUM.sub(lambda m: repr(m[1]), line.split("non-default args:", 1)[1].strip())
        try:
            commands.append(ast.literal_eval(text))
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"serve.log non-default args record is not a literal: {exc}") from None
    if not commands or any(command != commands[0] for command in commands):
        raise ValueError("serve.log must contain one consistent non-default args record")
    return commands[0]


def without_paths(command):
    command = dict(command)
    for key in sorted(IGNORED_ARGS):
        command.pop(key, None)
    overrides = command.get("hf_overrides")
    if isinstance(overrides, dict):
        command["hf_overrides"] = {k: v for k, v in overrides.items() if k not in PATH_OVERRIDES}
    return command


def compare_commands(actual, expected):
    actual, expected = without_paths(actual), without_paths(expected)
    print("command: allowed " + ", ".join(sorted(IGNORED_ARGS)) + " and hf_overrides."
          + ", hf_overrides.".join(sorted(PATH_OVERRIDES)) + " (locations)")
    failures = []
    for key in sorted(set(actual) | set(expected)):
        mine, theirs = actual.get(key), expected.get(key)
        if key in RESOLVED_CONFIGS and isinstance(mine, dict) and isinstance(theirs, dict):
            differ = sorted(k for k in mine if k not in theirs or theirs[k] != mine[k])
            if differ:
                failures.append(f"command: {key}: " + ", ".join(
                    f"{k}: validation={theirs.get(k)!r}, recipe={mine[k]!r}" for k in differ))
            else:
                print(f"command: MATCH {key} on the keys the launch sets: {', '.join(sorted(mine))}")
            continue
        if key not in expected and key in ENGINE_DEFAULT_ARGS and mine == ENGINE_DEFAULT_ARGS[key]:
            print(f"command: allowed {key}={mine!r} (vLLM's default, which its non-default args record omits)")
            continue
        if key not in actual or key not in expected or mine != theirs:
            failures.append(f"command: {key}: validation={theirs!r}, recipe={mine!r}")
        else:
            print(f"command: MATCH {key}={mine!r}")
    return failures


def compare_environment(actual, expected, getters, engine):
    """The environment mismatches of the recipe's launch (ACTUAL) against a validation leg's (EXPECTED), with the
    released ENGINE (release_engine) and its GETTERS (engine_getters)."""
    at = dict(engine["sources"])
    failures, matched, allowed = [], 0, 0
    for key in sorted(set(actual) | set(expected)):
        if key in actual and key in expected and actual[key] == expected[key]:
            print(f"environment: MATCH {key}" + (" (path)" if key in PATH_KEYS else f"={actual[key]!r}"))
            matched += 1
            continue
        reason = LAB_KEYS.get(key)
        if key in PATH_KEYS:
            reason = "installation, checkpoint, cache or executable search path"
        elif key.startswith(IMAGE_PREFIXES):
            reason = "base-image metadata"
        elif key in RECIPE_KEYS and key not in expected:
            reason = RECIPE_KEYS[key]
        elif key in actual and key in expected and same_tree_file(actual[key], expected[key], expected,
                                                                  engine["image"]):
            reason = f"the same engine file, {actual[key]}"
        # A name on one side only can be equivalent only through the pinned fork getter.
        if reason is None and ((key in actual) != (key in expected)) and key in getters:
            value = actual[key] if key in actual else expected[key]
            filename, source, getter = getters[key]
            equal, default = default_equal(getter, key, value)
            if equal:
                expression = " ".join(ast.get_source_segment(source, getter).split())
                reason = (f"explicit {value!r} equals fork default {default!r}; "
                          f"{filename}:{getter.lineno} at {at[filename][:10]}: {expression}")
        if reason is not None:
            print(f"environment: allowed {key} ({reason})")
            allowed += 1
        else:
            failures.append(f"environment: {key}: validation={expected.get(key)!r}, recipe={actual.get(key)!r}")
    print(f"environment: {matched} matches; {allowed} allowed differences; {len(failures)} mismatches")
    return failures


def same_tree_file(mine, theirs, expected, image):
    """The image's file MINE and the same file THEIRS of the validation leg's engine: IMAGE's code directories (the
    engine tree; an assembled engine's engine and plugin) and the leg's PYTHONPATH entries, one for one."""
    trees = [t.rstrip("/") for t in expected.get("PYTHONPATH", "").split(":")]
    if len(trees) != len(image) or not all(t.startswith("/") for t in trees):
        return False
    for place, tree in zip(image, trees):
        if mine.startswith(place + "/"):
            return theirs.startswith(tree + "/") and mine[len(place):] == theirs[len(tree):]
    return False


def check_effective(root, path, layout, fork, serve_log=None, serve=None):
    expected = json.loads(path.read_text())
    if not isinstance(expected, dict) or not all(isinstance(v, str) for v in expected.values()):
        raise ValueError("effective environment must be a complete string-valued JSON object")
    engine = release_engine(root)
    getters = engine_getters(fork, engine)
    actual, command, _ = dry_run(root, layout, serve)
    logged = log_command(serve_log or path.with_name("serve.log"))
    print(f"Full environment: layout={layout}; DRY=1; fork={engine['label']}")
    failures = compare_environment(actual, expected, getters, engine)
    command_failures = compare_commands(command, logged)
    print(f"command: {len(command_failures)} mismatches")
    failures.extend(command_failures)
    if failures:
        print("\n".join(failures))
    print(f"Full environment match: {'FAIL' if failures else 'PASS'} ({layout}; {len(failures)} mismatches)")
    return int(bool(failures))
