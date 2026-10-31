#!/usr/bin/env python3
"""Install an assembled engine in the image's venv from the engine's own files (nothing is built), as an installed
engine is: /opt/venv/bin/vllm is the engine's console script, and /opt/venv/bin/python imports the engine and its
plugin and reads vllm's distribution (version, entry points) with no PYTHONPATH. The release's serve.sh clears
PYTHONPATH and execs /opt/venv/bin/vllm. Writes, under OPT (the stage's /opt):

  venv/lib/python3.X/site-packages/vllm-assembly.pth     the engine and plugin directories, /opt/assembly/engine
                                                         and /opt/assembly/plugin, on the venv's path
  venv/lib/python3.X/site-packages/<name>-<version>.dist-info/
                                                         METADATA, entry_points.txt and RECORD: the name and the
                                                         entry points of engine/pyproject.toml, the version of
                                                         engine/vllm/_version.py
  venv/bin/<script>                                      each [project.scripts] script as uv writes one in a
                                                         relocatable venv

A venv that already holds the engine's package or distribution is refused: it would shadow the assembly's.
Every build of the release image runs this same file on its stage."""
import argparse
import ast
import base64
import hashlib
import os
from pathlib import Path
import re
import sys
import tomllib

ROOT = "/opt/assembly"
PATHS = (ROOT + "/engine", ROOT + "/plugin")
PTH = "vllm-assembly.pth"
PLUGIN_PACKAGE = "vllm_sm80"
# uv's console script in a relocatable venv: the interpreter beside the script
SCRIPT = """#!/bin/sh
'''exec' "$(dirname -- "$(realpath -- "$0")")"/'python3' "$0" "$@"
' '''
# -*- coding: utf-8 -*-
import sys
from {module} import {head}
if __name__ == "__main__":
    if sys.argv[0].endswith("-script.pyw"):
        sys.argv[0] = sys.argv[0][:-11]
    elif sys.argv[0].endswith(".exe"):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit({call}())
"""
ENTRY_POINT = re.compile(r"([\w.]+):([\w.]+)")


def engine_version(engine):
    """The version engine/vllm/_version.py assigns to `version` (the queue commit's)."""
    path = engine / "vllm/_version.py"
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and \
                isinstance(node.value.value, str) and any(isinstance(t, ast.Name) and t.id == "version"
                                                          for t in node.targets):
            return node.value.value
    raise ValueError(f"{path} assigns no version string")


def script(value):
    match = ENTRY_POINT.fullmatch(value)
    if not match:
        raise ValueError(f"console script {value!r} is not module:attribute")
    module, call = match.groups()
    return SCRIPT.format(module=module, head=call.split(".")[0], call=call).encode()


def entry_points(project):
    """entry_points.txt of a [project] table: its scripts as console_scripts, then each entry-points group."""
    if project.get("gui-scripts"):
        raise ValueError("engine/pyproject.toml declares gui-scripts: not an image entry")
    groups = [("console_scripts", project.get("scripts") or {})]
    groups += list((project.get("entry-points") or {}).items())
    return "".join("[%s]\n%s\n" % (g, "".join("%s = %s\n" % kv for kv in t.items())) for g, t in groups if t)


def record_line(rel, data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{rel},sha256={digest},{len(data)}\n"


def install(opt):
    """Write the engine install under OPT (module doc); {path below OPT: bytes} written."""
    opt = Path(opt)
    engine = opt / "assembly/engine"
    project = tomllib.loads((engine / "pyproject.toml").read_text())["project"]
    name, version = project["name"], engine_version(engine)
    scripts = project.get("scripts") or {}
    if "vllm" not in scripts:
        raise ValueError("engine/pyproject.toml declares no vllm script")
    sites = sorted(opt.glob("venv/lib/python3.*/site-packages"))
    if len(sites) != 1:
        raise ValueError(f"{opt}/venv holds {len(sites)} site-packages directories, not one")
    site = sites[0]
    dist = re.sub(r"[-_.]+", "_", name).lower()
    held = sorted(p.name for p in site.iterdir() if p.name in (name, dist, PLUGIN_PACKAGE, PTH)
                  or re.fullmatch(rf"{re.escape(dist)}-.+\.(dist|egg)-info", p.name, re.I))
    if held:
        raise ValueError(f"{site} already holds {', '.join(held)}: it would shadow the assembly's engine")
    info = f"{dist}-{version}.dist-info"
    files = {f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode(),
             f"{info}/entry_points.txt": entry_points(project).encode(),
             PTH: "".join(p + "\n" for p in PATHS).encode()}
    bins = {f"../../../bin/{s}": script(v) for s, v in scripts.items()}
    files.update(bins)
    lines = sorted(record_line(rel, data) for rel, data in files.items()) + [f"{info}/RECORD,,\n"]
    files[f"{info}/RECORD"] = "".join(lines).encode()
    paths = {rel: Path(os.path.normpath(site / rel)) for rel in files}
    present = sorted(str(p) for p in paths.values() if p.exists())
    if present:
        raise ValueError(f"{', '.join(present)} exists: the venv is not the dependency lock's alone")
    (site / info).mkdir()
    (site / info).chmod(0o755)
    for rel, path in paths.items():
        path.write_bytes(files[rel])
        path.chmod(0o755 if rel in bins else 0o644)
    return {str(path.relative_to(opt)): files[rel] for rel, path in paths.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("opt", type=Path, help="the stage's /opt (venv/ and assembly/engine/ beside each other)")
    a = p.parse_args(argv)
    try:
        written = install(a.opt)
    except (OSError, KeyError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"venv_entry: {error}", file=sys.stderr)
        return 1
    for rel in sorted(written):
        print(f"venv_entry: /opt/{rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
