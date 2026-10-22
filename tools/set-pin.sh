#!/usr/bin/env bash
# Move the current release pin in defaults, links and documentation together.
# Earlier changelog entries retain their historical engine and image pins.
set -euo pipefail
if [ "$#" -ne 1 ]; then
    echo "usage: tools/set-pin.sh <40-hex commit>" >&2
    exit 2
fi
RECIPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 - "$RECIPE_ROOT" "$1" <<'PYTHON'
from pathlib import Path
import re
import subprocess
import sys

root, pin = Path(sys.argv[1]), sys.argv[2]
if not re.fullmatch(r"[0-9a-fA-F]{40}", pin):
    sys.exit("pin must be a full 40-hex commit, never a branch or short hash")
pin = pin.lower()
slot = chr(123) * 2 + "PIN" + chr(125) * 2
patterns = {
    "start.sh": r"^RELEASE_VLLM_COMMIT=[\"']?([^\"'\n]+)",
    "docker/build.sh": r"^RELEASE_VLLM_COMMIT=[\"']?([^\"'\n]+)",
    "docker/Dockerfile": r"^ARG VLLM_COMMIT=(.+)$",
}
texts = {}
for name, pattern in patterns.items():
    text = (root / name).read_text()
    matches = re.findall(pattern, text, re.M)
    if len(matches) != 1:
        sys.exit("expected one release pin in " + name)
    texts[name] = matches[0]
values = set(texts.values())
if len(values) != 1:
    sys.exit("release pins disagree; no files changed")
old = values.pop()
if old != slot and not re.fullmatch(r"[0-9a-f]{40}", old):
    sys.exit("existing release pin is neither a full commit nor the release slot")
files = subprocess.check_output(
    ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
).decode().split("\0")
updates = {}
for name in sorted(set(files) - {"", "tools/set-pin.sh"}):
    path = root / name
    if not path.is_file() or path.is_symlink():
        continue
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        continue
    current, history = text, ""
    if name == "CHANGELOG.md":
        sections = list(re.finditer(r"^## ", text, re.M))
        if len(sections) > 1:
            end = sections[1].start()
            current, history = text[:end], text[end:]
    # Image tags carry ten characters (docker/image_identity.py), also from the slot.
    changed = current.replace("engine-" + old, "engine-" + pin[:10]).replace(old, pin)
    if old != slot:
        changed = re.sub(r"(?<![0-9a-f])" + old[:10] + r"(?![0-9a-f])", pin[:10], changed)
    if changed != current:
        updates[path] = changed + history
for path, text in updates.items():
    path.write_text(text)
for name, pattern in patterns.items():
    if re.findall(pattern, (root / name).read_text(), re.M) != [pin]:
        sys.exit("pin verification failed in " + name)
print("Current release pin: " + pin)
print("grep -nF proof (historical changelog entries are preserved):")
for name in sorted(set(files) - {"", "tools/set-pin.sh"}):
    path = root / name
    if not path.is_file() or path.is_symlink():
        continue
    result = subprocess.run(["grep", "-nIF", "--", pin, str(path.relative_to(root))],
                            cwd=root, text=True, capture_output=True)
    if result.returncode not in (0, 1):
        sys.exit("grep failed for " + name)
    for line in result.stdout.splitlines():
        print(name + ":" + line)
PYTHON
