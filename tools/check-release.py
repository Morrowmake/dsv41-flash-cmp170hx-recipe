#!/usr/bin/env python3
"""Check release wording and commit identity without contacting a remote."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Encoded screening terms prevent the scanner from matching its own literals.
PATTERN = re.compile(bytes.fromhex("616861726f6e697c616861726f627c657974616e7c626563686f727c2f686f6d652f62617c2f55736572732f7c657079637c6261405b612d7a5d7c7569643d5c642b5c2862615c297c7e62612f7c636f2d617574686f7265647c636c617564657c636f6465787c6b6f70747c6175746f2d6770757c63616d706169676e7c617373697374616e747c5c626167656e74733f5c62").decode(), re.I)
IDENTITY = "Morrowmake <noreply@github.com>|Morrowmake <noreply@github.com>"
# This exact frozen dependency line is a third-party package name.
ALLOWLIST = {("docker/CONSTRAINTS", bytes.fromhex("6167656e742d6465746563746f723d3d322e302e30").decode())}


def allowed(name, line):
    return (name, line) in ALLOWLIST


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)


def main():
    release_range = sys.argv[1] if len(sys.argv) > 1 else "next-release..HEAD"
    failures = []
    commits = git("rev-list", release_range).splitlines()
    for commit in commits:
        identity = git("show", "-s", "--format=%an <%ae>|%cn <%ce>", commit).strip()
        if identity != IDENTITY:
            failures.append(commit[:10] + ": identity")
        message = git("show", "-s", "--format=%B", commit)
        if PATTERN.search(message) or re.search(r"^[A-Za-z-]+: .+", message, re.M):
            failures.append(commit[:10] + ": message")
        name = ""
        for line in git("show", "--format=", "--unified=0", commit).splitlines():
            if line.startswith("+++ b/"):
                name = line[6:]
            elif line.startswith("+") and not line.startswith("+++"):
                if PATTERN.search(line[1:]) and not allowed(name, line[1:]):
                    failures.append(commit[:10] + ": added line in " + name)
    name = ""
    for line in git("diff", "next-release", "--unified=0").splitlines():
        if line.startswith("+++ b/"):
            name = line[6:]
        elif line.startswith("+") and not line.startswith("+++"):
            if PATTERN.search(line[1:]) and not allowed(name, line[1:]):
                failures.append("release diff: " + name)
    exceptions = 0
    files = set(git("ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0")) - {""}
    for name in sorted(files):
        path = ROOT / name
        if not path.is_file():
            continue
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if PATTERN.search(line):
                if allowed(name, line):
                    exceptions += 1
                else:
                    failures.append(f"{name}:{number}")
    if failures:
        print("Release scan: FAIL")
        print("\n".join(failures))
        return 1
    print(f"Release scan: PASS ({len(commits)} commits; {len(files)} files; "
          f"{exceptions} exact package-line exception; zero unreviewed hits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
