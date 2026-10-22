#!/usr/bin/env python3
"""Validate every mount destination in a read-only rootfs launch, without CUDA."""
import argparse
from pathlib import Path, PurePosixPath


# Fail closed when the launch adds an option whose semantics are not covered.
ZERO = {"--clearenv", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
        "--unshare-all", "--die-with-parent"}
ONE = {"--hostname", "--size", "--chdir"}
MOUNTS = {"--ro-bind", "--bind", "--dev-bind"}
NEW = {"--tmpfs", "--proc", "--dev", "--dir"}


def root_path(rootfs, target):
    parts = list(PurePosixPath(target).parts[1:])
    current = rootfs
    links = 0
    while parts:
        part = parts.pop(0)
        if part in ("", "."):
            continue
        if part == "..":
            if current == rootfs:
                raise ValueError("destination escapes rootfs: " + target)
            current = current.parent
            continue
        current = current / part
        if current.is_symlink():
            links += 1
            if links > 40:
                raise ValueError("destination symlink loop: " + target)
            link = current.readlink()
            current = rootfs if link.is_absolute() else current.parent
            parts = list(link.parts[1:] if link.is_absolute() else link.parts) + parts
    return current


def validate(rootfs, argv):
    rootfs = rootfs.resolve()
    overlays = []
    mounts = []
    def writable_at(target):
        path = PurePosixPath(target)
        for prefix, writable in reversed(overlays):
            if path == prefix or prefix in path.parents:
                return writable
        return False
    def destination(target):
        if not target.startswith("/") or ".." in PurePosixPath(target).parts:
            raise ValueError("invalid mount destination: " + target)
        if not writable_at(target) and not root_path(rootfs, target).exists():
            raise ValueError("missing read-only rootfs mount target: " + target)
        mounts.append(target)
    i = 0
    while i < len(argv):
        option = argv[i]
        i += 1
        if option in ("setsid", "bwrap"):
            continue
        if option in ZERO:
            continue
        if option in ONE:
            i += 1
        elif option in MOUNTS:
            source, target = argv[i:i+2]
            if not Path(source).exists():
                raise ValueError("missing mount source: " + source)
            destination(target)
            overlays.append((PurePosixPath(target), False))
            i += 2
        elif option in NEW:
            target = argv[i]
            destination(target)
            overlays.append((PurePosixPath(target), True))
            i += 1
        elif option == "--symlink":
            destination(argv[i+1])
            i += 2
        elif option == "--setenv":
            i += 2
        elif option.startswith("--"):
            raise ValueError("unsupported bwrap option: " + option)
        else:
            break
    return mounts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootfs", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    targets = validate(args.rootfs, command)
    print(f"BWRAP TARGETS PASS: {len(targets)} mounts")
