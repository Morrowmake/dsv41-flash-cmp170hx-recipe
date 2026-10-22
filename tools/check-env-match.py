#!/usr/bin/env python3
"""Compare the recipe's launch (serve.sh under DRY=1, on CPU) with a validation environment.

  check-env-match.py REFERENCE.json
      REFERENCE holds {layout: {NAME: value}}: every name must be what the
      launch exports (or, for the launch arguments below, what its command
      sets); path-valued names are reported, not compared.
  check-env-match.py --effective ENV.json --layout tp4 --fork FORK [--serve-log LOG]
      ENV.json is a validation leg's complete server environment, LOG its
      serve.log (default: beside ENV.json); see tools/env_effective.py.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import env_effective as E  # noqa: E402

# Launch inputs a reference may name, compared with the command the launch runs.
ARGUMENTS = {"MAX_SEQS": "max_num_seqs", "MAX_BATCHED": "max_num_batched_tokens",
             "MAX_LEN": "max_model_len", "GPU_UTIL": "gpu_memory_utilization"}


def check(reference, serve=None):
    failures = []
    metadata = sorted(key for key in reference if key.startswith("_"))
    print("Ignored metadata: " + (", ".join(metadata) or "none"))
    layouts = sorted(key for key in reference if not key.startswith("_"))
    if layouts != list(E.LAYOUTS):
        print(f"Environment match: FAIL (reference layouts {layouts}, recipe layouts {list(E.LAYOUTS)})")
        return 1
    for layout in layouts:
        expected = reference[layout]
        actual, command, _ = E.dry_run(ROOT, layout, serve)
        ignored, compared = {}, 0
        for key, value in sorted(expected.items()):
            if key in ARGUMENTS:
                got = command.get(ARGUMENTS[key])
                if str(got) != str(value):
                    failures.append(f"{layout}: {key}: expected {value!r}, command has {got!r}")
                compared += 1
            elif isinstance(value, str) and value.startswith(("/", "~/")):
                ignored[key] = "path valued"
            elif actual.get(key) != str(value):
                failures.append(f"{layout}: {key}: expected {value!r}, got {actual.get(key)!r}")
                compared += 1
            else:
                compared += 1
        for key, reason in sorted(ignored.items()):
            print(f"{layout}: ignored {key} ({reason})")
        extras = sorted(set(actual) - set(expected))
        print(f"{layout}: exports the reference does not name: " + (", ".join(extras) or "none"))
        print(f"{layout}: compared {compared} settings")
    if failures:
        print("Environment match: FAIL\n" + "\n".join(failures))
        return 1
    print("Environment match: PASS (" + ", ".join(L.upper() for L in layouts) + ")")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path, nargs="?", help="validation JSON path")
    parser.add_argument("--effective", type=Path, help="full effective server environment JSON")
    parser.add_argument("--layout", choices=E.LAYOUTS)
    parser.add_argument("--fork", type=Path, help="fork checkout holding the pinned commit")
    parser.add_argument("--serve-log", type=Path, help="default: serve.log beside the effective JSON")
    parser.add_argument("--serve", type=Path, help="the launch to check (default: this checkout's serve.sh)")
    args = parser.parse_args()
    try:
        if args.effective:
            if args.reference or not args.layout or not args.fork:
                parser.error("--effective requires --layout and --fork and cannot take a reference")
            return E.check_effective(ROOT, args.effective, args.layout, args.fork, args.serve_log, args.serve)
        if not args.reference or args.layout or args.serve_log:
            parser.error("supply a reference, or --effective with --layout and --fork")
        return check(json.loads(args.reference.read_text()), args.serve)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"{'Full environment' if args.effective else 'Environment'} match: FAIL ({exc})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
