#!/usr/bin/env python3
"""Bind engine image tags to the full source revision in OCI labels."""
import argparse
import json
import re
import sys


def engine_tag(commit):
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Engine revision must be a full 40-character commit")
    return "engine-" + commit[:10]


def target_tag(target):
    repository, separator, tag = target.rpartition(":")
    if (not separator or not repository or "@" in target or
            any(c.isspace() for c in target) or
            not re.fullmatch(r"engine-[0-9a-f]{10}", tag)):
        raise ValueError("Image target must use an engine-<10-character-commit> tag")
    return tag


def check_config(config, target):
    labels = config["config"].get("Labels") or {}
    commit = labels.get("org.opencontainers.image.revision")
    tag = engine_tag(commit)
    if target_tag(target) != tag:
        raise ValueError("Image target differs from the full engine revision")
    if labels.get("org.opencontainers.image.version") != tag:
        raise ValueError("OCI version differs from the engine tag")
    return commit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tag", "target", "inspect"))
    parser.add_argument("value")
    args = parser.parse_args()
    try:
        if args.mode == "tag":
            print(engine_tag(args.value))
        elif args.mode == "target":
            print(target_tag(args.value))
        else:
            from layer_count import check, from_inspect
            inspection = json.load(sys.stdin)
            count = from_inspect(inspection)
            check_config({"config": inspection[0]["Config"]}, args.value)
            check(count)
    except (ValueError, KeyError, TypeError) as error:
        print("[image-identity] FAIL: " + str(error), file=sys.stderr)
        sys.exit(1)
