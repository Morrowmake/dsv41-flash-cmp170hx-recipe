#!/usr/bin/env python3
"""Bind engine image tags to the full source revision in OCI labels."""
import argparse
import json
import re
import sys


# An assembled engine (upstream + queue + plugin) is named by all three pins: a later release that changes only
# the plugin on the same queue gets a new tag.
ASSEMBLY_LABELS = ("org.morrowmake.upstream-commit", "org.morrowmake.queue-commit", "org.morrowmake.plugin-commit")


def _full(commit):
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Engine revision must be a full 40-character commit")
    return commit


def engine_tag(commit, upstream=None, plugin=None):
    """engine-<commit 10>; an assembly (UPSTREAM and PLUGIN given, COMMIT its queue): engine-<u10>-<q10>-<p10>."""
    if upstream is None and plugin is None:
        return "engine-" + _full(commit)[:10]
    return "engine-" + "-".join(_full(c)[:10] for c in (upstream, commit, plugin))


def target_tag(target):
    repository, separator, tag = target.rpartition(":")
    if (not separator or not repository or "@" in target or
            any(c.isspace() for c in target) or
            not re.fullmatch(r"engine-[0-9a-f]{10}(-[0-9a-f]{10}-[0-9a-f]{10})?", tag)):
        raise ValueError("Image target must use an engine-<10-character-commit> tag "
                         "(an assembly: engine-<upstream 10>-<queue 10>-<plugin 10>)")
    return tag


def check_config(config, target):
    labels = config["config"].get("Labels") or {}
    commit = labels.get("org.opencontainers.image.revision")
    if any(k in labels for k in ASSEMBLY_LABELS):
        upstream, queue, plugin = (labels.get(k) for k in ASSEMBLY_LABELS)
        if queue != commit:
            raise ValueError("Assembly queue label differs from the engine revision")
        tag = engine_tag(commit, upstream, plugin)
    else:
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
