import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tomllib

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import image_identity
import oci_layers as oci
from test_release_image import PIN, base, pin_of, publication_image

TAG = "engine-" + PIN[:10]
TARGET = "ghcr.io/morrowmake/vllm-cmp170hx:" + TAG


def build_pins():
    build = (ROOT / "build.sh").read_text()
    return {k.lower(): v for k, v in re.findall(r"^RELEASE_([A-Z0-9_]+)='([^']*)'$", build, re.M)}


def test_rootless_tag_names_the_three_pins_and_ignores_recipe_version():
    prelude = (ROOT / "build.sh").read_text().split("\nPIP=", 1)[0]
    prelude = prelude.replace('HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
                              "HERE=" + shlex.quote(str(ROOT)))
    run = subprocess.run(["bash", "-c", prelude + '\nprintf "%s\n" "$TAG"\n'],
                         env=dict(os.environ, RELEASE="9.9.9"), capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    pins = build_pins()
    assert run.stdout.strip() == "engine-" + "-".join(
        pins[k][:10] for k in ("upstream_commit", "queue_commit", "plugin_commit"))


def test_assembly_tag_is_bound_to_its_three_pin_labels(tmp_path):
    """An assembly's tag names upstream, queue and plugin; the queue-only tag, a tag of another plugin or a
    target whose labels disagree are refused, before compression and at the target check."""
    upstream, queue, plugin = "c" * 40, PIN, "d" * 40
    labels = {"org.morrowmake.upstream-commit": upstream, "org.morrowmake.queue-commit": queue,
              "org.morrowmake.plugin-commit": plugin}
    tag = "engine-" + "-".join(c[:10] for c in (upstream, queue, plugin))
    assert image_identity.engine_tag(queue, upstream, plugin) == tag
    assert image_identity.target_tag("fixture:" + tag) == tag
    payload = tmp_path / "payload"
    for name in ("python", "dependencies", "native", "engine"):
        (payload / name).mkdir(parents=True)
    b = base(tmp_path)
    for wrong in ("engine-" + queue[:10], "engine-" + "-".join(c[:10] for c in (upstream, queue, "e" * 40))):
        with pytest.raises(ValueError, match="tag differs"):
            oci.assemble(b, payload, tmp_path / "out/oci", queue, wrong, {}, pin_of(b), parallel=False,
                         extra_labels=labels)
    oci.assemble(b, payload, tmp_path / "out/oci", queue, tag, {}, pin_of(b), parallel=False, extra_labels=labels)
    config = json.loads((tmp_path / "out/config.json").read_text())
    assert config["config"]["Labels"]["org.opencontainers.image.version"] == tag
    assert config["config"]["Labels"]["org.opencontainers.image.revision"] == queue
    assert image_identity.check_config(config, "fixture:" + tag) == queue
    with pytest.raises(ValueError, match="differs"):
        image_identity.check_config(config, "fixture:engine-" + queue[:10])
    config["config"]["Labels"]["org.morrowmake.queue-commit"] = "f" * 40
    with pytest.raises(ValueError, match="queue label differs"):
        image_identity.check_config(config, "fixture:" + tag)


def test_build_sh_pins_are_the_recipe_s_and_the_assembly_s():
    """build.sh's pins: the recipe contract's [pins] and the release's assembly.json."""
    pins = build_pins()
    for key in ("upstream_commit", "queue_commit", "plugin_commit"):
        assert re.fullmatch(r"[0-9a-f]{40}", pins[key]), key
    contract = tomllib.loads((ROOT.parent / "recipe.toml").read_text())["pins"]
    record = json.loads((ROOT / "assembly.json").read_text())
    for key in ("upstream_commit", "queue_commit", "plugin_commit", "native_build"):
        assert pins[key] == contract[key] == record[key], key
    assert pins["native_manifest_sha256"] == record["native_manifest_sha256"]


@pytest.mark.parametrize("tag", ["1.7.2-" + PIN[:10], "engine-" + "b" * 10, TAG + "+build1"])
def test_assembly_rejects_unbound_tag_before_compression(tmp_path, tag):
    out = tmp_path / "out/oci"
    payload = tmp_path / "payload"
    for name in ("python", "dependencies", "native", "engine"):
        (payload / name).mkdir(parents=True)
    b = base(tmp_path)
    with pytest.raises(ValueError, match="tag differs"):
        oci.assemble(b, payload, out, PIN, tag, {}, pin_of(b), parallel=False)
    assert not out.exists()


@pytest.mark.parametrize("target", ["fixture-image", "fixture-image:1.7.2-" + PIN[:10],
                                    "fixture-image:" + TAG + "+build1"])
def test_docker_rejects_non_engine_target_before_build(tmp_path, target):
    fake = tmp_path / "docker"
    calls = tmp_path / "calls"
    fake.write_text('#!/bin/sh\nprintf "%s\n" "$*" >> "$CALL_LOG"\n'
                    'if [ "$1" = image ]; then echo ' +
                    shlex.quote('[{"RootFS":{"Layers":[]}}]') + '; fi\n')
    fake.chmod(0o755)
    run = subprocess.run(["bash", str(ROOT / "build-docker.sh"), target],
                         env=dict(os.environ, PATH=str(tmp_path) + ":" + os.environ["PATH"],
                                  CALL_LOG=str(calls)), capture_output=True, text=True)
    assert run.returncode != 0
    assert not calls.exists()


@pytest.mark.parametrize("revision", ["b" * 40, PIN[:10]])
def test_docker_rejects_inspected_revision_mismatch(tmp_path, revision):
    fake = tmp_path / "docker"
    inspection = [dict(RootFS=dict(Layers=[]), Config=dict(Labels={
        "org.opencontainers.image.revision": revision,
        "org.opencontainers.image.version": TAG}))]
    fake.write_text('#!/bin/sh\nif [ "$1" = image ]; then echo ' +
                    shlex.quote(json.dumps(inspection)) + '; fi\n')
    fake.chmod(0o755)
    run = subprocess.run(["bash", str(ROOT / "build-docker.sh"), "fixture-image:" + TAG],
                         env=dict(os.environ, PATH=str(tmp_path) + ":" + os.environ["PATH"]),
                         capture_output=True, text=True)
    assert run.returncode != 0
    assert "[image-identity] FAIL" in run.stderr


@pytest.mark.parametrize("case", ["legacy-tag", "wrong-tag", "invalid-suffix", "missing-revision",
                                  "short-revision", "wrong-version", "missing-acceptance-commit",
                                  "same-prefix-different-commit"])
def test_push_rejects_unbound_engine_identity_before_auth(tmp_path, case):
    labels = {"org.opencontainers.image.revision": PIN, "org.opencontainers.image.version": TAG}
    target = TARGET
    accepted_commit = PIN
    if case == "legacy-tag":
        target = TARGET.replace(TAG, "1.7.2-" + PIN[:10])
    elif case == "wrong-tag":
        target = TARGET.replace(TAG, "engine-" + "b" * 10)
    elif case == "invalid-suffix":
        target += "+build1"
    elif case == "missing-revision":
        labels.pop("org.opencontainers.image.revision")
    elif case == "short-revision":
        labels["org.opencontainers.image.revision"] = PIN[:10]
    elif case == "wrong-version":
        labels["org.opencontainers.image.version"] = "1.7.2"
    elif case == "same-prefix-different-commit":
        accepted_commit = PIN[:10] + "b" * 30
    image = publication_image(tmp_path, labels)
    digest = json.loads((image / "index.json").read_text())["manifests"][0]["digest"]
    record = dict(image_digest=digest, passed=True, artifact_audit_passed=True)
    if case != "missing-acceptance-commit":
        record["engine_commit"] = accepted_commit
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(json.dumps(record))
    fake = tmp_path / "crane"
    calls = tmp_path / "calls"
    fake.write_text('#!/bin/sh\nprintf "%s\n" "$*" >> "$CALL_LOG"\n'
                    'if [ "$1" = auth ]; then cat >/dev/null; fi\n'
                    'if [ "$1" = digest ]; then printf "%s\n" "$EXPECTED"; fi\n')
    fake.chmod(0o755)
    run = subprocess.run(["bash", str(ROOT / "push.sh"), str(image), target, digest, str(acceptance)],
                         input="fixture-private-auth-value", capture_output=True, text=True,
                         env=dict(os.environ, IMAGE_PUBLISH="1", CRANE=str(fake),
                                  CALL_LOG=str(calls), EXPECTED=digest))
    assert run.returncode != 0
    assert not calls.exists()
