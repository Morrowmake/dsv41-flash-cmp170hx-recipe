import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import layer_count
import oci_layers as oci


def base(tmp_path, count):
    root = tmp_path/'base'
    (root/'blobs/sha256').mkdir(parents=True)
    desc = oci.put(root, b'tiny-base-layer', oci.OCI+'layer.v1.tar+gzip')
    config = dict(architecture='amd64', os='linux', config={},
                  rootfs=dict(type='layers', diff_ids=[desc['digest']] * count))
    cd = oci.put(root, oci.encoded(config), oci.OCI+'config.v1+json')
    md = oci.put(root, oci.encoded(dict(schemaVersion=2, config=cd, layers=[desc]*count)),
                 oci.OCI+'manifest.v1+json')
    (root/'index.json').write_text(json.dumps(dict(manifests=[md])))
    return root


@pytest.mark.parametrize('base_count,layered,seed,passed', [
    (119, True, False, True), (120, True, False, False),
    (118, True, True, True), (119, True, True, False),
    (122, False, True, True), (123, False, False, False)])
def test_total_layers_before_compression(tmp_path, monkeypatch, capsys,
                                         base_count, layered, seed, passed):
    monkeypatch.setenv('IMAGE_LAYER_COUNT_CHECK', '1')
    b = base(tmp_path, base_count)
    payload = tmp_path/'payload'
    for name in ['python', 'dependencies', 'native', 'engine'] + (['cache-seed'] if seed else []):
        (payload/name).mkdir(parents=True)
        (payload/name/'data').write_text(name)
    out = tmp_path/'out/oci'
    if passed:
        oci.assemble(b, payload, out, 'a'*40, 'engine-' + 'a'*10, layered=layered, parallel=False)
        manifest, config = oci.image(out)
        assert len(manifest['layers']) == len(config['rootfs']['diff_ids']) == 123
    else:
        def forbidden_pack(*args):
            pytest.fail('compression ran before rejecting layer count')
        monkeypatch.setattr(oci, 'pack', forbidden_pack)
        with pytest.raises(ValueError, match='maximum is 123'):
            oci.assemble(b, payload, out, 'a'*40, 'engine-' + 'a'*10, layered=layered)
        assert not out.exists()
    assert f'count={123 if passed else 124} limit=123 check=1' in capsys.readouterr().out


@pytest.mark.parametrize('count,enabled,success', [(123, '1', True), (124, '1', False), (124, '0', True)])
def test_docker_wrapper_checks_actual_final_layers(tmp_path, count, enabled, success):
    fake = tmp_path/'docker'
    calls = tmp_path/'calls'
    fake.write_text('''#!/usr/bin/env python3
import json,os,sys
with open(os.environ['CALL_LOG'], 'a') as f: f.write(' '.join(sys.argv[1:]) + '\\n')
if sys.argv[1:3] == ['image', 'inspect']:
    print(json.dumps([dict(RootFS=dict(Layers=['sha256:fixture']*int(os.environ['LAYER_COUNT'])),
                          Config=dict(Labels={'org.opencontainers.image.revision': 'a'*40,
                                              'org.opencontainers.image.version': 'engine-'+'a'*10}))]))
''')
    fake.chmod(0o755)
    run = subprocess.run(['bash', str(ROOT/'build-docker.sh'), 'fixture-image:engine-'+'a'*10, '--build-arg', 'IMAGE_LAYERED=0'],
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PATH=str(tmp_path)+':'+os.environ['PATH'],
                                  CALL_LOG=str(calls), LAYER_COUNT=str(count), IMAGE_LAYER_COUNT_CHECK=enabled),
                         capture_output=True, text=True)
    assert (run.returncode == 0) == success, run.stdout + run.stderr
    assert f'count={count} limit=123 check={enabled}' in run.stdout
    assert calls.read_text().splitlines() == [f'build -f {ROOT}/Dockerfile -t fixture-image:engine-{'a'*10} --build-arg IMAGE_LAYERED=0 --build-arg IMAGE_TAG=engine-{'a'*10} {ROOT}/..',
                                             'image inspect fixture-image:engine-'+'a'*10]


def test_layer_guard_switch_and_invalid_inspection(monkeypatch):
    monkeypatch.setenv('IMAGE_LAYER_COUNT_CHECK', '0')
    assert layer_count.check(124) == 124
    monkeypatch.setenv('IMAGE_LAYER_COUNT_CHECK', 'invalid')
    with pytest.raises(ValueError, match='must be 0 or 1'):
        layer_count.check(4)
    with pytest.raises(ValueError):
        layer_count.from_inspect([])
