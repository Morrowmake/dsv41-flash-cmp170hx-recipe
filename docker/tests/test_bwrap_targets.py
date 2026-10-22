import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from bwrap_targets import validate


@pytest.mark.parametrize('layout', ['tp4', 'pp4'])
@pytest.mark.parametrize('arm', ['cold', 'warm'])
def test_missing_readonly_mount_target(tmp_path, layout, arm):
    root = tmp_path / 'root'
    (root / 'cache').mkdir(parents=True)
    source = tmp_path / 'source'
    source.mkdir()
    command = ['setsid', 'bwrap', '--ro-bind', str(root), '/', '--bind', str(source), '/cache',
               '--setenv', 'LAYOUT', layout]
    if arm == 'warm':
        command += ['--ro-bind', str(source), '/opt/cache-seed']
        with pytest.raises(ValueError, match='missing.* /opt/cache-seed'):
            validate(root, command)
        (root / 'opt/cache-seed').mkdir(parents=True)
    assert validate(root, command) == ['/', '/cache'] + (['/opt/cache-seed'] if arm == 'warm' else [])


def test_writable_parent_then_readonly_overlay(tmp_path):
    root = tmp_path / 'root'
    (root / 'tmp').mkdir(parents=True)
    source = tmp_path / 'source'
    source.mkdir()
    assert validate(root, ['--tmpfs', '/tmp', '--ro-bind', str(source), '/tmp/new']) == ['/tmp', '/tmp/new']
    with pytest.raises(ValueError, match='missing.* /tmp/new/missing'):
        validate(root, ['--tmpfs', '/tmp', '--ro-bind', str(source), '/tmp/new',
                        '--ro-bind', str(source), '/tmp/new/missing'])


def test_root_relative_symlinks(tmp_path):
    root = tmp_path / 'root'
    (root / 'usr/bin').mkdir(parents=True)
    (root / 'bin').symlink_to('/usr/bin')
    source = tmp_path / 'source'
    source.write_text('fixture')
    with pytest.raises(ValueError, match='missing.* /bin/true'):
        validate(root, ['--ro-bind', str(source), '/bin/true'])
    (root / 'usr/bin/true').write_text('fixture')
    assert validate(root, ['--ro-bind', str(source), '/bin/true']) == ['/bin/true']


def test_new_options_fail_closed(tmp_path):
    with pytest.raises(ValueError, match='unsupported'):
        validate(tmp_path, ['--future-mount', '/missing'])


@pytest.mark.parametrize('layout', ['tp4', 'pp4'])
def test_exact_gpu_cache_leg_mounts(tmp_path, layout):
    required = ['TEST_GPU_CACHE_LEG', 'TEST_IMAGE_ROOTFS', 'TEST_IMAGE_CONFIG',
                'TEST_TARGET_MODEL', 'TEST_DRAFT_MODEL']
    if not all(os.environ.get(name) for name in required):
        pytest.skip('Set built image and GPU leg paths for exact mount validation')
    recipe = ROOT.parent
    run = subprocess.run(['bash', os.environ['TEST_GPU_CACHE_LEG'], os.environ['TEST_IMAGE_ROOTFS'],
                          str(recipe), os.environ['TEST_TARGET_MODEL'], os.environ['TEST_DRAFT_MODEL'],
                          str(tmp_path/'result'), layout], capture_output=True, text=True, timeout=30,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES='', IMAGE_CACHE_LEG_CPU_CHECK='1',
                                  IMAGE_OCI_CONFIG=os.environ['TEST_IMAGE_CONFIG']))
    assert run.returncode == 0, run.stdout + run.stderr
    for arm in ('cold', 'warm'):
        assert f'CPU MOUNTS PASS: {layout} {arm}' in run.stdout
