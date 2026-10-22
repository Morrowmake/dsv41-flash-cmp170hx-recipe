import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from p2p_build import build


def test_cpu_compiler_never_runs_probe(tmp_path):
    source = tmp_path/'input.cu'
    source.write_text('fixture')
    compiler = tmp_path/'nvcc'
    compiler.write_text('''#!/bin/sh
[ "$CUDA_VISIBLE_DEVICES" = "" ] || exit 2
[ "$PWD" != "${HOME}" ] || exit 2
[ "$6" = "probe.cu" ] || exit 3
printf '#!/bin/sh\\nexit 99\\n' > probe
''')
    compiler.chmod(0o755)
    build(source, tmp_path/'output', compiler)
    assert len(list((tmp_path/'output').glob('probe-*'))) == 1


def test_build_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setenv('IMAGE_P2P_PREBUILD', '0')
    build(tmp_path/'absent', tmp_path/'output', tmp_path/'no-compiler')
    assert list((tmp_path/'output').iterdir()) == []
    monkeypatch.setenv('IMAGE_P2P_PREBUILD', 'invalid')
    with pytest.raises(ValueError, match='must be 0 or 1'):
        build(tmp_path/'absent', tmp_path/'output', tmp_path/'no-compiler')


def test_built_probe_and_empty_seed_mount():
    import hashlib
    if not os.environ.get('TEST_IMAGE_ROOTFS'):
        pytest.skip('Set TEST_IMAGE_ROOTFS for built-probe inspection')
    rootfs = Path(os.environ['TEST_IMAGE_ROOTFS'])
    digest = hashlib.sha256((ROOT.parent/'p2p_probe.cu').read_bytes()).hexdigest()
    binary = rootfs/'opt/image-tools/p2p'/('probe-'+digest)
    assert binary.is_file() and os.access(binary, os.X_OK)
    with binary.open('rb') as stream:
        assert stream.read(4) == b'\x7fELF'
    seed = rootfs/'opt/cache-seed'
    assert seed.is_dir() and list(seed.iterdir()) == []
