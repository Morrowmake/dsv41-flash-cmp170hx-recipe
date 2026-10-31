"""The recipe contract every generated recipe keeps (recipe.toml, schema recipe-contract/1): one set of pins, the
generated files as generated, a container that never swaps, EXTRA_ARGS kept off the arguments the release fixes,
and speculative decoding switched off without breaking the boot check."""
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = tomllib.loads((ROOT / 'recipe.toml').read_text())
LAUNCH = CONTRACT['launch']
CONTROLLED = CONTRACT['inputs']['controlled_args']
# Stands in for the image's python: the peer check fails, the boot check records its arguments.
PYTHON = '''#!/bin/bash
case "$1" in
  */p2p_check.py) echo "check" >> "$CALL_LOG"; echo 0 ;;
  */boot_check.py) echo "boot:$*" >> "$CALL_LOG" ;;
  *) echo "unexpected:$*" >> "$CALL_LOG"; exit 1 ;;
esac
'''
# Stands in for the image's serve.sh: it reports what it inherited.
SERVE = '#!/bin/bash\necho serve >> "$CALL_LOG"\nenv | LC_ALL=C sort > "$CALL_LOG.env"\n'


@pytest.fixture
def stage(tmp_path):
    (tmp_path / 'venv/bin').mkdir(parents=True)
    (tmp_path / 'venv/bin/python').write_text(PYTHON)
    (tmp_path / 'venv/bin/python').chmod(0o755)
    (tmp_path / 'serve.sh').write_text(SERVE)
    (tmp_path / 'cgroup').mkdir()
    (tmp_path / 'meminfo').write_text(f'MemTotal: {512 * 1048576} kB\nMemAvailable: {300 * 1048576} kB\n')
    return tmp_path


def entry(stage, **settings):
    """docker/entry.sh as the container runs it, with stand-ins for the image's python and serve.sh."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(('VLLM_', 'NCCL_')) and k not in (
        'LAYOUT', 'DRY', 'P2P', 'EXTRA_ARGS', 'BOOT_CHECK', 'RECIPE_BOOT_CHECK_OWNER', 'RECIPE_CACHE_ROOT')}
    env.update(CUDA_VISIBLE_DEVICES='', RECIPE_DIR=str(ROOT), VENV=str(stage / 'venv'),
               SERVE_SH=str(stage / 'serve.sh'), CALL_LOG=str(stage / 'calls'),
               RECIPE_MEMINFO=str(stage / 'meminfo'), RECIPE_CGROUP=str(stage / 'cgroup'))
    env.update(settings)
    run = subprocess.run(['bash', str(ROOT / 'docker/entry.sh')], text=True, capture_output=True, env=env,
                         timeout=60)
    calls = (stage / 'calls').read_text().splitlines() if (stage / 'calls').exists() else []
    served = (stage / 'calls.env').read_text().splitlines() if (stage / 'calls.env').exists() else None
    for name in ('calls', 'calls.env'):
        (stage / name).unlink(missing_ok=True)
    return run, calls, served


def run_start(root, body, **env):
    """start.sh's definitions in a scratch checkout, then BODY instead of its main."""
    shutil.copytree(ROOT, root / 'checkout', ignore=shutil.ignore_patterns('.git', 'models', 'cache', 'logs'))
    script = root / 'checkout' / 'start.sh'
    script.write_text(script.read_text().rsplit('main "$@"', 1)[0] + body)
    (root / 'meminfo').write_text(f'MemTotal: {512 * 1048576} kB\nMemAvailable: {300 * 1048576} kB\n')
    (root / 'cgroup').mkdir(exist_ok=True)
    clean = {k: v for k, v in os.environ.items() if not k.startswith(('VLLM_', 'NCCL_', 'MODELS_DIR'))}
    run = subprocess.run(['bash', str(script)], capture_output=True, text=True, timeout=60,
                         env=dict(clean, HOME=str(root / 'home'), CUDA_VISIBLE_DEVICES='',
                                  RECIPE_MEMINFO=str(root / 'meminfo'), RECIPE_CGROUP=str(root / 'cgroup'), **env))
    shutil.rmtree(root / 'checkout')
    return run


def test_generated_files_are_as_generated():
    """A hand edit to a generated file is lost at the next generation; the recipe holds what was generated."""
    for rel, digest in CONTRACT['files'].items():
        if rel == 'recipe.toml':
            continue
        assert 'sha256:' + hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel


def test_one_set_of_pins():
    """start.sh, Docker Compose and recipe.toml name the same engine and image: a single-tree engine's commit
    (RELEASE_VLLM_COMMIT), or an assembled engine's four pins (RELEASE_UPSTREAM_COMMIT, RELEASE_QUEUE_COMMIT,
    RELEASE_PLUGIN_COMMIT, RELEASE_NATIVE_BUILD)."""
    start = (ROOT / 'start.sh').read_text()
    compose = (ROOT / 'docker-compose.yml').read_text()
    pins = CONTRACT['pins']
    assert re.findall(r'^RELEASE_IMAGE=(.+)$', start, re.M) == [pins['image']]
    engine = {k: pins[k] for k in ('vllm_commit', 'upstream_commit', 'queue_commit', 'plugin_commit', 'native_build')
              if k in pins}
    assert sorted(re.findall(r'^RELEASE_((?!IMAGE=)[A-Z_]+)=(.+)$', start, re.M)) == sorted(
        (k.upper(), v) for k, v in engine.items())
    assert set(engine) in ({'vllm_commit'}, {'upstream_commit', 'queue_commit', 'plugin_commit', 'native_build'})
    assert re.findall(r'\$\{IMAGE:-(.+)\}$', compose, re.M) == [pins['image']]
    assert re.fullmatch(r'ghcr\.io/morrowmake/vllm-cmp170hx@sha256:[0-9a-f]{64}', pins['image'])


def test_the_container_never_swaps(tmp_path):
    """Memory limit = memory+swap limit, for start.sh (the host's memory, or MEM_LIMIT) and for Compose."""
    run = run_start(tmp_path, 'container_args /env\nprintf "%s\\n" "${CRUN[@]}"\n', MEM_LIMIT='200g')
    assert run.returncode == 0, run.stderr
    out = run.stdout.splitlines()
    assert out[out.index('--memory') + 1] == out[out.index('--memory-swap') + 1] == '200g'
    run = run_start(tmp_path, 'container_args /env\nprintf "%s\\n" "${CRUN[@]}"\n')
    out = run.stdout.splitlines()
    assert out[out.index('--memory') + 1] == out[out.index('--memory-swap') + 1] == f'{512 * 1048576}k'
    compose = (ROOT / 'docker-compose.yml').read_text()
    assert re.findall(r'^\s+mem_limit: (.+)$', compose, re.M) == re.findall(r'^\s+memswap_limit: (.+)$', compose,
                                                                              re.M) == ['${MEM_LIMIT:-1024g}']


def test_every_cache_is_under_the_cache_mount():
    """Compose's container environment puts every cache and the temporary directory under /cache."""
    env = dict(line.split('=', 1) for line in (ROOT / 'docker/container.env').read_text().splitlines()
               if line and not line.startswith('#'))
    paths = {k: v for k, v in env.items() if k in ('HOME', 'TMPDIR') or k.endswith(('_DIR', '_HOME', '_ROOT',
                                                                                    '_PATH', '_BASE'))}
    assert {'HOME', 'TMPDIR', 'XDG_CACHE_HOME', 'HF_HOME', 'TRITON_CACHE_DIR', 'VLLM_CACHE_ROOT'} <= set(paths)
    assert all(v == '/cache' or v.startswith('/cache/') for v in paths.values()), paths


def controlled_forms():
    """Each way the engine's parser accepts a controlled argument: the option, `_` for `-`, a unique prefix,
    `=value`, a dotted JSON key, the short alias."""
    name = 'tensor-parallel-size' if 'tensor-parallel-size' in CONTROLLED else CONTROLLED[0]
    forms = [f'--{name} 2', f'--{name}=2', '--' + name.replace('-', '_') + ' 2', f'--{name[:-2]} 2',
             f'--{name}.key 2', f'--max-num-seqs 4 --{name} 2']
    if 'tensor-parallel-size' in CONTROLLED:
        forms += ['-tp 2', '-tp=2']
    return forms


@pytest.mark.parametrize('extra', controlled_forms())
def test_extra_args_cannot_set_a_controlled_argument(stage, tmp_path, extra):
    """The entry refuses before it checks the cards or starts anything; start.sh refuses before it launches."""
    run, calls, served = entry(stage, DRY='0', P2P='auto', BOOT_CHECK='1', EXTRA_ARGS=extra)
    assert run.returncode == 2 and 'EXTRA_ARGS cannot set' in run.stderr and calls == []
    run = run_start(tmp_path, 'preflight_launch\necho launched\n', EXTRA_ARGS=extra)
    assert run.returncode != 0 and 'EXTRA_ARGS cannot set' in run.stderr and 'launched' not in run.stdout


@pytest.mark.parametrize('extra', ['--max-num-seqs 4', '--speculative-config None', '--enable-prefix-caching',
                                   '--served-model-name other'])
def test_extra_args_that_the_release_does_not_fix_pass(stage, extra):
    run, calls, served = entry(stage, DRY='0', P2P='off', BOOT_CHECK='0', EXTRA_ARGS=extra)
    assert run.returncode == 0, run.stderr
    assert calls[-1] == 'serve' and f'EXTRA_ARGS={extra}' in served


@pytest.mark.parametrize('extra', ['--speculative-config None', '--speculative-config=null',
                                   '--speculative-config off'])
def test_drafting_off_boots_without_draft_counters(stage, extra):
    """Speculative decoding off in EXTRA_ARGS: the boot check runs without its draft-counter checks, the
    settings only the drafter reads are cleared, and one line says so."""
    drafter = {name: '/drafter-only' for name in LAUNCH['drafting_off_clears']}
    run, calls, served = entry(stage, DRY='0', P2P='off', BOOT_CHECK='1', EXTRA_ARGS=extra, **drafter)
    assert run.returncode == 0, run.stderr
    assert 'Speculative decoding: off (EXTRA_ARGS)' in run.stdout
    boot = [c for c in calls if c.startswith('boot:')]
    assert len(boot) == 1 and '--no-draft' in boot[0] and boot[0].index('--no-draft') < boot[0].index('--serve')
    run, calls, served = entry(stage, DRY='0', P2P='off', BOOT_CHECK='0', EXTRA_ARGS=extra, **drafter)
    for name in drafter:
        assert f'{name}=' in served, name


def test_drafting_on_keeps_the_draft_counters(stage):
    drafter = {name: '/drafter-only' for name in LAUNCH['drafting_off_clears']}
    run, calls, served = entry(stage, DRY='0', P2P='off', BOOT_CHECK='1', **drafter)
    assert run.returncode == 0, run.stderr
    boot = [c for c in calls if c.startswith('boot:')]
    assert len(boot) == 1 and '--no-draft' not in boot[0]
    assert f'--model {LAUNCH["served_model_name"]}' in boot[0]
    run, calls, served = entry(stage, DRY='0', P2P='off', BOOT_CHECK='0', **drafter)
    for name in drafter:
        assert f'{name}=/drafter-only' in served, name
