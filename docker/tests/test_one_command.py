"""One command from a fresh clone: Compose, the checkpoints, the hybrid build, start.sh's container."""
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREPARE = ROOT / 'docker' / 'prepare.sh'
MODELS = ('DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw', 'DeepSeek-V4.1-Flash-1017-engram',
          'DeepSeek-V4.1-Flash-1017-hybrid-exl3-fp8')
HOSTS = ('DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw', 'DeepSeek-V4.1-Flash-1017',
         'DeepSeek-V4.1-Flash-1017-hybrid-exl3-fp8')


def release_image():
    return re.search(r'^RELEASE_IMAGE=(.+)$', (ROOT / 'start.sh').read_text(), re.M)[1]


def default(text, key):
    return re.search(r'^' + key + r'="\$\{' + key + r':-(.+)\}"', text, re.M)[1]


def test_every_default_image_is_the_start_pin():
    image = release_image()
    compose = (ROOT / 'docker-compose.yml').read_text()
    assert re.findall(r'\$\{IMAGE:-(.+)\}$', compose, re.M) == [image]
    files = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '--cached', '--others', '--exclude-standard'],
                           capture_output=True, text=True, check=True).stdout.split()
    for path in (ROOT / name for name in files):
        if path.name == 'CHANGELOG.md' or 'tests' in path.parts or not path.is_file():
            continue
        for found in re.findall(r'ghcr\.io/morrowmake/vllm-cmp170hx@sha256:(?:[0-9a-f]{64}|\{\{IMAGE_DIGEST\}\})',
                                path.read_text(errors='replace')):
            assert found == image, path


def compose_value(value, env):
    """VALUE as Compose interpolates `${NAME:-default}`: the default ends at the first `}`, the rest is literal."""
    match = re.fullmatch(r'\$\{(\w+):-([^}]*)\}(.*)', value)
    return (env.get(match[1]) or match[2]) + match[3]


def test_the_compose_image_is_one_digest_reference_with_or_without_image_set():
    image = release_image()
    assert re.fullmatch(r'ghcr\.io/morrowmake/vllm-cmp170hx@sha256:[0-9a-f]{64}', image), image
    value = re.search(r'^x-image: &image (.+)$', (ROOT / 'docker-compose.yml').read_text(), re.M)[1]
    assert compose_value(value, {}) == image
    local = 'sha256:' + '0' * 64
    assert compose_value(value, {'IMAGE': local}) == local


def test_compose_runs_the_start_container_after_prepare():
    yaml = pytest.importorskip('yaml')
    spec = yaml.safe_load((ROOT / 'docker-compose.yml').read_text())
    prepare, server = spec['services']['prepare'], spec['services']['dsv41-flash']
    assert server['depends_on'] == {'prepare': {'condition': 'service_completed_successfully'}}
    assert prepare['image'] == server['image'] == '${IMAGE:-' + release_image() + '}'
    assert prepare['entrypoint'] == ['/bin/bash', '/recipe/prepare.sh'] and prepare['command'] == ['prepare']
    assert server['entrypoint'] == ['/bin/bash', '/recipe/prepare.sh', 'serve']
    assert 'command' not in server and server['shm_size'] == '16g'
    # Swap included, the same limit: the server never swaps.
    assert server['mem_limit'] == server['memswap_limit'] == '${MEM_LIMIT:-1024g}'
    assert server['env_file'] == ['docker/container.env']
    for name in ('docker/entry.sh', 'boot_check.py', 'p2p_check.py', 'p2p_check.sh', 'p2p_probe.cu',
                 'engram_placement.sh'):
        assert f'./{name}:/recipe/{name.split("/")[-1]}:ro' in server['volumes']
    for host, mount in zip(HOSTS, MODELS):
        assert f'${{MODELS_DIR:-./models}}/{host}:/models/{mount}' in prepare['volumes']
        assert f'${{MODELS_DIR:-./models}}/{host}:/models/{mount}:ro' in server['volumes']
    assert './cache:/cache' in prepare['volumes'] and './cache:/cache' in server['volumes']
    assert 'HOST_UID' not in str(server)
    # The server comes back after a crash; prepare must finish, never loop.
    assert server['restart'] == 'on-failure:3' and 'restart' not in prepare


def test_image_and_launch_name_the_same_checkpoint_paths():
    """prepare.sh, start.sh and the image's model directories agree."""
    prepare, start = PREPARE.read_text(), (ROOT / 'start.sh').read_text()
    dockerfile, build = (ROOT / 'docker/Dockerfile').read_text(), (ROOT / 'docker/build.sh').read_text()
    for name, role in zip(MODELS, ('EXPORT', 'OFFICIAL', 'HYBRID')):
        assert f'{role}_DIR=/models/{name}\n' in prepare
        assert f'{role}_MOUNT=/models/{name}\n' in start
        assert f'/models/{name}' in dockerfile and f'models/{name}' in build


def test_checkpoints_are_pinned_alike_for_start_and_compose():
    start, prepare = (ROOT / 'start.sh').read_text(), PREPARE.read_text()
    advanced = (ROOT / '.env.advanced.example').read_text()
    for key in ('EXPORT_REPO', 'OFFICIAL_REPO'):
        assert default(prepare, key) == default(start, key)
    for key in ('EXPORT_REPO', 'EXPORT_REVISION', 'OFFICIAL_REPO', 'OFFICIAL_REVISION'):
        assert re.findall(r'^# ' + key + r'=(.+)$', advanced, re.M) == [default(prepare, key)]
    for key in ('EXPORT_REVISION', 'OFFICIAL_REVISION'):
        assert re.fullmatch(r'[0-9a-f]{40}', default(prepare, key)), key
    assert default(prepare, 'NEED_DISK_GB') == default(start, 'NEED_DISK_GB')


def write_checkpoint(path, shards=('a.safetensors', 'b.safetensors'), extra=()):
    path.mkdir(parents=True)
    weight_map = ', '.join(f'"layer.{i}.weight": "{name}"' for i, name in enumerate(shards))
    (path / 'model.safetensors.index.json').write_text('{"weight_map": {' + weight_map + '}}')
    for name in shards:
        (path / name).write_bytes(b'x')
    for name in ('config.json', *extra):
        (path / name).write_text(name)
    return path


def prepare_functions(body, **env):
    """Run prepare.sh's functions with stat and chown replaced by recorders."""
    source = PREPARE.read_text().split('case "${1:-}" in', 1)[0]
    script = source + '''
stat() { case "$3" in *root*|*checkout*) echo "${FAKE_OWNER:-0}" ;; *) echo 1000 ;; esac; }
chown() { echo "chown $*"; }
''' + body
    return subprocess.run(['bash', '-c', script], capture_output=True, text=True,
                          env=dict(os.environ, **env))


def check(path, name):
    return subprocess.run(['bash', str(PREPARE), 'check', str(path), name]).returncode


def required(name):
    return subprocess.run(['bash', str(PREPARE), 'required', name], capture_output=True, text=True,
                          check=True).stdout.split()


@pytest.mark.parametrize('name', ['export', 'official'])
def test_a_download_is_complete_with_every_file_and_shard(tmp_path, name):
    extra = [n for n in required(name) if n not in ('config.json', 'model.safetensors.index.json')]
    good = write_checkpoint(tmp_path / 'good', extra=extra)
    assert check(good, name) == 0
    shard = write_checkpoint(tmp_path / 'shard', extra=extra)
    (shard / 'b.safetensors').unlink()
    assert check(shard, name) == 1
    partial = write_checkpoint(tmp_path / 'partial', extra=extra)
    download = partial / '.cache' / 'huggingface' / 'download' / 'sub'
    download.mkdir(parents=True)
    (download / 'c.safetensors.incomplete').write_bytes(b'')
    assert check(partial, name) == 1
    for file in required(name):
        missing = write_checkpoint(tmp_path / ('missing-' + file), extra=extra)
        (missing / file).unlink()
        assert check(missing, name) == 1, file
    assert check(tmp_path / 'absent', name) == 1


def hybrid_pins():
    text = PREPARE.read_text().split('HYBRID_SHA256="', 1)[1].split('"', 1)[0]
    return dict(line.split('=', 1) for line in text.splitlines())


def write_hybrid(path, target):
    """A hybrid link directory whose shard points at TARGET; returns its written files' sha256."""
    path.mkdir(parents=True)
    (path / 'model.safetensors.index.json').write_text('{"weight_map": {"a": "model-00001.safetensors"}}')
    for name in ('config.json', 'quantization_config.json', 'tokenizer.json', 'tokenizer_config.json',
                 'hybrid-build.json'):
        (path / name).write_text(name)
    (path / 'model-00001.safetensors').symlink_to(target / 'model-00001.safetensors')
    return {n: hashlib.sha256((path / n).read_bytes()).hexdigest()
            for n in ('config.json', 'model.safetensors.index.json', 'quantization_config.json')}


def test_hybrid_needs_the_validated_bytes_and_links(tmp_path):
    official = tmp_path / 'official'
    official.mkdir()
    (official / 'model-00001.safetensors').write_bytes(b'x')
    actual = write_hybrid(tmp_path / 'hybrid', official)
    pins = '\n'.join(f'{k}={v}' for k, v in actual.items())
    body = f'''HYBRID_SHA256="{pins}"; OFFICIAL_DIR="{official}"
complete "{tmp_path / 'hybrid'}" hybrid && echo complete || echo incomplete
'''
    assert prepare_functions(body).stdout.strip() == 'complete'
    # A link that does not resolve where the targets are is not a built hybrid.
    (official / 'model-00001.safetensors').unlink()
    assert prepare_functions(body).stdout.strip() == 'incomplete'
    # Outside the container (no /models/...), only the links are checked.
    body_host = body.replace(f'OFFICIAL_DIR="{official}"', f'OFFICIAL_DIR="{tmp_path / "absent"}"')
    assert prepare_functions(body_host).stdout.strip() == 'complete'
    # Other bytes are another checkpoint.
    (tmp_path / 'hybrid' / 'config.json').write_text('changed')
    assert prepare_functions(body_host).stdout.strip() == 'incomplete'
    assert set(hybrid_pins()) == set(actual)
    assert all(re.fullmatch(r'[0-9a-f]{64}', v) for v in hybrid_pins().values())


def test_hybrid_build_runs_the_engine_builder_into_an_empty_directory(tmp_path):
    out = tmp_path / 'hybrid'
    out.mkdir()
    builder = tmp_path / 'builder.py'
    builder.write_text('')
    python = tmp_path / 'python'
    python.write_text('#!/bin/sh\necho "build $*" >> "$LOG"\n')
    python.chmod(0o755)
    run = prepare_functions(f'''PY="{python}"; BUILDER="{builder}"; HYBRID_DIR="{out}"
EXPORT_DIR=/models/export; OFFICIAL_DIR=/models/official
hybrid
''', LOG=str(tmp_path / 'log'))
    # The stand-in builder writes nothing, so the result is refused.
    assert run.returncode == 1 and 'validated bytes' in run.stderr
    call = (tmp_path / 'log').read_text()
    assert f'{builder} --export /models/export --source /models/official' in call
    assert "--exl3-modules ^(layers|mtp)\\.\\d+\\.ffn\\.experts(\\.|$)" in call
    assert '--drop \\.engram\\.embed\\. --method deepseek_v4_fp8' in call and f'--out {out}' in call
    # A directory holding anything else is never built into or changed.
    (out / 'other').write_text('mine')
    (tmp_path / 'log').unlink()
    run = prepare_functions(f'''PY="{python}"; BUILDER="{builder}"; HYBRID_DIR="{out}"
hybrid
''', LOG=str(tmp_path / 'log'))
    assert run.returncode == 1 and 'different or partial' in run.stderr
    assert not (tmp_path / 'log').exists() and (out / 'other').read_text() == 'mine'


def test_fetch_downloads_only_what_the_server_reads_at_the_pin(tmp_path):
    hf = tmp_path / 'hf'
    hf.write_text('#!/bin/sh\necho "hf $*"\n')
    hf.chmod(0o755)
    target = write_checkpoint(tmp_path / 'target', extra=['quantization_config.json', 'tokenizer.json',
                                                          'tokenizer_config.json'])
    run = prepare_functions(f'HF="{hf}"\nfetch export org/repo rev1 "{target}"\nfetch official org/base rev2 "{target}"\n')
    assert run.returncode == 0, run.stderr
    lines = [line for line in run.stdout.splitlines() if line.startswith('hf ')]
    assert lines == [
        f'hf download org/repo --revision rev1 --local-dir {target} --include config.json '
        'model.safetensors.index.json quantization_config.json tokenizer.json tokenizer_config.json model-*.safetensors',
        f'hf download org/base --revision rev2 --local-dir {target} --include config.json '
        'model.safetensors.index.json model-*.safetensors']


def test_refresh_fetches_complete_downloads_again(tmp_path):
    hf = tmp_path / 'hf'
    hf.write_text('#!/bin/sh\necho "hf $*"\n')
    hf.chmod(0o755)
    names = ['quantization_config.json', 'tokenizer.json', 'tokenizer_config.json']
    export = write_checkpoint(tmp_path / 'export', extra=names)
    official = write_checkpoint(tmp_path / 'official')
    body = f'''HF="{hf}"; EXPORT_DIR="{export}"; OFFICIAL_DIR="{official}"; hybrid() {{ echo hybrid; }}
df() {{ printf 'Avail\\n9999G\\n'; }}
models
'''
    quiet = prepare_functions(body, REFRESH_WEIGHTS='0')
    assert quiet.returncode == 0, quiet.stderr
    assert 'hf ' not in quiet.stdout and 'hybrid' in quiet.stdout
    again = prepare_functions(body, REFRESH_WEIGHTS='1')
    assert again.returncode == 0, again.stderr
    assert again.stdout.count('hf download') == 2


def test_claim_changes_only_empty_directories_docker_created(tmp_path):
    fresh = tmp_path / 'root-fresh'
    fresh.mkdir()
    other = tmp_path / 'root-other'
    other.mkdir()
    (other / 'model-00001.safetensors').write_text('x')
    user = tmp_path / 'user-owned'
    user.mkdir()
    run = prepare_functions(f'''SERVE_UID=1234; SERVE_GID=4321
claim "{fresh}"
claim "{other}"
claim "{user}"
''')
    assert run.returncode == 0, run.stderr
    assert f'chown 1234:4321 {fresh}' in run.stdout
    assert str(other) not in run.stdout and str(user) not in run.stdout


@pytest.mark.parametrize('env,expected', [({'FAKE_OWNER': '1001'}, '1001 1001'),
                                          ({'HOST_UID': '1500', 'HOST_GID': '1600'}, '1500 1600'),
                                          ({'FAKE_OWNER': '0'}, '0 0')])
def test_serving_user_is_the_checkout_owner(tmp_path, env, expected):
    (tmp_path / 'checkout').mkdir()
    run = prepare_functions(f'''CHECKOUT="{tmp_path}/checkout"
serving_ids; echo "$SERVE_UID $SERVE_GID"
as_serving_user false; echo returned
''', **{'HOST_UID': '', 'HOST_GID': '', **env})
    assert run.stdout.splitlines()[0] == expected, run.stderr
    assert ('returned' in run.stdout) == expected.startswith('0 ')


def test_rejects_non_numeric_override():
    run = prepare_functions('serving_ids', HOST_UID='ba', HOST_GID='')
    assert run.returncode == 1 and 'numeric' in run.stderr


def run_start(root, body, **env):
    source = (ROOT / 'start.sh').read_text().rsplit('main "$@"', 1)[0]
    script = root / 'start.sh'
    script.write_text(source + body)
    (root / 'docker').mkdir(exist_ok=True)
    (root / 'docker' / 'prepare.sh').write_text(PREPARE.read_text())
    (root / 'engram_placement.sh').write_text((ROOT / 'engram_placement.sh').read_text())
    if not (root / '.env.example').exists():
        (root / '.env.example').write_text('')
    clean = {k: v for k, v in os.environ.items() if not k.startswith(('VLLM_', 'NCCL_', 'MODELS_DIR', 'NEED_RAM_GIB'))}
    return subprocess.run(['bash', str(script)], capture_output=True, text=True,
                          env=dict(clean, HOME=str(root / 'home'), CUDA_VISIBLE_DEVICES='', **env), timeout=60)


def test_start_download_runs_prepare_models_with_the_three_directories(tmp_path):
    run = run_start(tmp_path, '''models_present() { return 1; }
container_image_present() { return 0; }
docker() { printf '%s\\n' "$@"; }
do_download
''', MODELS_DIR='store', REFRESH_WEIGHTS='1')
    assert run.returncode == 0, run.stderr
    args = run.stdout.splitlines()
    store = tmp_path / 'store'
    for host, mount in zip(HOSTS, MODELS):
        assert f'{store}/{host}:/models/{mount}:rw' in args
        assert (store / host).is_dir()
    entry = args.index('/recipe/prepare.sh')
    assert args[entry - 1:entry + 2] == ['/bin/bash' if args[entry - 1] == '/bin/bash' else args[entry - 1],
                                         '/recipe/prepare.sh', 'models']
    assert 'REFRESH_WEIGHTS=1' in args
    skip = run_start(tmp_path, 'models_present() { return 0; }\ndocker() { echo called; }\ndo_download\n')
    assert 'called' not in skip.stdout and 'skipping' in skip.stdout


def test_container_runs_the_entry_with_read_only_checkpoints(tmp_path):
    run = run_start(tmp_path, 'container_env\ncontainer_args /env\nprintf "%s\\n" "${CRUN[@]}"\n',
                    P2P='off', API_KEY='k', EXTRA_ARGS='--max-model-len 131072', VLLM_ENGRAM_PLACEMENT='ssd',
                    NCCL_P2P_LEVEL='LOC')
    assert run.returncode == 0, run.stderr
    out = run.stdout.splitlines()
    for line in ('LAYOUT=tp4', 'HOST=0.0.0.0', 'PORT=8000', 'P2P=off', 'API_KEY=k',
                 'EXTRA_ARGS=--max-model-len 131072', 'VLLM_ENGRAM_PLACEMENT=ssd', 'NCCL_P2P_LEVEL=LOC',
                 'RECIPE_BOOT_CHECK_OWNER=start', 'RECIPE_CACHE_ROOT=/cache'):
        assert line in out, line
    for host, mount in zip(HOSTS, MODELS):
        assert f'{tmp_path}/models/{host}:/models/{mount}:ro' in out, mount
    assert out[-4:-2] == ['--entrypoint', '/bin/bash'] and out[-1] == '/recipe/entry.sh'
    assert not any('serve.sh' in a for a in out)


def test_container_never_swaps(tmp_path):
    """The container's memory limit, swap included, is the host's memory, so the Engram tables are never
    swapped out; MEM_LIMIT overrides."""
    body = 'container_args /env\nprintf "%s\\n" "${CRUN[@]}"\n'
    out = run_start(tmp_path, body, **host_memory(tmp_path, 300)).stdout.splitlines()
    total = f'{512 * 1048576}k'
    assert out[out.index('--memory') + 1] == out[out.index('--memory-swap') + 1] == total
    out = run_start(tmp_path, body, MEM_LIMIT='200g').stdout.splitlines()
    assert out[out.index('--memory') + 1] == out[out.index('--memory-swap') + 1] == '200g'
    (tmp_path / 'empty').write_text('')
    run = run_start(tmp_path, body, RECIPE_MEMINFO=str(tmp_path / 'empty'))
    assert run.returncode != 0 and 'set MEM_LIMIT' in run.stderr


def host_memory(root, avail_gib):
    """Fake /proc/meminfo and an empty cgroup directory (no limit) for engram_placement.sh."""
    (root / 'meminfo').write_text(f'MemTotal: {512 * 1048576} kB\nMemAvailable: {avail_gib * 1048576} kB\n')
    (root / 'cgroup').mkdir(exist_ok=True)
    return {'RECIPE_MEMINFO': str(root / 'meminfo'), 'RECIPE_CGROUP': str(root / 'cgroup')}


PREFLIGHT = 'server_is_ours() { return 1; }\npreflight_ram\necho "where=$(engram_where)"\n'
SSD_100 = ('Engram tables: ssd (auto: 100 GiB available, RAM placement needs ~212 GiB; slower), '
           'row cache 17 GiB per card, filled at start, direct reads')


@pytest.mark.parametrize('placement,avail,ok,line', [
    (None, 300, True, 'Engram tables: ram (auto: 300 GiB available, RAM placement needs ~212 GiB)'),
    (None, 100, True, SSD_100),
    ('auto', 64, True, 'row cache 8 GiB per card, not filled at start (under 12 GiB), direct reads'),
    ('ram', 300, True, 'Engram tables: ram (VLLM_ENGRAM_PLACEMENT=ram: 300 GiB available'),
    ('ram', 100, False, None),
    ('ssd', 100, True, 'Engram tables: ssd (VLLM_ENGRAM_PLACEMENT=ssd: 100 GiB available'),
])
def test_host_memory_for_the_engram_tables(tmp_path, placement, avail, ok, line):
    """auto decides as the container's entry will and never stops; an explicit ram stops below the need."""
    settings = host_memory(tmp_path, avail)
    if placement:
        settings['VLLM_ENGRAM_PLACEMENT'] = placement
    run = run_start(tmp_path, PREFLIGHT, **settings)
    assert (run.returncode == 0) is ok, run.stdout + run.stderr
    if ok:
        assert line in run.stdout
        where = {(None, 300): 'ram (auto)', (None, 100): 'ssd (auto)', ('auto', 64): 'ssd (auto)',
                 ('ram', 300): 'ram', ('ssd', 100): 'ssd'}[placement, avail]
        assert f'where={where}\n' in run.stdout
    else:
        assert 'VLLM_ENGRAM_PLACEMENT=ssd' in run.stderr and '100 GiB available' in run.stderr
        dry = run_start(tmp_path, PREFLIGHT, DRY='1', **settings)
        assert dry.returncode == 0 and 'not fatal under DRY' in dry.stderr


def test_launch_line_names_the_decision(tmp_path):
    """The launch log line reads engram_where; before a decision it is the setting."""
    start = (ROOT / 'start.sh').read_text()
    assert 'Engram tables in $(engram_where), $(publish_addr)' in start
    run = run_start(tmp_path, 'echo "where=$(engram_where)"\n')
    assert 'where=auto\n' in run.stdout


def test_dry_launch_names_the_decision_and_keeps_the_launch(tmp_path):
    """DRY says what the entry will decide; the launch it prints stays the release's own, whatever the host's
    memory, so a recipe DRY fixture does not depend on it."""
    (tmp_path / 'serve.sh').write_text('#!/bin/bash\nenv | grep ^VLLM_ENGRAM_ | sort\necho served\n')
    body = 'server_is_ours() { return 1; }\ncontainer_image_present() { return 1; }\npreflight_ram\ncontainer_launch\n'
    run = run_start(tmp_path, body, DRY='1', **host_memory(tmp_path, 100))
    assert run.returncode == 0, run.stderr
    assert "serve.sh would run (the entry first sets the Engram placement: ssd)" in run.stdout
    assert '    served' in run.stdout.splitlines()
    assert not [l for l in run.stdout.splitlines() if l.startswith('    VLLM_ENGRAM_')]


def test_status_shows_the_setting_and_the_last_start(tmp_path):
    log = tmp_path / 'serve.log'
    log.write_text('Engram tables: ram (auto: 300 GiB available, RAM placement needs ~212 GiB)\n' + SSD_100 + '\n')
    body = ('read_cid() { :; }\nhealth_ok() { return 1; }\nkv_line() { :; }\ninstall_done() { return 0; }\n'
            'models_present() { return 0; }\ndo_status\n')
    run = run_start(tmp_path, body, SERVE_LOG=str(log))
    assert run.returncode == 0, run.stderr
    assert 'VLLM_ENGRAM_PLACEMENT=auto; last start: ssd (auto: 100 GiB available' in run.stdout


def test_need_reaches_the_container_only_when_set(tmp_path):
    run = run_start(tmp_path, 'container_env\n')
    assert run.returncode == 0 and 'NEED_RAM_GIB' not in run.stdout
    run = run_start(tmp_path, 'container_env\n', NEED_RAM_GIB='150')
    assert 'NEED_RAM_GIB=150' in run.stdout.splitlines()


def test_drafting_off_reaches_the_container_without_the_adaptive_depth(tmp_path):
    """EXTRA_ARGS turning speculative decoding off clears VLLM_DSPARK_ADAPTIVE_DEPTH in the container's
    environment (the drafter's state is what it works on) and the preflight says so in one line."""
    policy = {'VLLM_DSPARK_ADAPTIVE_DEPTH': '/policy.json'}
    run = run_start(tmp_path, 'container_env\n', EXTRA_ARGS='--speculative-config None', **policy)
    assert run.returncode == 0, run.stderr
    assert 'VLLM_DSPARK_ADAPTIVE_DEPTH=' in run.stdout.splitlines()
    run = run_start(tmp_path, 'container_env\n', EXTRA_ARGS='--max-model-len 8192', **policy)
    assert 'VLLM_DSPARK_ADAPTIVE_DEPTH=/policy.json' in run.stdout.splitlines()
    body = 'server_is_ours() { return 1; }\npreflight_ram\n'
    run = run_start(tmp_path, body, EXTRA_ARGS='--speculative-config=off', **host_memory(tmp_path, 300))
    assert run.returncode == 0, run.stderr
    assert 'Speculative decoding: off (EXTRA_ARGS)' in run.stdout
    start = (ROOT / 'start.sh').read_text()
    assert '$(drafting_off && echo --no-draft) &' in start


def test_unknown_placement_and_layout_refuse(tmp_path):
    run = run_start(tmp_path, ':\n', VLLM_ENGRAM_PLACEMENT='disk')
    assert run.returncode == 1 and 'must be auto, ram or ssd' in run.stderr
    assert run_start(tmp_path, ':\n', NEED_RAM_GIB='lots').returncode == 1
    assert run_start(tmp_path, ':\n', LAYOUT='pp4').returncode == 1


def test_dry_start_needs_no_engine_on_the_host(tmp_path):
    """Without the image, the DRY launch shows this checkout's serve.sh under DRY=1."""
    checkout = tmp_path / 'checkout'
    shutil.copytree(ROOT, checkout, ignore=shutil.ignore_patterns(
        '.git', '.env', 'models', 'cache', 'logs', '__pycache__'))
    (checkout / 'serve.sh').write_text('#!/bin/bash\necho "launch DRY=$DRY LAYOUT=$LAYOUT PORT=$PORT HOST=$HOST"\n')
    shim = tmp_path / 'shim'
    shim.mkdir()
    for name in ('nvidia-smi', 'docker'):
        (shim / name).write_text('#!/bin/sh\nexit 1\n')
        (shim / name).chmod(0o755)
    run = subprocess.run(['bash', str(checkout / 'start.sh')], capture_output=True, text=True,
                         env={'HOME': str(tmp_path), 'PATH': f'{shim}:/usr/bin:/bin',
                              'DRY': '1', 'NEED_DISK_GB': '0', 'NEED_RAM_GIB': '0'}, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert 'launch DRY=1 LAYOUT=tp4 PORT=8000 HOST=0.0.0.0' in run.stdout
    assert "this checkout's serve.sh" in run.stdout
    assert not (checkout / 'logs' / 'container.env.dry').exists()
