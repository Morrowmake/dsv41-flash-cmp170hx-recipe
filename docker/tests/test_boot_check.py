import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

ROOT = Path(__file__).parents[2]
spec = importlib.util.spec_from_file_location('boot_check', ROOT / 'boot_check.py')
boot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boot)


@contextlib.contextmanager
def endpoint(mode='ok'):
    state = dict(posts=[], metrics=0, finished=0, ready=False, auth=[])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            state['auth'].append(self.headers.get('Authorization'))
            self.send_response(200)
            self.end_headers()
            if self.path == '/metrics':
                state['metrics'] += 1
                values = (73, 219, 17) if state['ready'] else (9, 27, 5)
                if mode == 'zero':
                    values = (73, 219, 0) if state['ready'] else (9, 27, 0)
                if mode == 'short':
                    values = (10, 28, 6)
                if mode == 'reset' and state['ready']:
                    values = (0, 0, 0)
                rows = [f'{name}{{model_name="fixture",engine="0"}} {value}'
                        for name, value in zip(boot.COUNTERS, values)]
                # Unrelated model's traffic must never make a failed check pass.
                rows += [f'{name}{{model_name="other"}} 99999' for name in boot.COUNTERS]
                rows += [f'{boot.SUCCESS}{{model_name="fixture",finished_reason="stop"}} {state["finished"]}']
                if mode == 'missing':
                    rows = rows[1:]
                self.wfile.write(('\n'.join(rows) + '\n').encode())

        def do_POST(self):
            state['auth'].append(self.headers.get('Authorization'))
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            state['posts'].append(body)
            if mode == 'http-error':
                self.send_response(500)
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            if not body.get('stream'):
                state['finished'] += 1
                answer = 'wrong' if mode == 'wrong' else ('ok' if mode == 'lowercase' else ' \nOK\t')
                self.wfile.write(json.dumps(dict(choices=[dict(message=dict(content=answer))])).encode())
            else:
                data = dict(error='fixture') if mode == 'stream-error' else dict(choices=[dict(delta=dict(content='code'))])
                self.wfile.write(b'data: ' + json.dumps(data).encode() + b'\n\n')
                if mode != 'incomplete':
                    self.wfile.write(b'data: [DONE]\n\n')
                state['ready'] = True

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield boot.Client(f'http://127.0.0.1:{server.server_port}', 'fixture', 'fixture-key'), state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_two_requests_and_auth():
    with endpoint() as (client, state):
        boot.check(client, metrics_wait=0)
        assert len(state['posts']) == 2
        first, second = state['posts']
        assert first['messages'][0]['content'] == boot.OK_PROMPT
        assert first['temperature'] == second['temperature'] == 0
        assert first['reasoning_effort'] == second['reasoning_effort'] == 'low'
        assert second['stream'] and second['min_tokens'] == second['max_tokens'] == 768
        assert second['ignore_eos'] is True
        assert set(state['auth']) == {'Bearer fixture-key'}


@pytest.mark.parametrize('mode', ['wrong', 'lowercase', 'zero', 'short', 'reset', 'missing',
                                 'incomplete', 'stream-error', 'http-error'])
def test_failures(mode):
    with endpoint(mode) as (client, state):
        with pytest.raises(Exception):
            boot.check(client, metrics_wait=0)
        assert len(state['posts']) <= 2


def test_delayed_counters_without_extra_generation(monkeypatch):
    with endpoint() as (client, state):
        original = client.metrics
        calls = 0

        def delayed():
            nonlocal calls
            calls += 1
            if calls == 2:
                state['finished'] = 0
            if calls == 3:
                state['finished'] = 1
            ready = state['ready']
            if calls == 4:
                state['ready'] = False
            result = original()
            state['ready'] = ready
            return result

        monkeypatch.setattr(client, 'metrics', delayed)
        monkeypatch.setattr(boot.time, 'sleep', lambda _: None)
        boot.check(client)
        assert calls == 5 and len(state['posts']) == 2


@pytest.mark.parametrize('value', ['NaN', '+Inf', '-1'])
def test_invalid_metrics(value):
    data = '\n'.join(f'{name} {value}' for name in boot.TRACKED)
    with pytest.raises(boot.CheckError):
        boot.parse_metrics(data, 'fixture')


def test_start_owns_check_once_and_stops_on_failure(tmp_path):
    (tmp_path/'.env.example').write_text('')
    source = (ROOT/'start.sh').read_text().rsplit('main "$@"', 1)[0]
    script = tmp_path/'start.sh'
    script.write_text(source + '''
SCRIPT_DIR="$TEST_RECIPE"
CLIENT_HOST=127.0.0.1
SERVED_MODEL_NAME=fixture
container_is_ours() { return 0; }
SERVE_LOG=/dev/null
health_ok() { return 0; }
served_id() { echo fixture; }
do_stop() { echo stopped > "$STOP_RECORD"; }
wait_ready
''')
    for mode, launched, enabled, posts, stopped in [('ok', '1', '1', 2, False),
                                                  ('wrong', '1', '1', 1, True),
                                                  ('ok', '0', '1', 0, False),
                                                  ('ok', '1', '0', 0, False)]:
        record = tmp_path/'stop'
        record.unlink(missing_ok=True)
        with endpoint(mode) as (client, state):
            # Configuration initialises LAUNCHED=0; set it just before wait_ready.
            runscript = tmp_path/'run.sh'
            runscript.write_text(script.read_text().replace('\nwait_ready\n', '\nLAUNCHED="$TEST_LAUNCHED"\nwait_ready\n'))
            run = subprocess.run(['bash', str(runscript)], capture_output=True, text=True,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PORT=client.base.rsplit(':', 1)[1],
                                          TEST_RECIPE=str(ROOT), TEST_LAUNCHED=launched,
                                          STOP_RECORD=str(record), BOOT_CHECK=enabled, API_KEY='fixture-key'))
            assert (run.returncode != 0) == stopped, run.stdout + run.stderr
            assert len(state['posts']) == posts
            assert record.exists() == stopped


def test_one_failed_liveness_query_does_not_stop_the_launch(tmp_path):
    # A single failed ownership check (docker returned nothing) while the
    # server loads is retried; two in a row mean the server is gone.
    (tmp_path/'.env.example').write_text('')
    source = (ROOT/'start.sh').read_text().rsplit('main "$@"', 1)[0]
    queries = tmp_path/'queries'
    record = tmp_path/'stop'
    for failures, healthy in [(1, True), (2, False)]:
        queries.write_text('')
        record.unlink(missing_ok=True)
        script = tmp_path/'run.sh'
        script.write_text(source + '''
SCRIPT_DIR="$TEST_RECIPE"
SERVE_LOG=/dev/null
sleep() { :; }
container_is_ours() {
    echo q >>"$QUERIES"
    [ "$(wc -l <"$QUERIES")" -gt "$FAILURES" ]
}
health_ok() { [ "$(wc -l <"$QUERIES")" -ge 3 ]; }
served_id() { echo fixture; }
do_stop() { echo stopped >"$STOP_RECORD"; }
LAUNCHED=1
BOOT_CHECK=0
wait_ready
''')
        run = subprocess.run(['bash', str(script)], capture_output=True, text=True, timeout=60,
                             env=dict(os.environ, CUDA_VISIBLE_DEVICES='', TEST_RECIPE=str(ROOT),
                                      QUERIES=str(queries), FAILURES=str(failures),
                                      STOP_RECORD=str(record), READY_TIMEOUT='60'))
        out = run.stdout + run.stderr
        assert (run.returncode == 0) == healthy, out
        assert ('healthy after' in out) == healthy, out
        assert ('the server exited' in out) != healthy, out
        assert not record.exists()


def test_supervisor_failure_cleans_owned_child(tmp_path):
    pidfile = tmp_path/'pid'
    childcode = 'import os,time,pathlib;pathlib.Path(' + repr(str(pidfile)) + ').write_text(str(os.getpid()));time.sleep(120)'
    with endpoint('wrong') as (client, state):
        run = subprocess.run([sys.executable, str(ROOT/'boot_check.py'), '--base', client.base,
                              '--model', 'fixture', '--serve', sys.executable, '-c', childcode],
                             env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), capture_output=True, text=True, timeout=10)
        assert run.returncode == 1 and '[boot-check] FAIL' in run.stdout
    if pidfile.exists():
        with pytest.raises(ProcessLookupError):
            os.kill(int(pidfile.read_text()), 0)


def test_supervisor_success_and_signal_cleanup(tmp_path):
    pidfile = tmp_path/'pid'
    childcode = 'import os,time,pathlib;pathlib.Path(' + repr(str(pidfile)) + ').write_text(str(os.getpid()));time.sleep(120)'
    with endpoint() as (client, state):
        proc = subprocess.Popen([sys.executable, '-u', str(ROOT/'boot_check.py'), '--base', client.base,
                                 '--model', 'fixture', '--serve', sys.executable, '-c', childcode],
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=subprocess.PIPE, text=True)
        try:
            assert 'enabled' in proc.stdout.readline()
            assert 'PASS' in proc.stdout.readline()
            deadline = time.monotonic() + 5
            while not pidfile.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert pidfile.exists()
            proc.send_signal(signal.SIGTERM)
            assert proc.wait(timeout=10) == 1
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


@pytest.mark.parametrize('p2p,off,container,output,fail', [
    (True, True, False, '', True),
    (False, True, True, '', False),
    (True, False, True, '', False),
    (True, False, True, "nvcc fatal: Could not open output file '/tmp/tmpxft_test'", True),
    (True, False, True, 'Using /models/target as model source', False),
    (True, False, True, 'nvcc -I/opt/include /opt/source.cu -o /cache/tmp/run/kernel.o', False),
    (True, False, True, 'nvcc -o /opt/extension/kernel.o /opt/source.cu', True),
    (True, False, True, 'g++ --output-file=/root/kernel.so /opt/source.cpp', True),
    (True, False, True, 'nvcc -o /cache/../tmp/kernel.o', True),
    (True, False, True, 'nvcc -o/tmp/kernel.o', True),
    (True, False, True, 'Emitting ninja build file /root/extensions/build.ninja...', True),
    (True, False, True, 'ninja: Entering directory `/tmp/build`', True),
    (True, False, True, 'Using /cache/compiled/key/.cache/torch_extensions as PyTorch extensions root', False),
    (True, False, True, 'FAILED: /tmp/kernel.o', True),
    (True, False, True, 'PermissionError: [Errno 13] Permission denied', True),
])
def test_boot_log_gates(tmp_path, p2p, off, container, output, fail):
    log = tmp_path / 'serve.log'
    text = '[p2p] P2P enabled: content check passed\n' if p2p else '[p2p] P2P disabled\n'
    if off:
        text += ('Custom all-reduce flags-in-data path requested '
                 '(VLLM_CUSTOM_ALLREDUCE_FLAGS=1) but off: extension did not build on every rank\n')
    log.write_text(text + output)
    if fail:
        with pytest.raises(boot.CheckError):
            boot.check_log(log, container)
    else:
        boot.check_log(log, container)


def test_boot_log_ignores_previous_launch(tmp_path):
    log = tmp_path / 'serve.log'
    old = b'nvcc -o /tmp/old.o\n'
    log.write_bytes(old + b'nvcc -o /cache/tmp/new.o\n')
    boot.check_log(log, container=True, offset=len(old))


def test_start_stops_log_failure_before_health(tmp_path):
    (tmp_path / '.env.example').write_text('')
    log = tmp_path / 'serve.log'
    log.write_text('[p2p] P2P enabled: content check passed\n'
                   'Custom all-reduce flags-in-data path requested '
                   '(VLLM_CUSTOM_ALLREDUCE_FLAGS=1) but off: fixture\n')
    source = (ROOT / 'start.sh').read_text().rsplit('main "$@"', 1)[0]
    script = tmp_path / 'start.sh'
    script.write_text(source + '''
SCRIPT_DIR="$TEST_RECIPE"
SERVE_LOG="$TEST_LOG"
LAUNCHED=1
health_ok() { echo contacted > "$HEALTH_RECORD"; return 0; }
do_stop() { echo stopped > "$STOP_RECORD"; }
wait_ready
''')
    run = subprocess.run(['bash', str(script)], capture_output=True, text=True,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES='',
                                  TEST_RECIPE=str(ROOT), TEST_LOG=str(log), BOOT_CHECK='1',
                                  HEALTH_RECORD=str(tmp_path / 'health'), STOP_RECORD=str(tmp_path / 'stop')))
    assert run.returncode == 1 and '[boot-check] FAIL' in run.stdout + run.stderr
    assert (tmp_path / 'stop').exists() and not (tmp_path / 'health').exists()
