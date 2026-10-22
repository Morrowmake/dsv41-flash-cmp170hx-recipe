"""Readiness retries must stop at the original deadline and first reply."""
import errno
import importlib.util
import io
import http.server
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
import urllib.error
from unittest.mock import Mock

import pytest

ROOT = Path(os.environ.get('RECIPE_TEST_ROOT', Path(__file__).resolve().parents[2]))
spec = importlib.util.spec_from_file_location('recipe_boot_check', ROOT / 'boot_check.py')
boot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boot)


@pytest.fixture
def client(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(boot.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(boot.time, 'sleep', lambda duration: now.__setitem__(0, now[0] + duration))
    result = boot.Client('http://127.0.0.1:8000', 'test')
    result.ready_deadline = 13.0
    result.opener = Mock()
    return result


def unavailable(path='/metrics', code=503):
    return urllib.error.HTTPError(path, code, 'not ready', {}, io.BytesIO())


@pytest.mark.parametrize('path,body', [('/health', None), ('/metrics', None),
                                     ('/v1/chat/completions', {'model': 'test'})])
def test_startup_503_then_ready(client, path, body):
    response = object()
    client.opener.open.side_effect = [unavailable(path), response]
    assert client.open(path, body) is response
    assert client.opener.open.call_count == 2
    assert client.opener.open.call_args.kwargs['timeout'] == 2.0


@pytest.mark.parametrize('body', [None, {'model': 'test'}])
def test_refused_connection_then_ready(client, body):
    response = object()
    client.opener.open.side_effect = [urllib.error.URLError(
        OSError(errno.ECONNREFUSED, 'connection refused')), response]
    assert client.open('/v1/chat/completions', body) is response
    assert client.opener.open.call_count == 2


def test_unavailable_expires_without_resetting_deadline(client):
    client.opener.open.side_effect = unavailable()
    with pytest.raises(boot.CheckError):
        client.open('/metrics')
    assert client.opener.open.call_count == 3
    assert client.ready_deadline == 13.0


@pytest.mark.parametrize('code', [400, 401, 403, 404, 500])
def test_semantic_errors_are_immediately_fatal(client, code):
    client.opener.open.side_effect = unavailable(code=code)
    with pytest.raises(urllib.error.HTTPError):
        client.open('/metrics')
    assert client.opener.open.call_count == 1


def test_uncertain_post_is_never_replayed(client):
    client.opener.open.side_effect = ConnectionResetError(errno.ECONNRESET, 'reset')
    with pytest.raises(ConnectionResetError):
        client.open('/v1/chat/completions', {'model': 'test'})
    assert client.opener.open.call_count == 1


def test_after_first_reply_503_is_fatal(client):
    client.ready_retry = False
    client.opener.open.side_effect = unavailable()
    with pytest.raises(urllib.error.HTTPError):
        client.open('/metrics')
    assert client.opener.open.call_count == 1


def test_without_startup_mode_503_is_fatal(client):
    client.ready_deadline = None
    client.opener.open.side_effect = unavailable()
    with pytest.raises(urllib.error.HTTPError):
        client.open('/metrics')
    assert client.opener.open.call_count == 1


@pytest.mark.parametrize('failure', ['transport', '503'])
def test_slow_frontend_handoff(failure, tmp_path):
    """Run the actual owner and boot-check CLI against a warming HTTP frontend."""
    state = {'ready_at': None, 'errors': 0, 'posts': 0, 'counts': [0, 0, 0, 0]}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, text, status=200):
            self.send_response(status)
            self.end_headers()
            self.wfile.write(text.encode())

        def do_GET(self):
            if self.path == '/health':
                self.send('{}')
                return
            if state['ready_at'] is None:
                state['ready_at'] = time.monotonic() + 1.2
            if time.monotonic() < state['ready_at']:
                state['errors'] += 1
                if failure == '503':
                    self.send('{}', 503)
                else:
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                return
            names = ('spec_decode_num_drafts_total', 'spec_decode_num_draft_tokens_total',
                     'spec_decode_num_accepted_tokens_total', 'request_success_total')
            self.send('\n'.join(f'vllm:{key} {value}'
                                for key, value in zip(names, state['counts'])))

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            state['posts'] += 1
            if body.get('stream'):
                state['counts'] = [64, 64, 1, 2]
                self.send('data: {"choices":[{"delta":{"content":"code"}}]}\n\n'
                          'data: [DONE]\n\n')
            else:
                state['counts'][3] += 1
                self.send('{"choices":[{"message":{"content":"OK"}}]}')

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source = (ROOT / 'start.sh').read_text()
    function = '\n'.join(re.search(r'^' + name + r'\(\) \{\n.*?^\}', source, re.M | re.S).group()
                         for name in ('container_alive', 'wait_ready'))
    log = tmp_path / 'serve.log'
    log.touch()
    script = '''set -euo pipefail
SCRIPT_DIR="$1"; PORT="$2"; PID="$3"; SERVE_LOG="$4"; STOPFILE="$5"
CLIENT_HOST=127.0.0.1; HOST=127.0.0.1; SERVED_MODEL_NAME=test
LAUNCHED=1; BOOT_CHECK=1; READY_TIMEOUT=8; DRY=""
log() { echo "$*"; }; warn() { echo "$*" >&2; }
container_is_ours() { kill -0 "$PID" 2>/dev/null; }
health_ok() { curl -fsS -m 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }
do_stop() { echo stopped >> "$STOPFILE"; }
kv_line() { :; }; served_id() { echo test; }
''' + function + '\nwait_ready\n'
    stop = tmp_path / 'stop'
    try:
        result = subprocess.run(
            ['bash', '-c', script, 'owner', str(ROOT), str(server.server_port),
             str(os.getpid()), str(log), str(stop)],
            env={**os.environ, 'API_KEY': 'readiness-test'},
            capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'PASS:' in result.stdout
        assert state['errors'] > 0
        assert state['posts'] == 2
        assert not stop.exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
