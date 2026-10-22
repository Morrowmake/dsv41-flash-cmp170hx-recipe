#!/usr/bin/env python3
"""Two-request startup check; also supervises a directly launched server."""
import argparse
import errno
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import tempfile
import threading
from pathlib import Path
import urllib.error
import urllib.request

OK_PROMPT = 'Reply with exactly OK, with no punctuation or other text.'
DRAFT_PROMPT = ('Write a Python module with 100 simple arithmetic functions named '
                'add_1 through add_100. Each function adds its number to its argument. '
                'Output only the code, one function after another.')
COUNTERS = ('vllm:spec_decode_num_drafts_total',
            'vllm:spec_decode_num_draft_tokens_total',
            'vllm:spec_decode_num_accepted_tokens_total')
SUCCESS = 'vllm:request_success_total'
TRACKED = (*COUNTERS, SUCCESS)


class CheckError(RuntimeError):
    pass


def banner(message):
    print('[boot-check] ' + message, flush=True)


def check_log(path, container=False, offset=0):
    with Path(path).open('rb') as source:
        source.seek(offset)
        text = source.read().decode(errors='replace')
    p2p = re.search(r'\[p2p\] P2P enabled:', text)
    requested_off = re.search(
        r'Custom all-reduce flags-in-data path requested[^\n]*'
        r'VLLM_CUSTOM_ALLREDUCE_FLAGS=1[^\n]*but off', text)
    if p2p and requested_off:
        raise CheckError('P2P enabled but requested flags-in-data all-reduce is off')
    if not container:
        return
    for line in text.splitlines():
        # Output options only: compiler inputs and system include paths are read-only.
        outputs = re.findall(
            r"(?:Could not open output file\s+|(?:^|\s)(?:(?:-o|-MF|-odir)(?:=|\s*)|(?:--output-file|--output-directory)(?:=|\s+)))['\"]?(/[^'\"\s]+)", line)
        outputs += re.findall(r'FAILED:\s+(/[^\s]+)', line)
        outputs += re.findall(
            r"(?:Emitting ninja build file\s+|ninja: Entering directory\s+|ninja\s+-C\s+)['\"`]?(/[^'\"`\s]+)", line)
        outputs += re.findall(r'Using (/[^\s]+) as PyTorch extensions root', line)
        for output in outputs:
            if not Path(output).is_relative_to('/cache') or '..' in Path(output).parts:
                raise CheckError('JIT compiler output outside /cache: ' + output)
        if re.search(r'(?:nvcc|ptxas|gcc|g\+\+).*fatal|PermissionError:|Permission denied', line, re.I):
            raise CheckError('compiler or filesystem failure in container log')


class Client:
    def __init__(self, base, model, key=''):
        self.base, self.model, self.key = base.rstrip('/'), model, key
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.ready_deadline = None
        self.ready_guard = None
        self.ready_retry = True

    def open(self, path, body=None, timeout=120):
        headers = {}
        if self.key:
            headers['Authorization'] = 'Bearer ' + self.key
        if body is not None:
            headers['Content-Type'] = 'application/json'
            body = json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, body, headers)
        while True:
            if self.ready_guard:
                self.ready_guard()
            remaining = (self.ready_deadline - time.monotonic()
                         if self.ready_deadline is not None else None)
            if remaining is not None and remaining <= 0:
                raise CheckError('server readiness timeout')
            try:
                return self.opener.open(request, timeout=min(timeout, remaining)
                                        if remaining is not None else timeout)
            except urllib.error.HTTPError as error:
                # An explicit unavailable response rejects the request during
                # warm-up. Auth and other API errors remain fatal.
                if (error.code != 503 or remaining is None
                        or not self.ready_retry):
                    raise
                error.close()
                time.sleep(min(1, max(0, self.ready_deadline - time.monotonic())))
            except (OSError, urllib.error.URLError) as error:
                reason = getattr(error, 'reason', error)
                # Never replay a POST whose delivery is uncertain. Only a refused
                # connection proves the generation request was not sent.
                retryable = body is None or (isinstance(reason, OSError)
                                            and reason.errno == errno.ECONNREFUSED)
                if remaining is None or not self.ready_retry or not retryable:
                    raise
                if self.ready_guard:
                    self.ready_guard()
                time.sleep(min(1, max(0, self.ready_deadline - time.monotonic())))

    def metrics(self):
        with self.open('/metrics', timeout=5) as response:
            return parse_metrics(response.read().decode(), self.model)

    def request(self, prompt, **kwargs):
        return self.open('/v1/chat/completions', dict(
            model=self.model, messages=[dict(role='user', content=prompt)],
            temperature=0, reasoning_effort='low', **kwargs))


def parse_metrics(text, model):
    result = {}
    for line in text.splitlines():
        match = re.fullmatch(r'([^\s{]+)(\{.*\})?\s+([^\s]+)(?:\s+\d+)?', line)
        if not match or match[1] not in TRACKED:
            continue
        labels = match[2] or ''
        label = re.search(r'(?:\{|,)model_name=("(?:\\.|[^"\\])*")', labels)
        if label and json.loads(label[1]) != model:
            continue
        value = float(match[3])
        if not math.isfinite(value) or value < 0:
            raise CheckError('invalid spec-decode counter')
        result[(match[1], labels)] = value
    if {key[0] for key in result} != set(TRACKED):
        raise CheckError('missing spec-decode counters for served model')
    return result


def deltas(before, after):
    if before.keys() != after.keys() or any(after[k] < v for k, v in before.items()):
        raise CheckError('spec-decode counters reset or changed series during boot')
    return tuple(sum(after[k] - before[k] for k in before if k[0] == name) for name in COUNTERS)


def check(client, metrics_wait=60, log_check=None):
    banner('enabled: two temperature-0 requests')
    if log_check:
        log_check()
    initial = client.metrics()
    with client.request(OK_PROMPT, max_tokens=128) as response:
        # A generation response completes the frontend handoff. Later transport
        # failures are engine/check failures, not startup readiness races.
        client.ready_retry = False
        answer = json.load(response)['choices'][0]['message'].get('content')
    if not isinstance(answer, str) or answer.strip() != 'OK':
        raise CheckError('expected exactly OK after stripping surrounding whitespace')
    # Snapshot after the short reply: only the long request should satisfy the gate.
    deadline = time.monotonic() + metrics_wait
    while True:
        before = client.metrics()
        deltas(initial, before)
        if sum(before[k] - initial[k] for k in initial if k[0] == SUCCESS) >= 1:
            break
        if time.monotonic() >= deadline:
            raise CheckError('OK request completion did not reach metrics')
        time.sleep(1)
    done, chunks = False, 0
    stream_deadline = time.monotonic() + 180
    with client.request(DRAFT_PROMPT, stream=True, max_tokens=768,
                        min_tokens=768, ignore_eos=True) as response:
        for line in response:
            if time.monotonic() >= stream_deadline:
                raise CheckError('generation stream timeout')
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                done = True
                break
            event = json.loads(data)
            if 'error' in event:
                raise CheckError('generation stream returned an error')
            if event.get('choices'):
                chunks += 1
    if not done or not chunks:
        raise CheckError('generation stream was empty or incomplete')
    deadline = time.monotonic() + metrics_wait
    while True:
        drafts, proposed, accepted = deltas(before, client.metrics())
        if drafts >= 64 and proposed >= 64 and accepted > 0:
            if log_check:
                log_check()
            banner(f'PASS: OK; drafts={drafts:g} proposed={proposed:g} accepted={accepted:g}')
            return
        if time.monotonic() >= deadline:
            raise CheckError(f'insufficient DFlash activity: drafts={drafts:g} '
                             f'proposed={proposed:g} accepted={accepted:g}; '
                             'need >=64 drafts and proposed tokens, >0 accepted')
        time.sleep(1)


def supervise(client, command, ready_timeout, container=False):
    log = tempfile.NamedTemporaryFile(prefix='boot-', suffix='.log')
    child = subprocess.Popen(command, start_new_session=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def relay():
        for line in iter(child.stdout.readline, b''):
            log.write(line)
            log.flush()
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()

    reader = threading.Thread(target=relay, daemon=True)
    reader.start()
    log_check = lambda: check_log(log.name, container)

    def interrupted(signum, frame):
        raise CheckError('server supervisor interrupted')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        client.ready_deadline = time.monotonic() + ready_timeout

        def ready_guard():
            log_check()
            if child.poll() is not None:
                raise CheckError('server exited before readiness')

        client.ready_guard = ready_guard
        with client.open('/health', timeout=5):
            pass
        check(client, log_check=log_check)
        client.ready_deadline = None
        client.ready_guard = None
        return child.wait()
    finally:
        # This group was created here; it contains only this launch's descendants.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        reader.join(timeout=5)
        child.stdout.close()
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base')
    parser.add_argument('--model')
    parser.add_argument('--log', type=Path)
    parser.add_argument('--log-offset', type=int, default=0)
    parser.add_argument('--log-only', action='store_true')
    parser.add_argument('--container', action='store_true')
    parser.add_argument('--ready-timeout', type=int, default=1800)
    parser.add_argument('--wait-ready', action='store_true')
    parser.add_argument('--serve', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.log_only and args.log is None:
        parser.error('--log-only requires --log')
    if not args.log_only and (not args.base or not args.model):
        parser.error('--base and --model are required for generation checks')
    log_check = (lambda: check_log(args.log, args.container, args.log_offset)) if args.log else None
    try:
        if args.log_only:
            log_check()
            return 0
        client = Client(args.base, args.model, os.getenv('API_KEY', os.getenv('VLLM_API_KEY', '')))
        if args.serve:
            return supervise(client, args.serve, args.ready_timeout, args.container)
        if args.wait_ready:
            client.ready_deadline = time.monotonic() + args.ready_timeout
            client.ready_guard = log_check
            with client.open('/health', timeout=5):
                pass
        check(client, log_check=log_check)
        return 0
    except Exception as error:
        # HTTP errors must not echo request headers or credentials.
        banner('FAIL: ' + (str(error) if isinstance(error, CheckError) else type(error).__name__))
        return 1


if __name__ == '__main__':
    sys.exit(main())
