"""Run the installed-style entry points against isolated databases."""

import json
import os
import subprocess
import sys

import pytest


@pytest.fixture
def cli_env(tmp_path):
    env = os.environ.copy()
    for name in list(env):
        if name.startswith('MEMORIA_') or name == 'ANTHROPIC_API_KEY':
            env.pop(name)
    env.update({
        'MEMORIA_DB': str(tmp_path / 'memory.db'), 'MEMORIA_SCOPE': 'global',
        'MEMORIA_NO_MODELS': '1', 'MEMORIA_NO_LLM': '1',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'MEMORIA_HOOK_STATE_DIR': str(tmp_path / 'hook-state'),
    })
    return env


def run_cli(env, *args):
    return subprocess.run([sys.executable, '-m', 'memoria', *args], env=env,
                          text=True, capture_output=True, timeout=20)


def test_remember_recall_backup_check_roundtrip(cli_env, tmp_path):
    remembered = run_cli(cli_env, 'remember', 'remember the zephyrmarker')
    assert remembered.returncode == 0, remembered.stderr
    source = json.loads(remembered.stdout)['conversation_id']
    recalled = run_cli(cli_env, 'recall', 'zephyrmarker', '--json')
    assert recalled.returncode == 0, recalled.stderr
    assert json.loads(recalled.stdout)['results'][0]['provenance']['id'] == source
    backup = run_cli(cli_env, 'storage', 'backup', str(tmp_path / 'snapshot.db'))
    assert backup.returncode == 0, backup.stderr
    assert json.loads(backup.stdout)['verified'] is True
    checked = run_cli(cli_env, 'storage', 'check')
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)['ok'] is True


def test_cli_errors_are_nonzero_and_concise(cli_env):
    result = run_cli(cli_env, 'recall', 'query', '-k', '-1')
    assert result.returncode == 1
    assert 'top_k' in result.stderr
    assert 'Traceback' not in result.stderr
    result = run_cli(cli_env, 'ingest', '/nonexistent-memoria-test-path')
    assert result.returncode == 1


def test_mcp_subprocess_roundtrip_and_invalid_message_recovery(cli_env):
    messages = [
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2025-11-25'}},
        {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
        {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call', 'params': {
            'name': 'memoria_remember', 'arguments': {'text': 'remember the zephyrmarker'},
        }},
        {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {
            'name': 'memoria_recall', 'arguments': {'query': 'zephyrmarker', 'format': 'json'},
        }},
    ]
    payload = 'not json\n' + '\n'.join(json.dumps(message) for message in messages) + '\n'
    result = subprocess.run([sys.executable, '-m', 'memoria', 'serve'], env=cli_env,
                            input=payload, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(responses) == 4  # malformed message + three requests, no notification response
    assert responses[0]['error']['code'] == -32700
    output = json.loads(responses[-1]['result']['content'][0]['text'])
    assert output['count'] == 1
    assert output['results'][0]['provenance']['content'] == 'remember the zephyrmarker'


def test_http_subprocess_honors_cli_database_and_returns_provenance(cli_env, tmp_path):
    from http.client import HTTPConnection
    import selectors
    import signal
    target = tmp_path / 'explicit.db'
    process = subprocess.Popen(
        [sys.executable, '-m', 'memoria', '--db', str(target), '--no-models',
         'serve-http', '--port', '0'], env=cli_env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    started = False
    try:
        with selectors.DefaultSelector() as ready:
            ready.register(process.stdout, selectors.EVENT_READ)
            assert ready.select(timeout=10), 'HTTP server did not become ready'
        line = process.stdout.readline()
        assert 'http://127.0.0.1:' in line, line
        port = int(line.strip().rsplit(':', 1)[1])
        started = True

        def post(path, payload):
            connection = HTTPConnection('127.0.0.1', port, timeout=5)
            try:
                connection.request('POST', path, json.dumps(payload), {'Content-Type': 'application/json'})
                response = connection.getresponse()
                output = json.loads(response.read())
                assert response.status == 200, output
                return output
            finally:
                connection.close()
        source = post('/remember', {'text': 'remember the zephyrmarker'})['conversation_id']
        response = post('/recall', {'query': 'zephyrmarker', 'format': 'json'})
        assert response['results'][0]['provenance']['id'] == source
        assert target.exists()
        assert not os.path.exists(cli_env['MEMORIA_DB'])
    finally:
        if started:
            process.send_signal(signal.SIGINT)
        else:
            # Preserve a readiness assertion instead of masking it with a
            # second timeout while interrupting a process that never started.
            process.kill()
        try:
            _, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
    assert process.returncode == 0, stderr
