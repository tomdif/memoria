"""Wire-level MCP framing and argument validation without model downloads."""

import io
import json

import pytest

import memoria.mcp_server as mcp


def rpc(method, **params):
    return {'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params}


def test_initialize_negotiates_versions_and_ping():
    for version in mcp.SUPPORTED_PROTOCOL_VERSIONS:
        response = mcp._rpc_response(rpc('initialize', protocolVersion=version))
        assert response['result']['protocolVersion'] == version
    assert mcp._rpc_response(rpc('ping'))['result'] == {}


@pytest.mark.parametrize('envelope', [None, [], 42, {}, {'method': 'ping'}, {'jsonrpc': '2.0', 'method': 2},
    {'jsonrpc': '2.0', 'method': 'ping', 'id': True}])
def test_invalid_envelope_returns_protocol_error(envelope):
    assert mcp._rpc_response(envelope)['error']['code'] == -32600


@pytest.mark.parametrize('name,arguments', [
    ('memoria_remember', {}), ('memoria_recall', []),
    ('memoria_recall', {'query': 'x', 'top_k': True}),
    ('memoria_recall', {'query': 'x', 'mode': 'unknown'}),
    ('memoria_storage', {'action': 'retain', 'keep_latest': 0, 'apply': 'false'}),
    ('memoria_storage', {'vacuum': 1}), ('memoria_recall', {'query': 'x', 'typo': 2}),
    ('unknown', {}), ([], {}),
])
def test_invalid_arguments_do_not_initialize_database(name, arguments, monkeypatch):
    def fail():
        raise AssertionError('database initialized for invalid request')
    monkeypatch.setattr(mcp, 'get_memoria', fail)
    result = mcp._rpc_response(rpc('tools/call', name=name, arguments=arguments))
    assert result['error']['code'] == -32602


def test_notifications_do_not_respond_or_mutate(monkeypatch):
    def fail(*args):
        raise AssertionError('notification invoked a tool')
    monkeypatch.setattr(mcp, 'handle_tool', fail)
    for method in ['notifications/initialized', 'notifications/cancelled', 'tools/call', 'unknown']:
        assert mcp._rpc_response({'jsonrpc': '2.0', 'method': method}) is None


def test_tool_failures_are_marked_as_errors(monkeypatch):
    monkeypatch.setattr(mcp, 'handle_tool', lambda *_: {'error': 'entity missing'})
    response = mcp._rpc_response(rpc('tools/call', name='memoria_entity', arguments={'name': 'Missing'}))
    assert response['result']['isError'] is True


def test_stdio_survives_bad_json_and_oversized_request(monkeypatch):
    lines = ['{', 'null', '[' + ' ' * 200 + ']', json.dumps(rpc('ping'))]
    stdin = io.StringIO('\n'.join(lines) + '\n')
    stdout = io.StringIO()
    monkeypatch.setattr(mcp.sys, 'stdin', stdin)
    monkeypatch.setattr(mcp.sys, 'stdout', stdout)
    monkeypatch.setattr(mcp, 'MAX_REQUEST_CHARS', 150)
    monkeypatch.setattr(mcp, '_memoria', None)
    mcp.run_mcp_server()
    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [r['error']['code'] for r in responses[:3]] == [-32700, -32600, -32600]
    assert responses[-1]['result'] == {}


def test_tool_output_cannot_pollute_jsonrpc_stdout(monkeypatch, capsys):
    def noisy_tool(*args):
        print('model loading progress')
        return {'ok': True}
    monkeypatch.setattr(mcp, 'handle_tool', noisy_tool)
    response = mcp._rpc_response(rpc('tools/call', name='memoria_stats'))
    assert response['result']['content']
    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'model loading progress' in captured.err
