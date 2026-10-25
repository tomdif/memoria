"""Exercise HTTP request parsing, authentication, and dispatch over real sockets."""

from http.client import HTTPConnection
from http.server import HTTPServer
import json
import threading

import pytest

import memoria.http_server as http
from tests.test_interfaces import FakeMemoria


@pytest.fixture
def api(monkeypatch):
    fake = FakeMemoria()
    monkeypatch.setattr(http, '_memoria', fake)
    monkeypatch.delenv('MEMORIA_HTTP_TOKEN', raising=False)
    monkeypatch.delenv('MEMORIA_ALLOWED_ORIGINS', raising=False)
    server = HTTPServer(('127.0.0.1', 0), http.MemoriaHandler)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01})
    thread.start()

    def request(path='/health', method='GET', body=None, headers=None):
        connection = HTTPConnection('127.0.0.1', server.server_port, timeout=2)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), json.loads(response.read())
        finally:
            connection.close()
    yield request, fake
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


@pytest.mark.parametrize('body', ['[1]', 'null', 'true', '42', '"query"', '{', '{"top_k": NaN}', '{"query": "x", "top_k": Infinity}'])
def test_invalid_json_never_reaches_backend(api, body):
    request, fake = api
    status, _, payload = request('/recall', 'POST', body, {'Content-Type': 'application/json'})
    assert status == 400
    assert 'error' in payload
    assert fake.calls == []


@pytest.mark.parametrize('body', [{}, {'query': ''}, {'query': []}, {'query': 'x', 'top_k': True},
    {'query': 'x', 'top_k': -1}, {'query': 'x', 'mode': 'invalid'}, {'query': 'x', 'format': 'xml'}])
def test_invalid_recall_parameters_are_client_errors(api, body):
    request, fake = api
    assert request('/recall', 'POST', json.dumps(body), {'Content-Type': 'application/json'})[0] == 400
    assert not fake.calls


@pytest.mark.parametrize('path,body', [
    ('/storage/retain', {'keep_latest': 0, 'apply': 'false'}),
    ('/storage/restore', {'archive_path': '/tmp/archive', 'apply': 1}),
    ('/storage/maintain', {'vacuum': 'false'}),
    ('/storage/maintain', {'apply_stale_cleanup': 'false'}),
])
def test_string_boolean_cannot_apply_storage_mutations(api, path, body):
    request, fake = api
    assert request(path, 'POST', json.dumps(body), {'Content-Type': 'application/json'})[0] == 400
    assert not fake.calls


@pytest.mark.parametrize('path', ['/', '/recall/extra', '/storage/retain/extra', '/unknown'])
def test_unknown_post_routes_are_404(api, path):
    request, fake = api
    assert request(path, 'POST')[0] == 404
    assert not fake.calls


def test_health_does_not_open_database_or_load_models(api, monkeypatch):
    request, _ = api
    def fail():
        raise AssertionError('health opened a database')
    monkeypatch.setattr(http, 'get_memoria', fail)
    status, headers, payload = request()
    assert status == 200
    assert payload['status'] == 'ok'
    assert 'Access-Control-Allow-Origin' not in headers
    assert headers['Cache-Control'] == 'no-store'


def test_body_limits_and_content_type(api):
    request, _ = api
    assert request('/recall', 'POST', '{}', {'Content-Type': 'text/plain'})[0] == 415
    assert request('/recall', 'POST', '', {'Content-Length': str(http.MAX_BODY_BYTES + 1)})[0] == 413
    assert request('/recall', 'POST', '', {'Content-Length': '-1'})[0] == 400
    assert request('/recall', 'POST', '', {'Content-Length': 'oops'})[0] == 400
    assert request('/recall', 'POST', '', {'Transfer-Encoding': 'chunked'})[0] == 400


def test_browser_origins_are_denied_unless_explicitly_allowed(api, monkeypatch):
    request, _ = api
    assert request(headers={'Origin': 'https://untrusted.example'})[0] == 403
    monkeypatch.setenv('MEMORIA_ALLOWED_ORIGINS', 'https://trusted.example')
    status, headers, _ = request(headers={'Origin': 'https://trusted.example'})
    assert status == 200
    assert headers['Access-Control-Allow-Origin'] == 'https://trusted.example'
    assert headers['Vary'] == 'Origin'
    assert request(method='OPTIONS', headers={'Origin': 'https://untrusted.example'})[0] == 403


def test_rebinding_host_is_rejected(api):
    request, _ = api
    assert request(headers={'Host': 'untrusted.example'})[0] == 403


def test_bearer_authentication_covers_health_and_mutations(api, monkeypatch):
    request, fake = api
    monkeypatch.setenv('MEMORIA_HTTP_TOKEN', 'test-only-token')
    assert request()[0] == 401
    assert request(headers={'Authorization': 'Bearer wrong'})[0] == 401
    assert request('/storage/retain', 'POST', '{}', {'Content-Type': 'application/json'})[0] == 401
    assert request(headers={'Authorization': 'Bearer test-only-token'})[0] == 200
    assert not fake.calls


def test_non_loopback_binding_requires_authentication(monkeypatch):
    monkeypatch.delenv('MEMORIA_HTTP_TOKEN', raising=False)
    with pytest.raises(ValueError, match='MEMORIA_HTTP_TOKEN'):
        http.run_http_server(host='0.0.0.0', port=0)


def test_internal_errors_do_not_expose_server_paths(api, monkeypatch):
    request, fake = api
    def fail():
        raise RuntimeError('/private/internal/path and sensitive details')
    monkeypatch.setattr(fake, 'graph_stats', fail)
    status, _, payload = request('/stats')
    assert status == 500
    assert payload == {'error': 'Internal server error'}
