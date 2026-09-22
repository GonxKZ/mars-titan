import importlib
import socket

import pytest

URL = "https://www.fool.com/investing/2022/06/01/results/"


def module():
    try:
        return importlib.import_module("mars_titan.data.news_fetch")
    except ModuleNotFoundError:
        pytest.fail("Falta la adquisición recuperable de capturas")


def fetcher(tmp_path, monkeypatch, responses, **kwargs):
    mod = module()
    requests = []

    def request(url, *, timeout, max_bytes):
        requests.append(url)
        result = responses[url]
        if isinstance(result, Exception):
            raise result
        status, headers, body = result
        if len(body) > max_bytes:
            raise mod.SourceUnavailable("response_too_large")
        return status, headers, body

    monkeypatch.setattr(mod, "_request", request)
    client = mod.ArticleFetcher(tmp_path / "evidence", **kwargs)
    return client, requests


def responses(**changes):
    return {
        "https://www.fool.com/robots.txt": (200, {}, b"User-agent: *\nAllow: /\n"),
        URL: (200, {"content-type": "text/html"}, b"<html>prueba</html>"),
        **changes,
    }


def test_capture_is_reused_after_restart_without_network(tmp_path, monkeypatch):
    client, requests = fetcher(tmp_path, monkeypatch, responses(), min_interval=0)
    capture = client.fetch(URL)
    assert capture.body == b"<html>prueba</html>"
    again = module().ArticleFetcher(tmp_path / "evidence").cached(URL)
    assert capture == again
    assert client.fetch(URL) == capture
    assert len(requests) == 2


@pytest.mark.parametrize(
    "url",
    [
        "http://www.fool.com/a",
        "https://127.0.0.1/a",
        "https://localhost/a",
        "https://www.fool.com.evil.test/a",
        "https://user:secret@www.fool.com/a",
        "https://www.fool.com:8443/a",
        "https://www.fool.com/a#fragment",
    ],
)
def test_unsafe_destination_is_rejected_before_transport(tmp_path, monkeypatch, url):
    client, requests = fetcher(tmp_path, monkeypatch, responses())
    with pytest.raises(ValueError):
        client.fetch(url)
    assert requests == []


def test_external_redirect_is_not_followed(tmp_path, monkeypatch):
    client, requests = fetcher(
        tmp_path,
        monkeypatch,
        responses(
            **{
                URL: (302, {"location": "https://example.test/secret"}, b""),
            }
        ),
        min_interval=0,
    )
    with pytest.raises(ValueError):
        client.fetch(URL)
    assert requests == ["https://www.fool.com/robots.txt", URL]


def test_robots_disallow_prevents_article_request(tmp_path, monkeypatch):
    reply = responses(
        **{
            "https://www.fool.com/robots.txt": (
                200,
                {},
                b"User-agent: *\nDisallow: /investing/\n",
            )
        }
    )
    client, requests = fetcher(tmp_path, monkeypatch, reply, min_interval=0)
    with pytest.raises(module().SourceUnavailable, match="robots_disallow"):
        client.fetch(URL)
    assert len(requests) == 1


def test_crawl_delay_survives_restart(tmp_path, monkeypatch):
    reply = responses(
        **{
            "https://www.fool.com/robots.txt": (
                200,
                {},
                b"User-agent: *\nCrawl-delay: 30\n",
            )
        }
    )
    client, requests = fetcher(tmp_path, monkeypatch, reply, min_interval=0)
    with pytest.raises(module().SourceUnavailable, match="rate_limit") as caught:
        client.fetch(URL)
    assert caught.value.retry_at > 0
    again = module().ArticleFetcher(tmp_path / "evidence", min_interval=0)
    with pytest.raises(module().SourceUnavailable, match="rate_limit"):
        again.fetch(URL)
    assert len(requests) == 1


@pytest.mark.parametrize("status", [403, 429, 503])
def test_http_refusal_is_persisted_without_retries_or_fake_evidence(tmp_path, monkeypatch, status):
    client, requests = fetcher(
        tmp_path,
        monkeypatch,
        responses(
            **{
                URL: (status, {"retry-after": "120"}, b"rechazado"),
            }
        ),
        min_interval=0,
    )
    with pytest.raises(module().SourceUnavailable, match=f"http_{status}") as caught:
        client.fetch(URL)
    assert caught.value.retry_at > 0
    assert client.cached(URL) is None
    with pytest.raises(module().SourceUnavailable):
        client.fetch(URL)
    assert len(requests) == 2


def test_timeout_and_oversized_html_do_not_become_evidence(tmp_path, monkeypatch):
    client, _ = fetcher(tmp_path, monkeypatch, responses(**{URL: TimeoutError()}), min_interval=0)
    with pytest.raises(module().SourceUnavailable, match="network_error"):
        client.fetch(URL)
    assert client.cached(URL) is None


def test_private_dns_address_is_rejected_before_connect(monkeypatch):
    mod = module()
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
        ],
    )
    with pytest.raises(ValueError, match="pública"):
        mod._request(URL, timeout=1, max_bytes=128)


def test_wrong_content_type_is_not_cached(tmp_path, monkeypatch):
    client, _ = fetcher(
        tmp_path,
        monkeypatch,
        responses(
            **{
                URL: (200, {"content-type": "application/octet-stream"}, b"datos"),
            }
        ),
        min_interval=0,
    )
    with pytest.raises(module().SourceUnavailable, match="content_type"):
        client.fetch(URL)
    assert client.cached(URL) is None


def test_compressed_capture_corruption_is_detected(tmp_path, monkeypatch):
    import sqlite3

    client, _ = fetcher(tmp_path, monkeypatch, responses(), min_interval=0)
    client.fetch(URL)
    with sqlite3.connect(tmp_path / "evidence" / "captures.sqlite") as db:
        db.execute("UPDATE captures SET sha256=?", ("0" * 64,))
    with pytest.raises(ValueError, match="huella"):
        client.cached(URL)


def test_database_quota_applies_to_capture_writes_on_new_connections(tmp_path, monkeypatch):
    import random

    client, _ = fetcher(
        tmp_path,
        monkeypatch,
        responses(
            **{
                URL: (200, {"content-type": "text/html"}, random.Random(7).randbytes(200_000)),
            }
        ),
        min_interval=0,
        quota_bytes=65536,
    )
    with pytest.raises(module().SourceUnavailable, match="cache_write_error"):
        client.fetch(URL)
    assert client.cached(URL) is None
    assert client.path.stat().st_size <= 65536


def test_retry_after_http_date_is_respected():
    from email.utils import formatdate

    now = 1000
    assert module()._retry_time(formatdate(1120, usegmt=True), now) == 1120


def test_oversized_response_is_not_cached(tmp_path, monkeypatch):
    client, _ = fetcher(
        tmp_path,
        monkeypatch,
        responses(
            **{
                URL: (200, {"content-type": "text/html"}, b"a" * (4 * 1024**2 + 1)),
            }
        ),
        min_interval=0,
    )
    with pytest.raises(module().SourceUnavailable, match="response_too_large"):
        client.fetch(URL)
    assert client.cached(URL) is None


def test_robots_wildcard_blocks_matching_article_path(tmp_path, monkeypatch):
    client, requests = fetcher(
        tmp_path,
        monkeypatch,
        responses(
            **{
                "https://www.fool.com/robots.txt": (
                    200,
                    {},
                    b"User-agent: *\nDisallow: /investing/*\n",
                ),
            }
        ),
        min_interval=0,
    )
    with pytest.raises(module().SourceUnavailable, match="robots_disallow"):
        client.fetch(URL)
    assert len(requests) == 1


def test_interrupted_robots_storage_does_not_confirm_partial_policy(tmp_path, monkeypatch):
    import sqlite3

    client, _ = fetcher(tmp_path, monkeypatch, responses(), min_interval=0)
    with sqlite3.connect(client.path) as db:
        db.execute(
            "CREATE TRIGGER interrupt_robots BEFORE INSERT ON state "
            "WHEN NEW.key='robots_fetched' BEGIN SELECT RAISE(ABORT,'corte simulado'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        client.fetch(URL)
    with sqlite3.connect(client.path) as db:
        keys = {r[0] for r in db.execute("SELECT key FROM state WHERE key LIKE 'robots_%'")}
    assert keys in (set(), {"robots_text", "robots_fetched", "robots_expires"})


def transport(monkeypatch, response):
    import io
    import ssl

    class Socket:
        def __init__(self, *args):
            self.destination = None
            self.server_hostname = None
            self.sent = b""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def connect(self, destination):
            self.destination = destination

        def settimeout(self, value):
            assert 0 < value <= 5

        def sendall(self, value):
            self.sent += value

        def makefile(self, mode):
            assert mode == "rb"
            return io.BytesIO(response)

        def close(self):
            pass

    channel = Socket()

    class TLS:
        def wrap_socket(self, raw, *, server_hostname):
            assert raw is channel
            channel.server_hostname = server_hostname
            return channel

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
        ],
    )
    monkeypatch.setattr(socket, "socket", lambda *a, **k: channel)
    monkeypatch.setattr(ssl, "create_default_context", TLS)
    return channel


def test_transport_keeps_sni_and_pinned_address_with_bounded_response(monkeypatch):
    channel = transport(monkeypatch, b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nbody")
    status, _, body = module()._request(URL, timeout=5, max_bytes=8)
    assert status == 200 and body == b"body"
    assert channel.destination == ("93.184.216.34", 443)
    assert channel.server_hostname == "www.fool.com"
    assert b"Host: www.fool.com" in channel.sent


@pytest.mark.parametrize(
    "headers,body,reason",
    [
        (b"Content-Length: 9\r\n", b"body", "response_too_large"),
        (b"", b"a" * 9, "response_too_large"),
        (b"Content-Encoding: gzip\r\n", b"body", "unexpected_encoding"),
        (b"Content-Length: 8\r\n", b"body", "truncated_response"),
    ],
)
def test_transport_rejects_incomplete_or_unbounded_payload(monkeypatch, headers, body, reason):
    transport(monkeypatch, b"HTTP/1.1 200 OK\r\n" + headers + b"\r\n" + body)
    with pytest.raises(module().SourceUnavailable, match=reason):
        module()._request(URL, timeout=5, max_bytes=8)
