"""Servidor en directo del observatorio: solo lectura, límites y desconexiones."""

import json
import math
import os
import socket
import struct
import threading
import time
from email.utils import formatdate
from http.client import HTTPConnection
from pathlib import Path

import pytest

from mars_titan.observatory.live_server import Hub, Limits, LiveObservatory, campaign_state
from mars_titan.observatory.telemetry import FIELDS, SystemProbe, TelemetryRing
from mars_titan.observatory.traces import write_trace_bundle


class FakeNvml:
    name = "GPU de prueba"

    def sample(self):
        return dict.fromkeys(FIELDS[:7], math.nan) | {"gpu_temp_c": 61.0}


class FakeProbe:
    nvml = FakeNvml()
    disk_path = "/"

    def __init__(self):
        self.calls = 0

    def sample(self):
        self.calls += 1
        return {name: float(self.calls) for name in FIELDS}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def tree(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<!doctype html><title>prueba</title>")
    (site / "app.js").write_text("export {};")
    public = tmp_path / "public"
    dump(
        public / "observatory.json",
        {
            "generated_at": "2026-10-10T00:00:00Z",
            "runs": [],
            "pagination": {"total_runs": 0, "pages": []},
        },
    )
    page = "a" * 64
    dump(public / "pages" / f"{page}.json", {"runs": []})
    (tmp_path / "secret.txt").write_text("privado")
    return tmp_path


def start(tree, **options):
    options.setdefault("probe", FakeProbe())
    server = LiveObservatory(
        ("127.0.0.1", 0), site_dir=tree / "site", public_dir=tree / "public", **options
    )
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    return server, thread


def stop(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(5)


def request(server, path, headers=None, method="GET", host=None):
    connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    connection.putrequest(method, path, skip_host=True)
    connection.putheader("Host", host or f"127.0.0.1:{server.server_address[1]}")
    for name, value in (headers or {}).items():
        connection.putheader(name, value)
    connection.endheaders()
    response = connection.getresponse()
    body = response.read()
    connection.close()
    return response, body


class EventStream:
    """Cliente SSE mínimo sobre un socket, para controlar lecturas y cierres."""

    def __init__(self, server):
        self.socket = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=5)
        port = server.server_address[1]
        self.socket.sendall(f"GET /api/events HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode())
        self.buffer = b""
        head = self._until(b"\r\n\r\n")
        self.status = int(head.split(b" ")[1])

    def _until(self, marker):
        while marker not in self.buffer:
            chunk = self.socket.recv(65536)
            if not chunk:
                raise ConnectionError("El servidor cerró la conexión")
            self.buffer += chunk
        part, self.buffer = self.buffer.split(marker, 1)
        return part

    def next(self):
        while True:
            block = self._until(b"\n\n").decode()
            fields = dict(
                line.split(": ", 1) for line in block.splitlines() if not line.startswith(":")
            )
            if "event" in fields:
                return fields["event"], json.loads(fields["data"]), fields.get("id")

    def close(self):
        self.socket.close()


def test_static_files_use_validators_and_conditional_requests(tree):
    server, thread = start(tree)
    try:
        response, body = request(server, "/")
        assert response.status == 200 and b"prueba" in body
        assert "frame-ancestors 'none'" in response.getheader("Content-Security-Policy")
        response, body = request(server, "/data/observatory.json")
        tag, modified = response.getheader("ETag"), response.getheader("Last-Modified")
        assert response.status == 200 and response.getheader("Cache-Control") == "no-cache"
        again, body = request(server, "/data/observatory.json", {"If-None-Match": tag})
        assert again.status == 304 and body == b""
        since, _ = request(server, "/data/observatory.json", {"If-Modified-Since": modified})
        assert since.status == 304
        older = formatdate(time.time() - 86400 * 400, usegmt=True)
        stale, _ = request(server, "/data/observatory.json", {"If-Modified-Since": older})
        assert stale.status == 200
        page, _ = request(server, f"/data/pages/{'a' * 64}.json")
        assert "immutable" in page.getheader("Cache-Control")
        head, body = request(server, "/data/observatory.json", method="HEAD")
        assert head.status == 200 and body == b""
    finally:
        stop(server, thread)


@pytest.mark.parametrize(
    "path",
    [
        "/../secret.txt",
        "/%2e%2e/secret.txt",
        "/data/../secret.txt",
        "/data/pages/abc.json",
        "/site/app.js",
        "/.git/config",
    ],
)
def test_paths_outside_the_allowlist_are_not_served(tree, path):
    server, thread = start(tree)
    try:
        response, body = request(server, path)
        assert response.status == 404 and b"privado" not in body
    finally:
        stop(server, thread)


def test_symlinks_and_foreign_hosts_are_rejected(tree):
    os.symlink(tree / "secret.txt", tree / "site" / "linked.js")
    server, thread = start(tree)
    try:
        response, body = request(server, "/linked.js")
        assert response.status == 403 and b"privado" not in body
        response, _ = request(server, "/", host="attacker.example:80")
        assert response.status == 421
        response, _ = request(server, "/", method="POST")
        assert response.status == 405
    finally:
        stop(server, thread)


def test_default_bind_is_loopback_only(tree):
    with pytest.raises(ValueError, match="interfaz local"):
        LiveObservatory(("0.0.0.0", 0), site_dir=tree / "site", probe=FakeProbe())


def test_sync_then_index_change_is_pushed(tree):
    server, thread = start(tree, limits=Limits(source_seconds=0.5, heartbeat_seconds=2))
    stream = EventStream(server)
    try:
        assert stream.status == 200
        name, payload, _ = stream.next()
        assert name == "sync" and payload["status"]["mode"] == "live"
        assert set(payload["backlog"]) == {"t", *FIELDS}
        while "index" not in payload.get("topics", {}):
            name, event, _ = stream.next()
            if name == "index":
                payload = {"topics": {"index": event}}
        first = payload["topics"]["index"]["etag"]
        time.sleep(0.01)
        dump(
            tree / "public" / "observatory.json",
            {"generated_at": "2026-10-10T00:01:00Z", "runs": [{"status": "running"}]},
        )
        while True:
            name, event, _ = stream.next()
            if name == "index":
                break
        assert event["etag"] != first and event["running"] == 1
    finally:
        stream.close()
        stop(server, thread)


def test_client_limit_rejects_extra_streams_and_releases_on_disconnect(tree):
    server, thread = start(tree, limits=Limits(max_clients=1, heartbeat_seconds=1))
    first = EventStream(server)
    try:
        assert first.status == 200
        second = EventStream(server)
        assert second.status == 503
        second.close()
        first.close()
        # La desconexión se detecta en el siguiente latido como máximo.
        deadline = time.monotonic() + 5
        while server.hub.clients and time.monotonic() < deadline:
            time.sleep(0.1)
        assert server.hub.clients == 0
        third = EventStream(server)
        assert third.status == 200
        third.close()
    finally:
        stop(server, thread)


def test_event_rate_is_limited_and_changes_are_coalesced(tree):
    limits = Limits(events_per_second=2.0, heartbeat_seconds=5)
    server, thread = start(tree, limits=limits)
    stream = EventStream(server)
    try:
        stream.next()
        started = time.monotonic()
        for value in range(40):
            server.hub.publish("probe", {"value": value})
            time.sleep(0.02)
        received = []
        while not received or received[-1] != 39:
            name, event, _ = stream.next()
            if name == "probe":
                received.append(event["value"])
        elapsed = time.monotonic() - started
        assert len(received) <= math.ceil(elapsed * limits.events_per_second) + 1
        assert len(received) < 40 and received == sorted(received)
    finally:
        stream.close()
        stop(server, thread)


def test_connection_budget_answers_503_without_spawning_threads(tree):
    limits = Limits(max_clients=1, max_connections=2)
    server, thread = start(tree, limits=limits)
    held = [
        socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=5)
        for _ in range(2)
    ]
    try:
        time.sleep(0.2)
        extra = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=5)
        assert extra.recv(64).startswith(b"HTTP/1.1 503")
        extra.close()
    finally:
        for connection in held:
            connection.close()
        stop(server, thread)


def test_limits_reject_unbounded_configurations():
    for options in (
        {"max_clients": 0},
        {"events_per_second": 100},
        {"max_connections": 2, "max_clients": 4},
    ):
        with pytest.raises(ValueError):
            Limits(**options)


def test_hub_keeps_only_the_latest_version_and_skips_identical_payloads():
    hub = Hub(Limits())
    assert hub.publish("topic", {"a": 1}) is True
    assert hub.publish("topic", {"a": 1}) is False
    hub.publish("topic", {"a": 2})
    pending = hub.changes({}, 0.01)
    assert [(name, version, json.loads(body)) for name, version, body in pending] == [
        ("topic", 2, {"a": 2})
    ]
    with pytest.raises(ValueError):
        Hub(Limits(max_event_bytes=1024)).publish("big", {"x": "y" * 2048})


def test_campaign_state_reads_jobs_without_locks(tmp_path):
    folder = tmp_path / "campaign"
    jobs = {
        "US+CN/fold-000/gru/search-gru-00": True,
        "US+CN/fold-000/gru/finalist-s43": False,
        "US/fold-001/transformer_compact/search-transformer-01": False,
    }
    dump(
        folder / "summary.json",
        {"kind": "historical_masked_campaign_run", "status": "running", "jobs": jobs},
    )
    dump(folder / "jobs/US+CN/fold-000/gru/search-gru-00/receipt.json", {"status": "completed"})
    attempt = folder / "jobs/US+CN/fold-000/gru/finalist-s43/attempt-0001"
    dump(
        attempt / "run.json",
        {
            "global_step": 9,
            "epochs": [
                {
                    "epoch": 1,
                    "train": {"mae": 0.2, "samples_per_second": 10},
                    "validation": {"mae": -1},
                }
            ],
        },
    )
    (folder / ".lock").write_text("")
    state = campaign_state("a", folder, Limits())
    states = {tuple(cell[:4]): cell[4] for cell in state["cells"]}
    assert sorted(states.values()) == ["attempt", "done", "pending"]
    assert state["vocabulary"]["scopes"] == ["US+CN", "US"]
    assert state["active"][0]["epochs"][0] == {
        "epoch": 1,
        "train_mae": 0.2,
        "mae": None,
        "session_mae": None,
        "train_samples_per_second": 10,
        "train_seconds": None,
    }
    done = next(cell for cell in state["cells"] if cell[4] == "done")
    assert done[5].endswith("Z")
    dump(folder / "summary.json", {"jobs": {"../escape/x/y": True}})
    with pytest.raises(ValueError):
        campaign_state("a", folder, Limits())


def test_telemetry_ring_is_bounded_and_chronological():
    ring = TelemetryRing(3)
    for index in range(5):
        ring.append(index, {"gpu_temp_c": float(index), "ram_used_mib": math.nan})
    latest = ring.latest(10)
    assert latest["t"] == [2, 3, 4] and latest["gpu_temp_c"] == [2, 3, 4]
    assert latest["ram_used_mib"] == [None, None, None]
    assert ring.latest(1)["t"] == [4]


def test_system_probe_reports_absent_gpu_as_null_not_zero(tmp_path):
    class Missing:
        name = None

        def sample(self):
            return dict.fromkeys(FIELDS[:7], math.nan)

    probe = SystemProbe(tmp_path, nvml=Missing())
    probe.sample()
    # El uso de CPU compara dos lecturas, así que necesita que avancen los contadores.
    time.sleep(0.2)
    values = probe.sample()
    assert math.isnan(values["gpu_temp_c"]) and math.isnan(values["gpu_mem_used_mib"])
    assert values["ram_total_mib"] > 0 and values["disk_free_gib"] >= 0
    assert 0 <= values["cpu_util_pct"] <= 100


def test_trace_bundles_round_trip_with_alignment_and_absences(tmp_path):
    folder = tmp_path / "traces"
    manifest = write_trace_bundle(
        folder,
        "fixture-titans",
        run_id="run-1",
        attempt_id="attempt-0001",
        model_id="titans_mac_online",
        provenance="fixture",
        x_unit="optimizer_step",
        series=[
            {
                "id": "titans.surprise",
                "group": "titans",
                "label": "Sorpresa",
                "unit": "pérdida",
                "x": [0, 64, 128],
                "y": [1.5, math.nan, 0.25],
            }
        ],
        matrices=[
            {
                "id": "titans.forget_gate",
                "group": "titans",
                "label": "Olvido",
                "unit": "fracción",
                "rows": ["capa 1", "capa 2"],
                "x": [0, 64, 128],
                "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "range": [0, 1],
            }
        ],
    )
    blob = (folder / manifest["blob"]).read_bytes()
    assert len(blob) == manifest["blob_bytes"]
    series = manifest["series"][0]
    for reference in (series["x"], series["y"], manifest["matrices"][0]["values"]):
        assert reference["offset"] % 8 == 0
    y = struct.unpack_from("<3f", blob, series["y"]["offset"])
    assert y[0] == 1.5 and math.isnan(y[1]) and y[2] == 0.25
    index = json.loads((folder / "index.json").read_text())
    assert [entry["name"] for entry in index["bundles"]] == ["fixture-titans"]
    old = manifest["blob"]
    updated = write_trace_bundle(
        folder,
        "fixture-titans",
        run_id="run-1",
        attempt_id="attempt-0001",
        model_id="m",
        provenance="fixture",
        x_unit="epoch",
        series=[{"id": "loss", "group": "optimization", "label": "Pérdida", "x": [1], "y": [2]}],
    )
    assert updated["blob"] != old and not (folder / old).exists()


@pytest.mark.parametrize(
    "item",
    [
        {"id": "a", "group": "titans", "label": "x", "x": [1, 1], "y": [0, 0]},
        {"id": "a", "group": "titans", "label": "x", "x": [1, 2], "y": [0, math.inf]},
        {"id": "a", "group": "other", "label": "x", "x": [1], "y": [0]},
        {"id": "A b", "group": "titans", "label": "x", "x": [1], "y": [0]},
    ],
)
def test_trace_bundles_reject_invalid_series(tmp_path, item):
    with pytest.raises(ValueError):
        write_trace_bundle(
            tmp_path,
            "bad",
            run_id="r",
            attempt_id="a",
            model_id="m",
            provenance="measured",
            x_unit="epoch",
            series=[item],
        )


def test_traces_are_served_with_their_own_byte_budget(tree):
    folder = tree / "traces"
    write_trace_bundle(
        folder,
        "bundle",
        run_id="r",
        attempt_id="a",
        model_id="m",
        provenance="fixture",
        x_unit="epoch",
        series=[
            {
                "id": "loss",
                "group": "optimization",
                "label": "Pérdida",
                "x": list(range(1000)),
                "y": [0.5] * 1000,
            }
        ],
    )
    server, thread = start(tree, traces_dir=folder)
    try:
        response, body = request(server, "/data/traces/index.json")
        manifest_name = json.loads(body)["bundles"][0]["manifest"]
        response, body = request(server, f"/data/traces/{manifest_name}")
        blob = json.loads(body)["blob"]
        response, body = request(server, f"/data/traces/{blob}")
        assert (
            response.status == 200
            and response.getheader("Content-Type") == "application/octet-stream"
        )
        assert len(body) == json.loads((folder / manifest_name).read_text())["blob_bytes"]
    finally:
        stop(server, thread)


def test_cli_refuses_remote_hosts_without_explicit_permission(tree):
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[2] / "scripts/serve_observatory.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--site",
            str(tree / "site"),
            "--host",
            "0.0.0.0",
            "--port",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=os.environ | {"PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
    )
    assert result.returncode == 2 and "interfaz local" in result.stderr
