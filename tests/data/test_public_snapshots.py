"""Capturas públicas idempotentes: reutilización, 304, interrupciones e índice, sin red."""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mars_titan.data import public_snapshots
from scripts import refresh_public_sources

VIX = b"DATE,OPEN,HIGH,LOW,CLOSE\n09/18/2026,15,16,14,15.5\n"
VIX_NEW = VIX + b"09/19/2026,15.5,17,15,16.5\n"
ETAG = '"vix-1"'
MODIFIED = "Fri, 18 Sep 2026 14:00:00 GMT"
URL = "https://example.org/data.csv"


@pytest.fixture
def refresh(tmp_path, monkeypatch):
    monkeypatch.setattr(refresh_public_sources, "ROOT", tmp_path)
    monkeypatch.setattr(refresh_public_sources.time, "sleep", lambda _: None)
    return refresh_public_sources


def catalog_file(tmp_path, *identifiers, **policy):
    entries = [
        {
            "id": identifier,
            "provider": "Proveedor público",
            "url": "https://example.org/catalog",
            "download_url": URL,
            "format": "csv",
            "validator": "vix_csv",
            "status": "downloaded_validated",
            "auth": "none",
        }
        for identifier in identifiers or ("cboe_vix",)
    ]
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps({"schema_version": 1, "sources": entries, "download_policy": policy}),
        encoding="utf-8",
    )
    return path


def response(body, *, status=200, etag=None, last_modified=None, conditional=False):
    return body, {
        "http_status": status,
        "effective_url": URL,
        "content_type": "text/csv",
        "received_bytes": len(body),
        "elapsed_seconds": 0.1,
        "conditional_request": conditional,
        "etag": etag,
        "last_modified": last_modified,
    }


class Provider:
    """Proveedor falso que anota los validadores recibidos en cada petición."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.conditions = []

    def __call__(self, url, max_bytes, timeout, user_agent, conditions=None):
        self.conditions.append(conditions)
        return self.responses.pop(0)


def at(minute):
    return datetime(2026, 10, 10, 9, minute, 0, tzinfo=UTC)


def run(refresh, tmp_path, monkeypatch, provider, minute, catalog=None):
    monkeypatch.setattr(refresh, "fetch_url", provider)
    catalog = catalog or catalog_file(tmp_path)
    return refresh.run_refresh(catalog, tmp_path / "data/external", run_at=at(minute))


def tree(root, *excluded):
    """Huella de cada archivo del proyecto falso, salvo las rutas excluidas."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not any(path.is_relative_to(root / item) for item in excluded)
    }


def test_identical_download_references_the_confirmed_bytes(refresh, tmp_path, monkeypatch):
    first = run(refresh, tmp_path, monkeypatch, Provider(response(VIX)), 0)
    second = run(refresh, tmp_path, monkeypatch, Provider(response(VIX)), 1)
    original = first["manifest"]["records"][0]
    record = second["manifest"]["records"][0]
    assert record["status"] == "downloaded_validated"
    assert record["sha256"] == original["sha256"] == hashlib.sha256(VIX).hexdigest()
    assert record["local_path"] == original["local_path"]
    assert record["content_reused"] is True
    assert record["reused_from_run_id"] == first["manifest"]["run_id"]
    assert second["manifest"]["retained_bytes"] == 0
    assert second["manifest"]["reused_bytes"] == len(VIX)
    # La segunda ejecución solo confirma su manifiesto: no hay otra copia de los datos.
    assert [p.name for p in second["manifest_path"].parent.iterdir()] == ["manifest.json"]
    index = json.loads(second["index_path"].read_text())
    assert [capture["run_id"] for capture in index["captures"]] == [
        first["manifest"]["run_id"],
        second["manifest"]["run_id"],
    ]
    assert index["referenced_bytes"] == 2 * len(VIX)
    assert index["distinct_bytes"] == index["stored_bytes"] == len(VIX)


def test_not_modified_revalidates_and_references_the_previous_content(
    refresh, tmp_path, monkeypatch, capsys
):
    first = run(
        refresh,
        tmp_path,
        monkeypatch,
        Provider(response(VIX, etag=ETAG, last_modified=MODIFIED)),
        0,
    )
    provider = Provider(response(b"", status=304, conditional=True))
    monkeypatch.setattr(refresh, "CATALOG", catalog_file(tmp_path))
    monkeypatch.setattr(refresh, "OUTPUT_ROOT", tmp_path / "data/external")
    monkeypatch.setattr(refresh, "fetch_url", provider)
    # Un 304 con el contenido anterior validado no hace fallar la orden.
    assert refresh.main(["--source", "cboe_vix"]) == 0
    assert provider.conditions == [
        {
            "etag": ETAG,
            "last_modified": MODIFIED,
            "sha256": hashlib.sha256(VIX).hexdigest(),
            "run_id": first["manifest"]["run_id"],
        }
    ]
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    manifest = json.loads((tmp_path / summary["manifest"]).read_text())
    record = manifest["records"][0]
    assert record["status"] == "not_modified" and record["http_status"] == 304
    assert record["conditional_request"] is True
    assert record["format_validation"]["valid"] is True
    assert record["local_path"] == first["manifest"]["records"][0]["local_path"]
    assert record["content_reused"] is True
    # La respuesta 304 no trae validadores propios y se conservan los anteriores.
    assert (record["etag"], record["last_modified"]) == (ETAG, MODIFIED)
    assert record["available_at_utc"] is None and record["benchmark_eligible"] is False
    assert manifest["retained_bytes"] == 0


def test_a_304_without_conditions_is_a_failure(refresh, tmp_path, monkeypatch):
    result = run(refresh, tmp_path, monkeypatch, Provider(response(b"", status=304)), 0)
    record = result["manifest"]["records"][0]
    assert record["status"] == "failed" and record["local_path"] is None
    assert record["failure_reason"] == "Estado HTTP no satisfactorio"


def test_changed_content_after_conditional_request_is_a_new_snapshot(
    refresh, tmp_path, monkeypatch
):
    run(refresh, tmp_path, monkeypatch, Provider(response(VIX, etag=ETAG)), 0)
    provider = Provider(response(VIX_NEW, etag='"vix-2"', conditional=True))
    result = run(refresh, tmp_path, monkeypatch, provider, 1)
    assert provider.conditions[0]["etag"] == ETAG
    record = result["manifest"]["records"][0]
    assert record["content_reused"] is False and record["etag"] == '"vix-2"'
    assert (tmp_path / record["local_path"]).read_bytes() == VIX_NEW
    # La tercera petición compara con la versión más reciente, no con la primera.
    third = Provider(response(b"", status=304, conditional=True))
    final = run(refresh, tmp_path, monkeypatch, third, 2)
    assert third.conditions[0]["etag"] == '"vix-2"'
    assert final["manifest"]["records"][0]["sha256"] == hashlib.sha256(VIX_NEW).hexdigest()


def test_only_the_latest_valid_capture_provides_validators(refresh, tmp_path, monkeypatch):
    run(refresh, tmp_path, monkeypatch, Provider(response(VIX, etag=ETAG)), 0)
    run(refresh, tmp_path, monkeypatch, Provider(response(VIX_NEW)), 1)
    failed = Provider(response(b"<html>error</html>", etag='"html"'))
    run(refresh, tmp_path, monkeypatch, failed, 2)
    provider = Provider(response(VIX_NEW))
    run(refresh, tmp_path, monkeypatch, provider, 3)
    # La última captura válida no trajo validadores y la fallida no cuenta.
    assert failed.conditions == [None] and provider.conditions == [None]


def test_validators_only_apply_to_the_same_source_and_url(refresh, tmp_path, monkeypatch):
    first = run(refresh, tmp_path, monkeypatch, Provider(response(VIX, etag=ETAG)), 0)
    index = json.loads(first["index_path"].read_text())
    assert public_snapshots.validators(index, "cboe_vix", URL)["etag"] == ETAG
    # Las consultas con fechas en la URL, como BCE o Treasury, piden otra respuesta.
    assert public_snapshots.validators(index, "cboe_vix", URL + "?endPeriod=2026-10-11") is None
    assert public_snapshots.validators(index, "otra_fuente", URL) is None


def test_corrupted_previous_content_is_downloaded_and_stored_again(refresh, tmp_path, monkeypatch):
    first = run(refresh, tmp_path, monkeypatch, Provider(response(VIX, etag=ETAG)), 0)
    stored = tmp_path / first["manifest"]["records"][0]["local_path"]
    stored.write_bytes(VIX.replace(b"15.5", b"99.9"))
    provider = Provider(response(VIX, etag=ETAG))
    second = run(refresh, tmp_path, monkeypatch, provider, 1)
    # Sin contenido íntegro no se envían validadores ni se acepta remitir a esa copia.
    assert provider.conditions == [None]
    record = second["manifest"]["records"][0]
    assert record["content_reused"] is False
    assert (tmp_path / record["local_path"]).read_bytes() == VIX
    content = json.loads(second["index_path"].read_text())["contents"][record["sha256"]]
    assert content["local_path"] == record["local_path"] and content["available"] is True
    assert content["first_run_id"] == first["manifest"]["run_id"]


def test_content_that_changes_after_the_index_is_not_served_for_a_304(
    refresh, tmp_path, monkeypatch
):
    first = run(refresh, tmp_path, monkeypatch, Provider(response(VIX, etag=ETAG)), 0)
    stored = tmp_path / first["manifest"]["records"][0]["local_path"]

    def provider(url, *args):
        # El archivo cambia entre la construcción del índice y la respuesta 304.
        stored.write_bytes(b"truncado")
        return response(b"", status=304, conditional=True)

    result = run(refresh, tmp_path, monkeypatch, provider, 1)
    record = result["manifest"]["records"][0]
    assert record["status"] == "failed" and record["local_path"] is None
    assert "ya no está disponible" in record["failure_reason"]


def test_interrupted_run_is_reported_and_its_files_are_never_reused(refresh, tmp_path, monkeypatch):
    publish = refresh.write_new_file

    def interrupted(destination, content):
        if destination.name == "manifest.json":
            raise RuntimeError("Corte antes de confirmar el manifiesto")
        publish(destination, content)

    monkeypatch.setattr(refresh, "write_new_file", interrupted)
    with pytest.raises(RuntimeError, match="Corte"):
        run(refresh, tmp_path, monkeypatch, Provider(response(VIX, etag=ETAG)), 0)
    monkeypatch.setattr(refresh, "write_new_file", publish)
    lost = tmp_path / "data/external" / at(0).strftime("%Y%m%dT%H%M%S.%fZ")
    assert (lost / "cboe_vix.csv").read_bytes() == VIX and not (lost / "manifest.json").exists()
    provider = Provider(response(VIX, etag=ETAG))
    result = run(refresh, tmp_path, monkeypatch, provider, 1)
    # El bloqueo se liberó, la ejecución sin manifiesto no aporta validadores ni bytes y
    # el fallo previo queda a la vista.
    assert provider.conditions == [None]
    assert result["manifest"]["previous_interrupted_runs"] == [lost.name]
    record = result["manifest"]["records"][0]
    assert record["content_reused"] is False
    assert not record["local_path"].startswith(f"data/external/{lost.name}/")
    index = json.loads(result["index_path"].read_text())
    assert index["interrupted_runs"] == [lost.name]
    assert index["interrupted_bytes"] == len(VIX)
    assert [capture["run_id"] for capture in index["captures"]] == [result["manifest"]["run_id"]]


def test_a_failed_index_write_leaves_a_confirmed_capture_for_the_next_run(
    refresh, tmp_path, monkeypatch
):
    def broken(root, index):
        raise OSError("Disco lleno al escribir el índice")

    write_index = public_snapshots.write_index
    monkeypatch.setattr(public_snapshots, "write_index", broken)
    with pytest.raises(OSError, match="índice"):
        run(refresh, tmp_path, monkeypatch, Provider(response(VIX)), 0)
    monkeypatch.setattr(public_snapshots, "write_index", write_index)
    second = run(refresh, tmp_path, monkeypatch, Provider(response(VIX)), 1)
    assert second["manifest"]["records"][0]["content_reused"] is True
    index = json.loads(second["index_path"].read_text())
    assert len(index["captures"]) == 2 and index["interrupted_runs"] == []


def test_index_includes_the_initial_manifest_and_is_a_pure_rebuild(refresh, tmp_path, monkeypatch):
    initial_file = tmp_path / "data/external/2026-09-18/cboe-vix-history.csv"
    initial_file.parent.mkdir(parents=True)
    initial_file.write_bytes(VIX)
    initial = tmp_path / public_snapshots.INITIAL
    initial.parent.mkdir(parents=True)
    initial.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "records": [
                    {
                        "source_id": "cboe_vix",
                        "status": "downloaded_validated",
                        "http_status": 200,
                        "requested_url": URL,
                        "sha256": hashlib.sha256(VIX).hexdigest(),
                        "bytes": len(VIX),
                        "local_path": "data/external/2026-09-18/cboe-vix-history.csv",
                    },
                    {"source_id": "sec", "status": "failed", "http_status": 403},
                ],
            }
        ),
        encoding="utf-8",
    )
    before = initial.read_bytes()
    result = run(refresh, tmp_path, monkeypatch, Provider(response(VIX)), 0)
    record = result["manifest"]["records"][0]
    assert record["reused_from_run_id"] == public_snapshots.INITIAL_RUN
    assert record["local_path"] == "data/external/2026-09-18/cboe-vix-history.csv"
    assert initial.read_bytes() == before
    written = json.loads(result["index_path"].read_text())
    assert written["initial_manifest"] == public_snapshots.INITIAL
    assert written["captures"][0]["manifest_sha256"] == hashlib.sha256(before).hexdigest()
    assert written["benchmark_eligible"] is False
    rebuilt = public_snapshots.build_index(tmp_path, tmp_path / "data/external")
    assert json.loads(json.dumps(rebuilt)) == written


def test_disk_limit_stops_new_bytes_but_allows_references(refresh, tmp_path, monkeypatch):
    catalog = catalog_file(tmp_path, max_store_bytes=len(VIX) + 8)
    run(refresh, tmp_path, monkeypatch, Provider(response(VIX)), 0, catalog)
    grown = run(refresh, tmp_path, monkeypatch, Provider(response(VIX_NEW)), 1, catalog)
    record = grown["manifest"]["records"][0]
    assert record["status"] == "failed" and "disco" in record["failure_reason"]
    assert [p.name for p in grown["manifest_path"].parent.iterdir()] == ["manifest.json"]
    same = run(refresh, tmp_path, monkeypatch, Provider(response(VIX)), 2, catalog)
    assert same["manifest"]["records"][0]["content_reused"] is True
    assert same["manifest"]["limits"]["max_store_bytes"] == len(VIX) + 8


def test_disk_limit_counts_interrupted_runs(refresh, tmp_path, monkeypatch):
    lost = tmp_path / "data/external/20261010T080000.000000Z"
    lost.mkdir(parents=True)
    (lost / ".pending-truncado").write_bytes(b"x" * 40)
    catalog = catalog_file(tmp_path, max_store_bytes=len(VIX) + 39)
    result = run(refresh, tmp_path, monkeypatch, Provider(response(VIX)), 0, catalog)
    assert "disco" in result["manifest"]["records"][0]["failure_reason"]


def test_a_second_update_on_the_same_store_is_rejected(refresh, tmp_path, monkeypatch):
    provider = Provider(response(VIX))
    with public_snapshots.exclusive_update(tmp_path / "data/external"):
        with pytest.raises(ValueError, match="otra actualización"):
            run(refresh, tmp_path, monkeypatch, provider, 0)
    assert provider.conditions == []
    assert [p.name for p in (tmp_path / "data/external").iterdir()] == [public_snapshots.LOCK]


def test_a_corrupt_confirmed_manifest_stops_before_any_request(refresh, tmp_path, monkeypatch):
    folder = tmp_path / "data/external/20261010T080000.000000Z"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({"run_id": "otro"}), encoding="utf-8")
    provider = Provider(response(VIX))
    with pytest.raises(ValueError, match="no corresponde"):
        run(refresh, tmp_path, monkeypatch, provider, 0)
    assert provider.conditions == []


@pytest.mark.parametrize(
    "local_path", ["/etc/passwd", "../fuera.csv", "data/external/../../fuera.csv"]
)
def test_manifest_paths_cannot_leave_the_project(tmp_path, local_path):
    folder = tmp_path / "data/external/20261010T080000.000000Z"
    folder.mkdir(parents=True)
    record = dict(
        source_id="cboe_vix",
        status="downloaded_validated",
        sha256="0" * 64,
        bytes=1,
        local_path=local_path,
    )
    document = dict(schema_version=1, run_id=folder.name, records=[record])
    (folder / "manifest.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="fuera del proyecto"):
        public_snapshots.build_index(tmp_path, tmp_path / "data/external")


def test_refresh_only_adds_its_capture_and_the_index(refresh, tmp_path, monkeypatch):
    frozen = tmp_path / "dataset/benchmark.parquet"
    frozen.parent.mkdir()
    frozen.write_bytes(b"benchmark congelado")
    catalog_file(tmp_path)
    before = tree(tmp_path)
    run(refresh, tmp_path, monkeypatch, Provider(response(VIX)), 0)
    run(refresh, tmp_path, monkeypatch, Provider(response(VIX_NEW)), 1)
    assert tree(tmp_path, "data/external", public_snapshots.INDEX) == before


def test_stored_validators_with_line_breaks_are_never_sent(tmp_path):
    document = dict(
        schema_version=1,
        records=[
            dict(
                source_id="cboe_vix",
                status="downloaded_validated",
                requested_url=URL,
                sha256=hashlib.sha256(VIX).hexdigest(),
                bytes=len(VIX),
                local_path="data/vix.csv",
                etag='"a"\r\nX-Inyectada: 1',
            )
        ],
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "data/vix.csv").write_bytes(VIX)
    initial = tmp_path / "initial.json"
    initial.write_text(json.dumps(document), encoding="utf-8")
    index = public_snapshots.build_index(
        tmp_path, tmp_path / "data/external", initial="initial.json"
    )
    assert index["captures"][0]["records"][0]["etag"] is None
    assert public_snapshots.validators(index, "cboe_vix", URL) is None
    with pytest.raises(ValueError, match="no admitido"):
        refresh_public_sources.conditional_headers({"etag": '"a"\r\nX-Inyectada: 1'})


def test_response_validators_come_from_the_final_response():
    raw = (
        'HTTP/1.1 301 Moved Permanently\r\nLocation: https://example.org/b\r\nETag: "old"\r\n'
        "\r\nHTTP/2 200\r\ncontent-type: text/csv\r\n"
        f"last-modified: {MODIFIED}\r\n\r\n"
    )
    assert refresh_public_sources.response_validators(raw) == {
        "etag": None,
        "last_modified": MODIFIED,
    }
    assert refresh_public_sources.response_validators("") == {"etag": None, "last_modified": None}


def test_transport_sends_conditions_and_reads_a_304_without_body(monkeypatch):
    commands = []

    def curl_fixture(command, **kwargs):
        commands.append(command)
        Path(command[command.index("--dump-header") + 1]).write_text(
            f"HTTP/2 304\r\netag: {ETAG}\r\nlast-modified: {MODIFIED}\r\n\r\n", encoding="latin-1"
        )
        written = {
            "http_code": 304,
            "url_effective": URL,
            "content_type": None,
            "size_download": 0,
            "time_total": 0.01,
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(written), "")

    monkeypatch.setattr(refresh_public_sources.subprocess, "run", curl_fixture)
    conditions = {"etag": ETAG, "last_modified": MODIFIED, "sha256": "0" * 64, "run_id": "x"}
    body, receipt = refresh_public_sources.fetch_url(URL, 1024, 45, "research", conditions)
    assert body == b"" and receipt["http_status"] == 304
    assert receipt["conditional_request"] is True
    assert (receipt["etag"], receipt["last_modified"]) == (ETAG, MODIFIED)
    command = commands[0]
    sent = [command[i + 1] for i, item in enumerate(command) if item == "--header"]
    assert sent == [f"If-None-Match: {ETAG}", f"If-Modified-Since: {MODIFIED}"]
    # Sin validadores la petición es completa y un 304 sin cuerpo no se acepta.
    with pytest.raises(refresh_public_sources.FetchError, match="ausente"):
        refresh_public_sources.fetch_url(URL, 1024, 45, "research")
    assert "--header" not in commands[1]
