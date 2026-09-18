"""Salida pública del observatorio, sin datos científicos ni conexión de red."""

import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

SCRIPT = Path(
    os.environ.get(
        "OBSERVATORY_MODULE_PATH",
        Path(__file__).resolve().parents[2] / "scripts" / "export_observatory.py",
    )
)
NOW = datetime(2020, 1, 2, 12, tzinfo=UTC)


@pytest.fixture
def exporter():
    specification = importlib.util.spec_from_file_location("observatory_under_test", SCRIPT)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def snapshot(**changes):
    return {
        "schema_version": 1,
        "run_id": "run-01",
        "attempt_id": "attempt-02",
        "model_id": "M3",
        "variant_id": "compact-v1",
        "status": "running",
        "phase": "train",
        "heartbeat_at": "2020-01-02T12:00:00Z",
        "started_at": "2020-01-02T11:00:00Z",
        "updated_at": "2020-01-02T12:00:00Z",
        "completed_steps": 12,
        "total_steps": 40,
        "epoch": 2,
        "max_epochs": 5,
        "seed": 17,
        "fold": "fold-01",
        "comparison_contract": {
            "target": "residual-oc-v1",
            "universe": "universe-128-v1",
            "splits": "splits-v1",
            "evaluation_regime": "online-mature-labels-v1",
        },
        "metrics": {"mae": 0.02, "rank_ic": -0.1, "loss": 0.03},
        "history": [{"step": 12, "recorded_at": "2020-01-02T12:00:00Z", "loss": 0.03}],
        "checkpoint": {"step": 10, "saved_at": "2020-01-02T11:59:00Z", "resumable": True},
    } | changes


def write_snapshot(tmp_path, value, directory=None):
    path = tmp_path / "runs" / (directory or value["run_id"]) / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def export(exporter, tmp_path, **options):
    destination = tmp_path / "public" / "observatory.json"
    exporter.export_snapshot(tmp_path / "runs", destination, now=NOW, **options)
    return json.loads(destination.read_text(encoding="utf-8"))


def run_cli(tmp_path, *options):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-dir",
            str(tmp_path / "runs"),
            "--output",
            str(tmp_path / "public" / "observatory.json"),
            *options,
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )


def test_empty_directory_exports_catalog_without_inventing_runs(tmp_path):
    """Detecta un exportador que inventa entrenamientos o requiere fuentes inexistentes."""
    result = run_cli(tmp_path)

    assert result.returncode == 0, result.stderr
    document = json.loads((tmp_path / "public" / "observatory.json").read_text())
    assert document["schema_version"] == 1
    assert document["project"] == "MARS-TITAN"
    assert document["source_status"] == "no_runs_registered"
    assert document["runs"] == []
    assert [model["id"] for model in document["models"]] == [
        "B0",
        "B1",
        "B2",
        "B3",
        "B6",
        "M0",
        "M1",
        "M2",
        "M3",
    ]


def test_export_keeps_observed_values_and_uses_null_for_unknowns(exporter, tmp_path):
    """Detecta ceros fabricados, cambios de intento y pérdida de una métrica negativa válida."""
    write_snapshot(tmp_path, snapshot())
    document = export(exporter, tmp_path)
    run = document["runs"][0]

    assert document["source_status"] == "available"
    assert document["generated_at"] == "2020-01-02T12:00:00Z"
    assert run["attempt_id"] == "attempt-02"
    assert run["metrics"]["mae"] == 0.02
    assert run["metrics"]["rank_ic"] == -0.1
    assert run["metrics"]["vram_peak_mib"] is None
    assert run["history"] == [
        {"step": 12, "recorded_at": "2020-01-02T12:00:00Z", "loss": 0.03, "mae": None}
    ]
    assert run["checkpoint"] == {"step": 10, "saved_at": "2020-01-02T11:59:00Z", "resumable": True}
    assert run["test_released"] is False


def test_public_allowlist_discards_sensitive_fields_everywhere(exporter, tmp_path):
    """Detecta copiar el diccionario privado completo o metadatos arbitrarios anidados."""
    value = snapshot(secret="PRIVATE_SECRET", error="PRIVATE_SECRET", path="/PRIVATE_SECRET")
    value["metrics"]["token"] = "PRIVATE_SECRET"
    value["history"][0]["message"] = "PRIVATE_SECRET"
    value["checkpoint"]["path"] = "/PRIVATE_SECRET"
    value["notes"] = ["PRIVATE_SECRET"]
    write_snapshot(tmp_path, value)

    document = export(exporter, tmp_path)

    assert "PRIVATE_SECRET" not in json.dumps(document)
    assert set(document["runs"][0]) == {
        "run_id",
        "attempt_id",
        "model_id",
        "variant_id",
        "status",
        "phase",
        "heartbeat_at",
        "started_at",
        "updated_at",
        "completed_steps",
        "total_steps",
        "epoch",
        "max_epochs",
        "seed",
        "fold",
        "comparison_group",
        "metrics",
        "history",
        "checkpoint",
        "test_released",
    }


@pytest.mark.parametrize("phase", ["test", "evaluation", None])
def test_reserved_or_unidentified_phase_hides_all_metrics(exporter, tmp_path, phase):
    """Detecta que test_released privado o una fase ausente permitan filtrar el test."""
    write_snapshot(tmp_path, snapshot(phase=phase, status="completed", test_released=True))

    run = export(exporter, tmp_path)["runs"][0]

    assert all(value is None for value in run["metrics"].values())
    assert run["history"] == []
    assert run["test_released"] is False


def test_release_requires_explicit_cli_flag_and_terminal_run(tmp_path):
    """Detecta publicación implícita y liberación del test mientras todavía se ejecuta."""
    value = snapshot(phase="test", status="completed", test_released=True)
    write_snapshot(tmp_path, value)
    assert run_cli(tmp_path).returncode == 0
    destination = tmp_path / "public" / "observatory.json"
    assert json.loads(destination.read_text())["runs"][0]["metrics"]["mae"] is None

    released = run_cli(tmp_path, "--release-test")
    assert released.returncode == 0, released.stderr
    run = json.loads(destination.read_text())["runs"][0]
    assert run["test_released"] is True
    assert run["metrics"]["mae"] == 0.02

    previous = destination.read_bytes()
    write_snapshot(tmp_path, value | {"status": "running"})
    refused = run_cli(tmp_path, "--release-test")
    assert refused.returncode != 0
    assert destination.read_bytes() == previous


def test_comparison_group_changes_with_every_scientific_contract(exporter, tmp_path):
    """Detecta comparar fases o poblaciones diferentes mediante una etiqueta compartida."""
    value = snapshot(comparison_group="UNTRUSTED_GROUP")
    write_snapshot(tmp_path, value)
    original = export(exporter, tmp_path)["runs"][0]["comparison_group"]
    assert original and original != "UNTRUSTED_GROUP"

    groups = {original}
    for field in ("target", "universe", "splits", "evaluation_regime"):
        changed = snapshot()
        changed["comparison_contract"][field] += "-other"
        write_snapshot(tmp_path, changed)
        groups.add(export(exporter, tmp_path)["runs"][0]["comparison_group"])
    write_snapshot(tmp_path, snapshot(phase="validation"))
    groups.add(export(exporter, tmp_path)["runs"][0]["comparison_group"])
    assert len(groups) == 6

    write_snapshot(tmp_path, snapshot(comparison_contract=None))
    assert export(exporter, tmp_path)["runs"][0]["comparison_group"] is None


def test_stale_heartbeat_warns_without_fabricating_a_pause(exporter, tmp_path):
    """Detecta convertir ausencia de telemetría en un estado confirmado."""
    write_snapshot(tmp_path, snapshot(heartbeat_at="2020-01-02T11:00:00Z"))
    document = export(exporter, tmp_path)

    assert document["runs"][0]["status"] == "running"
    assert document["notes"]

    write_snapshot(tmp_path, snapshot(status="paused", heartbeat_at=None))
    assert export(exporter, tmp_path)["runs"][0]["status"] == "paused"


def test_history_is_bounded_and_attempts_are_not_merged(exporter, tmp_path):
    """Detecta crecimiento no acotado o arrastrar una curva del intento previo."""
    history = [
        {"step": step, "recorded_at": "2020-01-02T12:00:00Z", "loss": step / 1000}
        for step in range(503)
    ]
    write_snapshot(tmp_path, snapshot(history=history, completed_steps=502, total_steps=600))
    first = export(exporter, tmp_path)["runs"][0]
    assert len(first["history"]) == 500
    assert first["history"][0]["step"] == 3
    assert first["history"][-1]["step"] == 502

    write_snapshot(tmp_path, snapshot(attempt_id="attempt-03", history=[]))
    second = export(exporter, tmp_path)["runs"][0]
    assert second["attempt_id"] == "attempt-03"
    assert second["history"] == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("model_id", "UNKNOWN"),
        ("run_id", "../../private"),
        ("attempt_id", "<script>"),
        ("status", "alive"),
        ("phase", "secret"),
        ("completed_steps", -1),
        ("completed_steps", True),
        ("completed_steps", 41),
        ("epoch", 6),
        ("heartbeat_at", "2020-01-02T12:00:00"),
        ("heartbeat_at", "2020-01-03T12:00:00Z"),
        ("updated_at", "2020-01-02T10:00:00Z"),
        ("metrics", {"mae": float("nan")}),
        ("metrics", {"mae": float("inf")}),
        ("metrics", {"mae": -1}),
        ("metrics", {"mae": True}),
        ("metrics", {"rank_ic": 1.1}),
        ("metrics", {"coverage_80": 80}),
        ("checkpoint", {"step": 13}),
        ("checkpoint", {"resumable": "yes"}),
        ("checkpoint", {"resumable": True}),
        ("checkpoint", {"resumable": True, "step": 10}),
        ("checkpoint", {"resumable": True, "saved_at": "2020-01-02T11:59:00Z"}),
        ("history", [{"step": 12, "recorded_at": "2020-01-02T12:00:00Z", "phase": "test"}]),
        ("history", [{"step": 12, "recorded_at": "2020-01-02T12:00:00Z", "attempt_id": "old"}]),
        ("history", [{"step": 13, "recorded_at": "2020-01-02T12:00:00Z"}]),
    ],
)
def test_invalid_source_preserves_previous_snapshot(exporter, tmp_path, field, value):
    """Detecta validación omitida o reemplazo parcial de una instantánea válida."""
    destination = tmp_path / "public" / "observatory.json"
    destination.parent.mkdir()
    destination.write_bytes(b"PREVIOUS_VALID_SNAPSHOT")
    write_snapshot(tmp_path, snapshot(**{field: value}), directory="run-01")

    with pytest.raises(ValueError):
        export(exporter, tmp_path)

    assert destination.read_bytes() == b"PREVIOUS_VALID_SNAPSHOT"
    assert sorted(path.name for path in destination.parent.iterdir()) == ["observatory.json"]


@pytest.mark.parametrize("content", ["{", '{"schema_version":1,"schema_version":2}', "[[[[[[[[["])
def test_invalid_json_fails_cli_without_overwriting_or_echoing_source(tmp_path, content):
    """Detecta errores de parser publicados o escritura previa a validar toda la fuente."""
    path = write_snapshot(tmp_path, snapshot())
    path.write_text(content + "PRIVATE_SECRET")
    destination = tmp_path / "public" / "observatory.json"
    destination.parent.mkdir()
    destination.write_bytes(b"PREVIOUS")

    result = run_cli(tmp_path)

    assert result.returncode != 0
    assert destination.read_bytes() == b"PREVIOUS"
    assert "PRIVATE_SECRET" not in result.stderr + result.stdout


@pytest.mark.parametrize("link_kind", ["root", "run", "file"])
def test_symbolic_links_are_rejected(exporter, tmp_path, link_kind):
    """Detecta seguir un enlace hacia datos ajenos al directorio autorizado."""
    external = tmp_path / "external"
    external.mkdir()
    (external / "status.json").write_text(json.dumps(snapshot()))
    if link_kind == "root":
        (tmp_path / "runs").symlink_to(external, target_is_directory=True)
    else:
        (tmp_path / "runs").mkdir()
        run = tmp_path / "runs" / "run-01"
        if link_kind == "run":
            run.symlink_to(external, target_is_directory=True)
        else:
            run.mkdir()
            (run / "status.json").symlink_to(external / "status.json")

    with pytest.raises(ValueError):
        export(exporter, tmp_path)


def test_limits_reject_oversized_files_excess_runs_and_expired_budget(exporter, tmp_path):
    """Detecta lectura ilimitada o exportación después de agotar el presupuesto local."""
    write_snapshot(tmp_path, snapshot())
    with pytest.raises(ValueError):
        export(exporter, tmp_path, max_file_bytes=32)
    write_snapshot(tmp_path, snapshot(run_id="run-02"))
    with pytest.raises(ValueError):
        export(exporter, tmp_path, max_runs=1)
    with pytest.raises(ValueError):
        export(exporter, tmp_path, timeout_seconds=1e-12)


def test_output_is_deterministic_and_normalizes_timezone(exporter, tmp_path):
    """Detecta ordenar según el sistema de archivos o tratar horas locales como UTC."""
    write_snapshot(tmp_path, snapshot(run_id="run-02"))
    write_snapshot(tmp_path, snapshot(heartbeat_at="2020-01-02T13:00:00+01:00"))
    first = export(exporter, tmp_path)
    first_bytes = (tmp_path / "public" / "observatory.json").read_bytes()
    second = export(exporter, tmp_path)

    assert first == second
    assert (tmp_path / "public" / "observatory.json").read_bytes() == first_bytes
    assert [run["run_id"] for run in first["runs"]] == ["run-01", "run-02"]
    assert first["runs"][0]["heartbeat_at"] == "2020-01-02T12:00:00Z"


def test_missing_optional_observations_remain_null(exporter, tmp_path):
    """Detecta inventar progreso, estado o capacidad de reanudar."""
    write_snapshot(
        tmp_path,
        {"schema_version": 1, "run_id": "run-01", "attempt_id": "attempt-01", "model_id": "B0"},
    )
    run = export(exporter, tmp_path)["runs"][0]

    for field in ("status", "phase", "completed_steps", "total_steps", "epoch", "seed", "fold"):
        assert run[field] is None
    assert run["checkpoint"] == {"step": None, "saved_at": None, "resumable": None}
    assert run["history"] == []


def test_nested_private_data_cannot_bypass_depth_limit(exporter, tmp_path):
    """Detecta JSON profundamente anidado aunque el campo no sea exportable."""
    nested = None
    for _ in range(40):
        nested = {"private": nested}
    write_snapshot(tmp_path, snapshot(private=nested))

    with pytest.raises(ValueError):
        export(exporter, tmp_path)


def test_destination_cannot_replace_source_snapshot(exporter, tmp_path):
    """Detecta destruir el estado privado al seleccionar una salida dentro de la fuente."""
    source = write_snapshot(tmp_path, snapshot())
    previous = source.read_bytes()

    with pytest.raises(ValueError):
        exporter.export_snapshot(tmp_path / "runs", source, now=NOW)

    assert source.read_bytes() == previous


def test_output_parent_cannot_redirect_export_through_symlink(exporter, tmp_path):
    """Detecta crear la salida fuera del destino explícito siguiendo un enlace."""
    write_snapshot(tmp_path, snapshot())
    external = tmp_path / "external"
    external.mkdir()
    (tmp_path / "public").symlink_to(external, target_is_directory=True)

    with pytest.raises((ValueError, OSError)):
        export(exporter, tmp_path)

    assert list(external.iterdir()) == []


def test_release_parameter_cannot_be_a_truthy_string(exporter, tmp_path):
    """Detecta que el texto false se interprete como permiso para publicar el test."""
    write_snapshot(tmp_path, snapshot(phase="test", status="completed"))

    with pytest.raises(ValueError):
        export(exporter, tmp_path, release_test="false")


def test_output_size_limit_preserves_previous_snapshot(exporter, tmp_path):
    """Detecta publicar una salida que excede el presupuesto configurado."""
    write_snapshot(tmp_path, snapshot())
    export(exporter, tmp_path)
    destination = tmp_path / "public" / "observatory.json"
    previous = destination.read_bytes()

    with pytest.raises(ValueError):
        export(exporter, tmp_path, max_output_bytes=32)

    assert destination.read_bytes() == previous


def test_unknown_fold_does_not_claim_comparability(exporter, tmp_path):
    """Detecta agrupar métricas sin conocer la ventana evaluada."""
    write_snapshot(tmp_path, snapshot(fold=None))

    assert export(exporter, tmp_path)["runs"][0]["comparison_group"] is None


def test_duplicate_json_keys_are_rejected_before_publication(exporter, tmp_path):
    """Detecta aceptar dos valores competidores del mismo campo del contrato."""
    path = write_snapshot(tmp_path, snapshot())
    body = path.read_text()
    path.write_text(body.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'))

    with pytest.raises(ValueError):
        export(exporter, tmp_path)


def test_signed_finite_loss_is_preserved_in_metrics_and_history(exporter, tmp_path):
    """Detecta rechazar una pérdida válida con signo o alterar su curva."""
    value = snapshot(metrics={"loss": -0.75, "mae": 0.02})
    value["history"][0]["loss"] = -0.5
    write_snapshot(tmp_path, value)

    run = export(exporter, tmp_path)["runs"][0]

    assert run["metrics"]["loss"] == -0.75
    assert run["history"][0]["loss"] == -0.5
    assert run["metrics"]["mae"] == 0.02


def test_cli_snapshot_is_accepted_by_real_frontend_validator(tmp_path):
    """Detecta divergencias del contrato público entre el exportador y la web."""
    state = Path(__file__).resolve().parents[2] / "site" / "state.mjs"
    observed = snapshot(metrics={"loss": -0.75, "mae": 0.02})
    observed["history"][0]["loss"] = -0.5
    write_snapshot(tmp_path, observed)
    write_snapshot(
        tmp_path,
        {"schema_version": 1, "run_id": "run-02", "attempt_id": "attempt-01", "model_id": "B0"},
    )
    write_snapshot(
        tmp_path,
        snapshot(run_id="run-03", phase="test", status="completed", test_released=True),
    )
    exported = run_cli(tmp_path)
    assert exported.returncode == 0, exported.stderr
    destination = tmp_path / "public" / "observatory.json"
    javascript = """
        import { readFileSync } from 'node:fs';
        const { validateSnapshot } = await import(process.argv[1]);
        const result = validateSnapshot(JSON.parse(readFileSync(process.argv[2], 'utf8')));
        process.stdout.write(JSON.stringify(result));
    """
    validated = subprocess.run(
        ["node", "--input-type=module", "-e", javascript, state.as_uri(), str(destination)],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert validated.returncode == 0, validated.stderr
    known, unknown, reserved = json.loads(validated.stdout)["runs"]
    assert known["metrics"]["loss"] == -0.75
    assert known["history"][0]["loss"] == -0.5
    assert unknown["variant_id"] is None
    assert unknown["updated_at"] is None
    assert unknown["checkpoint"] == {"step": None, "saved_at": None, "resumable": None}
    assert reserved["test_released"] is False
    assert all(value is None for value in reserved["metrics"].values())
    assert reserved["history"] == []
