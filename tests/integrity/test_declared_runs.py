"""Ejecuciones declaradas frente a los recibos en disco, sobre recibos escritos en la prueba.

Los recibos e informes tienen solo los campos que lee `integrity.declared_runs`. No se ejecuta
ninguna campaña, así que estas pruebas corren con la protección de aprendizaje vigente. Las
mismas comprobaciones se recorren sobre la campaña reducida en
`tests/posttraining/test_publication_from_aggregates.py` cuando el bloqueo se levante.
"""

import json

import pytest

from mars_titan.integrity import declared_runs as runs

COMMON = dict(scope="US", window="fold-000", arm="gru", kind="fit")
JOBS = [
    dict(COMMON, id="US/fold-000/gru/search-a", stage="search"),
    dict(COMMON, id="US/fold-000/gru/search-b", stage="search"),
    dict(COMMON, id="US/fold-000/gru/finalist-s43", stage="finalist"),
]


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))


def _confirm(output, job_id, *, attempts=1, status="completed", report=None):
    """Intentos de un trabajo y su recibo. El último intento es el que confirma el recibo."""
    folder = output / "jobs" / job_id
    for index in range(1, attempts + 1):
        (folder / f"attempt-{index:04d}").mkdir(parents=True)
    receipt = dict(status=status)
    if report is not None:
        path = folder / f"attempt-{attempts:04d}" / "run.json"
        _write(path, report)
        receipt["report"] = dict(path=str(path.relative_to(output)))
    _write(folder / "receipt.json", receipt)


@pytest.fixture
def output(tmp_path):
    _confirm(tmp_path, JOBS[0]["id"])
    _confirm(tmp_path, JOBS[1]["id"], attempts=2, report=dict(stopped_early=True, stop_reason="x"))
    _confirm(tmp_path, JOBS[2]["id"], status="plateau_confirmed")
    return tmp_path


def test_every_declared_job_is_reported_with_its_attempts_and_its_stop(output):
    result = runs.check_output(JOBS, output)
    assert result["passed"] and result["planned"] == result["reported"] == 3
    assert result["confirmed"] == 3 and result["missing"] == result["undeclared"] == []
    # El segundo trabajo tuvo un intento que no se confirmó y paró antes de su presupuesto.
    assert result["unconfirmed_attempts"] == 1 and result["stopped_early"] == 1
    entry = next(entry for entry in result["jobs"] if entry["id"] == JOBS[1]["id"])
    assert entry["attempts"] == 2 and entry["unconfirmed_attempts"] == 1
    assert entry["stop_reason"] == "x"


def test_a_failed_job_counts_every_attempt_and_stays_missing(output):
    _confirm(output, JOBS[0]["id"], attempts=0, status="failed")
    (output / "jobs" / JOBS[0]["id"] / "attempt-0002").mkdir()
    result = runs.check_output(JOBS, output)
    assert not result["passed"] and result["missing"] == [JOBS[0]["id"]]
    # Los dos intentos del trabajo fallido y el no confirmado del segundo.
    assert result["unconfirmed_attempts"] == 3


def test_a_declared_job_without_its_receipt_fails(output):
    (output / "jobs" / JOBS[2]["id"] / "receipt.json").unlink()
    result = runs.check_output(JOBS, output)
    assert not result["passed"] and result["missing"] == [JOBS[2]["id"]]
    assert next(entry for entry in result["jobs"] if entry["id"] == JOBS[2]["id"])["status"] == (
        "missing"
    )


def test_a_receipt_of_an_undeclared_job_fails_and_attempt_receipts_do_not_count(output):
    # Un intento puede guardar su propio recibo sin ser otro trabajo.
    _write(output / "jobs" / JOBS[0]["id"] / "attempt-0001" / "receipt.json", {})
    assert runs.check_output(JOBS, output)["passed"]
    _confirm(output, "US/fold-000/gru/search-z")
    result = runs.check_output(JOBS, output)
    assert not result["passed"] and result["undeclared"] == ["US/fold-000/gru/search-z"]


def test_a_chain_window_counts_only_with_its_confirmed_selection(output):
    chains = [dict(id="US/fold-001/gru/chain", kind="chain_selection")]
    result = runs.check_output(JOBS, output, chains=chains, selected=lambda job: True)
    assert result["passed"] and result["planned"] == result["confirmed"] == 4
    assert result["jobs"][-1] == dict(id=chains[0]["id"], kind="chain_selection", status="selected")
    result = runs.check_output(JOBS, output, chains=chains, selected=lambda job: False)
    assert not result["passed"] and result["missing"] == [chains[0]["id"]]


def _published():
    windows = {"fold-000": {}, "fold-001": {}}
    sources = dict(
        windows=windows,
        arms={"gru": dict(seeds=[42, 43]), "ridge": dict(seeds=[])},
    )
    report = dict(
        arms={
            "gru": dict(seeds={seed: dict(windows=dict(windows)) for seed in ("42", "43")}),
            # Un modelo determinista se publica con una sola entrada.
            "ridge": dict(seeds={"deterministic": dict(windows=dict(windows))}),
        }
    )
    return report, sources


@pytest.mark.parametrize("defect", ["seed", "window", "model"])
def test_a_published_comparison_keeps_every_declared_seed_and_window(defect):
    report, sources = _published()
    assert runs.check_published(report, sources) == dict(arms=2, differing=[], passed=True)
    seeds = report["arms"]["gru"]["seeds"]
    if defect == "seed":
        seeds.pop("43")
        expected = "gru: semillas publicadas distintas de las declaradas"
    elif defect == "window":
        seeds["43"]["windows"].pop("fold-001")
        expected = "gru semilla 43: ventanas distintas de las declaradas"
    else:
        report["arms"]["gru_best"] = report["arms"]["gru"]
        expected = "gru_best: modelo publicado sin declarar"
    result = runs.check_published(report, sources)
    assert not result["passed"] and result["differing"] == [expected]
