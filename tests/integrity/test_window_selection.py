"""Selección con la validación anterior a cada ventana, sobre recibos escritos en la prueba.

Los recibos e informes tienen solo los campos que lee `integrity.window_selection`, y los
recibos de ventana son los del protocolo US v2 versionado. No hay modelos ajustados ni pasos
de optimizador, así que estas pruebas corren con la protección de aprendizaje vigente.
"""

import json

import pytest

from mars_titan.integrity import window_selection as selection
from tests.environments import walk_forward_fixture as receipts


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))


def _fit(output, job_id, score, case, *, published=None):
    """Recibo y su informe de un ajuste, con la puntuación de su validación.

    `published` es la puntuación que guarda el informe si difiere de la del recibo.
    """
    validation = score if published is None else published
    report = dict(predictions=dict(validation=dict(metrics=dict(session_mae=validation))))
    _write(output / "jobs" / job_id / "attempt-0001" / "run.json", report)
    receipt = dict(
        status="completed",
        score=score,
        identity=dict(case=case, seed=case["seed"]),
        parent=dict(id=job_id, sha256=f"{case['seed']:064x}"),
        report=dict(path=f"jobs/{job_id}/attempt-0001/run.json"),
    )
    _write(output / "jobs" / job_id / "receipt.json", receipt)
    return receipt


def test_a_finalist_must_repeat_the_case_chosen_by_validation(tmp_path):
    common = dict(scope="US", window="fold-000", arm="gru", kind="fit", anchor="fold-000")
    jobs = [
        dict(common, id="US/fold-000/gru/search-a", stage="search", seed=42),
        dict(common, id="US/fold-000/gru/search-b", stage="search", seed=42),
        dict(common, id="US/fold-000/gru/finalist-s43", stage="finalist", seed=43),
    ]
    _fit(tmp_path, jobs[0]["id"], 0.03, dict(rate=0.1, seed=42))
    _fit(tmp_path, jobs[1]["id"], 0.02, dict(rate=0.01, seed=42))
    _fit(tmp_path, jobs[2]["id"], 0.025, dict(rate=0.01, seed=43))
    campaign = dict(comparison_config=dict(arms={}))
    result = selection.check_selection(campaign, jobs, tmp_path)
    assert result["passed"], result["failures"]
    assert result["chosen"] == {"US/fold-000/gru": dict(job=jobs[1]["id"], searches=2)}
    # El finalista repite la tasa de la búsqueda que no ganó.
    _fit(tmp_path, jobs[2]["id"], 0.025, dict(rate=0.1, seed=43))
    result = selection.check_selection(campaign, jobs, tmp_path)
    assert result["failures"] == [f"{jobs[2]['id']}: el finalista no repite el caso elegido"]


# Ventana fold-000 del protocolo US v2, con dos búsquedas de la semilla 42 y su finalista 43.
COMMON = dict(scope="US", window="fold-000", arm="gru", kind="fit", anchor="fold-000")
JOBS = [
    dict(COMMON, id="US/fold-000/gru/search-a", stage="search", seed=42),
    dict(COMMON, id="US/fold-000/gru/search-b", stage="search", seed=42),
    dict(COMMON, id="US/fold-000/gru/finalist-s43", stage="finalist", seed=43),
]
COMPARED = dict(comparison_config=dict(arms={"gru": "gru"}))


def _window_receipt(output, seed, parent, **options):
    """Recibo de la ventana de una semilla, con el estado elegido como padre."""
    receipt = receipts.receipt("US", index=0, **options)
    receipt["parent"] = dict(parent)
    _write(output / "windows" / "US" / "fold-000" / "gru" / f"seed-{seed}" / "US.json", receipt)
    return receipt


@pytest.fixture
def chosen(tmp_path):
    _fit(tmp_path, JOBS[0]["id"], 0.03, dict(rate=0.1, seed=42))
    winner = _fit(tmp_path, JOBS[1]["id"], 0.02, dict(rate=0.01, seed=42))
    finalist = _fit(tmp_path, JOBS[2]["id"], 0.025, dict(rate=0.01, seed=43))
    _window_receipt(tmp_path, 42, winner["parent"])
    _window_receipt(tmp_path, 43, finalist["parent"])
    return tmp_path


def test_each_seed_reads_a_window_receipt_of_the_state_chosen_for_it(chosen):
    result = selection.check_selection(COMPARED, JOBS, chosen)
    assert result["passed"], result["failures"]
    assert result["window_receipts"] == 2
    assert result["chosen"] == {"US/fold-000/gru": dict(job=JOBS[1]["id"], searches=2)}


def _labels_at_evaluation(output):
    start = receipts.microseconds(receipts.fold("US", 0)["evaluation"][0])
    parent = json.loads((output / "jobs" / JOBS[1]["id"] / "receipt.json").read_text())["parent"]
    _window_receipt(output, 42, parent, until=start)


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        (
            # El recibo dice que ganó la otra búsqueda sin que lo diga su validación.
            lambda output: _fit(
                output, JOBS[0]["id"], 0.0, dict(rate=0.1, seed=42), published=0.03
            ),
            f"{JOBS[0]['id']}: la puntuación del recibo no es la de su validación",
        ),
        (
            _labels_at_evaluation,
            "US/fold-000/gru/seed-42 US: La última etiqueta usada debe madurar antes de la "
            "evaluación",
        ),
        (
            lambda output: _window_receipt(output, 43, dict(id=JOBS[2]["id"], sha256="0" * 64)),
            "US/fold-000/gru/seed-43 US: el recibo no nombra el estado elegido",
        ),
        (
            lambda output: (output / "windows/US/fold-000/gru/seed-43/US.json").unlink(),
            "US/fold-000/gru/seed-43: sin recibos de ventana",
        ),
    ],
    ids=["score", "labels", "parent", "no-receipt"],
)
def test_a_state_chosen_or_declared_against_the_rule_fails(chosen, defect, message):
    defect(chosen)
    result = selection.check_selection(COMPARED, JOBS, chosen)
    assert not result["passed"] and message in result["failures"], result["failures"]


def _partitions():
    fold = receipts.fold("US", 0)
    windows = {
        "fold-000": {
            name: fold[name] for name in ("train", "validation", "calibration", "evaluation")
        }
    }
    return dict(
        partition="evaluation",
        calibration=dict(partition="calibration"),
        resolved_scopes={"US": dict(windows=windows)},
    )


def test_partitions_out_of_order_or_another_evaluated_partition_fail():
    config = _partitions()
    assert selection.check_partitions(config, ["US"]) == dict(windows=1, failures=[], passed=True)
    window = config["resolved_scopes"]["US"]["windows"]["fold-000"]
    window["calibration"], window["evaluation"] = window["evaluation"], window["calibration"]
    assert selection.check_partitions(config, ["US"])["failures"] == [
        "US/fold-000: tramos fuera de orden"
    ]
    config = dict(_partitions(), partition="calibration")
    assert selection.check_partitions(config, ["US"])["failures"] == [
        "La comparación no evalúa la evaluación con la calibración previa"
    ]
