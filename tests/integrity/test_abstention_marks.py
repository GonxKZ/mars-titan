"""Un modelo que casi siempre se abstiene no gana ninguna métrica publicada (#442).

En el estudio de juguete de la comparación walk-forward, `ridge` solo llama en una fila por
sesión, la de mayor objetivo positivo, y acierta siempre. Su precisión al alza es perfecta.
Las pruebas comprueban que sus contrastes de precisión quedan marcados no elegibles en la
comparación y en la matriz, que su exactitud de dirección y su exactitud a la cobertura fija
quedan por debajo de las de los modelos que llaman, y que el informe de integridad detecta
una marca o una cobertura cambiadas. Las predicciones se escriben en la prueba y no hay
modelos ajustados ni pasos de optimizador.
"""

import copy
import json

import numpy as np
import pytest

from mars_titan.evaluation import abstention_coverage as abstention
from mars_titan.evaluation import comparison_matrix as matrix_module
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.integrity import abstention_marks as marks
from mars_titan.integrity import comparison_integrity as integrity
from mars_titan.integrity import publication_gate as gate
from tests.evaluation.liquidity_fixture import section as liquidity_section
from tests.evaluation.liquidity_fixture import study_edition
from tests.evaluation.test_comparison_matrix import sources, toy_matrix
from tests.evaluation.test_walk_forward_comparison import Study

METRICS = ["mae", "direction_accuracy", "up_precision", "down_precision"]
VIEWS = ("US+CN", "US", "CN")


def abstain(arm, seed, window, partition, values):
    """`ridge` predice cero salvo en la fila de mayor objetivo positivo de cada sesión."""
    if arm != "ridge":
        return
    target = np.asarray(values["target"], dtype=np.float64)
    moments = values["prediction_at"].to_numpy(zero_copy_only=False)
    prediction = np.zeros(len(target))
    for market, moment in {(m, t) for m, t in zip(values["market"], moments, strict=True)}:
        rows = [
            i
            for i, (m, t) in enumerate(zip(values["market"], moments, strict=True))
            if (m, t) == (market, moment) and target[i] > 0
        ]
        if rows:
            best = max(rows, key=lambda i: target[i])
            prediction[best] = target[best]
    values["prediction"] = prediction


@pytest.fixture(scope="module")
def published(tmp_path_factory):
    root = tmp_path_factory.mktemp("abstention")
    study = Study(root / "study")
    study.changes.append(abstain)
    study.config["comparison"]["metrics"] = METRICS
    # La matriz contrasta el MSE, que solo se publica junto a sus estratos de liquidez.
    study.config[walk.LIQUIDITY_FIELD] = liquidity_section(min_rows=1, min_sessions=1)
    study.publish()
    output = root / "walk"
    edition = study_edition(root / "edition")
    report = walk.write_walk_forward(
        study.config_path, study.sources_path, "US+CN", output, edition=edition
    )
    matrix = toy_matrix(root, study.config_path)
    manifest = sources(root, [("walk_forward_comparison_report", output / "comparison.json")])
    matrix_path = root / "matrix-out"
    matrix_module.write(matrix, manifest, "US+CN", matrix_path)
    return study, report, output, matrix_path / "matrix.json", edition


def _call_rows(contrasts):
    for view, families in contrasts.items():
        for family, metrics in families.items():
            for metric in abstention.CALL_CONDITIONED:
                for row in metrics[metric].get("contrasts", []):
                    yield view, family, row


def test_the_abstaining_model_has_perfect_precision_but_no_eligible_contrast(published):
    _, report, _, _, _ = published
    declared = report["abstention"]
    assert {key: declared[key] for key in abstention.declaration()} == abstention.declaration()
    for view in VIEWS:
        coverage = declared["call_coverage"][view]
        assert coverage["ridge"] < abstention.MIN_CALL_COVERAGE <= coverage["gru"]
        assert coverage["zero"] == 0.0
    seeds = report["arms"]["ridge"]["seeds"]["42"]["overall"]["US+CN"]["summary"]
    assert seeds["point"]["up_precision"] == 1.0
    rows = list(_call_rows(report["contrasts"]))
    assert rows
    for view, family, row in rows:
        involved = {arm for arm, value in row["coefficients"].items() if value != 0}
        assert row["eligible"] is not bool(involved & {"ridge", "zero"}), (view, family, row)
        assert set(row["call_coverage"]) == involved
    # Sin marcas en las métricas que juzgan todas las filas.
    mae = report["contrasts"]["US+CN"]["references_vs_zero"]["mae"]["contrasts"]
    assert all("eligible" not in row for row in mae)


def _summary(report, arm, seed):
    return report["arms"][arm]["seeds"][seed]["overall"]["US+CN"]["summary"]


def test_abstaining_does_not_win_direction_at_any_coverage(published):
    _, report, _, _, _ = published
    fixed = abstention.declaration()["coverages"].index(abstention.FIXED_COVERAGE)
    ridge = _summary(report, "ridge", "42")
    for seed in ("42", "43"):
        gru = _summary(report, "gru", seed)
        assert ridge["point"]["direction_accuracy"] < gru["point"]["direction_accuracy"]
        ours = ridge["selective_direction"]["curve"][fixed]
        theirs = gru["selective_direction"]["curve"][fixed]
        assert ours["coverage"] == theirs["coverage"] == abstention.FIXED_COVERAGE
        assert ours["retained_rows"] == theirs["retained_rows"]
        assert ours["accuracy"] < theirs["accuracy"]
    # El control cero, que se abstiene siempre, no acierta a ninguna cobertura.
    zero = _summary(report, "zero", "deterministic")["selective_direction"]["curve"]
    assert all(point["hits"] == 0 for point in zero)


def test_the_matrix_carries_the_same_marks(published):
    _, report, _, matrix_path, _ = published
    matrix = json.loads(matrix_path.read_text())
    assert matrix["abstention"] == abstention.declaration()
    every = matrix["views"]["forecast"]["US+CN"]["up_precision"]["every"]["contrasts"]
    assert {row["name"]: row["eligible"] for row in every} == {
        "gru-ridge": False,
        "titans-ridge": False,
        "titans-gru": True,
    }
    coverage = report["abstention"]["call_coverage"]
    assert marks.check_matrix(matrix, coverage)["passed"]


def _rewrite(path, change):
    document = json.loads(path.read_text())
    change(document)
    path.write_text(json.dumps(document))


def _first_ineligible(report):
    return next(row for _, _, row in _call_rows(report["contrasts"]) if not row["eligible"])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: _first_ineligible(r).update(eligible=True), "marcado True"),
        (lambda r: _first_ineligible(r).pop("eligible"), "marcado None"),
        (
            lambda r: _first_ineligible(r)["call_coverage"].update(ridge=0.9),
            "otra cobertura",
        ),
        (
            lambda r: r["abstention"]["call_coverage"]["US+CN"].update(ridge=0.9),
            "cobertura de llamadas publicada",
        ),
        (lambda r: r["abstention"].update(min_call_coverage=0.0), "reglas de abstención"),
        (
            lambda r: r["arms"]["ridge"]["seeds"]["42"]["overall"]["US"]["summary"][
                "selective_direction"
            ]["curve"][4].update(hits=0),
            "filas o aciertos distintos",
        ),
        (
            lambda r: r["arms"]["gru"]["seeds"]["43"]["overall"]["CN"]["summary"][
                "selective_direction"
            ]["curve"][2].update(retained_rows=1),
            "filas o aciertos distintos",
        ),
    ],
    ids=["eligible", "unmarked", "row-coverage", "coverage", "rules", "curve-hits", "curve-rows"],
)
def test_the_integrity_report_catches_a_changed_mark(published, tmp_path, change, message):
    study, _, output, _, edition = published
    folder = tmp_path / "walk"
    folder.mkdir()
    for name in ("comparison.json", "sessions.parquet"):
        (folder / name).write_bytes((output / name).read_bytes())
    assert integrity.published(
        study.config_path, study.sources_path, "US+CN", folder, edition=edition
    )["passed"]
    _rewrite(folder / "comparison.json", change)
    result = integrity.published(
        study.config_path, study.sources_path, "US+CN", folder, edition=edition
    )
    assert not result["passed"] and not result["abstention"]["passed"]
    assert any(message in failure for failure in result["abstention"]["failures"]), result[
        "abstention"
    ]["failures"]


def test_the_recheck_recomputes_the_selective_hits_of_every_file(published):
    # Las sesiones publicadas y el recálculo independiente dan los mismos aciertos.
    study, _, output, _, edition = published
    result = integrity.published(
        study.config_path, study.sources_path, "US+CN", output, edition=edition
    )
    files = result["scores"]["files"]
    assert files and {check["arm"] for check in files} >= {"ridge", "gru", "zero"}
    for check in files:
        assert check["columns"]["selective_hits"] == dict(mismatched_sessions=0), check


def test_the_gate_checks_the_matrix_of_each_walk_forward_comparison(published, tmp_path):
    study, _, output, matrix_path, edition = published
    entry = dict(
        label="walk-forward/US+CN",
        config=study.config_path,
        sources=study.sources_path,
        scope="US+CN",
        folder=output,
        matrix=matrix_path,
        edition=edition,
    )
    checked = gate._comparison(entry)
    assert checked["passed"] and checked["matrix"]["contrasts"] > 0
    changed = tmp_path / "matrix.json"
    matrix = json.loads(matrix_path.read_text())
    every = matrix["views"]["forecast"]["US"]["down_precision"]["every"]["contrasts"]
    next(row for row in every if not row["eligible"]).update(eligible=True)
    changed.write_text(json.dumps(matrix))
    checked = gate._comparison(dict(copy.deepcopy(entry), matrix=changed))
    assert not checked["passed"] and not checked["matrix"]["passed"]
