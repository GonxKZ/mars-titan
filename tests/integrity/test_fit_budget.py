"""Presupuesto declarado y consumido por cada modelo de la campaña (#442).

Las diferencias declaradas se comprueban sobre la campaña A v2 del repositorio y su
publicación. El consumido se comprueba con historiales escritos en la prueba para cada regla
de parada y con informes con la forma de cada ejecutor. Dos pruebas leen además informes
reales de la referencia neuronal y de las cuatro variantes de Titans-MAC, ejecutados con el
optimizador que solo registra gradientes. Como esos ejecutores se detienen con la protección,
se omiten mientras rige el bloqueo de aprendizaje (#546).
"""

import copy
import json
from pathlib import Path

import pytest

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.integrity import fit_budget as budget
from mars_titan.training import campaign_plan as plan
from mars_titan.training import campaign_publication as publication
from tests.learning_hold_lowering import lowers_learning_hold
from tests.training.test_masked_reference_run import (  # noqa: F401
    cpu_runner,
    masked_case,
    masked_view,
    run_masked,
)
from tests.training.test_titans_walk_forward import (  # noqa: F401
    RULE,
    base,
    learning_doubles_module,
)

ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT / "configs/evaluation/historical-masked-publication-a-v2.json"
FIXED = dict(metric="session_mae", stopping="fixed_budget", patience=1, min_delta=0.0)
PLATEAU = dict(FIXED, stopping="validation_plateau")
JOINT = dict(FIXED, stopping="joint_plateau")


@pytest.fixture(scope="module")
def loaded():
    return publication.load_publication(DECLARATION)


def test_the_a_v2_publication_declares_every_budget_difference(loaded):
    campaign = loaded["campaign_config"]
    result = budget.check_declared(campaign, loaded["budget_differences"])
    assert result["passed"], (result["undeclared"], result["stale"])
    budgets = result["budgets"]
    # Fija la referencia el primer modelo propio, de MARS-TITAN. Entre los propios solo
    # difieren los seis B6, que trasladan el padre congelado sin ajustar y lo declaran.
    assert budgets[result["reference"]]["family"] == plan.MARS
    own = {arm for arm, item in budgets.items() if item["family"] in budget.OWN}
    differences = result["differences"]
    carried = {arm for arm in own if arm.startswith("mars_titan_b6")}
    assert len(carried) == 6 and own - carried
    assert not (own - carried) & set(differences)
    assert all(differences[arm] == ["iterations"] for arm in carried)
    assert differences["ridge"] == ["iterations", "search_cases", "seeds"]
    assert differences["xgboost"] == ["iterations", "search_cases", "seeds"]
    assert differences["transformer_compact_online"] == ["iterations", "search_cases"]
    for arm in ("rnn", "lstm", "gru", "dlinear", "transformer_compact"):
        assert differences[arm] == ["iterations"]
    assert budgets["xgboost"]["iterations"]["unit"] == "round"
    assert budgets["xgboost"]["iterations"]["max_rounds"] == 2000
    assert budgets["ridge"]["iterations"] == dict(unit="closed_form")
    assert budgets["gru"]["iterations"]["trainer"] == "row_batches"
    assert budgets["gru"]["iterations"]["batch_rows"] == 256


@pytest.mark.parametrize("defect", ["undeclared", "stale"])
def test_a_difference_that_is_not_the_campaign_one_stops_the_declaration(loaded, tmp_path, defect):
    declarations = copy.deepcopy(loaded["budget_differences"])
    if defect == "undeclared":
        declarations["ridge"].pop("seeds")
        expected = dict(undeclared=["ridge/seeds"], stale=[])
    else:
        declarations["titans_mac_online"] = dict(search_cases="Otra búsqueda")
        expected = dict(undeclared=[], stale=["titans_mac_online/search_cases"])
    result = budget.check_declared(loaded["campaign_config"], declarations)
    assert not result["passed"]
    assert {key: result[key] for key in expected} == expected
    # La misma declaración en un archivo no llega a cargarse.
    document = json.loads(DECLARATION.read_text())
    folder = DECLARATION.parent
    for field in ("campaign", "comparison_matrix", "posttraining_comparison"):
        document[field] = str((folder / document[field]).resolve())
    document["budget_differences"] = declarations
    path = tmp_path / "publication.json"
    atomic_json(path, document)
    with pytest.raises(ValueError, match="diferencias de presupuesto declaradas"):
        publication.load_publication(path)


def test_a_declaration_needs_a_reason_for_each_field(loaded):
    declarations = copy.deepcopy(loaded["budget_differences"])
    declarations["ridge"]["seeds"] = " "
    with pytest.raises(ValueError, match="cada campo distinto con su motivo"):
        budget.check_declared(loaded["campaign_config"], declarations)


def run_document(scores, rule, *, first=1, best=None, steps=10, joint=None):
    """Registro de un entrenador cronológico con la selección que la regla daría."""
    entries = [
        dict(epoch=epoch, global_step=epoch * steps, validation=dict(session_mae=score))
        for epoch, score in enumerate(scores, start=first)
    ]
    selection, _ = budget.replay(list(enumerate(scores, start=first)), rule, joint)
    selection = dict(selection, best_epoch=selection["best_epoch"] if best is None else best)
    document = dict(history=entries, selection=selection, global_step=entries[-1]["global_step"])
    if joint is not None:
        document["joint_stop_epoch"] = joint
    return document


def epochs(rule, maximum):
    return dict(rule, max_epochs=maximum)


SCRATCH = dict(parent=None)
CHILD = dict(parent="US/fold-000/titans_mac_online/search-a")


@pytest.mark.parametrize(
    ("scores", "rule", "first", "options", "job", "failure"),
    [
        ([0.3, 0.2, 0.1], epochs(FIXED, 3), 1, {}, SCRATCH, None),
        ([0.3, 0.2], epochs(FIXED, 3), 1, {}, SCRATCH, "antes de lo que manda su regla"),
        ([0.1, 0.2, 0.3, 0.4], epochs(PLATEAU, 5), 1, {}, SCRATCH, "sigue tras la época 2"),
        ([0.3, 0.2, 0.25], epochs(PLATEAU, 5), 1, {}, SCRATCH, None),
        ([0.3, 0.2, 0.1], epochs(FIXED, 3), 1, dict(best=2), SCRATCH, "entrega la época 2"),
        ([0.1, 0.2, 0.3, 0.4], epochs(FIXED, 3), 0, {}, SCRATCH, "estado inicial sin ajustar"),
        ([0.1, 0.2, 0.3, 0.4], epochs(FIXED, 3), 0, {}, CHILD, None),
        ([0.3, 0.2, 0.1], epochs(FIXED, 3), 1, dict(steps=0), SCRATCH, "declara 0 actualizaciones"),
        ([0.1, 0.2, 0.3, 0.4], epochs(JOINT, 5), 1, dict(joint=4), SCRATCH, None),
        (
            [0.1, 0.2, 0.3, 0.4],
            epochs(JOINT, 5),
            1,
            dict(joint=3),
            SCRATCH,
            "sigue tras la época 3",
        ),
    ],
    ids=[
        "fixed-complete",
        "fixed-truncated",
        "plateau-beyond",
        "plateau-stops",
        "other-epoch",
        "initial-state",
        "parent-kept",
        "no-updates",
        "joint-epoch",
        "beyond-joint",
    ],
)
def test_each_fit_ends_and_selects_where_its_declared_rule_says(
    scores, rule, first, options, job, failure
):
    document = run_document(scores, rule, first=first, **options)
    result = budget.check_epochs(job, rule, document, {})
    if failure is None:
        assert result["failures"] == []
    else:
        assert any(failure in text for text in result["failures"]), result["failures"]


@lowers_learning_hold(
    "tests/integrity/test_fit_budget.py::"
    "test_a_chronological_fit_with_unchanged_weights_keeps_its_initial_state"
)
def test_a_chronological_fit_with_unchanged_weights_keeps_its_initial_state(base):  # noqa: F811
    """Las cuatro variantes reales de Titans-MAC con un optimizador que no cambia pesos.

    Sin cambios en los pesos, ninguna época mejora la validación de la época 0, así que el
    ajuste entrega su estado inicial y el informe lo marca sin ajustar. Las cuatro aplican
    las mismas actualizaciones.
    """
    rule = dict(RULE, max_epochs=2)
    steps = set()
    for variant in ("transformer_direct", "mac_disabled", "mac_frozen", "mac_online"):
        folder = base["root"] / "runs" / variant
        path = folder / "run.json"
        receipt = dict(report=dict(path=path.name, sha256=sha256(path)))
        report, run = budget._fit_documents(folder, receipt)
        assert run is not report and run["history"][0]["epoch"] == 0
        result = budget.check_epochs(SCRATCH, rule, run, report)
        assert result["fitted"] is False and result["best_epoch"] == 0
        assert result["failures"] == ["entrega su estado inicial sin ajustar"]
        steps.add(result["global_step"])
    assert len(steps) == 1 and steps.pop() > 0


@lowers_learning_hold(
    "tests/integrity/test_fit_budget.py::test_a_neural_reference_from_scratch_counts_as_fitted"
)
def test_a_neural_reference_from_scratch_counts_as_fitted(tmp_path, cpu_runner):  # noqa: F811
    case = masked_case(epochs=2)
    report = run_masked(masked_view(tmp_path), tmp_path / "run", case)
    rule = dict(case["selection"], max_epochs=case["epochs"])
    path = tmp_path / "run" / "run.json"
    receipt = dict(report=dict(path="run.json", sha256=sha256(path)))
    stored, run = budget._fit_documents(tmp_path / "run", receipt)
    assert stored is run and run["epochs"][0]["epoch"] == 1
    result = budget.check_epochs(SCRATCH, rule, run, stored)
    assert result["failures"] == [] and result["fitted"]
    assert result["global_step"] == report["global_step"] > 0


@pytest.mark.parametrize("key", ["fit", "run"])
def test_the_trainer_record_of_a_window_is_read_with_its_digest(tmp_path, key):
    """Titans-MAC, MARS-TITAN y CM-v1 guardan su registro en `fit` y la GRU episódica en `run`."""
    folder = tmp_path / "attempt"
    (folder / "trainer").mkdir(parents=True)
    run = run_document([0.3, 0.2, 0.1], epochs(FIXED, 2), first=0)
    atomic_json(folder / "trainer" / "run.json", run)
    record = dict(path="trainer/run.json", sha256=sha256(folder / "trainer" / "run.json"))
    atomic_json(folder / "window.json", {key: record, "status": "completed"})
    receipt = dict(report=dict(path="attempt/window.json", sha256=sha256(folder / "window.json")))
    report, found = budget._fit_documents(tmp_path, receipt)
    assert found == run and report[key] == record
    result = budget.check_epochs(SCRATCH, epochs(FIXED, 2), found, report)
    assert result["failures"] == [] and result["best_epoch"] == 2
    # Un registro cambiado después de confirmar la ventana no se lee.
    atomic_json(folder / "trainer" / "run.json", dict(run, global_step=1))
    with pytest.raises(ValueError, match="no coincide con su huella"):
        budget._fit_documents(tmp_path, receipt)


@pytest.mark.parametrize(
    ("change", "failure"),
    [
        ({}, None),
        (dict(completed_rounds=2001), "fuera de 1 a 2000"),
        (dict(selected_round=0), "sin ajustar"),
        (dict(stop_reason=None), "sin declarar por qué"),
    ],
    ids=["valid", "beyond", "unfitted", "silent-stop"],
)
def test_xgboost_consumes_at_most_its_rounds_and_selects_a_fitted_one(change, failure):
    declared = dict(unit="round", max_rounds=2000, minimum_rounds=200, patience_rounds=100)
    report = dict(completed_rounds=700, selected_round=600, stop_reason="validation_plateau")
    result = budget._xgboost_fit(declared, dict(report, **change))
    if failure is None:
        assert result["failures"] == []
    else:
        assert failure in result["failures"][0]


@pytest.mark.parametrize(
    ("used", "cap", "applied", "failure"),
    [
        (40, 40, 40, None),
        (41, 40, 40, "usa 41 etiquetas con 40 escrituras"),
        (30, 40, 50, "el tope aplicado no es el del banco"),
    ],
)
def test_the_online_control_never_uses_more_labels_than_the_bank_writes(
    used, cap, applied, failure
):
    report = dict(
        bank=dict(admitted=dict(validation=cap)),
        predictions=dict(validation=dict(metrics=dict(labels_used=used, cap=applied))),
    )
    result = budget._online_fit(report)
    if failure is None:
        assert result["failures"] == []
    else:
        assert any(failure in text for text in result["failures"])


def test_an_online_report_without_labels_per_partition_fails():
    result = budget._online_fit(dict(bank=dict(admitted=dict(validation=40)), predictions={}))
    assert result["failures"] == ["el informe no declara las etiquetas usadas por tramo"]


@pytest.fixture(scope="module")
def own_searches(loaded):
    """Las dos búsquedas de Titans-MAC en línea en la primera ventana conjunta de A v2."""
    campaign = loaded["campaign_config"]
    jobs = [
        job
        for job in plan.plan_campaign(campaign)
        if (job["kind"], job["arm"], job["scope"], job["window"], job["stage"])
        == (plan.FIT, "titans_mac_online", "US+CN", "fold-000", "search")
    ]
    assert len(jobs) == 2 and all(job.get("parent") is None for job in jobs)
    return campaign, jobs


def _confirm(output, job, document, *, status="completed"):
    """Recibo confirmado de un ajuste con su informe, escrito sin ejecutar nada."""
    folder = output / "jobs" / job["id"] / "attempt-0001"
    atomic_json(folder / "run.json", document)
    path = folder / "run.json"
    report = dict(path=str(path.relative_to(output)), sha256=sha256(path))
    atomic_json(output / "jobs" / job["id"] / "receipt.json", dict(status=status, report=report))


def _full(rule, *, steps=10, scores=None):
    scores = scores or [1.0 - 0.01 * epoch for epoch in range(rule["max_epochs"] + 1)]
    return run_document(scores, rule, first=0, steps=steps)


@pytest.mark.parametrize(
    ("defect", "failing", "unequal"),
    [
        (None, False, False),
        ("steps", False, True),
        # Un ajuste truncado aplica además menos actualizaciones que el otro.
        ("truncated", True, True),
        ("altered", True, False),
        ("unconfirmed", False, False),
    ],
)
def test_confirmed_fits_spend_their_rule_and_the_same_updates(
    own_searches, tmp_path, defect, failing, unequal
):
    campaign, jobs = own_searches
    rule = budget._rule(campaign, "titans_mac_online")
    assert rule["stopping"] == "fixed_budget" and rule["max_epochs"] == 30
    _confirm(tmp_path, jobs[0], _full(rule))
    second = _full(rule)
    if defect == "steps":
        second = _full(rule, steps=11)
    elif defect == "truncated":
        second = _full(rule, scores=[1.0 - 0.01 * epoch for epoch in range(20)])
    _confirm(tmp_path, jobs[1], second, status="failed" if defect == "unconfirmed" else "completed")
    if defect == "altered":
        changed = dict(_full(rule), note="cambiado tras confirmar el recibo")
        atomic_json(tmp_path / "jobs" / jobs[1]["id"] / "attempt-0001" / "run.json", changed)
    result = budget.check_consumed(campaign, jobs, tmp_path)
    # Un ajuste sin confirmar ya lo cuenta `declared_runs` y aquí no se lee.
    assert result["fits"] == (1 if defect == "unconfirmed" else 2)
    assert (result["failing"] == [jobs[1]["id"]]) is failing
    assert bool(result["unequal_updates"]) is unequal
    assert result["passed"] is not (failing or unequal)
    if defect == "steps":
        assert result["unequal_updates"] == [
            f"{jobs[0]['model']}/US+CN/fold-000: actualizaciones [300, 330]"
        ]
    if defect == "altered":
        assert "no es el que confirma su recibo" in result["jobs"][1]["failures"][0]


def test_a_joint_group_must_end_at_the_same_epoch(own_searches, tmp_path, monkeypatch):
    """Dos ajustes de un grupo de parada conjunta, con la regla sustituida por la conjunta."""
    campaign, jobs = own_searches
    rule = epochs(JOINT, 5)
    monkeypatch.setattr(budget, "_rule", lambda campaign, arm: rule)
    grouped = [dict(job, joint_group=["US+CN/fold-000/titans_mac_online"]) for job in jobs]
    scores = [0.1, 0.2, 0.3, 0.4]
    _confirm(tmp_path, grouped[0], run_document(scores, rule, joint=4))
    _confirm(tmp_path, grouped[1], run_document(scores, rule, joint=4))
    assert budget.check_consumed(campaign, grouped, tmp_path)["passed"]
    _confirm(tmp_path, grouped[1], run_document(scores[:3], rule, joint=3))
    result = budget.check_consumed(campaign, grouped, tmp_path)
    assert result["failing"] == [] and not result["passed"]
    assert result["unequal_updates"] == ["grupo de US+CN/fold-000/titans_mac_online: épocas [3, 4]"]


def test_plateau_probes_are_not_read_as_fits(own_searches, tmp_path):
    campaign, jobs = own_searches
    probe = dict(jobs[0], phase=plan.PLATEAU)
    atomic_json(tmp_path / "jobs" / probe["id"] / "receipt.json", dict(status="completed"))
    assert budget.check_consumed(campaign, [probe], tmp_path) == dict(
        fits=0, unfitted=[], failing=[], unequal_updates=[], jobs=[], passed=True
    )


def test_a_reference_initialized_from_its_parent_may_keep_its_initial_state():
    """Referencia neuronal con `initial_validation`: la época 0 compite con las ajustadas."""
    rule = epochs(FIXED, 2)
    run = dict(
        epochs=[
            dict(epoch=1, validation=dict(session_mae=0.3)),
            dict(epoch=2, validation=dict(session_mae=0.2)),
        ],
        initial_validation=dict(session_mae=0.1),
        selection=dict(best_epoch=0),
        global_step=20,
    )
    result = budget.check_epochs(CHILD, rule, run, run)
    assert result["failures"] == [] and result["best_epoch"] == 0 and result["fitted"]


def test_the_report_needs_both_the_declared_and_the_consumed_budget(loaded, own_searches, tmp_path):
    campaign, jobs = own_searches
    rule = budget._rule(campaign, "titans_mac_online")
    for job in jobs:
        _confirm(tmp_path, job, _full(rule))
    declarations = loaded["budget_differences"]
    assert budget.check(campaign, jobs, tmp_path, declarations)["passed"]
    stale = dict(declarations, titans_mac_online=dict(seeds="Otra semilla"))
    result = budget.check(campaign, jobs, tmp_path, stale)
    assert result["consumed"]["passed"] and not result["passed"]
