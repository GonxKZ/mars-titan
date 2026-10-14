"""Preprocesado ajustado solo con el pasado de cada corte (#442).

Las auditorías del código recorren las implementaciones reales del residualizador y de las
rutas de régimen con precios deterministas del propio módulo, y cada prueba introduce a
propósito un ajuste con datos futuros que debe hacer fallar la comprobación. Los recibos,
informes, normalizaciones, ediciones y calendarios se escriben en la prueba. No se ajusta
ningún modelo ni se ejecuta ningún paso de optimizador.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mars_titan.data.budget_targets import residual_targets
from mars_titan.data.residual_arrays import residual_targets_array
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation import regime_calendar
from mars_titan.integrity import past_only_preprocessing as past
from mars_titan.posttraining import chronological_matrix
from mars_titan.training import campaign_plan as plan
from mars_titan.training.campaign_data_policy import REAL_EDITION_ONLY
from tests.evaluation.test_walk_forward_comparison import Study
from tests.evaluation.test_walk_forward_retention import write_calendar

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs/evaluation"
COMPARISONS = (
    CONFIGS / "historical-masked-2000-comparison.json",
    CONFIGS / "historical-masked-2000-joint-comparison.json",
)
VALIDATION = ("2021-01-04", "2021-12-31")
START = past.micros(VALIDATION[0])


# Código: valores hasta cada corte fijos al alterar los precios posteriores


def test_the_label_and_regime_code_only_looks_at_the_past():
    result = past.audit_code()
    assert result["passed"]
    for name in ("residual_reference", "residual_arrays", "regimes"):
        assert result[name]["cuts"] == len(past.AUDIT_CUTS) and result[name]["compared"] > 0
    # Las rutas de la historia de comprobación no quedan todas sin clasificar.
    days, at, closes = past.regime_history()
    routes = regime_calendar.market_routes(
        "US", days, at, closes, np.zeros(len(days), dtype=bool), past.RegimeRule(past.REGIME_RULE)
    )
    assert len(set(routes["route"]) - {0}) >= 2


def _whole_sample_beta(asset, market, clock, *, cutoff):
    """Beta estimada con toda la serie, también con los rendimientos posteriores."""
    table = residual_targets_array(asset, market, clock, cutoff=cutoff)
    stock = np.diff(np.log(asset["close"].to_numpy()))
    factor = np.diff(np.log(market["close"].to_numpy()))
    table.loc[table["reason"] == "accepted", "beta"] = float(np.polyfit(factor, stock, 1)[0])
    return table


def _whole_sample_scale(asset, market, clock, *, cutoff):
    """Objetivo dividido por la desviación de todos los objetivos, también los futuros."""
    table = residual_targets(asset, market, clock, cutoff=cutoff)
    accepted = table["reason"] == "accepted"
    table.loc[accepted, "target"] = table.loc[accepted, "target"] / table["target"].std()
    return table


@pytest.mark.parametrize(
    ("implementation", "column"),
    [(_whole_sample_beta, "beta"), (_whole_sample_scale, "target")],
    ids=["beta-whole-sample", "target-scaled-whole-sample"],
)
def test_a_residualizer_fitted_with_the_future_fails(implementation, column):
    result = past.audit_residualizer(implementation)
    assert not result["passed"]
    assert any(item.endswith(column) for item in result["changed"]), result["changed"]


def _looks_ahead(market, days, at, closes, absent, rule):
    """Rutas calculadas con los cierres de cinco sesiones después."""
    shifted = np.concatenate([closes[:, 5:], np.repeat(closes[:, -1:], 5, axis=1)], axis=1)
    return regime_calendar.market_routes(market, days, at, shifted, absent, rule)


def _whole_sample_threshold(market, days, at, closes, absent, rule):
    """Turbulencia frente a la mediana de la volatilidad del mercado en toda la serie."""
    result = regime_calendar.market_routes(market, days, at, closes, absent, rule)
    returns = np.diff(np.log(closes), axis=1).mean(axis=0)
    volatility = pd.Series(returns).rolling(21, min_periods=2).std().to_numpy()
    turbulent = np.concatenate([[False], volatility > np.nanmedian(volatility)])
    result["route"] = [
        route if route == 0 else 1 + (route - 1) % 2 + 2 * int(busy)
        for route, busy in zip(result["route"], turbulent, strict=True)
    ]
    return result


@pytest.mark.parametrize(
    "routes",
    [_looks_ahead, _whole_sample_threshold],
    ids=["window-ahead", "threshold-whole-sample"],
)
def test_regime_routes_with_future_data_fail(routes):
    result = past.audit_regimes(routes)
    assert not result["passed"] and any(item.endswith("route") for item in result["changed"])


def test_a_failed_code_audit_blocks_the_section(monkeypatch):
    monkeypatch.setattr(
        past, "audit_regimes", lambda routes: dict(changed=["corte 1"], passed=False)
    )
    result = past.audit_code()
    assert result["residual_arrays"]["passed"] and not result["passed"]


def test_a_check_that_compares_no_rows_does_not_pass():
    frame = pd.DataFrame(dict(at=[1, 2, 3], value=[1.0, 2.0, 3.0]))
    result = past.past_only(
        lambda data: data, frame, lambda data, cut: data, [0], lambda table, cut: []
    )
    assert result["compared"] == 0 and not result["passed"]


# Ajustes confirmados de la campaña


def _confirm(output, job, report, *, fit=None):
    """Recibo confirmado con su informe y, si se indica, el registro de su entrenador."""
    folder = Path(output) / "jobs" / job["id"]
    folder.mkdir(parents=True)
    if fit is not None:
        atomic_json(folder / "fit" / "run.json", fit)
        report = dict(report, fit=dict(path="fit/run.json", sha256=sha256(folder / "fit/run.json")))
    atomic_json(folder / "run.json", report)
    record = dict(path=f"jobs/{job['id']}/run.json", sha256=sha256(folder / "run.json"))
    atomic_json(folder / "receipt.json", dict(status="completed", report=record))


def _job(name, family, *, kind=plan.FIT, **fields):
    return dict(id=f"US/fold-000/{name}/search-1", family=family, kind=kind, **fields)


def _mars_run(admission="m3", **changes):
    """Registro del entrenador de MARS-TITAN con la identidad que guarda `fit/run.json`."""
    phases = dict(
        train=dict(
            partition="train",
            warmup_start=10,
            decision_start=10,
            decision_end=100,
            close_at=100,
        ),
        validation=dict(
            partition="validation",
            warmup_start=10,
            decision_start=120,
            decision_end=200,
            close_at=200,
        ),
    )
    sources = {name: dict(index_sha256=name[0] * 64, phase=phase) for name, phase in phases.items()}
    scalers = dict(
        source_sha256="t" * 64,
        dataset_sha256="d" * 64,
        decision_start=10,
        decision_end=100,
        decisions=5,
    )
    identity = dict(
        admission=admission,
        dataset_sha256="d" * 64,
        sources=sources,
        retention=dict(scalers=scalers, capacity=1024) if admission == "m3" else dict(seed=1),
    )
    for change in changes.values():
        change(identity)
    return dict(identity=identity)


def _kalman_run(**changes):
    """Informe de una ventana B6 kalman con su tramo de estimación de σ² y ϱ."""
    estimation = dict(
        partition="train", warmup_start=10, decision_start=40, decision_end=100, close_at=100
    )
    identity = dict(
        phases=dict(validation=dict(partition="validation", decision_start=120)),
        cohort_noise=dict(phase=estimation, index=dict(sha256="i" * 64), estimate={}),
    )
    for change in changes.values():
        change(identity)
    return dict(identity=identity)


FITS = {
    "ridge": (plan.TABULAR, dict(samples=dict(train=10, validation=5), fitted_rows=10), None),
    "m3": (plan.MARS, {}, _mars_run()),
    "m1": (plan.MARS, {}, _mars_run("m1")),
    "kalman": (plan.MARS, _kalman_run(), None),
    "titans": (plan.TITANS, {}, None),
    "online": (plan.NEURAL, {}, None),
}


def _campaign_fits(output, **overrides):
    jobs = []
    for name, (family, report, fit) in {**FITS, **overrides}.items():
        kind = plan.ONLINE if name == "online" else plan.FIT
        job = _job(name, family, kind=kind)
        _confirm(output, job, report, fit=fit)
        jobs.append(job)
    return jobs


def test_fits_with_statistics_of_their_own_training_rows_pass(tmp_path):
    jobs = _campaign_fits(tmp_path)
    # Un ajuste pendiente, uno fallido y la meseta de un ajuste conjunto no se leen.
    jobs.append(_job("pending", plan.TABULAR))
    for name, status in (("failed", "failed"), ("plateau", "plateau_confirmed")):
        job = _job(name, plan.TABULAR, phase=plan.PLATEAU if name == "plateau" else None)
        atomic_json(tmp_path / "jobs" / job["id"] / "receipt.json", dict(status=status))
        jobs.append(job)
    result = past.check_fits(jobs, tmp_path)
    assert result["passed"] and result["fits"] == len(FITS)
    rules = {entry["id"].split("/")[2]: entry["rule"] for entry in result["jobs"]}
    assert rules == dict(
        ridge="train_rows",
        m3="write_scalers",
        m1="write_scalers",
        kalman="cohort_noise",
        titans="without_fitted",
        online="without_fitted",
    )


def _set(path, value):
    def change(identity):
        target = identity
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return change


@pytest.mark.parametrize(
    ("name", "fit", "message"),
    [
        (
            "ridge",
            (plan.TABULAR, dict(samples=dict(train=10, validation=5), fitted_rows=15), None),
            "recorren 15 filas de 10",
        ),
        ("ridge", (plan.TABULAR, dict(fitted_rows=10), None), "no declara sus filas"),
        (
            "m3",
            (
                plan.MARS,
                {},
                _mars_run(source=_set(("retention", "scalers", "source_sha256"), "v" * 64)),
            ),
            "no salen del índice de entrenamiento",
        ),
        (
            "m3",
            (
                plan.MARS,
                {},
                _mars_run(edition=_set(("retention", "scalers", "dataset_sha256"), "e" * 64)),
            ),
            "no salen del índice de entrenamiento",
        ),
        (
            "m3",
            (plan.MARS, {}, _mars_run(end=_set(("retention", "scalers", "decision_end"), 200))),
            "otras decisiones",
        ),
        (
            "m3",
            (plan.MARS, {}, _mars_run(late=_set(("sources", "train", "phase", "close_at"), 150))),
            "maduran después de empezar validación",
        ),
        (
            "m3",
            (plan.MARS, {}, _mars_run(gone=_set(("retention", "scalers"), None))),
            "sin escalas",
        ),
        (
            "m1",
            (plan.MARS, {}, _mars_run("m1", extra=_set(("retention", "scalers"), dict(a=1)))),
            "sin M3 lleva escalas",
        ),
        (
            "kalman",
            (plan.MARS, _kalman_run(late=_set(("cohort_noise", "phase", "close_at"), 150)), None),
            "estiman σ² y ϱ maduran después de empezar validación",
        ),
        (
            "kalman",
            (
                plan.MARS,
                _kalman_run(other=_set(("cohort_noise", "phase", "partition"), "validation")),
                None,
            ),
            "no salen del tramo de entrenamiento",
        ),
        ("liquidity", ("liquidity_reference", {}, None), "no está en el inventario"),
    ],
    ids=[
        "ridge-with-validation",
        "ridge-without-rows",
        "m3-from-validation",
        "m3-other-edition",
        "m3-other-decisions",
        "m3-late-labels",
        "m3-without-scalers",
        "m1-with-scalers",
        "kalman-late-labels",
        "kalman-from-validation",
        "unknown-family",
    ],
)
def test_a_fit_with_statistics_beyond_its_training_rows_fails(tmp_path, name, fit, message):
    jobs = _campaign_fits(tmp_path, **{name: fit})
    result = past.check_fits(jobs, tmp_path)
    assert not result["passed"]
    failing = [entry for entry in result["jobs"] if entry["failures"]]
    assert [entry["id"].split("/")[2] for entry in failing] == [name]
    assert any(message in failure for failure in failing[0]["failures"]), failing


def test_a_report_that_changed_after_its_receipt_fails(tmp_path):
    jobs = _campaign_fits(tmp_path)
    report = tmp_path / "jobs" / jobs[0]["id"] / "run.json"
    report.write_text(report.read_text().replace('"fitted_rows": 10', '"fitted_rows": 11'))
    result = past.check_fits(jobs, tmp_path)
    assert result["failing"] == [jobs[0]["id"]]


# Normalización de cada padre de la etapa de adaptadores


def _stage_campaign():
    windows = {"fold-000": dict(validation=list(VALIDATION))}
    return dict(comparison_config=dict(resolved_scopes={"US": dict(windows=windows)}))


def _normalization(**changes):
    statistics = dict(
        fit_partition="train",
        samples=40,
        window=dict(train_rows=40, train_bounds=dict(US=[1, START - 10, START - 5])),
    )
    for key, value in changes.items():
        if key in statistics["window"]:
            statistics["window"][key] = value
        else:
            statistics[key] = value
    return statistics


def _stage(output, statistics=None, *, family="neural_reference"):
    common = dict(scope="US", window="fold-000", base_arm="gru", seed=42, family=family)
    planned = [
        dict(common, id="US/fold-000/gru_frozen/frozen-s42", kind="frozen"),
        dict(common, id="US/fold-000/gru_lora/fit-s42", kind=plan.FIT),
        dict(common, id="US/fold-000/gru_bias/fit-s42", kind=plan.FIT),
    ]
    for job in planned:
        folder = Path(output) / "jobs" / job["id"]
        folder.mkdir(parents=True)
        atomic_json(folder / "receipt.json", dict(status="completed"))
    if statistics is not None:
        path = Path(output) / "windows-data/US/fold-000/parents/gru/seed-42/normalization.json"
        atomic_json(path, statistics)
    return planned


def test_each_adapter_parent_normalizes_with_its_training_rows(tmp_path):
    result = past.check_stage(_stage_campaign(), _stage(tmp_path, _normalization()), tmp_path)
    # Los dos ajustes comparten padre y la normalización se lee una vez.
    assert result["passed"] and result["parents"] == 1


@pytest.mark.parametrize(
    ("statistics", "message"),
    [
        (_normalization(fit_partition="validation"), "no se ajusta en el tramo de entrenamiento"),
        (_normalization(samples=55), "recorre 55 filas de 40"),
        (_normalization(train_bounds=dict(US=[1, START, START + 5])), "maduran después"),
        (None, "no guarda su normalización"),
    ],
    ids=["validation-partition", "more-rows", "late-labels", "missing"],
)
def test_a_parent_normalized_with_later_rows_fails(tmp_path, statistics, message):
    result = past.check_stage(_stage_campaign(), _stage(tmp_path, statistics), tmp_path)
    assert result["failing"] == ["US/fold-000/gru/42"]
    assert any(message in failure for failure in result["entries"][0]["failures"])


def test_chronological_stage_fits_reuse_the_scalers_of_their_parent(tmp_path):
    family = next(iter(chronological_matrix.CAMPAIGN_DESIGNS))
    result = past.check_stage(_stage_campaign(), _stage(tmp_path, family=family), tmp_path)
    assert result["passed"] and result["parents"] == 0


# Edición y calendario calculados con el código auditado


def _edition(path, **configuration):
    atomic_json(path, dict(kind="corpus_supervision", configuration=configuration))
    return {str(path): sha256(path)}


def _current_code():
    return {field: sha256(past.ROOT / name) for field, name in past.TARGET_CODE.items()}


def _calendar(path, code):
    write_calendar(path)
    calendar = json.loads(path.read_text())
    if code is not None:
        calendar["code_sha256"] = code
    path.write_text(json.dumps(calendar))
    return path


def test_an_edition_and_a_calendar_from_the_audited_code_pass(tmp_path):
    parents = _edition(tmp_path / "targets.json", **_current_code())
    calendar = _calendar(tmp_path / "calendar.json", regime_calendar.code_sha256())
    result = past.check_editions(parents, calendar, strict=True)
    assert result["passed"] and result["undeclared"] == []


@pytest.mark.parametrize(
    ("label", "change", "message"),
    [
        ("targets", lambda code: dict(code, target_implementation_sha256="0" * 64), "otro código"),
        ("targets", lambda code: dict(code, temporal_reference_sha256="0" * 64), "otro código"),
        ("calendar", None, "otro código"),
        ("sha", None, "no es la que fijan sus vistas"),
    ],
    ids=["targets-implementation", "targets-calendar-days", "calendar-code", "edition-hash"],
)
def test_an_edition_or_calendar_from_other_code_fails(tmp_path, label, change, message):
    code = _current_code()
    parents = _edition(tmp_path / "targets.json", **(change(code) if label == "targets" else code))
    calendar_code = regime_calendar.code_sha256()
    if label == "calendar":
        calendar_code = dict(calendar_code, **{"memory/regimes.py": "0" * 64})
    if label == "sha":
        parents = {path: "0" * 64 for path in parents}
    calendar = _calendar(tmp_path / "calendar.json", calendar_code)
    result = past.check_editions(parents, calendar, strict=False)
    assert not result["passed"] and any(message in failure for failure in result["failures"])


def test_undeclared_code_only_fails_with_the_real_edition_policy(tmp_path):
    parents = _edition(tmp_path / "targets.json")
    calendar = _calendar(tmp_path / "calendar.json", None)
    relaxed = past.check_editions(parents, calendar, strict=False)
    assert relaxed["passed"] and len(relaxed["undeclared"]) == 2
    strict = past.check_editions(parents, calendar, strict=True)
    assert not strict["passed"] and len(strict["failures"]) == 2
    # Con la política real, sin ninguna edición publicada tampoco pasa.
    assert not past.check_editions({}, None, strict=True)["passed"]


def test_the_editions_are_the_parents_of_the_published_views(tmp_path):
    study = Study(tmp_path / "study")
    entry = dict(
        label="walk-forward/US+CN",
        config=study.config_path,
        sources=study.sources_path,
        scope="US+CN",
    )
    found = past._parents([entry, dict(entry, label="posttraining/US+CN/gru")], study.policy)
    assert found == {"/unavailable/parent/manifest.json": "e" * 64}


# Reglas causales de cada sección de la comparación


@pytest.mark.parametrize("path", COMPARISONS, ids=["A", "A_v2"])
def test_every_section_of_both_comparisons_has_a_causal_rule(path):
    result = past.check_rules(path)
    assert result["passed"], result["failures"]
    assert set(result["rules"]) == {key for key, rule in past.SECTIONS.items() if rule} & set(
        json.loads(path.read_text())
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c.update(turnover_strata=dict(kind="volume_terciles_whole_sample")), "no está"),
        (
            lambda c: c["liquidity_strata"].update(source="volume_terciles_whole_sample"),
            "liquidity_strata",
        ),
        (
            lambda c: c["modality_strata"].update(presence_source="any_future_row"),
            "modality_strata",
        ),
        (lambda c: c["calibration"].update(partition="evaluation"), "calibration"),
        (lambda c: c["retention_interference"].update(regimes="centered_windows"), "retention"),
        (lambda c: c["long_short"].update(kind="whole_sample_quantiles"), "long_short"),
    ],
    ids=["new-section", "liquidity-strata", "presence", "calibration", "regimes", "long-short"],
)
def test_a_section_without_a_causal_rule_fails(tmp_path, change, message):
    config = json.loads(COMPARISONS[1].read_text())
    change(config)
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(config))
    result = past.check_rules(path)
    assert not result["passed"] and any(message in failure for failure in result["failures"])


# Sección completa


def test_the_section_gathers_code_editions_fits_rules_and_stage(tmp_path, monkeypatch):
    output, adapters = tmp_path / "campaign", tmp_path / "adapters"
    jobs = _campaign_fits(output)
    planned = _stage(adapters, _normalization())
    parents = _edition(tmp_path / "targets.json", **_current_code())
    calendar = _calendar(tmp_path / "calendar.json", regime_calendar.code_sha256())
    seen = []

    def published_parents(comparisons, policy):
        seen.append(comparisons)
        return parents

    from mars_titan.posttraining import campaign_stage

    monkeypatch.setattr(past, "_parents", published_parents)
    monkeypatch.setattr(campaign_stage, "plan_stage", lambda stage: planned)
    campaign = dict(
        _stage_campaign(),
        comparison_path=str(COMPARISONS[1]),
        input_policy="historical_masked_2000_v1",
        data_policy=REAL_EDITION_ONLY,
    )
    options = dict(posttraining=dict(stage={}), adapter_output=adapters, regimes=calendar)
    result = past.check(campaign, jobs, output, ["comparisons"], **options)
    assert result["passed"] and result["strict"] and seen == [["comparisons"]]
    assert set(result) >= {"code", "editions", "fits", "rules", "stage"}
    # Sin etapa publicada no hay sección de la etapa, y un ajuste con validación la tumba.
    result = past.check(campaign, jobs, output, [], regimes=calendar)
    assert result["passed"] and "stage" not in result
    report = output / "jobs" / jobs[0]["id"] / "run.json"
    receipt = output / "jobs" / jobs[0]["id"] / "receipt.json"
    document = json.loads(report.read_text())
    atomic_json(report, dict(document, fitted_rows=15))
    record = json.loads(receipt.read_text())
    record["report"]["sha256"] = sha256(report)
    atomic_json(receipt, record)
    assert not past.check(campaign, jobs, output, [], regimes=calendar)["passed"]
