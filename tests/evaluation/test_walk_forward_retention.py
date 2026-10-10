"""Retención e interferencia dentro de la comparación walk-forward.

Reutiliza el estudio sintético de ``test_walk_forward_comparison`` con veinte decisiones por
ventana y mercado y un calendario de regímenes escrito en la prueba. Las predicciones son
valores conocidos, no salen de ningún modelo ajustado y no se ejecuta ningún paso de
optimizador. La prueba comprueba la conexión: qué series entran, cómo se promedian las
semillas, qué sesiones pertenecen a cada ventana y qué identidad queda en el informe.
"""

import json
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from mars_titan.data.storage import sha256
from mars_titan.evaluation import regime_calendar as rc
from mars_titan.evaluation import retention_interference as ri
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.memory.regimes import REGIME_RULE, RegimeRule
from tests.evaluation import test_walk_forward_comparison as base
from tests.evaluation.test_walk_forward_predictive_ability import long_decisions

HOURS = {"US": 21, "CN": 7}
WARMUP, MEASURED = ri.STARTS


def section():
    declared = json.loads(base.CONFIG.read_text())[walk.RETENTION_FIELD]
    return dict(
        declared,
        warmup_months=1,
        entry_sessions=2,
        long_absence_sessions=4,
        absence_bins=[2, 4, 8],
        placebo_lag_sessions=5,
        min_sessions=1,
        pairs=dict(
            titans_over_gru=dict(
                memory="titans", control="gru", history=MEASURED, separates="fixture"
            ),
            gru_over_ridge=dict(memory="gru", control="ridge", history=WARMUP, separates="fixture"),
        ),
    )


def write_calendar(path, *, skip=None):
    """Una sesión diaria por mercado de septiembre a diciembre de 2023 con rachas fijas."""
    rng = np.random.default_rng(17)
    markets = {}
    for market, hour in HOURS.items():
        days = [date(2023, 9, 1) + timedelta(days=i) for i in range(122)]
        if skip is not None:
            days = [day for day in days if day != skip]
        routes = []
        while len(routes) < len(days):
            routes += [int(rng.integers(1, 5))] * int(rng.integers(1, 5))
        markets[market] = dict(
            sessions=[day.isoformat() for day in days],
            at=[
                int(datetime(d.year, d.month, d.day, hour, tzinfo=UTC).timestamp() * 1_000_000)
                for d in days
            ],
            route=routes[: len(days)],
            assets=[7] * len(days),
            returns=[63] * len(days),
        )
    calendar = dict(
        schema_version=rc.SCHEMA_VERSION,
        kind=rc.KIND,
        rule=RegimeRule(REGIME_RULE).identity(),
        final_test_opened=False,
        markets=markets,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calendar))
    return path


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, "decisions", long_decisions)
        plain = base.Study(tmp_path_factory.mktemp("plain"))
        without = plain.run()[0]
        study = base.Study(tmp_path_factory.mktemp("retention"))
        study.config[walk.RETENTION_FIELD] = section()
        study.publish()
        calendar = write_calendar(study.root / "regimes" / "calendar.json")
        report, _ = walk.evaluate_walk_forward(
            study.config_path, study.sources_path, study.scope, regimes=calendar
        )
        pending, _ = study.run()
        expected = {arm: session_mae(study, arm) for arm in ("ridge", "gru", "titans")}
    return study, report, pending, without, calendar, expected


def session_mae(study, arm):
    """MAE por sesión a mano: media por activo de cada semilla y media de las semillas."""
    per_seed = []
    for seed in base.ARMS[arm]["seeds"]:
        sessions = {}
        for fold in study.folds:
            table = study.table(arm, seed, fold, "evaluation").to_pydict()
            for market, moment, y, p in zip(
                table["market"],
                table["prediction_at"],
                table["target"],
                table["prediction"],
                strict=True,
            ):
                sessions.setdefault((market, moment), []).append(abs(p - y))
        per_seed.append({key: np.mean(errors) for key, errors in sessions.items()})
    return {key: np.mean([seed[key] for seed in per_seed]) for key in per_seed[0]}


def test_the_section_uses_the_seed_averaged_session_mae_of_each_pair(runs):
    study, report, _, _, calendar, expected = runs
    result = report[walk.RETENTION_FIELD]
    assert result["status"] == "computed" and result["declaration"] == section()
    assert result["calendar"] == dict(
        sha256=sha256(calendar), rule=RegimeRule(REGIME_RULE).identity()
    )
    resolved, _, _ = rc.read(calendar)
    windows = [tuple(fold["evaluation"]) for fold in study.folds]
    for market in ("US", "CN"):
        keys = sorted(key for key in expected["gru"] if key[0] == market)
        times = np.array([int(moment.timestamp() * 1_000_000) for _, moment in keys])
        for name, pair in section()["pairs"].items():
            benefit = np.array(
                [expected[pair["control"]][key] - expected[pair["memory"]][key] for key in keys]
            )
            kind, absence, _ = ri.session_classes(
                times, resolved[market], windows, section(), pair["history"]
            )
            revisit, novel = benefit[kind == 1], benefit[kind == 0]
            record = result["markets"][market]["pairs"][name]
            assert record["estimates"]["predictive_effect"] == pytest.approx(
                benefit[kind >= 0].mean()
            )
            assert record["estimates"]["retention"] == pytest.approx(revisit.mean() - novel.mean())
            long, short = (benefit[(kind == 1) & mask] for mask in (absence >= 4, absence < 4))
            assert record["estimates"]["interference"] == pytest.approx(long.mean() - short.mean())
            assert record["sessions"]["classified"] == len(keys) == 40
            assert set(record["decisions"]) == {"retention", "interference"}


def test_the_rest_of_the_report_does_not_change(runs):
    _, report, pending, without, _, _ = runs
    for key in ("contrasts", "arms", "interval_calibration", "sign_reliability"):
        assert report[key] == pending[key] == without[key]
    assert set(report["analysis_source_sha256"]) - set(without["analysis_source_sha256"]) == {
        "evaluation/retention_interference.py",
        "evaluation/regime_calendar.py",
    }


def test_without_a_calendar_the_section_stays_pending(runs):
    _, _, pending, without, _, _ = runs
    assert pending[walk.RETENTION_FIELD] == ri.pending(section())
    assert walk.RETENTION_FIELD not in without
    assert "evaluation/retention_interference.py" not in pending["analysis_source_sha256"]


def test_a_calendar_needs_the_declared_section_and_every_evaluated_session(runs, tmp_path):
    study, _, _, _, _, _ = runs
    plain = base.Study(tmp_path / "plain")
    calendar = write_calendar(tmp_path / "calendar.json")
    with pytest.raises(ValueError, match="no declara la retención"):
        walk.evaluate_walk_forward(
            plain.config_path, plain.sources_path, plain.scope, regimes=calendar
        )
    missing = write_calendar(tmp_path / "missing.json", skip=date(2023, 11, 3))
    with pytest.raises(ValueError, match="no está en el calendario"):
        walk.evaluate_walk_forward(
            study.config_path, study.sources_path, study.scope, regimes=missing
        )


def test_the_loader_validates_the_declared_section(tmp_path):
    study = base.Study(tmp_path / "invalid")
    wrong = section()
    wrong["pairs"]["titans_over_gru"]["control"] = "lstm"
    study.config[walk.RETENTION_FIELD] = wrong
    study.publish()
    with pytest.raises(ValueError, match="dos brazos distintos de la comparación"):
        walk.load_config(study.config_path)


def test_the_command_line_publishes_the_section(runs, tmp_path, capsys):
    study, report, _, _, calendar, _ = runs
    output = tmp_path / "comparison"
    assert (
        walk.main(
            [
                "--config",
                str(study.config_path),
                "--sources",
                str(study.sources_path),
                "--scope",
                study.scope,
                "--output",
                str(output),
                "--regimes",
                str(calendar),
            ]
        )
        == 0
    )
    published = json.loads((output / "comparison.json").read_text())
    assert published[walk.RETENTION_FIELD]["markets"] == json.loads(
        json.dumps(report[walk.RETENTION_FIELD]["markets"])
    )
    assert "Reserva final cerrada" in capsys.readouterr().out
