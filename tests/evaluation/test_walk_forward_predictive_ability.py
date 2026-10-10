"""Contrastes de capacidad predictiva dentro de la comparación walk-forward.

Reutiliza el estudio sintético de ``test_walk_forward_comparison`` con veinte días de
decisión por ventana, para que los remuestreos por bloques tengan días suficientes.
Las predicciones son valores conocidos escritos en la prueba, no salen de ningún modelo
ajustado y no se ejecuta ningún paso de optimizador.
"""

import copy
import json
from datetime import UTC, datetime, timedelta
from importlib.metadata import version

import numpy as np
import pytest
from arch.bootstrap import MCS, SPA
from statsmodels.tsa.stattools import diebold_mariano_test

from mars_titan.evaluation import predictive_ability as pa
from mars_titan.evaluation import walk_forward_comparison as walk
from tests.evaluation import test_walk_forward_comparison as base

DAYS = 20


def long_decisions(month):
    """Veinte instantes de decisión por mercado, al cierre de US y de CN del mismo día UTC."""
    first = datetime(2023, month, 2, tzinfo=UTC)
    return {
        market: [
            first + timedelta(days=day, hours=21 if market == "US" else 7) for day in range(DAYS)
        ]
        for market in ("US", "CN")
    }


def section():
    declared = json.loads(base.CONFIG.read_text())["predictive_ability"]
    return dict(declared, metrics=["mae", "pinball"], min_days=30)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, "decisions", long_decisions)
        plain = base.Study(tmp_path_factory.mktemp("plain"))
        without = plain.run()[0]
        study = base.Study(tmp_path_factory.mktemp("predictive"))
        study.config[walk.PREDICTIVE_FIELD] = section()
        study.publish()
        report, _ = study.run()
        daily = {arm: daily_mae(study, arm) for arm in base.ARMS}
    return study, report, without, daily


def daily_mae(study, arm):
    """MAE diario a mano: media por sesión, media de semillas y media de las sesiones del día."""
    seeds = base.ARMS[arm]["seeds"] or [None]
    per_seed = []
    for seed in seeds:
        sessions = {}
        for fold in study.folds:
            table = study.table(arm, seed, fold, "evaluation").to_pydict()
            prediction = [0.0] * len(table["target"]) if seed is None else table["prediction"]
            for market, moment, y, p in zip(
                table["market"], table["prediction_at"], table["target"], prediction, strict=True
            ):
                sessions.setdefault((market, moment), []).append(abs(p - y))
        per_seed.append({key: np.mean(errors) for key, errors in sessions.items()})
    keys = sorted(per_seed[0], key=lambda key: (key[1].date(), key[0]))
    days = {}
    for key in keys:
        days.setdefault(key[1].date(), []).append(np.mean([item[key] for item in per_seed]))
    return np.array([np.mean(values) for _, values in sorted(days.items())])


def test_the_section_is_optional_and_changes_no_other_output(runs):
    _, report, without, _ = runs
    assert walk.PREDICTIVE_FIELD not in without
    for key in ("arms", "contrasts", "interval_calibration", "sign_reliability"):
        assert report[key] == without[key], key
    assert report["versions"]["arch"] == version("arch") == "8.0.0"
    assert report["versions"]["statsmodels"] == version("statsmodels") == "0.15.0"
    assert "evaluation/predictive_ability.py" in report["analysis_source_sha256"]
    json.dumps(report, allow_nan=False)


def test_daily_losses_and_diebold_mariano_match_a_computation_by_hand(runs):
    _, report, _, daily = runs
    section = report[walk.PREDICTIVE_FIELD]
    assert list(section["views"]) == ["US+CN", "US", "CN"]
    family = section["views"]["US+CN"]["mae"]["references_vs_zero"]
    assert family["days"] == 2 * DAYS and family["excluded_days"] == 0
    for arm, value in family["mean_daily_loss"].items():
        assert value == pytest.approx(daily[arm].mean(), rel=1e-12), arm
    for row in family["diebold_mariano"]:
        variant, zero = daily[row["variant"]], daily[row["base"]]
        direct = diebold_mariano_test(
            np.zeros(2 * DAYS),
            variant,
            zero,
            lags=pa.hac_lags(2 * DAYS),
            criterion="mae",
            harvey_adj=True,
        )
        assert row["statistic"] == pytest.approx(float(direct.statistic), rel=1e-9)
        assert row["pvalue"] == pytest.approx(float(direct.pvalue), rel=1e-9, abs=1e-15)
        assert row["statistic"] < 0  # las predicciones mejoran al control cero


def test_spa_and_mcs_use_the_declared_resampling_on_the_same_days(runs):
    study, report, _, daily = runs
    comparison = study.config["comparison"]
    family = report[walk.PREDICTIVE_FIELD]["views"]["US+CN"]["mae"]["references_vs_zero"]
    spa = family["superior_predictive_ability"]
    assert spa["benchmark"] == "zero" and spa["variants"] == ["ridge", "gru", "titans"]
    direct = SPA(
        daily["zero"],
        np.column_stack([daily[arm] for arm in spa["variants"]]),
        block_size=comparison["block_length"],
        reps=comparison["replicates"],
        bootstrap="circular",
        studentize=False,
        seed=comparison["seed"],
    )
    direct.compute()
    for key, value in direct.pvalues.items():
        assert spa["pvalues"][key] == pytest.approx(float(value), abs=1e-12)
    assert spa["reality_check_pvalue"] == spa["pvalues"]["upper"]
    assert set(spa["stepm_superior"]) <= {"ridge", "gru", "titans"}
    mcs = family["model_confidence_set"]
    arms = family["arms"]
    direct = MCS(
        np.column_stack([daily[arm] for arm in arms]),
        size=0.1,
        reps=comparison["replicates"],
        block_size=comparison["block_length"],
        method="R",
        bootstrap="circular",
        seed=comparison["seed"],
    )
    direct.compute()
    assert mcs["included"] == [arms[i] for i in sorted(int(i) for i in direct.included)]
    assert "zero" not in mcs["included"]


def test_quantile_metrics_skip_families_with_point_arms(runs):
    _, report, _, _ = runs
    view = report[walk.PREDICTIVE_FIELD]["views"]["US"]["pinball"]
    assert "métrica" in view["references_vs_zero"]["reason"]
    assert "métrica" in view["levels"]["reason"]
    quantile = view["quantile_models"]
    assert quantile["days"] == 2 * DAYS and quantile["arms"] == ["titans", "gru"]
    assert quantile["diebold_mariano"][0]["name"] == "titans-gru"


def test_a_wrong_section_is_rejected_before_reading_predictions(tmp_path):
    study = base.Study(tmp_path)
    study.config[walk.PREDICTIVE_FIELD] = dict(section(), metrics=["mae", "rank_ic"])
    study.publish()
    with pytest.raises(ValueError, match="pérdidas"):
        study.run()
    broken = copy.deepcopy(study.config)
    broken[walk.PREDICTIVE_FIELD] = dict(section(), min_days=1)
    study.config = broken
    study.publish()
    with pytest.raises(ValueError, match="al menos 3"):
        study.run()


def test_campaign_comparisons_validate_with_the_section(tmp_path):
    for path in (base.CONFIG, base.CONFIG.parent / "historical-masked-2000-joint-comparison.json"):
        config = walk.load_config(path)
        assert config[walk.PREDICTIVE_FIELD]["metrics"] == ["mae"]
        assert config["schema_version"] in (4, 5)
