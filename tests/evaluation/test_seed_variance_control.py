"""Control de varianza entre semillas de los pilotos de memoria (PT7, #459).

Las series por sesión se construyen aquí con valores conocidos. Si las semillas del control
y la variante son múltiplos exactos de una misma serie, cada réplica del bootstrap conserva
sus cocientes, así que la banda, la diferencia y el intervalo se conocen de antemano.
Nada se ajusta ni se lee de la edición.
"""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from mars_titan.evaluation import seed_variance_control as svc
from mars_titan.evaluation.forecast_panel import SessionSeries
from mars_titan.evaluation.paired_comparisons import (
    compare_series,
    delta,
    interaction,
    relative_difference_draws,
)

ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT / "configs/evaluation/memory-pilot-seed-control.json"
PT1 = ROOT / "configs/evaluation/titans-memory-stability-comparison.json"
CONTROL, PT1_MODEL, TRIVIAL = (
    "titans_mac_online",
    "titans_mac_online_pt1",
    "titans_mac_online_pt1_init",
)
SESSIONS = 240


def session_mae(values, *, metric="mae", loss=True):
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    return SessionSeries(
        metric,
        loss,
        "cohort",
        ("US",),
        np.zeros(count, dtype=np.int64),
        np.arange(count, dtype=np.int64),
        values,
        np.ones(count, dtype=bool),
    )


def base_values(seed=7):
    return np.random.default_rng(seed).gamma(2.0, 0.01, SESSIONS)


@pytest.fixture
def declaration():
    value, _ = svc.load_declaration(DECLARATION)
    value = copy.deepcopy(value)
    value["metric"]["replicates"] = 400
    return value


def scaled(factors, values=None):
    values = base_values() if values is None else values
    return {seed: session_mae(values * factor) for seed, factor in factors.items()}


def models(variant=PT1_MODEL, base=CONTROL):
    return dict(variant=variant, base=base, control=CONTROL)


def test_the_repository_declaration_reads_pt1_and_leaves_the_later_pilots_pending():
    summary = svc.check_declaration(DECLARATION)
    assert summary["control"]["seeds"] == [42, 43, 44]
    assert summary["extra_control_seeds"] == [43, 44]
    assert summary["rules"]["pt1"]["models"] == models()
    assert summary["rules"]["pt1_trivial_alternative"]["models"] == models(base=TRIVIAL)
    assert summary["rules"]["pt1"]["kind"] == svc.NON_INFERIORITY
    # PT4 a PT6 aún no tienen sus modelos declarados: la regla queda fijada sin ellos.
    for name in ("pt4", "pt5", "pt6"):
        rule = summary["rules"][name]
        assert rule["kind"] == svc.SUPERIORITY and rule["seeds"] == [42]
        assert rule["models"] is None
    assert summary["rules"]["pt6"]["rows"] == "missing_modality"
    with pytest.raises(ValueError, match="aún no está declarada"):
        svc.proposal_models(svc.load_declaration(DECLARATION)[0], "pt4")


def test_pt1_declares_the_same_margins_seeds_and_resampling_as_the_seed_control():
    control, _ = svc.load_declaration(DECLARATION)
    pt1 = json.loads(PT1.read_text())
    rule, decision = control["rules"]["pt1"], pt1["decision_rule"]
    assert decision["keep_if"][0].endswith(f"<= {rule['keep_margin']}")
    assert decision["abandon_if"][0].endswith(f"> {rule['abandon_margin']}")
    # La banda condiciona las dos decisiones con los mismos márgenes.
    assert decision["keep_if"][-1].endswith(f"< {rule['keep_margin']}")
    assert decision["abandon_by_margin_requires"].endswith(f"< {rule['abandon_margin']}")
    assert decision["seed_variance_control"] == dict(
        declaration="configs/evaluation/memory-pilot-seed-control.json",
        rules=["pt1", "pt1_trivial_alternative"],
    )
    trivial = control["rules"]["pt1_trivial_alternative"]
    assert pt1["pilot"]["seeds"] == {
        PT1_MODEL: len(rule["seeds"]),
        TRIVIAL: len(trivial["seeds"]),
        CONTROL: 3,
    }
    assert pt1["pilot"]["seed_control_gpu_hours"] == control["pilot"]["gpu_hours_reserved"]
    assert pt1["pilot"]["validation_windows"] == control["pilot"]["validation_windows"]
    for key in ("block_length", "replicates", "confidence", "partition"):
        assert pt1["metric"][key] == control["metric"][key]
    assert set(pt1["equal"]) >= set(control["equal"])


def test_the_reserved_hours_come_from_the_slowest_measured_pass():
    """Las horas de las semillas añadidas salen de la medida versionada, no de una hipótesis."""
    control, _ = svc.load_declaration(DECLARATION)
    pt1 = json.loads(PT1.read_text())
    receipt = json.loads((ROOT / control["pilot"]["gpu_hours_basis"]).read_text())
    assert pt1["pilot"]["gpu_hours_basis"] == control["pilot"]["gpu_hours_basis"]
    assert receipt["kind"] == "memory_pilot_seed_control_cost"
    assert receipt["optimizer_steps"] == 0 and receipt["final_test_opened"] is False
    assert receipt["model"] == CONTROL and receipt["device"] == "cuda:0"
    assert len(receipt["inputs"]["assets"]) == 64 and receipt["inputs"]["max_event_rows"] <= 64
    slowest = receipt["estimate"]["slowest"]
    assumptions = slowest["assumptions"]
    assert assumptions["validation_windows"] == control["pilot"]["validation_windows"]
    assert assumptions["search_cases"] == len(receipt["search_cases"]) == 2
    rates = receipt["rates"]
    assert len(rates["train"]) == len(rates["inference"]) == receipt["repeats"] >= 3
    # La estimación usa la pasada más lenta de las medidas, que da la cota superior.
    expected = assumptions["validation_passes_per_case"] * assumptions["search_cases"]
    assert slowest["inference_rows_per_seed"] == slowest["validation_rows_per_pass"] * expected
    hours = (
        slowest["fit_rows_per_seed"] / min(rates["train"])
        + slowest["inference_rows_per_seed"] / min(rates["inference"])
    ) / 3600
    assert slowest["hours_per_seed"] == pytest.approx(hours, rel=1e-12)
    assert slowest["hours_per_seed"] >= receipt["estimate"]["median"]["hours_per_seed"]
    # Dos semillas añadidas del control en la reserva. PT1, su alternativa trivial y la
    # semilla 42 del control en el presupuesto de PT1, con el coste por semilla del control.
    assert control["pilot"]["gpu_hours_reserved"] == math.ceil(2 * slowest["hours_per_seed"])
    assert pt1["pilot"]["gpu_hours_max"] == math.ceil(3 * slowest["hours_per_seed"])


def _invalid():
    def rule(name, **changes):
        return lambda value: value["rules"][name].update(changes)

    return {
        "two_control_seeds": (lambda v: v["control"].update(seeds=[42, 43]), "al menos 3"),
        "repeated_seed": (lambda v: v["control"].update(seeds=[42, 42, 43]), "al menos 3"),
        "every_control_seed_paired": (rule("pt1", seeds=[42, 43, 44]), "dejarle semillas"),
        "foreign_seed": (rule("pt4", seeds=[7]), "dejarle semillas"),
        "selection_on_evaluation": (
            lambda v: v["metric"].update(partition="evaluation"),
            "en validación",
        ),
        "final_test": (lambda v: v.update(final_test_opened=True), "reserva de 2024"),
        "margins_reversed": (rule("pt1", keep_margin=0.005, abandon_margin=0.002), "abandonar"),
        "margin_on_superiority": (rule("pt4", keep_margin=0.002), "exactamente"),
        "unknown_kind": (rule("pt4", kind="better"), "tipo desconocido"),
        "control_as_variant": (rule("pt4", variant="control"), "innovación"),
        "missing_extra_seed": (
            lambda v: v["pilot"].update(extra_control_seeds=[43]),
            "no empareja",
        ),
        "no_reserve": (lambda v: v["pilot"].update(gpu_hours_reserved=0), "reservar"),
        "other_band": (lambda v: v["band"].update(statistic="mean_abs"), "máximo"),
        "executed_without_runs": (lambda v: v.update(status="executed"), "no concuerdan"),
        "unknown_sharing": (lambda v: v["pilot"].update(shared_by=["pt9"]), "compartirse"),
        "unknown_rows": (rule("pt6", rows="news_only"), "filas"),
        "integer_confidence": (lambda v: v["metric"].update(confidence=1), "confianza"),
        "no_discard_rule": (lambda v: v.pop("discard_if"), "descarta el control"),
        "extra_seeds_skip_selection": (
            lambda v: v["pilot"].update(extra_seed_procedure="selected_case_only"),
            "repetir la búsqueda",
        ),
    }


INVALID = _invalid()


@pytest.mark.parametrize("name", sorted(INVALID))
def test_invalid_declarations_are_rejected(name):
    change, message = INVALID[name]
    value = json.loads(DECLARATION.read_text())
    change(value)
    with pytest.raises(ValueError, match=message):
        svc.validate_declaration(value)


def test_a_proposal_with_another_control_cannot_use_the_band(tmp_path):
    value, _ = svc.load_declaration(DECLARATION)
    pt1 = json.loads(PT1.read_text())
    pt1["variant"] = "mac_frozen"
    (tmp_path / "configs/evaluation").mkdir(parents=True)
    (tmp_path / value["rules"]["pt1"]["declaration"]).write_text(json.dumps(pt1))
    with pytest.raises(ValueError, match="no es el del control de varianza"):
        svc.proposal_models(value, "pt1", root=tmp_path)


@pytest.mark.parametrize(
    ("interval", "band", "expected"),
    [
        ((-0.001, 0.0019), 0.001, "keep"),
        ((-0.001, 0.002), 0.001, "keep"),
        ((-0.001, 0.0019), 0.002, "not_interpretable"),
        ((-0.001, 0.003), 0.001, "undecided"),
        ((0.001, 0.006), 0.001, "abandon"),
        ((0.001, 0.006), 0.005, "not_interpretable"),
        ((-0.001, 0.005), 0.004, "undecided"),
    ],
)
def test_non_inferiority_never_widens_its_margin(interval, band, expected):
    decision, _ = svc.decide(
        svc.NON_INFERIORITY, interval, band, keep_margin=0.002, abandon_margin=0.005
    )
    assert decision == expected


@pytest.mark.parametrize(
    ("kind", "interval", "band", "expected"),
    [
        (svc.SUPERIORITY, (-0.01, -0.0031), 0.003, "keep"),
        (svc.SUPERIORITY, (-0.01, -0.003), 0.003, "abandon"),
        (svc.SUPERIORITY, (-0.01, -0.001), 0.0, "keep"),
        (svc.SUPERIORITY, (-0.01, 0.001), 0.0, "abandon"),
        (svc.EQUIVALENCE, (-0.002, 0.002), 0.002, "equivalent"),
        (svc.EQUIVALENCE, (-0.0021, 0.001), 0.002, "not_equivalent"),
        (svc.EQUIVALENCE, (-0.001, 0.0021), 0.002, "not_equivalent"),
    ],
)
def test_a_difference_or_an_equality_must_clear_the_band(kind, interval, band, expected):
    assert svc.decide(kind, interval, band)[0] == expected


def test_decide_rejects_a_reversed_interval_a_negative_band_or_an_unknown_kind():
    with pytest.raises(ValueError, match="no son válidos"):
        svc.decide(svc.SUPERIORITY, (0.01, -0.01), 0.001)
    with pytest.raises(ValueError, match="no son válidos"):
        svc.decide(svc.SUPERIORITY, (-0.01, 0.01), -0.001)
    with pytest.raises(ValueError, match="desconocido"):
        svc.decide("better", (-0.01, 0.01), 0.001)


def test_relative_draws_reproduce_the_relative_interval_of_compare_series():
    rng = np.random.default_rng(3)
    base = base_values()
    series = {
        "B": session_mae(base),
        "V": session_mae(base * (1 + rng.normal(0, 0.05, SESSIONS))),
        "W": session_mae(base * (1 + rng.normal(0.01, 0.05, SESSIONS))),
    }
    contrasts = {"d": delta("B", "V"), "e": delta("V", "W")}
    options = dict(block_length=5, replicates=300, seed=11)
    result = compare_series(series, contrasts, **options)
    estimates, draws = relative_difference_draws(series, contrasts, **options)
    assert draws.shape == (300, 2)
    for j, row in enumerate(result["contrasts"]):
        assert -100 * estimates[j] == pytest.approx(row["relative_improvement_percent"], rel=1e-12)
        bounds = np.quantile(-100 * draws[:, j], [0.025, 0.975], method="linear")
        assert list(bounds) == pytest.approx(row["relative_improvement_interval"], rel=1e-12)


def test_relative_draws_only_accept_differences_of_losses_with_enough_days():
    series = {"B": session_mae(base_values()), "V": session_mae(base_values(8))}
    options = dict(block_length=5, replicates=50, seed=1)
    with pytest.raises(ValueError, match="no es una diferencia entre dos modelos"):
        relative_difference_draws(
            dict(series, C=series["B"], D=series["V"]),
            {"i": interaction("B", "V", "C", "D")},
            **options,
        )
    ic = {
        name: session_mae(item.values, metric="rank_ic", loss=False)
        for name, item in series.items()
    }
    with pytest.raises(ValueError, match="solo se define para pérdidas"):
        relative_difference_draws(ic, {"d": delta("B", "V")}, **options)
    with pytest.raises(ValueError, match="más días"):
        relative_difference_draws(
            series, {"d": delta("B", "V")}, **dict(options, block_length=SESSIONS)
        )
    with pytest.raises(ValueError, match="es cero"):
        relative_difference_draws(
            dict(series, B=session_mae(np.zeros(SESSIONS))), {"d": delta("B", "V")}, **options
        )


def test_the_band_is_the_largest_difference_between_control_seeds(declaration):
    # Con múltiplos exactos, cada réplica repite los cocientes: la banda es la mayor de las
    # tres diferencias relativas y el intervalo de PT1 se reduce a su diferencia.
    series = {CONTROL: scaled({42: 1.0, 43: 1.004, 44: 0.999}), PT1_MODEL: scaled({42: 1.001})}
    report = svc.evaluate_rule(declaration, "pt1", series, models())
    pairs = {item["name"]: item["relative_difference"] for item in report["control_pairs"]}
    assert pairs == pytest.approx(
        {
            f"{CONTROL}@43-{CONTROL}@42": 0.004,
            f"{CONTROL}@44-{CONTROL}@42": -0.001,
            f"{CONTROL}@44-{CONTROL}@43": 0.999 / 1.004 - 1,
        },
        rel=1e-9,
    )
    # La pareja 43 y 44 se compara con la de menor MAE como base, en cualquier orden.
    widest = 1.004 / 0.999 - 1
    assert report["observed_seed_spread"] == pytest.approx(widest, rel=1e-9)
    assert report["minimum_interpretable_difference"] == pytest.approx(widest, rel=1e-9)
    assert report["relative_difference"] == pytest.approx(0.001, rel=1e-9)
    assert report["interval"] == pytest.approx([0.001, 0.001], rel=1e-9)
    # El límite superior no supera el 0,2 %, pero la banda sí: no se puede conservar. Sin la
    # banda se habría conservado, así que aquí el control de varianza cambia la decisión.
    assert report["decision"] == "not_interpretable"
    assert report["decision_without_band"] == "keep"
    assert report["sessions"] == SESSIONS and report["final_test_opened"] is False
    assert report["paired_seeds"] == [42] and report["control_seeds"] == [42, 43, 44]
    # Con semillas más parecidas la banda queda bajo el margen y PT1 se conserva.
    series[CONTROL] = scaled({42: 1.0, 43: 1.0005, 44: 0.9998})
    report = svc.evaluate_rule(declaration, "pt1", series, models())
    assert report["minimum_interpretable_difference"] == pytest.approx(
        1.0005 / 0.9998 - 1, rel=1e-9
    )
    assert report["decision"] == "keep"


def test_the_variant_is_paired_with_the_control_of_its_own_seed(declaration):
    # Las semillas 43 y 44 del control solo forman la banda. La base de PT1 es la 42.
    series = {CONTROL: scaled({42: 1.0, 43: 1.03, 44: 0.97}), PT1_MODEL: scaled({42: 1.0})}
    report = svc.evaluate_rule(declaration, "pt1", series, models())
    assert report["relative_difference"] == 0.0 and report["interval"] == [0.0, 0.0]
    assert report["minimum_interpretable_difference"] == pytest.approx(1.03 / 0.97 - 1)


@pytest.mark.parametrize(("spread", "expected"), [(0.004, "keep"), (0.012, "abandon")])
def test_an_improvement_counts_only_beyond_the_band(declaration, spread, expected):
    series = {
        CONTROL: scaled({42: 1.0, 43: 1.0 + spread, 44: 1.0}),
        "market_memory": scaled({42: 0.99}),
    }
    report = svc.evaluate_rule(declaration, "pt4", series, models(variant="market_memory"))
    assert report["relative_difference"] == pytest.approx(-0.01, rel=1e-9)
    assert report["minimum_interpretable_difference"] == pytest.approx(spread, rel=1e-9)
    assert report["decision"] == expected and report["rule_kind"] == "superiority"
    assert report["decision_without_band"] == "keep"


@pytest.mark.parametrize(
    ("spread", "expected"), [(0.004, "equivalent"), (0.0005, "not_equivalent")]
)
def test_the_trivial_alternative_is_equal_only_inside_the_band(declaration, spread, expected):
    series = {
        CONTROL: scaled({42: 1.0, 43: 1.0 + spread, 44: 1.0}),
        PT1_MODEL: scaled({42: 1.001}),
        TRIVIAL: scaled({42: 1.0}),
    }
    report = svc.evaluate_rule(declaration, "pt1_trivial_alternative", series, models(base=TRIVIAL))
    assert report["relative_difference"] == pytest.approx(0.001, rel=1e-9)
    assert report["decision"] == expected


def test_the_rule_records_its_rows_and_is_repeatable(declaration):
    rng = np.random.default_rng(5)
    base = base_values()
    noisy = {seed: base * (1 + rng.normal(0, 0.02, SESSIONS)) for seed in (42, 43, 44)}
    series = {
        CONTROL: {seed: session_mae(values) for seed, values in noisy.items()},
        "presence_film": {42: session_mae(base * (1 + rng.normal(-0.01, 0.02, SESSIONS)))},
        "wide_fusion": {42: session_mae(base * (1 + rng.normal(0, 0.02, SESSIONS)))},
    }
    chosen = models(variant="presence_film", base="wide_fusion")
    first = svc.evaluate_rule(declaration, "pt6", series, chosen)
    assert first["rows"] == "missing_modality" and first["base"] == "wide_fusion"
    assert svc.evaluate_rule(declaration, "pt6", series, chosen) == first
    lower, upper = first["interval"]
    assert lower < first["relative_difference"] < upper
    # La banda incluye el remuestreo de las sesiones, así que supera la dispersión observada.
    assert first["minimum_interpretable_difference"] > first["observed_seed_spread"]
    other = copy.deepcopy(declaration)
    other["metric"]["bootstrap_seed"] += 1
    assert svc.evaluate_rule(other, "pt6", series, chosen)["interval"] != first["interval"]


def test_the_rule_interval_is_the_relative_interval_of_compare_series(declaration):
    # Con ruido real por sesión, el intervalo de la regla es el de la mejora relativa de
    # `compare_series` con el mismo bloque, réplicas y semilla, con el signo cambiado.
    rng = np.random.default_rng(9)
    base = base_values()
    series = {
        CONTROL: {s: session_mae(base * (1 + rng.normal(0, 0.03, SESSIONS))) for s in (42, 43, 44)},
        PT1_MODEL: {42: session_mae(base * (1 + rng.normal(0.004, 0.03, SESSIONS)))},
    }
    report = svc.evaluate_rule(declaration, "pt1", series, models())
    metric = declaration["metric"]
    result = compare_series(
        {"base": series[CONTROL][42], "variant": series[PT1_MODEL][42]},
        {"proposal": delta("base", "variant")},
        block_length=metric["block_length"],
        replicates=metric["replicates"],
        seed=metric["bootstrap_seed"],
        confidence=metric["confidence"],
    )
    (row,) = result["contrasts"]
    lower, upper = (-value / 100 for value in reversed(row["relative_improvement_interval"]))
    assert report["interval"] == pytest.approx([lower, upper], rel=1e-12)
    assert report["relative_difference"] == pytest.approx(
        -row["relative_improvement_percent"] / 100, rel=1e-12
    )


def test_the_band_is_the_declared_quantile_of_the_widest_pair_in_each_replicate(declaration):
    # Con ruido por sesión la banda varía entre réplicas: es el cuantil declarado del
    # máximo por réplica, recalculado aquí con las mismas réplicas.
    rng = np.random.default_rng(13)
    base = base_values()
    noisy = {s: session_mae(base * (1 + rng.normal(0, 0.03, SESSIONS))) for s in (42, 43, 44)}
    series = {CONTROL: noisy, PT1_MODEL: {42: session_mae(base)}}
    report = svc.evaluate_rule(declaration, "pt1", series, models())
    metric = declaration["metric"]
    named = {f"c{seed}": value for seed, value in noisy.items()}
    pairs = {"a": delta("c42", "c43"), "b": delta("c42", "c44"), "c": delta("c43", "c44")}
    _, draws = relative_difference_draws(
        {"variant": series[PT1_MODEL][42], "base": noisy[42], **named},
        {"proposal": delta("base", "variant"), **pairs},
        block_length=metric["block_length"],
        replicates=metric["replicates"],
        seed=metric["bootstrap_seed"],
    )
    ratio = 1 + draws[:, 1:]
    widest = (np.maximum(ratio, 1 / ratio) - 1).max(axis=1)
    assert np.ptp(widest) > 0
    expected = np.quantile(widest, declaration["band"]["quantile"], method="linear")
    assert report["minimum_interpretable_difference"] == pytest.approx(expected, rel=1e-12)
    lower = copy.deepcopy(declaration)
    lower["band"]["quantile"] = 0.5
    median = svc.evaluate_rule(lower, "pt1", series, models())["minimum_interpretable_difference"]
    assert median == pytest.approx(np.quantile(widest, 0.5, method="linear"), rel=1e-12)
    assert median < report["minimum_interpretable_difference"]


def test_missing_seeds_a_variant_used_as_base_or_another_metric_are_rejected(declaration):
    series = {CONTROL: scaled({42: 1.0, 43: 1.001}), PT1_MODEL: scaled({42: 1.0})}
    with pytest.raises(ValueError, match="Faltan semillas de titans_mac_online"):
        svc.evaluate_rule(declaration, "pt1", series, models())
    series[CONTROL] = scaled({42: 1.0, 43: 1.001, 44: 1.0})
    with pytest.raises(ValueError, match="no puede ser la base"):
        svc.evaluate_rule(declaration, "pt1", series, models(variant=CONTROL))
    series[CONTROL] = {seed: session_mae(base_values(), metric="mse") for seed in (42, 43, 44)}
    with pytest.raises(ValueError, match="MAE por sesión"):
        svc.evaluate_rule(declaration, "pt1", series, models())


def test_every_seed_and_model_must_cover_the_same_sessions(declaration):
    # Una semilla del control con un día menos haría que la banda y el contraste vieran
    # sesiones distintas. Lo mismo con la variante.
    series = {CONTROL: scaled({42: 1.0, 43: 1.001, 44: 1.0}), PT1_MODEL: scaled({42: 1.0})}
    full = series[CONTROL][44]
    defined = full.defined.copy()
    defined[17] = False
    series[CONTROL][44] = SessionSeries(
        "mae",
        True,
        full.cohort_sha256,
        full.markets,
        full.session_market,
        full.session_period,
        np.where(defined, full.values, 0.0),
        defined,
    )
    with pytest.raises(ValueError, match="mismas sesiones"):
        svc.evaluate_rule(declaration, "pt1", series, models())
    series[CONTROL][44] = full
    assert svc.evaluate_rule(declaration, "pt1", series, models())["sessions"] == SESSIONS
    variant = series[PT1_MODEL][42]
    shifted = variant.session_period.copy()
    shifted[-1] += 1
    series[PT1_MODEL][42] = SessionSeries(
        "mae",
        True,
        variant.cohort_sha256,
        variant.markets,
        variant.session_market,
        shifted,
        variant.values,
        variant.defined,
    )
    with pytest.raises(ValueError, match="mismas sesiones"):
        svc.evaluate_rule(declaration, "pt1", series, models())


def _report(rule, decision, without):
    return dict(rule=rule, decision=decision, decision_without_band=without)


def test_the_control_is_discarded_only_if_the_band_changes_no_decision(declaration):
    same = [_report("pt1", "keep", "keep"), _report("pt4", "abandon", "abandon")]
    verdict = svc.discard_verdict(declaration, same)
    assert verdict == dict(
        criterion=svc.DISCARD, evaluated=["pt1", "pt4"], changed_by_band=[], discard=True
    )
    changed = [_report("pt1", "not_interpretable", "keep"), _report("pt4", "abandon", "abandon")]
    verdict = svc.discard_verdict(declaration, changed)
    assert verdict["changed_by_band"] == ["pt1"] and verdict["discard"] is False
    # El criterio se calcula con los informes reales de las reglas.
    series = {CONTROL: scaled({42: 1.0, 43: 1.004, 44: 0.999}), PT1_MODEL: scaled({42: 1.001})}
    report = svc.evaluate_rule(declaration, "pt1", series, models())
    assert svc.discard_verdict(declaration, [report])["discard"] is False
    for reports, message in (
        ([], "al menos una"),
        ([_report("pt1", "keep", "keep")] * 2, "una vez"),
        ([_report("pt9", "keep", "keep")], "declarada"),
    ):
        with pytest.raises(ValueError, match=message):
            svc.discard_verdict(declaration, reports)


def test_the_command_checks_the_declaration_without_predictions(capsys):
    svc.main(["check", "--declaration", str(DECLARATION)])
    printed = json.loads(capsys.readouterr().out)
    declared, digest = svc.load_declaration(DECLARATION)
    assert printed["sha256"] == digest and printed["rules"]["pt1"]["models"] == models()
    assert printed["gpu_hours_reserved"] == declared["pilot"]["gpu_hours_reserved"]
