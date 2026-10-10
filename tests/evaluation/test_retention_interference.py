"""Retención e interferencia en las revisitas de un régimen, con series fijadas a mano.

Las rutas y los beneficios por sesión se escriben en la prueba, así que cada clase y cada
estadístico se deducen de la declaración y no de una ejecución anterior. No hay modelos ni
pasos de optimizador: memoria y control son series de MAE conocidas.
"""

import copy
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from mars_titan.evaluation import retention_interference as ri
from mars_titan.evaluation.paired_comparisons import circular_block_counts, family_intervals
from mars_titan.training.walk_forward_phases import months_before

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/evaluation/historical-masked-2000-comparison.json"
JOINT = ROOT / "configs/evaluation/historical-masked-2000-joint-comparison.json"
NOVEL, REVISIT, CONTINUING = (ri.CLASSES.index(name) for name in ri.CLASSES)
OPTIONS = dict(block_length=3, replicates=400, seed=11, confidence=0.95)
DAY = 86_400_000_000
FIELD = "retention_interference"
WARMUP, MEASURED = ri.STARTS


def declared():
    return copy.deepcopy(json.loads(CONFIG.read_text())[FIELD])


def arms():
    return json.loads(CONFIG.read_text())["arms"]


def section(**changes):
    value = dict(
        declared(),
        warmup_months=1,
        entry_sessions=2,
        long_absence_sessions=4,
        absence_bins=[2, 4, 8],
        placebo_lag_sessions=5,
        min_sessions=1,
        pairs=dict(
            writes=dict(memory="memory", control="control", history=WARMUP, separates="writes"),
            same=dict(memory="control", control="twin", history=WARMUP, separates="nothing"),
        ),
    )
    value.update(changes)
    return value


def test_the_declared_section_is_valid_in_both_comparisons():
    for path in (CONFIG, JOINT):
        config = json.loads(path.read_text())
        assert ri.declaration(config[FIELD], config["arms"])
    pairs = declared()["pairs"]
    assert pairs["fast_weight_writes"] == dict(
        memory="titans_mac_online",
        control="titans_mac_frozen",
        history=WARMUP,
        separates="inference_writes_of_the_fast_weights_at_equal_capacity",
    )
    # El banco y el Transformer en línea solo reciben etiquetas en el tramo medido.
    for name, pair in pairs.items():
        writes_labels = name.startswith(("episodic_bank", "write_policy", "online"))
        assert pair["history"] == (MEASURED if writes_labels else WARMUP)
    assert {pair["control"] for pair in pairs.values()} >= {
        "titans_mac_frozen",
        "titans_mac_disabled",
        "mars_titan_m0",
        "mars_titan_m1",
        "transformer_compact",
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda s: s.pop("placebo"), "exactamente sus campos"),
        (lambda s: s.update(status="primary"), "análisis secundario"),
        (lambda s: s.update(declared_at="10-10-2026"), "análisis secundario"),
        (lambda s: s.update(hypothesis="  "), "análisis secundario"),
        (lambda s: s.update(placebo="shuffled_routes"), "calendario de regímenes"),
        (lambda s: s.update(regimes="calendar_month_v1"), "calendario de regímenes"),
        (lambda s: s.update(entry_sessions=0), "fijados de antemano"),
        (lambda s: s.update(entry_sessions=True), "fijados de antemano"),
        (lambda s: s.update(absence_bins=[126, 63]), "fijados de antemano"),
        (lambda s: s.update(long_absence_sessions=50), "fijados de antemano"),
        (lambda s: s.update(placebo_lag_sessions=0), "fijados de antemano"),
        (lambda s: s.update(pairs={}), "entre 1 y 16 pares"),
        (
            lambda s: s["pairs"]["fast_weight_writes"].update(control="titans_mac_online"),
            "dos brazos distintos",
        ),
        (
            lambda s: s["pairs"]["fast_weight_writes"].update(control="missing"),
            "dos brazos distintos",
        ),
        (lambda s: s["pairs"]["fast_weight_writes"].update(control="zero"), "control cero"),
        (
            lambda s: s["pairs"].update(copy=dict(s["pairs"]["fast_weight_writes"])),
            "está repetido",
        ),
        (lambda s: s["pairs"]["fast_weight_writes"].pop("separates"), "lo que separa"),
        (lambda s: s["pairs"]["fast_weight_writes"].update(history="always"), "comienzo"),
    ],
)
def test_declarations_that_change_the_analysis_are_rejected(change, message):
    value = declared()
    change(value)
    with pytest.raises(ValueError, match=message):
        ri.declaration(value, arms())


def test_spells_classes_absences_and_positions_follow_the_definition():
    #          0  1  2  3  4  5  6  7  8  9  10
    routes = [1, 1, 1, 2, 2, 0, 2, 1, 1, 3, 1]
    kind, absence, position = ri.classify(routes, 2)
    assert kind.tolist() == [
        NOVEL,
        NOVEL,
        CONTINUING,
        NOVEL,
        NOVEL,
        -1,
        CONTINUING,
        REVISIT,
        REVISIT,
        NOVEL,
        REVISIT,
    ]
    # El tramo 1 vuelve tras tres sesiones clasificadas del régimen 2 y la sin clasificar
    # ni cuenta ni corta su tramo.
    assert absence.tolist() == [0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 1]
    assert position.tolist() == [1, 2, 3, 1, 2, 0, 3, 1, 2, 1, 1]


def test_the_classes_of_a_session_only_depend_on_earlier_routes():
    rng = np.random.default_rng(3)
    routes = rng.integers(0, 5, 300)
    kind, absence, position = ri.classify(routes, 5)
    for cut in (1, 50, 173, 299):
        changed = routes.copy()
        changed[cut:] = rng.integers(0, 5, len(routes) - cut)
        other = ri.classify(changed, 5)
        for left, right in zip((kind, absence, position), other, strict=True):
            assert np.array_equal(left[:cut], right[:cut])


def test_the_placebo_takes_the_route_of_an_earlier_session():
    routes = np.array([1, 2, 3, 4, 1, 2])
    assert ri.placebo_routes(routes, 2).tolist() == [0, 0, 1, 2, 3, 4]
    assert ri.placebo_routes(routes, 9).tolist() == [0] * 6


def test_the_warmup_starts_where_the_memory_phases_start():
    day = date(2000, 1, 1)
    for _ in range(400):
        for months in (0, 1, 12, 60):
            assert ri._first_of_month_before(day, months).isoformat() == months_before(
                day.isoformat(), months
            )
        day += timedelta(days=23)


def micros(day):
    return int(datetime.combine(day, datetime.min.time(), UTC).timestamp() * 1_000_000)


def calendar(routes, first=date(2023, 1, 1)):
    """Una sesión diaria a las 21 h UTC desde `first` con las rutas indicadas."""
    days = [first + timedelta(days=i) for i in range(len(routes))]
    at = np.array([micros(day) + 21 * 3_600_000_000 for day in days], dtype=np.int64)
    return at, np.asarray(routes, dtype=np.int64)


def test_each_window_restarts_the_history_at_its_warmup():
    # Enero entero es régimen 1. Febrero empieza en 2 y vuelve a 1 el día 10.
    routes = [1] * 31 + [2] * 9 + [1] * 19
    at, route = calendar(routes)
    times = at[31:]
    windows = [("2023-02-01", "2023-03-01")]
    kind, absence, _ = ri.session_classes(times, (at, route), windows, section(), WARMUP)
    # Con un mes de calentamiento enero forma parte del recorrido y el 1 es una revisita.
    assert kind[9] == REVISIT and absence[9] == 9
    # Una memoria que solo escribe en el tramo medido no guardó enero: el 1 es nuevo.
    kind, _, _ = ri.session_classes(times, (at, route), windows, section(), MEASURED)
    assert kind[9] == NOVEL
    # Con la evaluación a mitad de mes, la historia empieza el primer día del mes situado
    # dos meses antes, como las fases de la memoria: el 15 de febrero sigue un tramo de 1.
    late = [("2023-02-15", "2023-03-01")]
    kind, absence, position = ri.session_classes(
        at[45:], (at, route), late, section(warmup_months=2), WARMUP
    )
    assert (kind[0], absence[0], position[0]) == (CONTINUING, 0, 6)
    with pytest.raises(ValueError, match="comienzo"):
        ri.session_classes(times, (at, route), windows, section(), "always")


def test_sessions_outside_the_calendar_or_the_windows_stop_the_analysis():
    at, route = calendar([1] * 60)
    windows = [("2023-01-15", "2023-02-01")]
    with pytest.raises(ValueError, match="no está en el calendario"):
        ri.session_classes(at[20:22] + 1, (at, route), windows, section(), WARMUP)
    with pytest.raises(ValueError, match="ninguna ventana"):
        ri.session_classes(at[40:42], (at, route), windows, section(), WARMUP)
    overlapping = [("2023-01-15", "2023-02-01"), ("2023-01-20", "2023-02-10")]
    with pytest.raises(ValueError, match="se solapan"):
        ri.session_classes(at[20:22], (at, route), overlapping, section(), WARMUP)


def scripted_study(benefit_of, months=12, seed=5, start=MEASURED):
    """Un año de sesiones diarias en ventanas mensuales con rachas de longitud variable.

    Cada ventana reinicia la historia, así que cada mes tiene regímenes nuevos y revisitas.
    `benefit_of` recibe las clases reales, las del placebo y el índice de cada sesión.
    """
    rng = np.random.default_rng(seed)
    first = date(2023, 1, 1)
    edges = [date(2023 + (m // 12), m % 12 + 1, 1) for m in range(months + 1)]
    days = (edges[-1] - first).days
    routes = []
    while len(routes) < days:
        routes += [int(rng.integers(1, 5))] * int(rng.integers(2, 7))
    at, route = calendar(routes[:days], first)
    times = at
    windows = [
        (low.isoformat(), high.isoformat()) for low, high in zip(edges, edges[1:], strict=False)
    ]
    value = section(warmup_months=1)
    for pair in value["pairs"].values():
        pair["history"] = start
    classes = ri.session_classes(times, (at, route), windows, value, start)
    placebo = ri.session_classes(times, (at, route), windows, value, start, placebo=True)
    control = 1.0 + rng.uniform(0, 0.2, len(times))
    memory = control - benefit_of(classes, placebo, np.arange(len(times)))
    return dict(
        at=at,
        route=route,
        times=times,
        windows=windows,
        section=value,
        classes=classes,
        placebo=placebo,
        control=control,
        memory=memory,
    )


def run_report(study, *, pairs=None, options=OPTIONS, defined=None):
    series = dict(memory=study["memory"], control=study["control"], twin=study["control"].copy())
    period = np.arange(len(study["times"]))
    value = dict(study["section"], pairs=pairs or study["section"]["pairs"])
    defined = np.ones(len(period), bool) if defined is None else defined

    def series_of(arm, market):
        if arm not in series:
            return None
        return series[arm], defined, study["times"], period

    return ri.report(
        value, options, study["windows"], series_of, {"US": (study["at"], study["route"])}, ["US"]
    )


def by_class(classes, values):
    kind, absence, _ = classes
    return dict(
        novel=values[kind == NOVEL].mean(),
        revisit=values[kind == REVISIT].mean(),
        continuing=values[kind == CONTINUING].mean(),
        classified=values[kind >= 0].mean(),
        long=values[(kind == REVISIT) & (absence >= 4)].mean(),
        short=values[(kind == REVISIT) & (absence < 4)].mean(),
    )


def test_estimates_are_the_declared_differences_of_class_means():
    def benefit(classes, placebo, index):
        kind, absence, _ = classes
        revisit = kind == REVISIT
        return np.select([revisit & (absence < 4), revisit], [0.3, 0.15], 0.0)

    study = scripted_study(benefit)
    result = run_report(study)["markets"]["US"]
    pair = result["pairs"]["writes"]
    b = study["control"] - study["memory"]
    real, fake = by_class(study["classes"], b), by_class(study["placebo"], b)
    assert pair["estimates"] == pytest.approx(
        dict(
            predictive_effect=real["classified"],
            retention=real["revisit"] - real["novel"],
            interference=real["long"] - real["short"],
            plasticity=real["novel"] - real["continuing"],
        ),
        abs=1e-12,
    )
    decisions = pair["decisions"]
    assert decisions["retention"]["difference_from_placebo"] == pytest.approx(
        (real["revisit"] - real["novel"]) - (fake["revisit"] - fake["novel"]), abs=1e-12
    )
    assert decisions["interference"]["difference_from_placebo"] == pytest.approx(
        (real["long"] - real["short"]) - (fake["long"] - fake["short"]), abs=1e-12
    )
    kind = study["classes"][0]
    assert pair["sessions"]["revisit_entry"] == int(np.sum(kind == REVISIT))
    assert result["classes"] == {
        MEASURED: {name: int(np.sum(kind == index)) for index, name in enumerate(ri.CLASSES)}
        | dict(unclassified=0)
    }
    assert decisions["retention"]["status"] == "supported"
    assert decisions["retention"]["reason"] is None
    assert decisions["interference"]["status"] == "detected"
    # Curvas descriptivas: posición en el tramo y ausencia de las revisitas.
    revisit_first = (kind == REVISIT) & (study["classes"][2] == 1)
    assert pair["curves"]["recovery"]["revisit_entry"][0] == dict(
        sessions=int(revisit_first.sum()), mean=pytest.approx(b[revisit_first].mean())
    )
    assert [row["from_sessions"] for row in pair["curves"]["absence"]] == [1, 2, 4, 8]
    assert sum(row["sessions"] for row in pair["curves"]["absence"]) == int(np.sum(kind == REVISIT))


def test_sessions_without_a_defined_metric_leave_every_class_and_curve():
    def benefit(classes, placebo, index):
        kind, absence, _ = classes
        return np.where(kind == REVISIT, 0.2, 0.0)

    study = scripted_study(benefit)
    defined = np.ones(len(study["times"]), bool)
    defined[::7] = False
    study["memory"][~defined] = np.nan
    pair = run_report(study, defined=defined)["markets"]["US"]["pairs"]["writes"]
    b = study["control"] - study["memory"]
    kept = [values[defined] for values in study["classes"]]
    real = by_class(kept, b[defined])
    assert pair["estimates"]["retention"] == pytest.approx(real["revisit"] - real["novel"])
    assert pair["sessions"]["classified"] == int(defined.sum())
    curves = pair["curves"]
    assert sum(row["sessions"] for row in curves["absence"]) == int(np.sum(kept[0] == REVISIT))
    assert all(
        row["mean"] is None or np.isfinite(row["mean"])
        for rows in curves["recovery"].values()
        for row in rows
    )


def replicate_statistics(study, b):
    """Retención e interferencia de cada réplica, reales y del placebo, calculadas a mano."""
    rng = np.random.default_rng(OPTIONS["seed"])
    groups = []
    for kind, absence, _ in (study["classes"], study["placebo"]):
        revisit = kind == REVISIT
        groups.append(
            np.stack(
                [revisit, kind == NOVEL, revisit & (absence >= 4), revisit & (absence < 4)], axis=1
            ).astype(float)
        )
    draws = []
    for offset in range(0, OPTIONS["replicates"], 256):
        size = min(256, OPTIONS["replicates"] - offset)
        counts = circular_block_counts(rng, size, len(b), OPTIONS["block_length"]).astype(float)
        rows = []
        for masks in groups:
            means = (counts @ (b[:, None] * masks)) / (counts @ masks)
            rows += [means[:, 0] - means[:, 1], means[:, 2] - means[:, 3]]
        draws.append(np.stack(rows, axis=1))
    return np.concatenate(draws)


def test_the_intervals_use_the_comparison_blocks_seed_and_family():
    def benefit(classes, placebo, index):
        kind, absence, _ = classes
        return np.where(kind == REVISIT, 0.2 - 0.02 * absence, 0.0) + 0.05 * np.sin(index)

    study = scripted_study(benefit)
    pair = run_report(study)["markets"]["US"]["pairs"]["writes"]
    b = study["control"] - study["memory"]
    draws = replicate_statistics(study, b)
    # Columnas: retención e interferencia reales y del placebo.
    family = np.stack(
        [draws[:, 0], draws[:, 0] - draws[:, 2], draws[:, 1], draws[:, 1] - draws[:, 3]], axis=1
    )
    decisions = pair["decisions"]
    estimate = np.array(
        [
            pair["estimates"]["retention"],
            decisions["retention"]["difference_from_placebo"],
            pair["estimates"]["interference"],
            decisions["interference"]["difference_from_placebo"],
        ]
    )
    # El par idéntico entra en la misma familia con varianza nula y no cambia el crítico.
    joint = family_intervals(estimate, family, OPTIONS["confidence"])
    for offset, key in ((0, "retention"), (2, "interference")):
        intervals = decisions[key]["intervals"]
        assert intervals[key] == pytest.approx(
            [joint["joint_lower"][offset], joint["joint_upper"][offset]]
        )
        assert intervals["difference_from_placebo"] == pytest.approx(
            [joint["joint_lower"][offset + 1], joint["joint_upper"][offset + 1]]
        )
        assert decisions[key]["critical"] == pytest.approx(joint["critical"])
    marginal = family_intervals(estimate[:1], draws[:, :1], OPTIONS["confidence"])
    assert pair["intervals"]["retention"] == pytest.approx(
        [marginal["lower"][0], marginal["upper"][0]]
    )


def test_a_pair_with_identical_series_confirms_nothing():
    study = scripted_study(lambda classes, placebo, index: np.zeros(len(index)))
    pair = run_report(study)["markets"]["US"]["pairs"]["same"]
    assert set(pair["estimates"].values()) == {0.0}
    retention, interference = (pair["decisions"][key] for key in ("retention", "interference"))
    assert retention["intervals"]["retention"] == [0.0, 0.0]
    assert retention["status"] == "discarded"
    assert (
        retention["reason"]
        == "El intervalo de retention no excluye el cero en el sentido declarado"
    )
    assert interference["status"] == "not_detected"


def test_a_gain_that_grows_through_each_window_is_not_retention():
    # Una mejora que solo crece con los días de cada ventana favorece a las revisitas, que
    # caen más tarde que los regímenes nuevos. La retención sale positiva con su intervalo,
    # pero el placebo con las mismas rachas la reproduce y la hipótesis se descarta.
    def benefit(classes, placebo, index):
        return np.array([0.01 * (date(2023, 1, 1) + timedelta(days=int(i))).day for i in index])

    study = scripted_study(benefit)
    pair = run_report(study)["markets"]["US"]["pairs"]["writes"]
    retention = pair["decisions"]["retention"]
    assert retention["intervals"]["retention"][0] > 0
    assert retention["status"] == "discarded"
    assert retention["intervals"]["difference_from_placebo"][0] <= 0
    assert retention["reason"] == "retention no se separa de la del placebo"


def test_missing_classes_short_series_and_absent_arms_leave_the_pair_undetermined():
    study = scripted_study(lambda classes, placebo, index: np.zeros(len(index)))
    many = dict(study, section=section(warmup_months=1, min_sessions=10_000))
    for decision in run_report(many)["markets"]["US"]["pairs"]["writes"]["decisions"].values():
        assert decision["status"] == "undetermined" and "menos sesiones" in decision["reason"]
        assert decision["intervals"] is None
    short = run_report(study, options=dict(OPTIONS, block_length=len(study["times"])))
    pair = short["markets"]["US"]["pairs"]["writes"]
    assert pair["decisions"]["retention"]["reason"].startswith("Se necesitan más días")
    assert set(pair["intervals"].values()) == {None}
    absent = dict(writes=dict(memory="memory", control="nobody", history=MEASURED, separates="x"))
    result = run_report(study, pairs=absent)["markets"]["US"]["pairs"]["writes"]
    assert result == dict(status="not_in_scope", reason=ri._NOT_IN_SCOPE)


def test_a_memory_that_saw_every_regime_in_the_warmup_can_only_test_interference():
    # Con la historia desde el calentamiento de un mes casi no quedan regímenes nuevos.
    def benefit(classes, placebo, index):
        kind, absence, _ = classes
        revisit = kind == REVISIT
        return np.select([revisit & (absence < 8), revisit], [0.3, 0.1], 0.0)

    study = scripted_study(benefit, start=WARMUP)
    study["section"].update(min_sessions=20, long_absence_sessions=8)
    pair = run_report(study)["markets"]["US"]["pairs"]["writes"]
    assert pair["sessions"]["novel_entry"] < 20 <= pair["sessions"]["revisit_long"]
    assert pair["decisions"]["retention"]["status"] == "undetermined"
    assert pair["decisions"]["interference"]["status"] == "detected"


def test_arms_that_do_not_share_sessions_are_rejected():
    study = scripted_study(lambda classes, placebo, index: np.zeros(len(index)))
    value = study["section"]

    def series_of(arm, market):
        times = study["times"] if arm == "memory" else study["times"] + DAY
        return study[arm], np.ones(len(times), bool), times, np.arange(len(times))

    with pytest.raises(ValueError, match="mismas sesiones"):
        ri.report(
            value,
            OPTIONS,
            study["windows"],
            series_of,
            {"US": (study["at"], study["route"])},
            ["US"],
        )


def test_pending_keeps_the_declaration():
    value = declared()
    assert ri.pending(value) == dict(
        declaration=value,
        status="not_computed",
        reason="Falta el calendario de regímenes de la edición evaluada",
    )
