"""Sensibilidad al preentrenamiento posterior, con comparaciones escritas a mano.

Las dos comparaciones publicadas son tablas por sesión con MAE conocido: la diferencia entre la
edición de control y la congelada se fija en la prueba, así que cada estadístico se deduce de
la declaración. No hay modelos, ediciones reales ni pasos de optimizador.
"""

import copy
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from mars_titan.evaluation import encoder_sensitivity as es
from mars_titan.evaluation.paired_comparisons import circular_block_counts, family_intervals
from mars_titan.evaluation.walk_forward_comparison import REPORT_KIND

ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT / "configs/encoders/pretraining-free-sensitivity.json"
COMPARISONS = [
    ROOT / "configs/evaluation/historical-masked-2000-comparison.json",
    ROOT / "configs/evaluation/historical-masked-2000-joint-comparison.json",
]
SEEDS = (42, 43, 44)
ARMS = ("gru", "titans_mac_online")


def declared(**changes):
    value = copy.deepcopy(json.loads(DECLARATION.read_text()))
    value.update(changes)
    return value


def section(**changes):
    """Declaración con un bootstrap pequeño para que las pruebas sean rápidas."""
    return declared(
        bootstrap=dict(block_length=5, replicates=400, seed=3, confidence=0.95), **changes
    )


def sessions(market="US"):
    """Días laborables de 2014 a 2023 con la hora de cierre de cada mercado."""
    hour = 21 if market == "US" else 7
    day, days = date(2014, 1, 1), []
    while day < date(2024, 1, 1):
        if day.weekday() < 5:
            days.append(datetime(day.year, day.month, day.day, hour, tzinfo=UTC))
        day += timedelta(days=1)
    return days


def micros(moment):
    return int(moment.timestamp() * 1_000_000)


def write_comparison(folder, deltas, *, edition, market="US", edit=None):
    """Comparación publicada con el MAE de cada brazo, semilla y sesión.

    `deltas` asigna a cada brazo una función del instante que da Δ. La edición congelada usa un
    MAE base y la de control le suma Δ. Las semillas se desplazan alrededor de la base con
    desplazamientos que suman cero, así que la media de las semillas es la base.
    """
    moments = sessions(market)
    base = 1.0 + np.random.default_rng(11).uniform(0, 0.2, len(moments))
    rows = []
    for arm, delta in deltas.items():
        shift = np.array([delta(m) for m in moments]) if edition == "control" else 0.0 * base
        for seed, offset in zip(SEEDS, (-0.05, 0.0, 0.05), strict=True):
            for index, moment in enumerate(moments):
                window = f"fold-{moment.year - 2014:03d}"
                mae = float(base[index] + offset + shift[index])
                rows.append((arm, seed, window, "raw", market, moment, 20, mae, 11, 9))
    names = (
        "arm",
        "seed",
        "window",
        "quantiles",
        "market",
        "prediction_at",
        "samples",
        "mae",
        "positive_targets",
        "negative_targets",
    )
    types = dict(
        prediction_at=pa.timestamp("us", tz="UTC"),
        samples=pa.int64(),
        mae=pa.float64(),
        positive_targets=pa.int64(),
        negative_targets=pa.int64(),
    )
    table = pa.table(
        {name: pa.array([row[i] for row in rows], types.get(name)) for i, name in enumerate(names)}
    )
    if edit is not None:
        table = edit(table)
    folder.mkdir(parents=True)
    pq.write_table(table, folder / "sessions.parquet")
    report = dict(
        schema_version=1,
        kind=REPORT_KIND,
        status="completed",
        final_test_opened=False,
        scope=market,
        markets=[market],
        configuration=dict(name="fixture", sha256="c" * 64),
        edition={market: ("f" if edition == "frozen" else "a") * 64},
        windows={
            f"fold-{year - 2014:03d}": dict(evaluation=[f"{year}-01-01", f"{year + 1}-01-01"])
            for year in range(2014, 2024)
        },
        artifacts={"sessions.parquet": sha256(folder / "sessions.parquet")},
    )
    (folder / "comparison.json").write_text(json.dumps(report))
    return folder


def pair(tmp_path, deltas, **options):
    frozen = write_comparison(tmp_path / "frozen", deltas, edition="frozen", **options)
    control = write_comparison(tmp_path / "control", deltas, edition="control", **options)
    return frozen, control


def block_of(moment):
    edges = es.block_edges(section())
    value = micros(moment)
    for name, low, high in zip(("placebo", "before", "after"), edges, edges[1:], strict=False):
        if low <= value < high:
            return name
    return None


def step(before, after):
    """Δ constante antes del corte de julio de 2021 y otro después."""
    return lambda moment: before if moment < datetime(2021, 7, 1, tzinfo=UTC) else after


def test_the_declaration_is_valid_and_matches_both_comparisons():
    value = es.load_declaration(DECLARATION)
    for path in COMPARISONS:
        config = json.loads(path.read_text())
        assert set(value["arms"]) <= set(config["arms"])
        assert value["bootstrap"] == {
            key: config["comparison"][key]
            for key in ("block_length", "replicates", "seed", "confidence")
        }
    assert [str(np.datetime64(edge, "us"))[:10] for edge in es.block_edges(value)] == [
        "2016-07-01",
        "2019-01-01",
        "2021-07-01",
        "2024-01-01",
    ]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.update(extra=1), "exactamente sus campos"),
        (lambda d: d.update(status="executed"), "antes de ejecutar"),
        (lambda d: d.update(does_not_measure=[]), "antes de ejecutar"),
        (lambda d: d.update(metric="session_mse"), "salto en el corte"),
        (lambda d: d["statistics"].pop("placebo_break"), "salto en el corte"),
        (lambda d: d.update(cutoff="2021-07-15"), "empezar un mes"),
        (lambda d: d.update(block_months=0), "empezar un mes"),
        (lambda d: d.update(min_sessions=0), "empezar un mes"),
        (lambda d: d.update(cutoff="2021-06-01"), "publicación verificada"),
        (lambda d: d.update(arms=[]), "sin repetir"),
        (lambda d: d.update(arms=["gru", "gru"]), "sin repetir"),
        (lambda d: d.update(scopes=["EU"]), "sin repetir"),
        (lambda d: d["bootstrap"].update(replicates=50), "bootstrap"),
        (lambda d: d["bootstrap"].update(confidence=1.0), "bootstrap"),
    ],
)
def test_declarations_that_change_the_analysis_are_rejected(change, message):
    value = declared()
    change(value)
    with pytest.raises(ValueError, match=message):
        es.declaration(value)


def test_estimates_are_the_declared_block_differences(tmp_path):
    deltas = {"gru": step(0.3, 0.0), "titans_mac_online": step(0.1, 0.1)}
    frozen, control = pair(tmp_path, deltas)
    report = es.evaluate(section(), frozen, control)
    assert report["scope"] == "US" and report["final_test_opened"] is False
    moments = sessions()
    for arm, delta in deltas.items():
        values = np.array([delta(m) for m in moments])
        blocks = np.array([block_of(m) for m in moments])
        mean = {name: values[blocks == name].mean() for name in ("placebo", "before", "after")}
        record = report["markets"]["US"][arm]
        assert record["estimates"] == pytest.approx(
            dict(
                effect=values.mean(),
                **{"break": mean["before"] - mean["after"]},
                placebo_break=mean["placebo"] - mean["before"],
            ),
            abs=1e-12,
        )
        assert record["sessions"] == dict(
            all=len(moments), **{name: int(np.sum(blocks == name)) for name in es._BLOCKS}
        )
    gru = report["markets"]["US"]["gru"]["decision"]
    assert gru["status"] == "anticipation_suspected" and gru["reason"] is None
    flat = report["markets"]["US"]["titans_mac_online"]["decision"]
    assert flat["status"] == "not_supported"
    assert flat["reason"] == "El intervalo del salto en el corte no queda por encima de cero"


def test_a_gain_that_decays_with_time_is_not_anticipation(tmp_path):
    def decay(moment):
        return 0.4 - 0.04 * (moment.year - 2014 + moment.timetuple().tm_yday / 366)

    frozen, control = pair(tmp_path, {"gru": decay, "titans_mac_online": decay})
    record = es.evaluate(section(), frozen, control)["markets"]["US"]["gru"]
    assert record["estimates"]["break"] > 0
    assert record["decision"]["intervals"]["break"][0] > 0
    assert record["decision"]["status"] == "not_supported"
    assert record["decision"]["reason"] == "El salto en el corte no se separa del salto placebo"


def test_intervals_follow_the_declared_bootstrap(tmp_path):
    rng = np.random.default_rng(5)
    noise = {m: rng.normal(0, 0.05) for m in sessions()}
    deltas = {
        "gru": lambda m: step(0.2, 0.0)(m) + noise[m],
        "titans_mac_online": lambda m: noise[m],
    }
    frozen, control = pair(tmp_path, deltas)
    value = section()
    report = es.evaluate(value, frozen, control)["markets"]["US"]
    moments = sessions()
    times = np.array([micros(m) for m in moments])
    days = np.unique(times // 86_400_000_000, return_inverse=True)[1]
    edges = es.block_edges(value)
    masks = [np.ones(len(times), bool)] + [
        (times >= low) & (times < high) for low, high in zip(edges, edges[1:], strict=False)
    ]
    options = value["bootstrap"]
    counts = circular_block_counts(
        np.random.default_rng(options["seed"]),
        options["replicates"],
        len(moments),
        options["block_length"],
    ).astype(float)
    estimates, columns = [], []
    for arm in ARMS:
        delta = np.array([deltas[arm](m) for m in moments])
        means = np.stack([counts @ (delta * mask) / (counts @ mask) for mask in masks], axis=1)
        whole = [delta[mask].mean() for mask in masks]
        jump = means[:, 2] - means[:, 3]
        placebo = means[:, 1] - means[:, 2]
        estimates += [whole[2] - whole[3], (whole[2] - whole[3]) - (whole[1] - whole[2])]
        columns += [jump, jump - placebo]
        marginal = family_intervals(np.array([whole[2] - whole[3]]), jump[:, None], 0.95)
        assert report[arm]["intervals"]["break"] == pytest.approx(
            [marginal["lower"][0], marginal["upper"][0]]
        )
    joint = family_intervals(np.array(estimates), np.stack(columns, axis=1), 0.95)
    for position, arm in enumerate(ARMS):
        decision = report[arm]["decision"]
        assert decision["intervals"]["break"] == pytest.approx(
            [joint["joint_lower"][2 * position], joint["joint_upper"][2 * position]]
        )
        assert decision["intervals"]["break_minus_placebo_break"] == pytest.approx(
            [joint["joint_lower"][2 * position + 1], joint["joint_upper"][2 * position + 1]]
        )
        assert decision["critical"] == pytest.approx(joint["critical"])
    assert days.max() + 1 == len(moments)


def test_short_blocks_or_series_leave_the_decision_undetermined(tmp_path):
    frozen, control = pair(tmp_path, {arm: step(0.3, 0.0) for arm in ARMS})
    many = es.evaluate(section(min_sessions=10_000), frozen, control)["markets"]["US"]
    assert many["gru"]["decision"]["status"] == "undetermined"
    assert many["gru"]["decision"]["reason"] == "Algún tramo tiene menos sesiones de las declaradas"
    value = section()
    value["bootstrap"]["block_length"] = 10_000
    short = es.evaluate(value, frozen, control)["markets"]["US"]["gru"]
    assert short["decision"]["reason"] == "Se necesitan más días que la longitud del bloque"
    assert set(short["intervals"].values()) == {None}


def test_a_series_that_ends_before_the_cutoff_cannot_decide(tmp_path):
    def before(table):
        moments = table["prediction_at"].cast("int64").to_numpy()
        return table.filter(pa.array(moments < micros(datetime(2021, 1, 1, tzinfo=UTC))))

    frozen, control = pair(tmp_path, {arm: step(0.3, 0.0) for arm in ARMS}, edit=before)
    record = es.evaluate(section(), frozen, control)["markets"]["US"]["gru"]
    assert record["sessions"]["after"] == 0 and record["estimates"]["break"] is None
    assert record["decision"]["status"] == "undetermined"


def rewrite(folder, change):
    report = json.loads((folder / "comparison.json").read_text())
    change(report)
    (folder / "comparison.json").write_text(json.dumps(report))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r.update(final_test_opened=True), "test final"),
        (lambda r: r.update(status="running"), "no está completa"),
        (lambda r: r.update(scope="CN"), "mismo ámbito"),
        (lambda r: r["configuration"].update(sha256="d" * 64), "misma configuración"),
        (
            lambda r: r["windows"]["fold-000"].update(evaluation=["2014-02-01", "2015-01-01"]),
            "ventanas",
        ),
        (lambda r: r.update(edition={"US": "f" * 64}), "misma edición"),
        (lambda r: r["artifacts"].update({"sessions.parquet": "0" * 64}), "no coincide"),
    ],
)
def test_comparisons_that_do_not_pair_are_rejected(tmp_path, change, message):
    frozen, control = pair(tmp_path, {arm: step(0.3, 0.0) for arm in ARMS})
    rewrite(control, change)
    with pytest.raises(ValueError, match=message):
        es.evaluate(section(), frozen, control)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda t: t.set_column(
                t.schema.get_field_index("samples"), "samples", pa.array(np.arange(t.num_rows))
            ),
            "mismas filas",
        ),
        (lambda t: t.slice(1), "mismas filas"),
        (lambda t: t.filter(pc.not_equal(t["arm"], "titans_mac_online")), "no está"),
    ],
)
def test_rows_that_differ_between_editions_are_rejected(tmp_path, edit, message):
    frozen = write_comparison(
        tmp_path / "frozen", {arm: step(0.3, 0.0) for arm in ARMS}, edition="frozen"
    )
    control = write_comparison(
        tmp_path / "control", {arm: step(0.3, 0.0) for arm in ARMS}, edition="control", edit=edit
    )
    with pytest.raises(ValueError, match=message):
        es.evaluate(section(), frozen, control)


def test_a_session_without_every_seed_is_rejected(tmp_path):
    def drop(table):
        keep = np.ones(table.num_rows, bool)
        keep[0] = False
        return table.filter(pa.array(keep))

    deltas = {arm: step(0.3, 0.0) for arm in ARMS}
    frozen = write_comparison(tmp_path / "frozen", deltas, edition="frozen", edit=drop)
    control = write_comparison(tmp_path / "control", deltas, edition="control", edit=drop)
    with pytest.raises(ValueError, match="todas sus semillas"):
        es.evaluate(section(), frozen, control)


def test_the_command_line_writes_a_new_report(tmp_path, capsys):
    frozen, control = pair(tmp_path, {arm: step(0.3, 0.0) for arm in ARMS})
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(section()))
    output = tmp_path / "out" / "sensitivity.json"
    arguments = ["--declaration", str(path), "--frozen", str(frozen), "--control", str(control)]
    assert es.main([*arguments, "--output", str(output)]) == 0
    written = json.loads(output.read_text())
    assert written["kind"] == es.REPORT_KIND
    assert written["comparisons"]["frozen"]["sha256"] == sha256(frozen / "comparison.json")
    assert "sensitivity.json" in capsys.readouterr().out
    with pytest.raises(ValueError, match="archivo nuevo"):
        es.main([*arguments, "--output", str(output)])
