"""Referencia tabular de la campaña: formato de campaña, aprendizaje, determinismo e informe."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("lightgbm")
ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tabular, report = load("dgx_tabular_reference"), load("dgx_screening_report")
ASSETS, SESSIONS = 48, 520
STAMP = pa.timestamp("us", tz="UTC")


def prices_file(path, days, opened, close, volume):
    path.parent.mkdir(parents=True, exist_ok=True)
    instants = (days.astype("datetime64[us]") + np.timedelta64(20, "h")).astype(np.int64)
    table = pa.table(
        dict(
            open=opened,
            high=np.maximum(opened, close) * 1.004,
            low=np.minimum(opened, close) * 0.996,
            close=close,
            volume=volume,
            available_at=pa.array(instants, type=pa.int64()).cast(STAMP),
        )
    )
    pq.write_table(table, path)
    return instants


@pytest.fixture(scope="module")
def state(tmp_path_factory):
    """Cohorte sintética: la etiqueta de cada sesión depende de la media intradía reciente."""
    root = tmp_path_factory.mktemp("tabular")
    random = np.random.default_rng(5)
    days = np.arange("2011-01-03", "2014-01-01", dtype="datetime64[D]")
    days = days[np.is_busday(days)][:SESSIONS]
    market = random.normal(0.0002, 0.008, SESSIONS)
    factor = root / "factor.parquet"
    prices_file(factor, days, np.full(SESSIONS, 100.0), 100 * (1 + market), np.full(SESSIONS, 1e7))
    assets = []
    for index in range(ASSETS):
        symbol = f"A{index:02d}"
        drift = np.repeat(random.normal(0, 0.004, SESSIONS // 20), 20)[:SESSIONS]
        intraday = drift + 1.1 * market + random.normal(0, 0.01, SESSIONS)
        close = 30 * np.exp(np.cumsum(random.normal(0, 0.015, SESSIONS)))
        instants = prices_file(
            root / "prepared/US" / symbol / "prices.parquet",
            days,
            close / (1 + intraday),
            close,
            random.uniform(1e5, 1e6, SESSIONS),
        )
        # La etiqueta de la sesión t es la deriva intradía de t+1 con ruido: se anticipa con
        # las medias intradía de las sesiones anteriores.
        target = np.r_[drift[1:], 0.0] + random.normal(0, 0.004, SESSIONS)
        partition = np.array([None] * SESSIONS, dtype=object)
        partition[260:400], partition[400:440] = "train", "validation"
        partition[440:470], partition[470:519] = "calibration", "evaluation"
        labels = root / "labels/US" / symbol / "labels.parquet"
        labels.parent.mkdir(parents=True)
        pq.write_table(
            pa.table(
                dict(
                    prediction_at=pa.array(instants, type=pa.int64()).cast(STAMP),
                    target=target,
                    partition=pa.array(partition.tolist(), type=pa.string()),
                )
            ),
            labels,
        )
        assets.append(dict(market="US", symbol=symbol))
    manifest = dict(
        roots=dict(prepared=str(root / "prepared"), labels=str(root / "labels"), samples=""),
        assets=assets,
        input_extensions=dict(technical="long_horizon_v2"),
        final_test_opened=False,
        market_factors=dict(US=dict(prices_path=str(factor))),
    )
    folder = root / "estado/cohorts/2013"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return root / "estado"


def build(state, output, **options):
    settings = dict(
        name="lightgbm_technical", seed=42, trees=[20, 60], threads=2, alpha_free=False, min_leaf=20
    )
    settings.update(options)
    return tabular.build(state, 2013, output, **settings)


def test_the_reference_learns_and_writes_the_predictions_of_a_campaign_model(state, tmp_path):
    run, receipt = build(state, tmp_path / "referencia")
    assert run == tmp_path / "referencia/runs/lightgbm_technical/2013/seed-42"
    assert receipt["rows"] == dict(
        train=ASSETS * 140, validation=ASSETS * 40, calibration=ASSETS * 30, evaluation=ASSETS * 49
    )
    assert receipt["technical"] == "long_horizon_v2" and receipt["final_test_access"] is False
    assert len(receipt["features"]) == 47 and receipt["features"][-1] == "market"
    assert receipt["trees"] in (20, 60) and set(receipt["validation_ic_by_trees"]) == {"20", "60"}
    # La señal sembrada se recupera fuera del tramo de ajuste.
    assert min(receipt["ic"].values()) > 0.15
    table = pq.read_table(run / "evaluation-predictions.parquet")
    assert table.schema.names == [
        "sample_id",
        "asset_id",
        "market",
        "prediction_at",
        "target",
        "prediction",
    ]
    assert table.num_rows == ASSETS * 49 and table.schema.field("prediction_at").type == STAMP
    first = table.slice(0, 1).to_pylist()[0]
    instant = int(table.column("prediction_at").cast("int64")[0].as_py())
    assert first["sample_id"] == f"{first['asset_id']}/{instant}" and first["market"] == "US"
    # El informe de la criba lo lee como un modelo terminado de otro estado.
    records = report.completed(tmp_path / "referencia", 2013, 42)
    assert list(records) == ["lightgbm_technical"] and records["lightgbm_technical"]["tier"]
    model = report.model_report(tmp_path / "referencia", records["lightgbm_technical"], 2013, 42)
    assert model["ic"]["evaluation"]["mean"] == pytest.approx(receipt["ic"]["evaluation"], abs=1e-9)
    assert model["ic"]["evaluation"]["sessions"] == 49 and model["epochs"] == []


def test_the_reference_is_deterministic_and_can_target_the_label_without_alpha(state, tmp_path):
    one, _ = build(state, tmp_path / "uno")
    two, _ = build(state, tmp_path / "dos")
    for split in tabular.SPLITS:
        left = pq.read_table(one / f"{split}-predictions.parquet")
        assert left.equals(pq.read_table(two / f"{split}-predictions.parquet"))
    run, receipt = build(state, tmp_path / "libre", name="lightgbm_technical_free", alpha_free=True)
    assert (
        receipt["target"] == "alpha_free_return_percent"
        and run.parts[-3] == "lightgbm_technical_free"
    )
    free = pq.read_table(run / "evaluation-predictions.parquet")
    label = pq.read_table(one / "evaluation-predictions.parquet")
    # Mismas filas y misma etiqueta: solo cambia la predicción, que resta el alfa conocido.
    assert free.column("sample_id").equals(label.column("sample_id"))
    assert free.column("target").equals(label.column("target"))
    assert not free.column("prediction").equals(label.column("prediction"))
    assert np.isfinite(free.column("prediction").to_numpy()).all()


def test_the_reference_needs_the_declared_features_and_a_closed_final_test(state, tmp_path):
    path = state / "cohorts/2013/manifest.json"
    manifest = json.loads(path.read_text())
    other = tmp_path / "estado/cohorts/2013"
    other.mkdir(parents=True)
    (other / "manifest.json").write_text(json.dumps(dict(manifest, input_extensions=None)))
    with pytest.raises(ValueError, match="rasgos técnicos"):
        build(tmp_path / "estado", tmp_path / "salida")
    (other / "manifest.json").write_text(json.dumps(dict(manifest, final_test_opened=True)))
    with pytest.raises(ValueError, match="test final"):
        build(tmp_path / "estado", tmp_path / "salida")
    assert not (tmp_path / "salida").exists()
    with pytest.raises(SystemExit):
        tabular.main(["--state", str(state), "--year", "2013", "--output", "x", "--threads", "0"])


def test_alpha_free_fails_closed_when_market_model_missing(state, tmp_path):
    """Bajo --target alpha_free, un mercado sin factor (has_market_model=0) lanza ValueError."""
    manifest = json.loads((state / "cohorts/2013/manifest.json").read_text())
    # Añadimos un activo CN sin factor de mercado declarado -> has_market_model == 0 para CN.
    cn_root = tmp_path / "no_factor"
    prepared = Path(manifest["roots"]["prepared"])
    labels = Path(manifest["roots"]["labels"])
    cn_prices = cn_root / "prepared/CN/C00/prices.parquet"
    cn_labels = cn_root / "labels/CN/C00/labels.parquet"
    cn_prices.parent.mkdir(parents=True)
    cn_labels.parent.mkdir(parents=True)
    pq.write_table(pq.read_table(prepared / "US/A00/prices.parquet"), cn_prices)
    pq.write_table(pq.read_table(labels / "US/A00/labels.parquet"), cn_labels)

    # Creamos un estado con US (con factor) y CN (sin factor).
    merged_prepared = cn_root / "prepared"
    merged_labels = cn_root / "labels"
    for sym in [a["symbol"] for a in manifest["assets"][:4]]:
        dst_p = merged_prepared / "US" / sym / "prices.parquet"
        dst_l = merged_labels / "US" / sym / "labels.parquet"
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        dst_l.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pq.read_table(prepared / "US" / sym / "prices.parquet"), dst_p)
        pq.write_table(pq.read_table(labels / "US" / sym / "labels.parquet"), dst_l)

    joint_manifest = dict(
        manifest,
        roots=dict(prepared=str(merged_prepared), labels=str(merged_labels), samples=""),
        assets=[*manifest["assets"][:4], dict(market="CN", symbol="C00")],
        market_factors=dict(US=manifest["market_factors"]["US"]),
    )
    cohort_dir = cn_root / "estado/cohorts/2013"
    cohort_dir.mkdir(parents=True)
    (cohort_dir / "manifest.json").write_text(json.dumps(joint_manifest))

    with pytest.raises(ValueError, match="alpha_free.*CN|has_market_model"):
        build(cn_root / "estado", tmp_path / "out_fail", alpha_free=True)


def test_a_market_factor_that_is_an_index_without_volume_is_read(state, tmp_path):
    # El factor de China es un índice: su fichero trae apertura, máximo, mínimo y cierre, sin
    # volumen. Los informes y la referencia solo necesitan la apertura y el cierre.
    days = np.arange("2011-01-03", "2011-03-01", dtype="datetime64[D]")
    days = days[np.is_busday(days)]
    instants = (days.astype("datetime64[us]") + np.timedelta64(7, "h")).astype(np.int64)
    opened = np.linspace(3000.0, 3100.0, len(days))
    close = opened * 1.01
    index = tmp_path / "indice.parquet"
    pq.write_table(
        pa.table(
            dict(
                session=[str(day) for day in days],
                open=opened,
                high=close * 1.001,
                low=opened * 0.999,
                close=close,
                available_at=pa.array(instants, type=pa.int64()).cast(STAMP),
                source_sha256=["0" * 64] * len(days),
            )
        ),
        index,
    )
    source = json.loads((state / "cohorts/2013/manifest.json").read_text())
    factors = dict(source["market_factors"], CN=dict(prices_path=str(index)))
    calendar, returns = tabular.market_factor(dict(source, market_factors=factors), "CN")
    assert calendar.tolist() == days.tolist()
    np.testing.assert_allclose(returns, 0.01, rtol=1e-12)
    # Un estado de EE. UU. cuya vista declara también ese factor no deja de poder medirse.
    joint = tmp_path / "estado/cohorts/2013"
    joint.mkdir(parents=True)
    (joint / "manifest.json").write_text(json.dumps(dict(source, market_factors=factors)))
    offset = report.KnownOffset(tmp_path / "estado", 2013)
    assert sorted(offset.factors) == ["CN", "US"]
    np.testing.assert_allclose(offset.factors["CN"][1], 0.01, rtol=1e-12)
    plain = report.KnownOffset(state, 2013)
    moment = np.array(["2013-06-03T20:00:00"], dtype="datetime64[us]")
    one, other = offset.known("US/A00", moment), plain.known("US/A00", moment)
    assert np.array_equal(one[0], other[0], equal_nan=True)
