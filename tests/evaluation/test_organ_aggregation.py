"""S1 y S2 de un origen: los órganos de memoria como componente del agregador causal."""

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.evaluation import online_aggregator as oa
from mars_titan.evaluation import organ_aggregation as og
from mars_titan.evaluation import posttraining_pilots as pilots

ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT / "configs" / "experiments" / "posttraining-aggregator-s2.json"
YEAR = 2018
DAY = 86_400_000_000
START = 1_514_851_200_000_000  # 2 de enero de 2018, 00:00 UTC
SESSIONS = dict(validation=10, calibration=40, evaluation=60)
ASSETS = 35
VIEW = "f" * 64


def declaration():
    return json.loads(DECLARATION.read_text(encoding="utf-8"))


def _write(folder, split, rows, prediction):
    folder.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        dict(
            sample_id=pa.array(rows["sample_id"]),
            asset_id=pa.array(rows["asset_id"]),
            market=pa.array([a.split("/", 1)[0] for a in rows["asset_id"]]),
            prediction_at=pa.array(rows["prediction_at"], pa.timestamp("us", "UTC")),
            target=pa.array(rows["target"], pa.float32()),
            prediction=pa.array(prediction, pa.float32()),
        )
    )
    pq.write_table(table, folder / f"{split}-predictions.parquet")


def campaign(tmp_path, *, tree=True, markets=("US",)):
    """Padre, árboles y brazo combinado sintéticos: solo la señal del órgano ve el tercer factor."""
    rng = np.random.default_rng(5)
    state, reference, organs = tmp_path / "estado", tmp_path / "referencia", tmp_path / "organos"
    parent = state / "runs" / "titans_mac_nogain" / str(YEAR) / "seed-42"
    trees = reference / "runs" / "lightgbm_technical" / str(YEAR) / "seed-42"
    combined = organs / "runs" / "titans_mac_nogain__combined" / str(YEAR) / "seed-42"
    known = dict(alpha={}, matures={})
    day = 0
    for split, count in SESSIONS.items():
        rows = {name: [] for name in ("sample_id", "asset_id", "prediction_at", "target")}
        own, other, organ, alpha = [], [], [], []
        for session in range(count):
            instant = START + (day + session) * DAY
            for market in markets:
                prefix = "" if market == "US" else market
                z1, z2, z3, noise, n1, n2, n3 = rng.standard_normal((7, ASSETS))
                rows["sample_id"] += [
                    f"{prefix}{split[0]}{session:03d}-{asset:02d}" for asset in range(ASSETS)
                ]
                rows["asset_id"] += [f"{market}/A{asset:02d}" for asset in range(ASSETS)]
                rows["prediction_at"] += [instant] * ASSETS
                # En China los árboles ven mucho más que el padre: una decisión que mezclara los
                # mercados elegiría los árboles.
                first, second = (0.3, 0.2) if market == "US" else (0.05, 0.6)
                rows["target"] += list(first * z1 + second * z2 + 0.3 * z3 + noise)
                own += list(z1 + 0.7 * n1)
                other += list(z2 + 0.7 * n2)
                organ += list(z1 + 0.7 * n1 + 0.05 * (z3 + 0.5 * n3))
                alpha += list(0.002 * rng.standard_normal(ASSETS))
        day += count + 2
        _write(parent, split, rows, own)
        if tree:
            _write(trees, split, rows, other)
        if split != "validation":
            _write(combined, split, rows, organ)
        known["alpha"][split] = dict(zip(rows["sample_id"], alpha, strict=True))
        known["matures"][split] = dict(
            zip(
                rows["sample_id"],
                np.asarray(rows["prediction_at"], dtype=np.int64) + DAY,
                strict=True,
            )
        )
    (parent / "run.json").write_text(
        json.dumps(
            dict(
                status="completed",
                final_test_opened=False,
                run_id="padre-1",
                request=dict(view_sha256=VIEW),
            )
        )
    )
    if tree:
        (trees / "run.json").write_text(
            json.dumps(dict(model="lightgbm_technical", seed=42, final_test_access=False))
        )
    emissions = {s: pilots._sha256(parent / f"{s}-predictions.parquet") for s in SESSIONS}
    outputs = {s: pilots._sha256(combined / f"{s}-predictions.parquet") for s in og.AGGREGATED}
    (combined / "run.json").write_text(
        json.dumps(
            dict(
                status="completed",
                final_test_opened=False,
                model="titans_mac_nogain__combined",
                origin=YEAR,
                seed=42,
                parent=dict(model="titans_mac_nogain", state=str(state), run_id="padre-1"),
                arm="combined",
                inputs=dict(emissions=emissions),
                outputs=outputs,
            )
        )
    )
    manifest = state / "cohorts" / str(YEAR) / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    return dict(state=state, reference=reference, organs=organs, known=known, organ=combined)


def _arguments(paths, output, *extra):
    return [
        "agregar",
        "--declaration",
        str(DECLARATION),
        "--state",
        str(paths["state"]),
        "--reference",
        str(paths["reference"]),
        "--organs",
        str(paths["organs"]),
        "--origin",
        str(YEAR),
        "--output",
        str(output),
        *extra,
    ]


def _known(paths, seen=None):
    def known_rows(state, origin, parent, view, tables, markets=None):
        if seen is not None:
            seen.append(view)

        def column(name, split):
            values = paths["known"][name][split]
            return np.asarray([values[i] for i in tables[split].column("sample_id").to_pylist()])

        return (
            {s: column("alpha", s) for s in tables},
            {s: column("matures", s) for s in tables},
        )

    return known_rows


def test_the_declaration_is_the_registered_one_and_its_grid_is_the_code_grid():
    document, digest = og.checked_declaration(declaration())
    assert digest == og.REGISTERED_SHA256
    assert tuple(document["aggregator"]["lambdas"]) == oa.RIDGE_LAMBDAS
    assert tuple(document["aggregator"]["half_lives"]) == oa.RIDGE_HALF_LIVES
    assert document["operable_scales"][0] == 0.001 and document["operable_scales"][-1] == 100.0


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(final_test_access=True),
        lambda d: d["placebo"].update(seed=1),
        lambda d: d["aggregator"].update(lambdas=[1.0]),
        lambda d: d.update(operable_scales=[0.01, 3.0]),
        lambda d: d.update(extra=True),
    ],
    ids=["test-final", "semilla-del-placebo", "rejilla", "factores", "clave-de-mas"],
)
def test_any_change_of_the_registered_configuration_is_refused(change):
    document = declaration()
    change(document)
    with pytest.raises(ValueError, match="no es la registrada"):
        og.checked_declaration(document)


def test_a_code_grid_that_is_not_the_declared_one_is_refused(monkeypatch):
    monkeypatch.setattr(oa, "RIDGE_LAMBDAS", (1.0,))
    with pytest.raises(ValueError, match="rejilla"):
        og.checked_declaration(declaration())


def test_without_the_trees_it_exits_with_code_75_and_says_what_is_missing(tmp_path, capsys):
    paths = campaign(tmp_path, tree=False)
    output = tmp_path / "agregador"
    assert og.main(_arguments(paths, output, "--wait-hours", "0")) == og.MISSING_EXIT
    error = capsys.readouterr().err
    assert "Faltan entradas" in error and "lightgbm_technical" in error
    assert not output.exists() or not any(output.iterdir())


def test_it_waits_for_the_inputs_that_another_unit_is_still_writing(tmp_path):
    paths = campaign(tmp_path, tree=False)
    late = campaign(tmp_path / "tarde")
    naps = []

    def nap(seconds):
        naps.append(seconds)
        shutil.copytree(late["reference"], paths["reference"])

    sources = og.folders(declaration(), paths["state"], paths["reference"], paths["organs"], YEAR)
    manifest = paths["state"] / "cohorts" / str(YEAR) / "manifest.json"
    found = og.wait_for_inputs(
        declaration(), sources, manifest, hours=1, poll=7, sleep=nap, say=lambda *_: None
    )
    assert found == [] and naps == [7]


def test_an_organ_that_has_not_finished_is_still_missing(tmp_path):
    paths = campaign(tmp_path)
    record = paths["organ"] / "run.json"
    record.write_text(json.dumps(dict(json.loads(record.read_text()), status="running")))
    sources = og.folders(declaration(), paths["state"], paths["reference"], paths["organs"], YEAR)
    manifest = paths["state"] / "cohorts" / str(YEAR) / "manifest.json"
    missing = og.missing_inputs(declaration(), sources, manifest)
    assert len(missing) == 1 and "no ha terminado" in missing[0]


def test_the_placebo_permutes_the_signal_among_the_assets_of_each_session():
    market = np.array(["US"] * 10)
    instant = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    values = np.arange(10, dtype=np.float64)
    first = og.shuffled_within_sessions(values, market, instant, 20261122)
    assert np.array_equal(first, og.shuffled_within_sessions(values, market, instant, 20261122))
    assert not np.array_equal(first, values)
    for start, end in ((0, 3), (3, 6), (6, 10)):
        assert sorted(first[start:end]) == list(values[start:end])


def test_s1_and_s2_are_measured_in_the_label_and_in_the_operable_return(
    tmp_path, monkeypatch, capsys
):
    paths = campaign(tmp_path)
    seen = []
    monkeypatch.setattr(og, "known_rows", _known(paths, seen))
    output = tmp_path / "agregador"
    assert og.main(_arguments(paths, output)) == 0
    assert seen == [VIEW]
    folder = output / str(YEAR)
    summary = json.loads((folder / "resumen.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["declaration_sha256"] == og.REGISTERED_SHA256
    assert set(summary["modes"]) == {"label", "operable"}
    for mode in ("label", "operable"):
        result = summary["modes"][mode]
        assert result["s1"]["best"] in ("titans_mac_nogain", "lightgbm_technical")
        assert result["s2"]["vs_s1"]["sessions"] == SESSIONS["evaluation"]
        assert result["s2"]["vs_s1"]["mean"] > 0 and result["s2"]["vs_placebo"]["mean"] > 0
        assert result["s2"]["weights"][og.SIGNAL] > result["s2"]["placebo_weights"][og.PLACEBO]
    assert set(summary["modes"]["operable"]["factors"]) == set(og.CONFIGS)
    assert summary["modes"]["label"]["factors"] is None
    table = pq.read_table(folder / "sesiones.parquet")
    assert table.num_rows == 2 * SESSIONS["evaluation"]
    assert {"mode", "s1", "s2", "s2_placebo", "titans_mac_nogain"} <= set(table.column_names)
    assert (folder / "resumen.md").is_file()
    assert [p.name for p in output.iterdir()] == [str(YEAR)]
    before = (folder / "resumen.json").stat().st_mtime_ns
    capsys.readouterr()
    assert og.main(_arguments(paths, output)) == 0
    assert (folder / "resumen.json").stat().st_mtime_ns == before
    assert "ya calculado" in capsys.readouterr().out
    trees = paths["reference"] / "runs" / "lightgbm_technical" / str(YEAR) / "seed-42" / "run.json"
    trees.write_text(json.dumps(dict(json.loads(trees.read_text()), rows=1)))
    assert og.main(_arguments(paths, output)) == 1
    assert "Las entradas cambiaron" in capsys.readouterr().err


def test_an_organ_emitted_from_other_parent_predictions_is_refused(tmp_path, monkeypatch, capsys):
    paths = campaign(tmp_path)
    monkeypatch.setattr(og, "known_rows", _known(paths))
    record = paths["organ"] / "run.json"
    data = json.loads(record.read_text())
    data["inputs"]["emissions"]["evaluation"] = "0" * 64
    record.write_text(json.dumps(data))
    assert og.main(_arguments(paths, tmp_path / "agregador")) == 1
    assert "otras predicciones del padre" in capsys.readouterr().err


def test_a_root_written_with_another_declaration_is_refused(tmp_path, monkeypatch, capsys):
    paths = campaign(tmp_path)
    monkeypatch.setattr(og, "known_rows", _known(paths))
    folder = tmp_path / "agregador" / str(YEAR)
    folder.mkdir(parents=True)
    (folder / "resumen.json").write_text(
        json.dumps(dict(status="completed", declaration_sha256="0" * 64))
    )
    assert og.main(_arguments(paths, tmp_path / "agregador")) == 1
    assert "otra declaración" in capsys.readouterr().err


def test_an_organ_of_another_fit_of_the_parent_is_refused(tmp_path, monkeypatch, capsys):
    paths = campaign(tmp_path)
    monkeypatch.setattr(og, "known_rows", _known(paths))
    record = paths["organ"] / "run.json"
    data = json.loads(record.read_text())
    data["parent"]["run_id"] = "otro"
    record.write_text(json.dumps(data))
    assert og.main(_arguments(paths, tmp_path / "agregador")) == 1
    assert "no es el brazo combinado de este ajuste" in capsys.readouterr().err


def test_each_measured_market_needs_its_declared_factor_with_its_hash(tmp_path):
    factor = tmp_path / "spy.parquet"
    pq.write_table(pa.table(dict(close=[1.0, 2.0])), factor)
    declared = dict(
        US=dict(prices_path=str(factor), prices_sha256=pilots._sha256(factor)),
        CN=dict(prices_path=str(tmp_path / "csi300.parquet"), prices_sha256="b" * 64),
        JP=dict(prices_path=str(factor), prices_sha256="0" * 64),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(market_factors=declared)))
    og.checked_factors(manifest, ["US"])
    with pytest.raises(ValueError, match="No está el factor de mercado de CN"):
        og.checked_factors(manifest, ["US", "CN"])
    with pytest.raises(ValueError, match="no conserva la huella"):
        og.checked_factors(manifest, ["JP"])
    with pytest.raises(ValueError, match="no declara el factor de mercado de KR"):
        og.checked_factors(manifest, ["KR"])


def test_a_joint_state_is_read_one_market_at_a_time(tmp_path, monkeypatch, capsys):
    paths = campaign(tmp_path, markets=("CN", "US"))
    monkeypatch.setattr(og, "known_rows", _known(paths))
    output = tmp_path / "agregador"
    assert og.main(_arguments(paths, output)) == 1
    assert "indica --market" in capsys.readouterr().err
    assert not output.exists() or not any(output.iterdir())
    assert og.main(_arguments(paths, output, "--market", "US")) == 0
    folder = output / f"{YEAR}-US"
    summary = json.loads((folder / "resumen.json").read_text(encoding="utf-8"))
    assert summary["markets"] == ["US"]
    assert summary["modes"]["label"]["s1"]["best"] == "titans_mac_nogain"
    assert pq.read_table(folder / "sesiones.parquet").column("market").unique().to_pylist() == [
        "US"
    ]
    assert og.main(_arguments(paths, output, "--market", "JP")) == 1
    assert "no tiene filas de ['JP']" in capsys.readouterr().err


def test_known_rows_takes_the_alpha_of_the_measured_market_rows(tmp_path, monkeypatch):
    parent = tmp_path / "padre"
    parent.mkdir()
    instants = [START, START, START + DAY, START + DAY]
    rows = pa.table(
        dict(
            sample_id=["u1", "c1", "u2", "c2"],
            asset_id=["US/A", "CN/B", "US/A", "CN/B"],
            market=["US", "CN", "US", "CN"],
            prediction_at=pa.array(instants, pa.timestamp("us", "UTC")),
            target=pa.array([0.0] * 4, pa.float32()),
            prediction=pa.array([0.0] * 4, pa.float32()),
        )
    )
    pq.write_table(rows, parent / "evaluation-predictions.parquet")
    every = rows.sort_by(og.ORDER)
    alpha_of = dict(u1=0.01, u2=0.02, c1=np.nan, c2=np.nan)

    class Known:
        def __init__(self, state, origin):
            pass

        def rows(self, run, split):
            ids = every.column("sample_id").to_pylist()
            return np.array([alpha_of[i] for i in ids]), None

    factor = tmp_path / "spy.parquet"
    pq.write_table(pa.table(dict(close=[1.0])), factor)
    manifest = tmp_path / "cohorts" / str(YEAR) / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            dict(
                market_factors=dict(
                    US=dict(prices_path=str(factor), prices_sha256=pilots._sha256(factor)),
                    CN=dict(prices_path=str(tmp_path / "csi300.parquet"), prices_sha256="b" * 64),
                )
            )
        )
    )
    maturity = dict(
        evaluation=pa.table(
            dict(
                asset_id=rows.column("asset_id"),
                prediction_at=rows.column("prediction_at"),
                matures_at=pa.array([t + DAY for t in instants], pa.timestamp("us", "UTC")),
            )
        )
    )
    monkeypatch.setattr(og, "_screening", lambda: type("Tool", (), dict(KnownOffset=Known)))
    monkeypatch.setattr(pilots, "_maturity", lambda manifest, view: maturity)
    tables = dict(evaluation=og.only(every, ["US"]))
    alpha, matures = og.known_rows(tmp_path, YEAR, parent, VIEW, tables, ["US"])
    assert alpha["evaluation"].tolist() == [0.01, 0.02]
    assert matures["evaluation"].tolist() == [START + DAY, START + 2 * DAY]
    with pytest.raises(ValueError, match="No está el factor de mercado de CN"):
        og.known_rows(tmp_path, YEAR, parent, VIEW, dict(evaluation=every), ["CN", "US"])


def test_the_v3_variant_renames_only_the_models_and_is_registered():
    base = json.loads(DECLARATION.read_text())
    variant = json.loads((DECLARATION.parent / "posttraining-aggregator-s2-v3.json").read_text())
    assert set(variant) == set(base)
    assert {key for key in base if base[key] != variant[key]} == {
        "parent",
        "tree",
        "organ",
        "description",
    }
    assert variant["parent"] == base["parent"] + "_v3"
    assert variant["tree"] == base["tree"] + "_v3"
    assert variant["organ"] == variant["parent"] + "__combined"
    _, digest = og.checked_declaration(variant)
    assert digest in og.VARIANT_SHA256 and digest != og.REGISTERED_SHA256
    # Cualquier otro cambio sobre la variante sigue rechazado.
    with pytest.raises(ValueError, match="no es la registrada"):
        og.checked_declaration(dict(variant, seed=43))
