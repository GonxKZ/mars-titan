"""Los contrastes con varias semillas promedian por sesión y no esconden semillas que faltan."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "dgx_seed_contrasts", ROOT / "scripts" / "dgx_seed_contrasts.py"
)
contrasts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contrasts)

START = 1_357_160_700_000_000
DAY = 86_400_000_000
SESSIONS = {"validation": 30, "calibration": 20, "evaluation": 40}
ASSETS = 40


def write_model(
    state, model, seed, *, signal, noise_seed, sessions=SESSIONS, year=2013, assets=ASSETS
):
    """Emisiones sintéticas: las etiquetas dependen solo de la sesión y el ruido, de la semilla."""
    folder = state / "runs" / model / str(year) / f"seed-{seed}"
    folder.mkdir(parents=True)
    offset = 0
    for split, count in sessions.items():
        rows = []
        for session in range(count):
            labels = np.random.default_rng(10_000 + offset + session).standard_normal(assets)
            noise = np.random.default_rng(noise_seed * 1000 + offset + session).standard_normal(
                assets
            )
            prediction = signal * labels + noise
            instant = START + (offset + session) * DAY
            for asset in range(assets):
                rows.append(
                    dict(
                        sample_id=f"US/A{asset:03d}/{instant}",
                        market="US",
                        prediction_at=instant,
                        target=float(labels[asset]),
                        prediction=float(prediction[asset]),
                    )
                )
        offset += count
        table = pa.Table.from_pylist(rows).cast(
            pa.schema(
                [
                    ("sample_id", pa.string()),
                    ("market", pa.string()),
                    ("prediction_at", pa.timestamp("us", tz="UTC")),
                    ("target", pa.float64()),
                    ("prediction", pa.float64()),
                ]
            )
        )
        pq.write_table(table, folder / f"{split}-predictions.parquet")


def build_state(tmp_path, seeds=(42, 43, 44)):
    state = tmp_path / "state"
    for seed in seeds:
        write_model(state, "base", seed, signal=0.3, noise_seed=seed)
        write_model(state, "mejor", seed, signal=0.6, noise_seed=seed)
        write_model(state, "igual", seed, signal=0.3, noise_seed=seed + 100)
    return state


def add_origin(state, year, seeds=(42, 43, 44), better=0.6, assets=ASSETS):
    """Otro origen del mismo estado, con su propio ruido."""
    for seed in seeds:
        write_model(
            state, "base", seed, signal=0.3, noise_seed=seed + year, year=year, assets=assets
        )
        write_model(
            state,
            "mejor",
            seed,
            signal=better,
            noise_seed=seed + 7 * year,
            year=year,
            assets=assets,
        )
    return state


def test_the_seeds_are_averaged_per_session_before_the_contrast(tmp_path):
    state = build_state(tmp_path)
    report = contrasts.build(state, 2013, [42, 43, 44], {"mejor": "base", "igual": "base"})
    row = report["contrasts"]["mejor"]
    assert row["control"] == "base" and row["seeds"] == [42, 43, 44]
    evaluation = row["evaluation"]
    per_seed = row["per_seed"]["evaluation"]
    # La media del contraste es la media de las diferencias de cada semilla.
    assert evaluation["mean"] == pytest.approx(np.mean(per_seed), rel=1e-12)
    assert evaluation["sessions"] == 40 and evaluation["mean"] > 0.1 and evaluation["t"] > 5
    assert row["signs"]["evaluation"] == "+++"
    # Un modelo con la misma señal que su control no se distingue de él.
    same = report["contrasts"]["igual"]["evaluation"]
    assert abs(same["t"]) < 2.5
    # La decisión junta validación y calibración, sin la evaluación.
    assert row["decision"]["sessions"] == 50
    assert report["models"]["mejor"]["evaluation"]["mean"] == pytest.approx(
        np.mean(report["models"]["mejor"]["per_seed"]["evaluation"]), rel=1e-12
    )


def test_identical_seeds_do_not_shrink_the_standard_error(tmp_path):
    state = tmp_path / "state"
    for seed in (42, 43, 44):
        # La misma semilla de ruido en las tres: tres copias del mismo ajuste.
        write_model(state, "base", seed, signal=0.3, noise_seed=1)
        write_model(state, "mejor", seed, signal=0.6, noise_seed=1)
    three = contrasts.build(state, 2013, [42, 43, 44], {"mejor": "base"})
    one = contrasts.build(state, 2013, [42], {"mejor": "base"})
    left, right = three["contrasts"]["mejor"]["evaluation"], one["contrasts"]["mejor"]["evaluation"]
    assert left["se"] == pytest.approx(right["se"], rel=1e-12)
    assert left["mean"] == pytest.approx(right["mean"], rel=1e-12)


def test_a_model_without_every_seed_is_reported_as_incomplete(tmp_path):
    state = build_state(tmp_path, seeds=(42, 43))
    write_model(state, "base", 44, signal=0.3, noise_seed=44)
    report = contrasts.build(state, 2013, [42, 43, 44], {"mejor": "base", "igual": "base"})
    assert report["contrasts"] == {}
    assert report["incomplete"] == {"igual": [44], "mejor": [44]}
    assert "base" in report["models"]


def test_seeds_that_do_not_share_sessions_are_rejected(tmp_path):
    state = build_state(tmp_path, seeds=(42, 43))
    shorter = dict(SESSIONS, evaluation=39)
    write_model(state, "base", 44, signal=0.3, noise_seed=44, sessions=shorter)
    write_model(state, "mejor", 44, signal=0.6, noise_seed=44, sessions=shorter)
    with pytest.raises(ValueError, match="no comparten sesiones"):
        contrasts.build(state, 2013, [42, 43, 44], {"mejor": "base"})


def _spearman(left, right):
    # Rangos medios en los empates: la suma de rangos enteros de tres semillas los tiene.
    return float(np.corrcoef(pd.Series(left).rank(), pd.Series(right).rank())[0, 1])


def test_the_ensemble_is_the_ic_of_the_mean_rank_of_the_seeds(tmp_path):
    state = build_state(tmp_path)
    report = contrasts.build(state, 2013, [42, 43, 44], {"mejor": "base"})
    # Recalculado a mano desde los ficheros: media de rangos por sesión y su IC con la etiqueta.
    tables = [
        pq.read_table(
            state / "runs" / "base" / "2013" / f"seed-{seed}" / "evaluation-predictions.parquet"
        ).to_pandas()
        for seed in (42, 43, 44)
    ]
    values = []
    for instant, first in tables[0].groupby("prediction_at"):
        first = first.sort_values("sample_id")
        total = np.zeros(len(first))
        for table in tables:
            rows = table[table["prediction_at"] == instant].sort_values("sample_id")
            total += np.argsort(np.argsort(rows["prediction"].to_numpy()))
        values.append(_spearman(total, first["target"].to_numpy()))
    ensemble = report["models"]["base"]["ensemble"]
    assert ensemble["evaluation"]["sessions"] == 40
    assert ensemble["evaluation"]["mean"] == pytest.approx(np.mean(values), rel=1e-9)
    # Con ruido independiente entre semillas, el conjunto supera a la semilla media.
    gain = report["models"]["base"]["ensemble_gain"]["evaluation"]
    single = report["models"]["base"]["evaluation"]["mean"]
    assert gain["mean"] == pytest.approx(ensemble["evaluation"]["mean"] - single, rel=1e-9)
    assert gain["mean"] > 0.02 and gain["t"] > 3
    # El contraste entre conjuntos compara lo que se usaría: el conjunto de cada modelo.
    between = report["contrasts"]["mejor"]["ensemble"]
    assert between["evaluation"]["mean"] == pytest.approx(
        report["models"]["mejor"]["ensemble"]["evaluation"]["mean"]
        - ensemble["evaluation"]["mean"],
        rel=1e-9,
    )
    assert between["decision"]["sessions"] == 50


def test_identical_seeds_gain_nothing_from_the_ensemble(tmp_path):
    state = tmp_path / "state"
    for seed in (42, 43, 44):
        write_model(state, "base", seed, signal=0.3, noise_seed=1)
        write_model(state, "mejor", seed, signal=0.6, noise_seed=1)
    report = contrasts.build(state, 2013, [42, 43, 44], {"mejor": "base"})
    for split in ("validation", "calibration", "evaluation"):
        assert report["models"]["base"]["ensemble_gain"][split]["mean"] == pytest.approx(
            0.0, abs=1e-12
        )


def test_seeds_with_other_rows_are_not_combined(tmp_path):
    state = build_state(tmp_path, seeds=(42, 43))
    # Las mismas sesiones, pero a la semilla 44 le falta un activo en cada una.
    write_model(state, "base", 44, signal=0.3, noise_seed=44, assets=ASSETS - 1)
    write_model(state, "mejor", 44, signal=0.6, noise_seed=44, assets=ASSETS - 1)
    with pytest.raises(ValueError, match="no comparten muestras"):
        contrasts.build(state, 2013, [42, 43, 44], {"mejor": "base"})


def test_several_origins_are_summarised_between_origins(tmp_path):
    from scipy import stats

    state = tmp_path / "state"
    years = [2014, 2015, 2016, 2017]
    for year, better in zip(years, (0.6, 0.5, 0.7, 0.45), strict=True):
        add_origin(state, year, better=better)
    document = contrasts.build_origins(state, years, [42, 43, 44], {"mejor": "base"})
    assert document["years"] == years and document["final_test_opened"] is False
    assert [origin["year"] for origin in document["origins"]] == years
    entry = document["across"]["mejor"]
    assert entry["control"] == "base" and entry["years"] == years and entry["missing"] == []
    for kind, read in (
        ("seeds", lambda origin, split: origin["contrasts"]["mejor"][split]["mean"]),
        ("ensemble", lambda origin, split: origin["contrasts"]["mejor"]["ensemble"][split]["mean"]),
    ):
        for split, sessions in (("evaluation", 40), ("decision", 50)):
            block = entry[kind][split]
            means = [read(origin, split) for origin in document["origins"]]
            assert block["origin_means"] == means
            # Cada origen es una réplica: media, error entre orígenes y t con K - 1 grados.
            between = block["between"]
            error = np.std(means, ddof=1) / np.sqrt(4)
            assert between["origins"] == 4 and between["df"] == 3 and between["positive"] == 4
            assert between["mean"] == pytest.approx(np.mean(means), rel=1e-12)
            assert between["se"] == pytest.approx(error, rel=1e-12)
            assert between["t"] == pytest.approx(np.mean(means) / error, rel=1e-12)
            assert between["p"] == pytest.approx(2 * stats.t.sf(between["t"], 3), rel=1e-9)
            # Y todas las sesiones juntas, sin cruzar orígenes en la autocorrelación.
            assert block["pooled"]["sessions"] == 4 * sessions
            assert block["pooled"]["mean"] == pytest.approx(np.mean(means), rel=1e-9)
    text = contrasts.markdown_origins(document)
    assert "## Entre orígenes" in text and "| mejor | base | 4 |" in text and "4 de 4" in text
    assert "| Modelo | 2014 | 2015 | 2016 | 2017 |" in text


def test_an_origin_without_every_seed_is_left_out_and_named(tmp_path):
    state = tmp_path / "state"
    for year in (2014, 2015, 2016):
        add_origin(state, year)
    # En 2015 al modelo le falta la semilla 44: ese origen no entra en su contraste.
    lost = state / "runs" / "mejor" / "2015" / "seed-44"
    for path in lost.iterdir():
        path.unlink()
    document = contrasts.build_origins(state, [2014, 2015, 2016], [42, 43, 44], {"mejor": "base"})
    entry = document["across"]["mejor"]
    assert entry["years"] == [2014, 2016] and entry["missing"] == [2015]
    assert entry["seeds"]["evaluation"]["between"]["origins"] == 2
    assert entry["seeds"]["evaluation"]["pooled"]["sessions"] == 80
    assert "sin 2015" in contrasts.markdown_origins(document)


def test_one_origin_has_no_error_between_origins(tmp_path):
    state = add_origin(tmp_path / "state", 2014)
    document = contrasts.build_origins(state, [2014], [42, 43, 44], {"mejor": "base"})
    between = document["across"]["mejor"]["seeds"]["evaluation"]["between"]
    assert between["origins"] == 1 and between["df"] == 0
    assert between["se"] is None and between["t"] is None and between["p"] is None


def test_the_command_with_several_years_writes_the_summary_between_origins(tmp_path, capsys):
    state = tmp_path / "state"
    for year in (2014, 2015, 2016):
        add_origin(state, year)
    declared = tmp_path / "contrasts.json"
    declared.write_text(json.dumps(dict(controls={"mejor": "base"})))
    output = tmp_path / "out"
    arguments = ["--state", str(state), "--contrasts", str(declared), "--output", str(output)]
    arguments += ["--seed", "42", "--seed", "43", "--seed", "44"]
    arguments += ["--year", "2014", "--year", "2015", "--year", "2016"]
    assert contrasts.main(arguments) == 0
    capsys.readouterr()
    document = json.loads((output / "seed-contrasts.json").read_text(encoding="utf-8"))
    assert document["years"] == [2014, 2015, 2016] and set(document["across"]) == {"mejor"}
    text = (output / "seed-contrasts.md").read_text(encoding="utf-8")
    assert "## Entre orígenes" in text and "| mejor | base | 3 |" in text


def test_a_reference_fitted_once_pairs_with_every_seed(tmp_path):
    state, other = build_state(tmp_path), tmp_path / "referencias"
    # La referencia vive en otro estado y solo tiene un ajuste, con la semilla 42.
    write_model(other, "arboles", 42, signal=0.2, noise_seed=900)
    controls = {"mejor": "base", "base": "arboles"}
    report = contrasts.build(
        state, 2013, [42, 43, 44], controls, references=[other], reference_seed=42
    )
    assert report["incomplete"] == {} and set(report["contrasts"]) == {"mejor", "base"}
    row = report["contrasts"]["base"]
    assert row["control"] == "arboles" and row["evaluation"]["sessions"] == 40
    assert row["evaluation"]["mean"] == pytest.approx(
        report["models"]["base"]["evaluation"]["mean"]
        - report["models"]["arboles"]["evaluation"]["mean"],
        rel=1e-9,
    )
    # Un solo ajuste: sus tres lecturas son la misma y su conjunto no gana nada.
    level = report["models"]["arboles"]
    assert len(set(level["per_seed"]["evaluation"])) == 1
    assert level["ensemble_gain"]["evaluation"]["mean"] == pytest.approx(0.0, abs=1e-12)
    # Sin decir qué semilla tiene la referencia, le faltan las otras dos y se nombra.
    strict = contrasts.build(state, 2013, [42, 43, 44], controls, references=[other])
    assert strict["incomplete"] == {"base": [43, 44]} and set(strict["contrasts"]) == {"mejor"}
    # Sin el estado de la referencia, el modelo no existe.
    alone = contrasts.build(state, 2013, [42, 43, 44], controls)
    assert alone["incomplete"] == {"base": [42, 43, 44]}


def test_the_campaign_state_wins_over_a_reference_with_the_same_model(tmp_path):
    state, other = build_state(tmp_path), tmp_path / "referencias"
    write_model(other, "base", 42, signal=0.9, noise_seed=901)
    plain = contrasts.build(state, 2013, [42, 43, 44], {"mejor": "base"})
    mixed = contrasts.build(
        state, 2013, [42, 43, 44], {"mejor": "base"}, references=[other], reference_seed=42
    )
    assert mixed["models"]["base"] == plain["models"]["base"]


def test_the_command_takes_references_and_extra_pairs(tmp_path, capsys):
    state, other = build_state(tmp_path), tmp_path / "referencias"
    write_model(other, "arboles", 42, signal=0.2, noise_seed=900)
    declared = tmp_path / "contrasts.json"
    declared.write_text(json.dumps(dict(controls={"mejor": "base"})))
    output = tmp_path / "out"
    arguments = ["--state", str(state), "--year", "2013", "--contrasts", str(declared)]
    arguments += ["--seed", "42", "--seed", "43", "--seed", "44"]
    arguments += ["--reference", str(other), "--reference-seed", "42"]
    assert contrasts.main([*arguments, "--pair", "base:arboles", "--output", str(output)]) == 0
    capsys.readouterr()
    document = json.loads((output / "seed-contrasts.json").read_text(encoding="utf-8"))
    assert set(document["contrasts"]) == {"mejor", "base"}
    assert document["contrasts"]["base"]["control"] == "arboles"
    # Un par que contradice al control declarado se rechaza, y uno mal escrito también.
    for pair in ("mejor:arboles", "base", "base:"):
        with pytest.raises(SystemExit):
            contrasts.main([*arguments, "--pair", pair, "--output", str(tmp_path / "otra")])
    assert not (tmp_path / "otra").exists()


class FakeOffset:
    """Alfa y liquidez conocidos de cada fila, función del activo y de la sesión."""

    def __init__(self):
        self.calls = 0

    def rows(self, run, split):
        self.calls += 1
        table = pq.read_table(run / f"{split}-predictions.parquet").sort_by(
            [("market", "ascending"), ("prediction_at", "ascending"), ("sample_id", "ascending")]
        )
        names = table.column("sample_id").to_pylist()
        assets = np.array([int(name.split("/")[1][1:]) for name in names])
        sessions = (np.array([int(name.split("/")[2]) for name in names]) - START) // DAY
        alpha = 0.8 * np.sin(0.7 * assets + 0.3 * sessions)
        # Un activo sin modelo de mercado: su alfa no se conoce y sus filas no cuentan.
        alpha[assets == 0] = np.nan
        return alpha, assets.astype(np.float64)


def operable_by_hand(state, model, seed, split, factor, liquid=False):
    """IC sin alfa por sesión de una semilla, calculado aparte con pandas."""
    path = state / "runs" / model / "2013" / f"seed-{seed}" / f"{split}-predictions.parquet"
    table = pq.read_table(path).to_pandas()
    table["asset"] = table["sample_id"].str.split("/").str[1].str[1:].astype(int)
    table["session"] = (table["sample_id"].str.split("/").str[2].astype("int64") - START) // DAY
    table["alpha"] = 0.8 * np.sin(0.7 * table["asset"] + 0.3 * table["session"])
    table = table[table["asset"] != 0]
    values = {}
    for instant, rows in table.groupby("prediction_at"):
        if liquid:
            rows = rows.sort_values("asset").iloc[len(rows) - len(rows) // 3 :]
        signal = factor * rows["prediction"] + rows["alpha"]
        values[instant] = signal.rank().corr((rows["target"] + rows["alpha"]).rank())
    return values


def test_the_operable_return_uses_the_scale_chosen_in_calibration(tmp_path):
    state = tmp_path / "state"
    for seed in (42, 43, 44):
        # Cien activos: el tercio líquido de los 99 con alfa pasa del mínimo de 30.
        write_model(state, "base", seed, signal=0.3, noise_seed=seed, assets=100)
        write_model(state, "mejor", seed, signal=0.6, noise_seed=seed, assets=100)
    # Las predicciones de un modelo, veinte veces mayores: sin factor, taparían al alfa.
    for path in (state / "runs" / "mejor").rglob("*.parquet"):
        table = pq.read_table(path)
        scaled = pa.array(table.column("prediction").to_numpy() * 20.0)
        pq.write_table(
            table.set_column(table.schema.get_field_index("prediction"), "prediction", scaled), path
        )
    provider = FakeOffset()
    report = contrasts.build(state, 2013, [42, 43, 44], {"mejor": "base"}, offset=provider)
    module = contrasts._report_module()
    for model in ("base", "mejor"):
        entry = report["models"][model]["operable"]
        assert len(entry["factors"]) == 3 and set(entry["factors"]) <= set(module.SCALES)
        per_seed = []
        for seed, factor in zip((42, 43, 44), entry["factors"], strict=True):
            # El factor es el que da más IC sin alfa en calibración, y solo en ella.
            calibration = {
                scale: np.mean(
                    list(operable_by_hand(state, model, seed, "calibration", scale).values())
                )
                for scale in module.SCALES
            }
            assert factor == max(module.SCALES, key=calibration.get)
            per_seed.append(operable_by_hand(state, model, seed, "evaluation", factor))
        expected = np.mean([np.mean([one[key] for one in per_seed]) for key in per_seed[0]])
        assert entry["evaluation"]["sessions"] == 40
        assert entry["evaluation"]["mean"] == pytest.approx(expected, rel=1e-9)
    assert report["models"]["mejor"]["operable"]["factors"][0] < 0.2
    row = report["contrasts"]["mejor"]["operable"]
    assert row["evaluation"]["mean"] == pytest.approx(
        report["models"]["mejor"]["operable"]["evaluation"]["mean"]
        - report["models"]["base"]["operable"]["evaluation"]["mean"],
        rel=1e-9,
    )
    assert row["decision"]["sessions"] == 50 and row["evaluation"]["mean"] > 0
    # El tercio más líquido de cada sesión, con el mismo factor.
    liquid = report["models"]["base"]["operable_liquid"]
    factor = report["models"]["base"]["operable"]["factors"][0]
    hand = operable_by_hand(state, "base", 42, "evaluation", factor, liquid=True)
    assert liquid["per_seed"]["evaluation"][0] == pytest.approx(
        np.mean(list(hand.values())), rel=1e-9
    )
    # El alfa de las filas es el mismo para todas las semillas y modelos: se lee una vez por tramo.
    assert provider.calls == 3
    text = contrasts.markdown(report)
    assert "## Rendimiento operable" in text and "| mejor | base |" in text
    # Sin el desplazamiento no se informa de lo operable.
    plain = contrasts.build(state, 2013, [42, 43, 44], {"mejor": "base"})
    assert (
        "operable" not in plain["models"]["base"] and "operable" not in plain["contrasts"]["mejor"]
    )
    assert "Rendimiento operable" not in contrasts.markdown(plain)


def test_a_small_universe_has_no_liquid_third_and_says_so_without_numbers(tmp_path):
    state = tmp_path / "state"
    for year in (2014, 2015):
        add_origin(state, year)
    # Cuarenta activos: el tercio líquido no llega a los 30 de una sesión válida.
    document = contrasts.build_origins(
        state, [2014, 2015], [42, 43, 44], {"mejor": "base"}, offsets=lambda year: FakeOffset()
    )
    entry = document["across"]["mejor"]
    assert entry["operable"]["evaluation"]["between"]["origins"] == 2
    liquid = entry["operable_liquid"]["evaluation"]
    assert liquid["origin_means"] == [None, None] and liquid["pooled"]["sessions"] == 0
    assert liquid["between"] == dict(
        origins=0, mean=None, se=None, t=None, df=0, p=None, positive=0
    )
    level = document["origins"][0]["models"]["base"]["operable_liquid"]
    assert level["evaluation"]["sessions"] == 0 and level["per_seed"]["evaluation"] == [None] * 3
    assert "— (—)" in contrasts.markdown_origins(document)


def test_the_summary_between_origins_carries_the_operable_return(tmp_path):
    state = tmp_path / "state"
    for year in (2014, 2015, 2016):
        add_origin(state, year, assets=100)
    document = contrasts.build_origins(
        state,
        [2014, 2015, 2016],
        [42, 43, 44],
        {"mejor": "base"},
        offsets=lambda year: FakeOffset(),
    )
    entry = document["across"]["mejor"]
    for kind in ("operable", "operable_liquid"):
        block = entry[kind]["evaluation"]
        means = [
            origin["contrasts"]["mejor"][kind]["evaluation"]["mean"]
            for origin in document["origins"]
        ]
        assert block["origin_means"] == means and block["between"]["origins"] == 3
        assert block["pooled"]["sessions"] == 120
    text = contrasts.markdown_origins(document)
    assert "## Entre orígenes: rendimiento operable" in text


def known_rows(state, model, seed, split):
    """Las filas con alfa de una emisión, con el alfa que les da `FakeOffset`."""
    path = state / "runs" / model / "2013" / f"seed-{seed}" / f"{split}-predictions.parquet"
    table = pq.read_table(path).to_pandas()
    table["asset"] = table["sample_id"].str.split("/").str[1].str[1:].astype(int)
    table["session"] = (table["sample_id"].str.split("/").str[2].astype("int64") - START) // DAY
    table["alpha"] = 0.8 * np.sin(0.7 * table["asset"] + 0.3 * table["session"])
    return table[table["asset"] != 0]


def label_by_hand(state, model, seed, split, factor=None):
    """IC de la etiqueta por sesión en las filas con alfa, calculado aparte con pandas.

    Sin factor, con la predicción tal cual. Con él, con la predicción de lo operable devuelta a
    la etiqueta: factor·(p + alfa) − alfa.
    """
    values = {}
    for instant, rows in known_rows(state, model, seed, split).groupby("prediction_at"):
        signal = rows["prediction"]
        if factor is not None:
            signal = factor * (rows["prediction"] + rows["alpha"]) - rows["alpha"]
        values[instant] = signal.rank().corr(rows["target"].rank())
    return values


def rewrite_predictions(state, model, change):
    """Cambia las predicciones emitidas de un modelo con `change(predicción, alfa)`."""
    for path in (state / "runs" / model).rglob("*.parquet"):
        table = pq.read_table(path)
        names = table.column("sample_id").to_pylist()
        assets = np.array([int(name.split("/")[1][1:]) for name in names])
        sessions = (np.array([int(name.split("/")[2]) for name in names]) - START) // DAY
        alpha = 0.8 * np.sin(0.7 * assets + 0.3 * sessions)
        changed = pa.array(change(table.column("prediction").to_numpy(), alpha))
        index = table.schema.get_field_index("prediction")
        pq.write_table(table.set_column(index, "prediction", changed), path)


def operable_state(tmp_path, seeds=(42, 43, 44)):
    """Dos emisiones del mismo ajuste: tal cual, y como predicción de lo operable a otra escala."""
    state = tmp_path / "state"
    for seed in seeds:
        write_model(state, "base", seed, signal=0.5, noise_seed=seed, assets=100)
        write_model(state, "operable", seed, signal=0.5, noise_seed=seed, assets=100)
    # 5·(p + alfa) − alfa: ordena lo operable, y con el factor 0,2 se devuelve exacta.
    rewrite_predictions(
        state, "operable", lambda prediction, alpha: 5.0 * (prediction + alpha) - alpha
    )
    return state


def test_a_model_fitted_for_the_operable_return_is_read_back_on_the_label(tmp_path):
    state = operable_state(tmp_path)
    report = contrasts.build(state, 2013, [42, 43, 44], {"operable": "base"}, offset=FakeOffset())
    module = contrasts._report_module()
    entry = report["models"]["operable"]["label_returned"]
    assert len(entry["factors"]) == 3
    per_seed = []
    for seed, factor in zip((42, 43, 44), entry["factors"], strict=True):
        # El factor es el que da más IC de la etiqueta en calibración, y solo en ella.
        calibration = {
            scale: np.mean(
                list(label_by_hand(state, "operable", seed, "calibration", scale).values())
            )
            for scale in module.SCALES
        }
        assert factor == max(module.SCALES, key=calibration.get)
        per_seed.append(label_by_hand(state, "operable", seed, "evaluation", factor))
    expected = np.mean([np.mean([one[key] for one in per_seed]) for key in per_seed[0]])
    assert entry["evaluation"]["sessions"] == 40
    assert entry["evaluation"]["mean"] == pytest.approx(expected, rel=1e-9)
    # La etiqueta con la predicción tal cual, en las mismas filas con alfa conocido.
    known = {}
    for model in ("base", "operable"):
        plain = [label_by_hand(state, model, seed, "evaluation") for seed in (42, 43, 44)]
        known[model] = np.mean([np.mean([one[key] for one in plain]) for key in plain[0]])
        level = report["models"][model]["label_known"]
        assert level["evaluation"]["mean"] == pytest.approx(known[model], rel=1e-9)
        assert "factors" not in level
    # Devolverla recupera lo que la inclinación hacia el alfa le quita en la etiqueta.
    assert entry["evaluation"]["mean"] > known["operable"] + 0.01
    # El contraste: la etiqueta devuelta del modelo frente a la etiqueta tal cual de su control.
    row = report["contrasts"]["operable"]["label_returned"]
    assert row["evaluation"]["mean"] == pytest.approx(
        entry["evaluation"]["mean"] - known["base"], rel=1e-9, abs=1e-12
    )
    assert row["evaluation"]["sessions"] == 40 and row["decision"]["sessions"] == 50
    text = contrasts.markdown(report)
    assert "Etiqueta devuelta" in text and "frente a la etiqueta del control" in text
    # Sin el desplazamiento no hay nada que devolver.
    plain = contrasts.build(state, 2013, [42, 43, 44], {"operable": "base"})
    assert "label_returned" not in plain["models"]["operable"]
    assert "label_returned" not in plain["contrasts"]["operable"]


def test_reading_back_with_the_exact_factor_recovers_the_prediction_of_the_label(tmp_path):
    state = operable_state(tmp_path, seeds=(42,))
    module = contrasts._report_module()
    provider = FakeOffset()
    base, tilted = (state / "runs" / name / "2013" / "seed-42" for name in ("base", "operable"))
    for split in ("calibration", "evaluation"):
        alpha, _ = provider.rows(base, split)
        emitted = module.load_split(base, split)
        plain = module.implied_sessions(
            emitted, alpha, free=False, prediction=emitted["prediction"]
        )
        back = module.returned_sessions(module.load_split(tilted, split), alpha, 0.2)
        assert back.keys() == plain.keys() and len(back) == SESSIONS[split]
        assert all(back[key] == pytest.approx(plain[key], abs=1e-9) for key in back)
        # Con otro factor queda parte del alfa y el IC de la etiqueta baja.
        worse = module.returned_sessions(module.load_split(tilted, split), alpha, 1.0)
        assert np.mean(list(worse.values())) < np.mean(list(plain.values())) - 0.01


def test_the_factor_needs_calibration_sessions_with_a_known_offset(tmp_path):
    state = tmp_path / "state"
    write_model(state, "base", 42, signal=0.5, noise_seed=42, assets=100)
    module = contrasts._report_module()
    run = state / "runs" / "base" / "2013" / "seed-42"
    splits = {split: module.load_split(run, split) for split in contrasts.SPLITS}
    unknown = {split: np.full(len(splits[split]["target"]), np.nan) for split in contrasts.SPLITS}
    for choose in (module.best_scale, module.best_return_scale):
        # Antes devolvía el primer factor de la rejilla sin avisar.
        with pytest.raises(ValueError, match="calibración"):
            choose(splits, unknown)


def test_the_grid_of_factors_keeps_its_old_points_and_reaches_further():
    module = contrasts._report_module()
    old = (0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
    assert set(old) < set(module.SCALES) and list(module.SCALES) == sorted(module.SCALES)
    assert min(module.SCALES) <= 0.001 and max(module.SCALES) >= 100.0


def test_a_factor_at_the_edge_of_the_grid_is_flagged(tmp_path):
    state = tmp_path / "state"
    for seed in (42, 43, 44):
        write_model(state, "base", seed, signal=0.5, noise_seed=seed, assets=100)
        write_model(state, "enorme", seed, signal=0.5, noise_seed=seed, assets=100)
    # Predicciones un millón de veces mayores: el mejor factor queda fuera de la rejilla.
    rewrite_predictions(state, "enorme", lambda prediction, alpha: 1e6 * prediction)
    report = contrasts.build(state, 2013, [42, 43, 44], {"enorme": "base"}, offset=FakeOffset())
    module = contrasts._report_module()
    assert report["models"]["enorme"]["operable"]["factors"] == [min(module.SCALES)] * 3
    assert report["models"]["enorme"]["operable"]["factor_at_edge"] is True
    assert report["models"]["base"]["operable"]["factor_at_edge"] is False
    assert "factor_at_edge" not in report["models"]["base"]["operable_liquid"]
    lines = [
        line for line in contrasts.markdown(report).split("\n") if "borde de la rejilla" in line
    ]
    assert len(lines) == 1 and "enorme" in lines[0] and "base" not in lines[0]
    # Sin ningún factor en el borde no se dice nada.
    inside = contrasts.build(state, 2013, [42, 43, 44], {"base": "base"}, offset=FakeOffset())
    assert "borde de la rejilla" not in contrasts.markdown(inside)


@pytest.mark.parametrize("column", ["prediction", "target"])
def test_an_emission_with_a_value_that_is_not_finite_is_refused(tmp_path, column):
    state = build_state(tmp_path)
    path = state / "runs" / "mejor" / "2013" / "seed-43" / "evaluation-predictions.parquet"
    table = pq.read_table(path)
    values = table.column(column).to_numpy().copy()
    values[7] = np.nan
    index = table.schema.get_field_index(column)
    pq.write_table(table.set_column(index, column, pa.array(values)), path)
    # Antes un NaN recibía el rango más alto de su sesión y entraba en el IC.
    with pytest.raises(ValueError, match="no finito"):
        contrasts.build(state, 2013, [42, 43, 44], {"mejor": "base"})


def test_a_universe_too_small_for_any_session_gives_no_numbers(tmp_path):
    state = tmp_path / "state"
    for seed in (42, 43, 44):
        write_model(state, "base", seed, signal=0.3, noise_seed=seed, assets=20)
        write_model(state, "mejor", seed, signal=0.6, noise_seed=seed, assets=20)
    # Veinte activos: ninguna sesión llega al mínimo de 30.
    report = contrasts.build(state, 2013, [42, 43, 44], {"mejor": "base"})
    level = report["models"]["base"]
    assert level["evaluation"]["sessions"] == 0 and level["per_seed"]["evaluation"] == [None] * 3
    row = report["contrasts"]["mejor"]
    assert row["per_seed"]["evaluation"] == [None] * 3 and row["signs"]["evaluation"] == "???"
    assert "| mejor | base |" in contrasts.markdown(report)


def write_record(state, model, seed, attempts, year=2013):
    queue = state / "queue"
    queue.mkdir(exist_ok=True)
    record = dict(status="completed", model=model, year=year, seed=seed, attempts=attempts)
    (queue / f"{model}--{year}--seed-{seed}.json").write_text(json.dumps(record))


def place(site, gpu, machine):
    return dict(
        execution=dict(code_commit="a" * 40),
        resources=dict(
            site=site,
            gpu_name=gpu,
            platform=dict(system="Linux", machine=machine),
            versions=dict(torch="2.14.0+cu130"),
        ),
    )


def test_the_document_says_where_each_model_was_fitted(tmp_path):
    state = build_state(tmp_path)
    dgx = place(None, "NVIDIA GB10", "aarch64")
    colab = place("google-colab-rtx-pro-6000", "NVIDIA RTX PRO 6000", "x86_64")
    for seed in (42, 43, 44):
        write_record(state, "base", seed, [dgx])
    # Un modelo con una semilla ajustada aquí y dos en Colab lleva los dos sitios.
    write_record(state, "mejor", 42, [dgx])
    write_record(state, "mejor", 43, [colab])
    write_record(state, "mejor", 44, [colab])
    report = contrasts.build(state, 2013, [42, 43, 44], {"mejor": "base", "igual": "base"})
    assert [item["site"] for item in report["models"]["base"]["provenance"]] == [None]
    assert [item["site"] for item in report["models"]["mejor"]["provenance"]] == [
        None,
        "google-colab-rtx-pro-6000",
    ]
    assert report["models"]["igual"]["provenance"] == []
    lines = contrasts.markdown(report).splitlines()
    section = lines[lines.index("## Procedencia de los ajustes") :]
    assert [line for line in section if line.startswith("- ")] == [
        "- NVIDIA GB10 (aarch64), sin etiqueta de equipo: `base`, `mejor`",
        "- NVIDIA RTX PRO 6000 (x86_64), equipo google-colab-rtx-pro-6000: `mejor`",
        "- sin telemetría del equipo: `igual`",
    ]


def test_the_summary_between_origins_says_the_place_of_each_origin(tmp_path):
    state = tmp_path / "state"
    for year in (2014, 2015):
        add_origin(state, year)
    for seed in (42, 43, 44):
        for model in ("base", "mejor"):
            write_record(state, model, seed, [place(None, "NVIDIA GB10", "aarch64")], year=2014)
            write_record(state, model, seed, [place("colab-a", "RTX", "x86_64")], year=2015)
    document = contrasts.build_origins(state, [2014, 2015], [42, 43, 44], {"mejor": "base"})
    lines = contrasts.markdown_origins(document).splitlines()
    section = lines[lines.index("## Procedencia de los ajustes") :]
    assert [line for line in section if line.startswith("- ")] == [
        "- Origen 2014, NVIDIA GB10 (aarch64), sin etiqueta de equipo: los 2 modelos",
        "- Origen 2015, RTX (x86_64), equipo colab-a: los 2 modelos",
    ]


def test_the_command_writes_the_table_and_the_document(tmp_path, capsys):
    state = build_state(tmp_path)
    declared = tmp_path / "contrasts.json"
    declared.write_text(json.dumps(dict(controls={"mejor": "base", "igual": "base"})))
    output = tmp_path / "out"
    arguments = ["--state", str(state), "--year", "2013", "--contrasts", str(declared)]
    arguments += ["--seed", "42", "--seed", "43", "--seed", "44", "--output", str(output)]
    assert contrasts.main(arguments) == 0
    capsys.readouterr()
    document = json.loads((output / "seed-contrasts.json").read_text(encoding="utf-8"))
    assert document["kind"] == "dgx_seed_contrasts_v1" and document["final_test_opened"] is False
    assert set(document["contrasts"]) == {"mejor", "igual"}
    text = (output / "seed-contrasts.md").read_text(encoding="utf-8")
    assert "| mejor | base | 3 |" in text and "+++" in text
    assert "Conjunto" in text and "Δ conjunto (t)" in text
    with pytest.raises(SystemExit):
        contrasts.main([*arguments[:-1], str(output), "--seed", "42"])
