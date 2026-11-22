"""Pilotos de post-entrenamiento sobre emisiones: carga, calibración, memoria y selección."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.evaluation import posttraining_pilots as pilots

DAY = 86_400_000_000
START = 1_333_310_700_000_000
LEVELS = (0.025, 0.1, 0.5, 0.9, 0.975)
SPLITS = ("validation", "calibration", "evaluation")


def synthetic(*, sessions=(40, 30, 60), assets=80, seed=0, scale=3.0, effect=0.6, width=0.2):
    """Emisiones con magnitud inflada, intervalos estrechos y un residuo persistente por activo."""
    rng = np.random.default_rng(seed)
    names = np.array([f"US/A{i:03d}" for i in range(assets)])
    persistent = rng.normal(0.0, 1.0, assets)
    tables, maturity, s = {}, {}, 0
    for name, count in zip(SPLITS, sessions, strict=True):
        cols = {k: [] for k in ("asset", "at", "target", "prediction", "matures")}
        for _ in range(count):
            signal = rng.normal(0.0, 1.0, assets)
            noise = rng.normal(0.0, 2.0, assets)
            cols["asset"].append(names)
            cols["at"].append(np.full(assets, START + s * DAY, dtype=np.int64))
            cols["target"].append(0.01 * (signal + effect * persistent + noise))
            cols["prediction"].append(scale * 0.01 * signal)
            cols["matures"].append(np.full(assets, START + (s + 1) * DAY, dtype=np.int64))
            s += 1
        arrays = {k: np.concatenate(v) for k, v in cols.items()}
        offsets = 0.01 * width * np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        quantiles = arrays["prediction"][:, None] + offsets[None, :]
        moments = pa.array(arrays["at"], type=pa.int64()).cast(pa.timestamp("us", tz="UTC"))
        sample = [f"{a}/{t}" for a, t in zip(arrays["asset"], arrays["at"], strict=True)]
        table = {
            "sample_id": pa.array(sample),
            "asset_id": pa.array(arrays["asset"]),
            "market": pa.array(["US"] * len(sample)),
            "prediction_at": moments,
            "target": arrays["target"],
            "prediction": arrays["prediction"],
            "zero": np.zeros(len(sample)),
        }
        for column, k in zip(pilots.QUANTILE_COLUMNS, range(5), strict=True):
            table[column] = quantiles[:, k]
        # La tabla emitida llega desordenada: la carga debe ordenarla.
        shuffled = rng.permutation(len(sample))
        tables[name] = pa.table(table).take(pa.array(shuffled))
        maturity[name] = pa.table(
            {
                "asset_id": pa.array(arrays["asset"]),
                "market": pa.array(["US"] * len(sample)),
                "prediction_at": moments,
                "target": arrays["target"],
                "matures_at": arrays["matures"],
            }
        )
    return tables, maturity


def splits_of(tables, maturity):
    return {name: pilots.load_split(tables[name], maturity[name], name) for name in SPLITS}


def write_run(folder, tables):
    folder.mkdir(parents=True)
    for name, table in tables.items():
        pq.write_table(table, folder / f"{name}-predictions.parquet")
    run = dict(
        status="completed",
        run_id="r" * 64,
        request=dict(view_sha256="v" * 64, window="fold-008", seed=42),
    )
    (folder / "run.json").write_text(json.dumps(run))


def declaration(**changes):
    document = {
        "schema_version": 1,
        "kind": pilots.KIND,
        "final_test_access": False,
        "description": "prueba",
        "market": "US",
        "min_assets": 30,
        "runs": [{"state": "s", "models": ["m"], "origins": [2013], "seeds": [42]}],
        "memory": {
            "value": "residual",
            "half_lives": [5, 21],
            "kappas": [0.0, 0.1, 0.4],
            "lab": {"half_life": 21, "kappa": 0.1},
            "placebo_seed": 7,
            "controls": ["label"],
        },
        "calibration": {
            "levels": list(LEVELS),
            "nominals": [0.8, 0.95],
            "min_rows": 200,
            "online_rates": [0.0, 0.01, 0.05],
            "regime_models": [],
        },
    }
    document.update(changes)
    return document


def test_loading_sorts_each_split_and_attaches_the_view_maturity():
    tables, maturity = synthetic(sessions=(3, 3, 3), assets=40)
    split = pilots.load_split(tables["calibration"], maturity["calibration"], "calibration")
    keys = list(zip(split.prediction_at.tolist(), split.asset.tolist(), strict=True))
    assert keys == sorted(keys)
    assert np.all(split.matures_at == split.prediction_at + DAY)
    assert [end - start for _, _, start, end in split.sessions] == [40, 40, 40]


def test_a_row_without_view_label_fails_with_its_coordinates():
    tables, maturity = synthetic(sessions=(2, 2, 2), assets=40)
    view = maturity["evaluation"]
    with pytest.raises(ValueError, match="US/A00"):
        pilots.load_split(tables["evaluation"], view.slice(1), "evaluation")


def test_a_target_that_differs_from_the_view_fails_with_its_coordinates():
    tables, maturity = synthetic(sessions=(2, 2, 2), assets=40)
    view = maturity["evaluation"]
    target = view["target"].to_numpy().copy()
    target[5] += 1e-12
    view = view.set_column(view.schema.get_field_index("target"), "target", pa.array(target))
    with pytest.raises(ValueError, match="objetivo"):
        pilots.load_split(tables["evaluation"], view, "evaluation")


def test_a_nan_prediction_and_crossed_quantiles_fail_with_their_coordinates():
    tables, maturity = synthetic(sessions=(2, 2, 2), assets=40)
    table = tables["evaluation"]
    values = table["prediction"].to_numpy().copy()
    values[3] = np.nan
    index = table.schema.get_field_index("prediction")
    broken = table.set_column(index, "prediction", pa.array(values))
    with pytest.raises(ValueError, match="no finit"):
        pilots.load_split(broken, maturity["evaluation"], "evaluation")
    low = table["quantile_0100"].to_numpy().copy()
    low[4] = table["quantile_0900"].to_numpy()[4] + 1.0
    index = table.schema.get_field_index("quantile_0100")
    crossed = table.set_column(index, "quantile_0100", pa.array(low))
    with pytest.raises(ValueError, match="cruzad"):
        pilots.load_split(crossed, maturity["evaluation"], "evaluation")


def test_static_scale_and_cqr_bring_coverage_and_magnitude_back():
    tables, maturity = synthetic(sessions=(30, 60, 60), assets=120, seed=3)
    splits = splits_of(tables, maturity)
    result = pilots.calibration_pilot(splits, declaration()["calibration"], regime=False)
    raw, fixed = result["methods"]["raw"], result["methods"]["scale_cqr_static"]
    assert raw["cov80"] < 0.5 and raw["mae_ratio"] > 1.0
    assert abs(fixed["cov80"] - 0.8) < 0.03 and abs(fixed["cov95"] - 0.95) < 0.02
    assert fixed["mae_ratio"] < raw["mae_ratio"]
    assert result["methods"]["cqr_static"]["mae_ratio"] == raw["mae_ratio"]
    assert fixed["pinball"] < raw["pinball"]
    assert result["contrasts"]["scale_cqr_static-raw"]["pinball"]["mean"] < 0


def test_online_calibration_only_reads_labels_already_matured():
    tables, maturity = synthetic(sessions=(10, 20, 20), assets=60, seed=4)
    splits = splits_of(tables, maturity)
    calibration = splits["calibration"]
    nominals = (0.8, 0.95)
    base = pilots.online_conformal(calibration, splits["evaluation"], 0.05, LEVELS, nominals, 200)
    evaluation = splits["evaluation"]
    k = 7
    start, end = evaluation.sessions[k][2], evaluation.sessions[k][3]
    target = evaluation.target.copy()
    target[start:end] += 0.5
    changed = pilots.replace_target(evaluation, target)
    after = pilots.online_conformal(calibration, changed, 0.05, LEVELS, nominals, 200)
    np.testing.assert_array_equal(after[:end], base[:end])
    following = slice(evaluation.sessions[k + 1][2], evaluation.sessions[k + 1][3])
    assert not np.array_equal(after[following], base[following])


def test_the_online_rate_is_chosen_without_reading_the_evaluation():
    tables, maturity = synthetic(sessions=(30, 30, 30), assets=60, seed=5)
    splits = splits_of(tables, maturity)
    section = declaration()["calibration"]
    chosen = pilots.select_online_rate(splits, section)
    evaluation = splits["evaluation"]
    noisy = pilots.replace_target(evaluation, evaluation.target * 50.0)
    assert pilots.select_online_rate(dict(splits, evaluation=noisy), section) == chosen


def test_the_memory_finds_a_planted_asset_effect_and_its_placebo_does_not():
    tables, maturity = synthetic(sessions=(40, 40, 80), assets=80, seed=6, effect=0.8)
    splits = splits_of(tables, maturity)
    result = pilots.memory_pilot(splits, declaration()["memory"], min_assets=30)
    contrasts = result["contrasts"]
    assert contrasts["lab-parent"]["mean"] > 0.02 and contrasts["lab-parent"]["t"] > 3
    assert contrasts["placebo_lab-parent"]["mean"] < 0.01
    assert contrasts["lab-placebo_lab"]["t"] > 3
    evaluation = result["evaluation"]
    assert evaluation["kappa_0"] == pytest.approx(evaluation["parent"], abs=1e-12)


def test_the_memory_configuration_is_chosen_without_reading_the_evaluation():
    tables, maturity = synthetic(sessions=(30, 40, 40), assets=60, seed=7, effect=0.8)
    splits = splits_of(tables, maturity)
    section = declaration()["memory"]
    chosen = pilots.memory_pilot(splits, section, min_assets=30)["selection"]
    assert chosen["kappa"] > 0
    # En una evaluación con el efecto invertido la mejor configuración sería otra.
    evaluation = splits["evaluation"]
    reversed_effect = pilots.replace_target(evaluation, -evaluation.target)
    again = pilots.memory_pilot(dict(splits, evaluation=reversed_effect), section, min_assets=30)
    assert again["selection"] == chosen
    assert again["evaluation"]["lab"] < again["evaluation"]["kappa_0"]


def test_the_placebo_map_is_a_reproducible_derangement():
    assets = [f"US/A{i:03d}" for i in range(50)]
    first = pilots.derangement(assets, 11)
    assert first == pilots.derangement(list(reversed(assets)), 11)
    assert sorted(first) == sorted(first.values()) == sorted(assets)
    assert all(key != value for key, value in first.items())
    assert first != pilots.derangement(assets, 12)


@pytest.mark.parametrize(
    "changes, message",
    [
        (dict(final_test_access=True), "test final"),
        (dict(extra=1), "campos"),
        (dict(kind="otra"), "posttraining_pilots_v1"),
    ],
)
def test_the_declaration_is_checked_before_reading_anything(changes, message):
    with pytest.raises(ValueError, match=message):
        pilots.checked_declaration(declaration(**changes))


def test_the_lab_configuration_must_belong_to_the_declared_grid():
    document = declaration()
    document["memory"] = dict(document["memory"], lab={"half_life": 63, "kappa": 0.1})
    with pytest.raises(ValueError, match="rejilla"):
        pilots.checked_declaration(document)


def test_a_run_report_has_finite_metrics_and_names_its_inputs(tmp_path):
    tables, maturity = synthetic(sessions=(30, 30, 40), assets=60, seed=8)
    folder = tmp_path / "runs" / "m" / "2013" / "seed-42"
    write_run(folder, tables)
    report, series = pilots.analyze_run(folder, maturity, pilots.checked_declaration(declaration()))
    assert report["final_test_opened"] is False
    expected = {f"{name}-predictions.parquet" for name in SPLITS} | {"run.json"}
    assert set(report["inputs"]) == expected
    for metrics in report["calibration"]["methods"].values():
        assert all(np.isfinite(v) for v in metrics.values() if isinstance(v, float))
    assert report["memory"]["selection"]["half_life"] in (5, 21)
    assert set(series.column_names) == {"split", "market", "instant", "variant", "metric", "value"}
    assert len(series) > 0



def study_section(**changes):
    section = {
        "asset": {
            "half_lives": [5, 21],
            "kappas": [0.05, 0.1, 0.2],
            "lab": {"half_life": 21, "kappa": 0.1},
        },
        "neighbours": {
            "window": 5, "k": 20, "tau": 0.2, "clip": 3.0, "kappas": [0.05, 0.1, 0.2, 0.4]
        },
        "combination": {"kappas": [0.0, 0.05, 0.1, 0.2]},
        "placebo_seed": 3,
    }
    section.update(changes)
    return section


def study_data(*, seed=0, effect=0.8, near=1.2, sessions=(30, 30, 50), assets=60):
    """Dos modelos, un efecto persistente por activo y otro que depende de una entrada."""
    rng = np.random.default_rng(seed)
    tables, maturity = synthetic(sessions=sessions, assets=assets, seed=seed, effect=effect)
    splits = splits_of(tables, maturity)
    rows = sum(split.rows for split in splits.values())
    latent = rng.normal(size=rows)
    noise = rng.normal(scale=0.3, size=(rows, 2))
    inputs = np.column_stack([latent, noise]).astype(np.float32)
    shift, offset = {}, 0
    for name in SPLITS:
        count = splits[name].rows
        shift[name] = pilots.replace_target(
            splits[name], splits[name].target + 0.01 * near * latent[offset : offset + count]
        )
        offset += count
    joined = np.concatenate([shift[name].prediction for name in SPLITS])
    predictions = {"a": joined, "b": joined / 3.0 + rng.normal(scale=0.002, size=rows)}
    keys = pilots.nrm.keys_from_inputs(inputs, pilots.nrm.projection(3, 4, seed=1))
    return shift, predictions, keys


def test_the_memory_study_finds_both_planted_effects_and_their_placebos_do_not():
    splits, predictions, keys = study_data()
    section = study_section()
    results, table = pilots.memory_study(splits, predictions, keys, section, log=lambda _: None)
    for model in ("a", "b"):
        contrasts = results[model]["contrasts"]
        assert contrasts["asset_selected-parent"]["mean"] > 0.02
        assert contrasts["asset_selected-parent"]["t"] > 3
        assert contrasts["neighbours_selected-parent"]["mean"] > 0.02
        assert contrasts["neighbours_selected-parent"]["t"] > 3
        assert contrasts["neighbours_placebo-parent"]["mean"] < 0.01
        assert contrasts["asset_placebo_selected-parent"]["mean"] < 0.01
        assert contrasts["combined-combined_asset_only"]["mean"] > 0
        assert contrasts["combined-combined_neighbours_only"]["mean"] > 0
    assert set(table.column_names) == {"model", "split", "market", "instant", "variant", "value"}


def test_the_memory_study_chooses_without_reading_the_evaluation():
    splits, predictions, keys = study_data(seed=1)
    section = study_section()
    first, _ = pilots.memory_study(splits, predictions, keys, section, log=lambda _: None)
    evaluation = splits["evaluation"]
    flipped = dict(splits, evaluation=pilots.replace_target(evaluation, -evaluation.target))
    again, _ = pilots.memory_study(flipped, predictions, keys, section, log=lambda _: None)
    for model in predictions:
        assert again[model]["selection"] == first[model]["selection"]
        assert again[model]["calibration"] == first[model]["calibration"]


def lab_cache(tmp_path, *, sessions=4, assets=5, seed=0):
    rng = np.random.default_rng(seed)
    cache = tmp_path / "cache"
    cache.mkdir()
    names = [f"US/A{i:03d}" for i in range(assets)]
    days = np.repeat(15000 + np.arange(sessions), assets).astype(np.int32)
    codes = np.tile(np.arange(assets), sessions).astype(np.int32)
    rows = sessions * assets
    np.save(cache / "X.npy", rng.normal(size=(rows, 46)).astype(np.float32))
    np.save(cache / "XR.npy", rng.normal(size=(rows, 46)).astype(np.float32))
    np.save(cache / "asset.npy", codes)
    np.save(cache / "day.npy", days)
    np.save(cache / "bounds.npy", np.arange(sessions + 1, dtype=np.int64) * assets)
    np.save(cache / "session_part.npy", np.array([0] + [1] * (sessions - 1), dtype=np.int8))
    (tmp_path / "names.json").write_text(json.dumps({"assets": names}))
    return cache, tmp_path / "names.json", names


def test_lab_inputs_follow_the_emitted_rows_and_name_a_missing_one(tmp_path):
    cache, names, assets = lab_cache(tmp_path)
    x, ranks = np.load(cache / "X.npy"), np.load(cache / "XR.npy")
    wanted = [(assets[3], 15002), (assets[0], 15001), (assets[4], 15003)]
    inputs, digests = pilots.lab_inputs(
        cache, names, np.array([a for a, _ in wanted]), np.array([d for _, d in wanted])
    )
    expected = [2 * 5 + 3, 1 * 5 + 0, 3 * 5 + 4]
    np.testing.assert_array_equal(inputs, np.hstack([x[expected], ranks[expected]]))
    assert inputs.dtype == np.float32 and "X.npy" in digests
    with pytest.raises(ValueError, match="US/A001"):
        pilots.lab_inputs(cache, names, np.array([assets[1]]), np.array([15000]))


def study_declaration(**changes):
    document = {
        "schema_version": 1,
        "kind": pilots.STUDY_KIND,
        "final_test_access": False,
        "description": "prueba",
        "market": "US",
        "min_assets": 30,
        "runs": [{"state": "v8", "origin": 2013, "seed": 42}],
        "models": ["titans_mac"],
        "inputs": {
            "source": "tabular_lab_cache_v1",
            "columns": 92,
            "projection_dim": 64,
            "projection_seed": 53,
        },
        **study_section(),
    }
    document.update(changes)
    return document


@pytest.mark.parametrize(
    "changes, message",
    [
        (dict(final_test_access=True), "test final"),
        (dict(combination={"kappas": [0.05, 0.1]}), "cero"),
        (
            dict(
                asset={"half_lives": [5], "kappas": [0.1], "lab": {"half_life": 21, "kappa": 0.1}}
            ),
            "rejilla",
        ),
    ],
)
def test_the_study_declaration_is_checked_before_reading_anything(changes, message):
    pilots.checked_study(study_declaration())
    with pytest.raises(ValueError, match=message):
        pilots.checked_study(study_declaration(**changes))
