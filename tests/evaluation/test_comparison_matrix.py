"""Matriz de comparaciones declarada y su evaluación con tablas por sesión sintéticas.

Las predicciones salen de los estudios sintéticos de la comparación walk-forward y de la
cartera, con valores conocidos. No hay modelos ajustados, datos de mercado ni pasos de
optimizador. Las pruebas de equivalencia comprueban que la matriz, al leer las tablas
publicadas, da exactamente los mismos contrastes que la comparación que las publicó.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from mars_titan.evaluation import comparison_matrix as matrix_module
from mars_titan.evaluation import long_short_comparison
from mars_titan.evaluation import session_table_contrasts as tables
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.evaluation.forecast_scores import score_sessions
from tests.evaluation.test_forecast_scores import random_panel
from tests.evaluation.test_walk_forward_comparison import Study

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs/evaluation/comparison-matrix-a.json"
REPORTS = ROOT / "reports/engineering/component-attribution-20261010"
HOURS = REPORTS / "arm-hours-a-v2-projected.json"
ALL_METRICS = list(walk.SERIES_METRICS)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))
    return sha256(path)


# Declaración de la campaña A


@pytest.fixture(scope="module")
def declared():
    return matrix_module.load_matrix(MATRIX)


def coefficients(declared, family, name):
    return declared["families"][family]["contrasts"][name]["coefficients"]


def test_the_declared_matrix_compiles_every_question_and_lineage(declared):
    summary = matrix_module.summary(declared)
    assert summary["families"] == 47 and summary["contrasts"] == 356
    families = declared["families"]
    assert len(families["between_families"]["contrasts"]) == 11 * 10 // 2
    assert len(families["mars_titan_vs_families__mars_titan_m1"]["contrasts"]) == 10
    assert len(families["mars_titan_vs_families__mars_titan_b6"]["contrasts"]) == 10
    assert set(families["staged_chain__gru"]["contrasts"]) == {
        "gru__chain-gru",
        "gru__chain-gru__frozen_parent",
        "gru__chain-gru__full_continuation",
        "gru__frozen_parent-gru",
        "gru__full_continuation-gru",
    }
    for entry in families.values():
        assert 1 <= len(entry["contrasts"]) <= 64
        for contrast in entry["contrasts"].values():
            # Toda la matriz compara brazos entre sí, así que los coeficientes de cada
            # contraste suman cero.
            assert sum(contrast["coefficients"].values()) == pytest.approx(0.0, abs=1e-12)
            assert contrast["unnamed"] == []


def test_the_titans_ladder_goes_from_the_compact_transformer_to_m3(declared):
    ladder = declared["families"]["titans__ladder"]["contrasts"]
    assert list(ladder) == [
        "+titans_encoder",
        "+mac_attention",
        "+memory_read",
        "+test_time_update",
        "+episodic_refiner",
        "+episodic_content",
        "+error_selection",
        "+anomaly_relevance",
        "total",
    ]
    assert coefficients(declared, "titans__ladder", "+test_time_update") == {
        "titans_mac_online": 1.0,
        "titans_mac_frozen": -1.0,
    }
    assert coefficients(declared, "titans__ladder", "total") == {
        "mars_titan_m3": 1.0,
        "transformer_compact": -1.0,
    }
    loo = declared["families"]["titans__leave_one_out"]["contrasts"]
    assert loo["-episodic_refiner"]["coefficients"] == {
        "mars_titan_m3": 1.0,
        "titans_mac_online": -1.0,
    }
    assert loo["-episodic_refiner"]["removes"] == [
        "episodic_refiner",
        "episodic_content",
        "error_selection",
        "anomaly_relevance",
    ]
    assert loo["-test_time_update"]["coefficients"] == {
        "mars_titan_m3": 1.0,
        "mars_titan_m3_frozen_core": -1.0,
    }
    effects = declared["families"]["titans__conditional_effects"]["contrasts"]
    assert effects["refinements_k4@mars_titan_m1"]["coefficients"] == {
        "mars_titan_m1_k4": 1.0,
        "mars_titan_m1": -1.0,
    }
    assert effects["codec_key@mars_titan_b6_bias"]["coefficients"] == {
        "mars_titan_b6": 1.0,
        "mars_titan_b6_bias": -1.0,
    }


def test_cm_v1_has_every_coalition_and_the_full_titans_game_is_a_limitation(declared):
    assert coefficients(declared, "cm_v1__shapley", "phi_contraction_penalty@cm_v1_b") == {
        "cm_v1_bc": 0.5,
        "cm_v1_b": -0.5,
        "cm_v1_bcm": 0.5,
        "cm_v1_bm": -0.5,
    }
    assert coefficients(
        declared, "cm_v1__interactions", "contraction_penaltyxanchored_consolidation@cm_v1_b"
    ) == {"cm_v1_bcm": 1.0, "cm_v1_bc": -1.0, "cm_v1_bm": -1.0, "cm_v1_b": 1.0}
    (limitation,) = declared["limitations"]
    assert limitation["lineage"] == "titans" and limitation["coalitions"] == 256
    assert limitation["structurally_invalid"] == 236


def test_the_missing_arms_report_is_reproducible_and_ranks_the_candidates(declared):
    hours = tables.load_hours(HOURS, "US+CN")
    report = matrix_module.missing_arms(declared, hours)
    assert report == json.loads((REPORTS / "missing-arms.json").read_text())
    totals = report["totals"]
    assert totals["contrasts"] == 356
    assert (
        totals["declared"]
        + totals["planned_elsewhere"]
        + totals["candidates"]
        + totals["unidentified"]
        == 356
    )
    first = report["candidates"][0]
    assert first["arm"] == "mars_titan_m2_k4" and first["tier"] == 1
    assert first["unlocked_alone"] == [
        "titans__conditional_effects/refinements_k4@mars_titan_m2",
        "titans__conditional_effects/error_selection@mars_titan_m1_k4",
        "titans__interactions/refinements_k4xerror_selection@mars_titan_m1",
    ]
    assert first["gpu_hours"] == hours["own"]["mars_titan_m1_k4"]
    tiers = [row["tier"] for row in report["candidates"]]
    assert tiers == sorted(tiers)
    frozen = next(row for row in report["candidates"] if row["arm"] == "mars_titan_m3_frozen_core")
    assert "titans__leave_one_out/-test_time_update" in frozen["unlocked_alone"]
    assert frozen["code"] == "requires_change" and frozen["tier"] == 2
    pair = next(b for b in report["bundles"] if "mars_titan_m1_frozen_core" in b["arms"])
    assert pair["arms"] == ["mars_titan_m0_frozen_core", "mars_titan_m1_frozen_core"]
    assert (
        "titans__interactions/test_time_updatexepisodic_content@mars_titan_m0_frozen_core"
        in (pair["contrasts"])
    )
    # El control en línea ya está declarado en la comparación, así que sus contrastes solo
    # esperan a otro brazo cuando también usan uno condicionado o derivado.
    assert "transformer_compact_online" not in report["conditional_arms"]
    assert (totals["declared"], totals["planned_elsewhere"]) == (196, 135)
    assert report["derived_arms"]["chain"]["contrasts"] == 3 * 22
    assert report["scientific_training_started"] is False


def edited(tmp_path, edit):
    declaration = json.loads(MATRIX.read_text())
    declaration["comparison"] = str(MATRIX.parent / declaration["comparison"])
    edit(declaration)
    path = tmp_path / "matrix.json"
    save(path, declaration)
    return path


def _set(path, value):
    def edit(declaration):
        target = declaration
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return edit


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_set(["status"], "draft"), "contrato"),
        (
            _set(["questions", "titans_reference_core", "pairs"], [["gru", "unknown_arm"]]),
            "sin declarar",
        ),
        (
            _set(["questions", "online_learning_control", "pairs"], [["{arm}", "gru"]]),
            "solo con each",
        ),
        (
            _set(["candidates", "mars_titan_m2_k4", "cost_like"], "mars_titan_m3_frozen_core"),
            "coste",
        ),
        (_set(["conditional_arms", "mars_titan_m2_k4"], dict(condition="x", issue=1)), "ya está"),
        (
            _set(["views", "forecast_calibrated", "metrics"], ["mae"]),
            "métricas por sesión",
        ),
        (
            _set(["views", "policies"], dict(source="session_metric_table", metrics=None, issue=1)),
            "pendiente",
        ),
        (
            _set(
                ["groups", "family_representatives"],
                [
                    "ridge",
                    "xgboost",
                    "rnn",
                    "lstm",
                    "gru",
                    "dlinear",
                    "transformer_compact",
                    "gru_episodic",
                    "titans_mac_online",
                    "mars_titan_m1",
                    "cm_v1_b",
                    "mars_titan_m3",
                ],
            ),
            "entre 1 y 64",
        ),
        (_set(["declared_at"], "mañana"), "fecha"),
        (
            _set(["questions", "titans_reference_core", "pairs"], [["gru", "unknown__chain"]]),
            "sin declarar",
        ),
        (
            _set(["questions", "titans_reference_core", "pairs"], [["gru", "gru__retrained"]]),
            "sin declarar",
        ),
    ],
)
def test_inconsistent_declarations_are_rejected(tmp_path, edit, message):
    with pytest.raises(ValueError, match=message):
        matrix_module.load_matrix(edited(tmp_path, edit))


def test_a_variant_listed_among_its_references_is_not_compared_with_itself(tmp_path):
    def edit(declaration):
        declaration["groups"]["references"].append("mars_titan_m1")

    families = matrix_module.load_matrix(edited(tmp_path, edit))["families"]
    assert len(families["mars_titan_vs_families__mars_titan_m1"]["contrasts"]) == 10
    assert len(families["mars_titan_vs_families__mars_titan_m0"]["contrasts"]) == 11
    assert (
        "mars_titan_m0-mars_titan_m1"
        in families["mars_titan_vs_families__mars_titan_m0"]["contrasts"]
    )


# Lectura de las tablas publicadas


def test_published_session_columns_give_the_same_series_as_the_scores():
    rng = np.random.default_rng(5)
    scores = score_sessions(random_panel(rng, rows=900, days=15))
    table = scores.to_table()
    for metric in ALL_METRICS:
        values, defined = tables.forecast_values(table, metric)
        series = scores.series(metric)
        assert np.array_equal(defined, series.defined), metric
        assert np.array_equal(np.where(defined, values, 0.0), series.values), metric
    point = score_sessions(random_panel(rng, quantiles=False)).to_table()
    assert tables.forecast_values(point, "pinball") is None
    assert tables.forecast_values(point, "sign_brier") is None
    assert tables.forecast_values(point, "interval_score@0.8") is None


# Equivalencia con la comparación walk-forward


def toy_matrix(root, comparison, **changes):
    declaration = dict(
        schema_version=1,
        kind="comparison_matrix",
        name="toy-matrix",
        status="declared_before_results",
        declared_at="2026-10-10",
        use=matrix_module.USE,
        comparison=str(comparison),
        conditional_arms=dict(gru_online=dict(condition="Control en línea", issue=1)),
        derived_arms=dict(chain=dict(condition="Cadena", issue=1)),
        groups=dict(models=["ridge", "gru", "titans"]),
        questions=dict(
            pair=dict(kind="pairs", pairs=[["gru", "titans"]], question="¿Titans mejora a GRU?"),
            versus_zero=dict(
                kind="pairs",
                pairs=[["zero", "ridge"], ["zero", "gru"], ["zero", "titans"]],
                question="¿Cada referencia mejora al control cero?",
            ),
            every=dict(kind="pairwise", arms="models", question="¿Quién mejora a quién?"),
            online=dict(
                kind="pairs", pairs=[["gru", "gru_online"]], question="¿Aprender en línea ayuda?"
            ),
            chain=dict(
                kind="pairs",
                each="models",
                pairs=[["{arm}", "{arm}__chain"]],
                question="¿La cadena mejora?",
            ),
        ),
        lineages=dict(
            core=dict(
                question="¿Cuánto aporta la memoria?",
                components=dict(
                    memory=dict(label="Memoria", requires=[]),
                    gate=dict(label="Puerta", requires=[]),
                ),
                arms=dict(gru=[], titans=["memory"]),
                ladder=["memory"],
                full=["memory"],
                interactions=[dict(first="memory", second="gate", context="gru")],
                games=[],
            )
        ),
        candidates={},
        views=dict(
            forecast=dict(
                source="walk_forward_comparison_report",
                quantiles="raw",
                metrics=[*ALL_METRICS, "sign_ece"],
            ),
            forecast_calibrated=dict(
                source="walk_forward_comparison_report",
                quantiles="calibrated",
                metrics=["pinball", "interval_score@0.8", "sign_ece"],
            ),
            portfolio=dict(
                source="long_short_comparison_report", statistics=["mean_net_return", "sharpe"]
            ),
            policies=dict(
                source="session_metric_table",
                metrics=dict(reward="gain", drawdown="loss"),
                issue=1,
            ),
            diagnostics=dict(
                source="session_metric_table", metrics=None, pending="Sin declarar", issue=1
            ),
        ),
    )
    declaration.update(changes)
    path = root / "matrix" / "matrix.json"
    save(path, declaration)
    return path


def sources(root, reports, hours=None, name="sources.json"):
    entries = []
    for kind, path, *view in reports:
        entry = dict(kind=kind, path=str(path), sha256=sha256(path))
        if view:
            entry["view"] = view[0]
        entries.append(entry)
    manifest = dict(
        schema_version=1,
        kind="comparison_matrix_sources",
        scope="US+CN",
        reports=entries,
        hours=None if hours is None else dict(path=str(hours), sha256=sha256(hours)),
    )
    path = root / "sources" / name
    save(path, manifest)
    return path


@pytest.fixture(scope="module")
def published(tmp_path_factory):
    root = tmp_path_factory.mktemp("matrix")
    study = Study(root / "study")
    study.config["comparison"]["metrics"] = ALL_METRICS
    study.publish()
    output = root / "walk"
    report = walk.write_walk_forward(study.config_path, study.sources_path, "US+CN", output)
    return root, study, report, output / "comparison.json"


@pytest.fixture(scope="module")
def evaluated(published):
    root, study, report, path = published
    matrix = toy_matrix(root, study.config_path)
    manifest = sources(root, [("walk_forward_comparison_report", path)])
    return matrix, manifest, matrix_module.evaluate(matrix, manifest, "US+CN")


def test_matrix_contrasts_equal_the_walk_forward_family_with_the_same_arms(published, evaluated):
    _, _, walk_report, _ = published
    _, _, report = evaluated
    assert report["final_test_opened"] is False and report["markets"] == ["US", "CN"]
    for view in ("US+CN", "US", "CN"):
        for metric in ALL_METRICS:
            for ours, theirs in (
                ("pair", "quantile_models"),
                ("versus_zero", "references_vs_zero"),
            ):
                expected = walk_report["contrasts"][view][theirs][metric]
                actual = report["views"]["forecast"][view][metric][ours]
                if "reason" in expected:
                    # El control cero no emite cuantiles, así que la familia entera queda
                    # sin estimar.
                    assert "cuantiles" in expected["reason"] and "reason" in actual, (view, metric)
                    continue
                assert actual["contrasts"] == expected["contrasts"], (view, metric, ours)
                assert actual["sensitivity"] == expected["sensitivity"], (view, metric, ours)
                assert actual["multiplicity"] == expected["multiplicity"], (view, metric, ours)


def test_contrasts_with_a_point_arm_are_not_applicable_to_quantile_metrics(evaluated):
    _, _, report = evaluated
    every = report["views"]["forecast"]["US+CN"]["pinball"]["every"]
    assert [row["name"] for row in every["contrasts"]] == ["titans-gru"]
    assert every["not_applicable"] == ["gru-ridge", "titans-ridge"]
    mae = report["views"]["forecast"]["US+CN"]["mae"]["every"]
    assert [row["name"] for row in mae["contrasts"]] == ["gru-ridge", "titans-ridge", "titans-gru"]
    assert mae["multiplicity"]["family_size"] == 3


def test_missing_arms_leave_their_contrasts_pending(evaluated):
    _, _, report = evaluated
    online = report["families"]["online"]
    assert online["estimable"] == [] and online["pending"] == [
        dict(name="gru_online-gru", missing=["gru_online"], unnamed=[])
    ]
    assert "online" not in report["views"]["forecast"]["US+CN"]["mae"]
    chain = report["families"]["chain__titans"]["pending"][0]
    assert chain["missing"] == ["titans__chain"]
    assert report["families"]["core__ladder"]["estimable"] == ["+memory", "total"]
    # Un conjunto sin brazo con nombre deja el contraste pendiente aunque haya coeficientes.
    assert report["families"]["core__interactions"] == dict(
        question="¿Cuánto aporta la memoria?",
        origin="lineage:core/interactions",
        estimable=[],
        pending=[
            dict(
                name="memoryxgate@gru",
                missing=[],
                unnamed=[["memory", "gate"], ["gate"]],
            )
        ],
    )
    assert report["views"]["diagnostics"] == dict(reason="Sin declarar", issue=1)


def test_the_ece_contrast_reuses_the_reliability_estimate_and_its_replicates(published, evaluated):
    _, _, walk_report, _ = published
    _, manifest, report = evaluated
    reliability = walk_report["sign_reliability"]["arms"]
    for view in ("US+CN", "US", "CN"):
        (row,) = report["views"]["forecast"][view]["sign_ece"]["pair"]["contrasts"]
        expected = reliability["titans"][view]["raw"]["estimate"]
        expected -= reliability["gru"][view]["raw"]["estimate"]
        assert row["estimate"] == pytest.approx(expected, abs=1e-15)
        assert row["interval"] is not None
    # Una familia con un solo nivel reproduce el intervalo percentil de la fiabilidad.
    matrix = matrix_module.load_matrix(evaluated[0])
    config = matrix["comparison_config"]
    loaded = tables.load_sources(manifest, matrix["views"], config, "US+CN")
    seeds = tables.forecast_arms(loaded)["raw"]["gru"]
    cells, _ = tables._ece_cells(seeds, None, "US+CN", ["US", "CN"])
    level = tables.ece_contrasts({"gru": cells}, {"gru": {"gru": 1.0}}, config["comparison"])
    (row,) = level["contrasts"]
    raw = reliability["gru"]["US+CN"]["raw"]
    assert row["estimate"] == pytest.approx(raw["estimate"], abs=1e-15)
    assert row["interval"] == pytest.approx(raw["interval"], abs=1e-15)


def test_calibrated_intervals_are_read_from_the_calibrated_sessions(evaluated):
    _, _, report = evaluated
    raw = report["views"]["forecast"]["US+CN"]["interval_score@0.8"]["pair"]["contrasts"][0]
    calibrated = report["views"]["forecast_calibrated"]["US+CN"]["interval_score@0.8"]["pair"]
    assert calibrated["contrasts"][0]["estimate"] is not None
    assert calibrated["contrasts"][0]["estimate"] != raw["estimate"]


def test_an_arm_must_match_across_reports_and_a_copy_changes_nothing(tmp_path, published):
    root, study, _, path = published
    matrix = toy_matrix(tmp_path, study.config_path)
    once = matrix_module.evaluate(
        matrix, sources(tmp_path, [("walk_forward_comparison_report", path)]), "US+CN"
    )
    twice = matrix_module.evaluate(
        matrix,
        sources(
            tmp_path,
            [("walk_forward_comparison_report", path)] * 2,
            name="twice.json",
        ),
        "US+CN",
    )
    assert twice["views"] == once["views"]
    other = Study(tmp_path / "other")
    other.config["comparison"]["metrics"] = ALL_METRICS

    def shift(arm, seed, fold, partition, values):
        if arm == "titans":
            values["prediction"] = values["prediction"] + 0.01

    other.changes.append(shift)
    other.publish()
    walk.write_walk_forward(other.config_path, other.sources_path, "US+CN", tmp_path / "w2")
    mixed = sources(
        tmp_path,
        [
            ("walk_forward_comparison_report", path),
            ("walk_forward_comparison_report", tmp_path / "w2" / "comparison.json"),
        ],
        name="mixed.json",
    )
    with pytest.raises(ValueError, match="difiere entre informes"):
        matrix_module.evaluate(matrix, mixed, "US+CN")


def test_a_tampered_session_table_is_rejected(tmp_path, published):
    _, study, _, path = published
    copy_dir = tmp_path / "walk"
    copy_dir.mkdir()
    for name in ("comparison.json", "sessions.parquet"):
        (copy_dir / name).write_bytes((path.parent / name).read_bytes())
    table = pq.read_table(copy_dir / "sessions.parquet")
    pq.write_table(table.slice(1), copy_dir / "sessions.parquet")
    manifest = sources(tmp_path, [("walk_forward_comparison_report", copy_dir / "comparison.json")])
    with pytest.raises(ValueError, match="intacta"):
        matrix_module.evaluate(toy_matrix(tmp_path, study.config_path), manifest, "US+CN")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("final_test_opened", True),
        ("status", "running"),
        ("scope", "US"),
        ("windows", "other"),
    ],
)
def test_reports_must_be_completed_sealed_and_share_their_views(tmp_path, published, field, value):
    _, study, report, path = published
    folder = tmp_path / "edited"
    folder.mkdir()
    (folder / "sessions.parquet").write_bytes((path.parent / "sessions.parquet").read_bytes())
    changed = copy.deepcopy(report)
    if field == "windows":
        changed["windows"]["fold-000"]["view_sha256"] = "0" * 64
    else:
        changed[field] = value
    save(folder / "comparison.json", changed)
    manifest = sources(
        tmp_path,
        [
            ("walk_forward_comparison_report", path),
            ("walk_forward_comparison_report", folder / "comparison.json"),
        ],
    )
    with pytest.raises(ValueError, match="completado|no comparte"):
        matrix_module.evaluate(toy_matrix(tmp_path, study.config_path), manifest, "US+CN")


def test_cost_adjusted_effects_divide_by_the_cumulative_gpu_hours(tmp_path, published):
    root, study, _, path = published
    hours = tmp_path / "hours.json"
    save(
        hours,
        dict(
            schema_version=1,
            kind="comparison_arm_hours",
            basis="measured",
            source="Horas de prueba",
            scope="US+CN",
            arms=dict(
                core=dict(gpu_hours=7.0, parents=[]),
                gru=dict(gpu_hours=10.0, parents=[]),
                titans=dict(gpu_hours=18.0, parents=["core"]),
            ),
        ),
    )
    manifest = sources(tmp_path, [("walk_forward_comparison_report", path)], hours=hours)
    report = matrix_module.evaluate(toy_matrix(tmp_path, study.config_path), manifest, "US+CN")
    assert report["hours"]["basis"] == "measured"
    (row,) = report["views"]["forecast"]["US+CN"]["mae"]["pair"]["contrasts"]
    cost = row["cost"]
    assert cost["gpu_hours_difference"] == pytest.approx(25.0 - 10.0)
    assert cost["improvement"] == -row["estimate"]
    assert cost["improvement_per_gpu_hour"] == pytest.approx(-row["estimate"] / 15.0)
    lower, upper = row["simultaneous_interval"]
    assert cost["improvement_per_gpu_hour_interval"] == pytest.approx(
        [-upper / 15.0, -lower / 15.0]
    )
    accuracy = report["views"]["forecast"]["US+CN"]["direction_accuracy"]["pair"]
    assert accuracy["contrasts"][0]["cost"]["improvement"] == accuracy["contrasts"][0]["estimate"]
    ridge = report["views"]["forecast"]["US+CN"]["mae"]["every"]["contrasts"][0]
    assert ridge["cost"]["reason"] == "Sin horas para ridge"


def policy_table(path, rows):
    pq.write_table(pa.table(rows), path)
    return path


def policy_rows(arms, seeds, metric, values, *, nulls=()):
    moments = [np.datetime64("2023-11-06T21:00", "us") + np.timedelta64(d, "D") for d in range(6)]
    rows = dict(arm=[], seed=[], market=[], prediction_at=[], metric=[], value=[])
    for arm in arms:
        for seed in seeds:
            for day, moment in enumerate(moments):
                rows["arm"].append(arm)
                rows["seed"].append(seed)
                rows["market"].append("US")
                rows["prediction_at"].append(moment)
                rows["metric"].append(metric)
                rows["value"].append(None if (arm, seed, day) in nulls else values(arm, seed, day))
    moments = np.array(rows["prediction_at"]).astype("datetime64[us]").astype(np.int64)
    rows["prediction_at"] = pa.array(moments).cast(pa.timestamp("us", tz="UTC"))
    rows["seed"] = pa.array(rows["seed"], pa.int64())
    rows["value"] = pa.array(rows["value"], pa.float64())
    return rows


def test_policy_tables_use_the_declared_orientation_and_the_same_resampling(tmp_path, published):
    _, study, _, path = published

    def reward(arm, seed, day):
        return (1.0 + 0.02 * (seed - 42) if arm == "titans" else 0.0) + 0.1 * day

    table = policy_table(
        tmp_path / "policies.parquet",
        policy_rows(["gru", "titans"], [42, 43], "reward", reward, nulls={("gru", 43, 2)}),
    )
    manifest = sources(
        tmp_path,
        [
            ("walk_forward_comparison_report", path),
            ("session_metric_table", table, "policies"),
        ],
    )
    report = matrix_module.evaluate(toy_matrix(tmp_path, study.config_path), manifest, "US+CN")
    (row,) = report["views"]["policies"]["US+CN"]["reward"]["pair"]["contrasts"]
    # La sesión 2 queda fuera porque una semilla de GRU no la define.
    assert row["estimate"] == pytest.approx(1.01) and row["sessions"] == 5
    assert report["views"]["policies"]["US+CN"]["drawdown"]["pair"]["reason"]
    negative = policy_table(
        tmp_path / "negative.parquet",
        policy_rows(["gru", "titans"], [42], "drawdown", lambda arm, seed, day: day - 3.0),
    )
    bad = sources(
        tmp_path,
        [("walk_forward_comparison_report", path), ("session_metric_table", negative, "policies")],
        name="bad.json",
    )
    with pytest.raises(ValueError, match="negativa"):
        matrix_module.evaluate(toy_matrix(tmp_path, study.config_path), bad, "US+CN")
    unknown = policy_table(
        tmp_path / "unknown.parquet",
        policy_rows(["gru"], [42], "sortino", lambda arm, seed, day: 1.0),
    )
    stray = sources(
        tmp_path,
        [("walk_forward_comparison_report", path), ("session_metric_table", unknown, "policies")],
        name="stray.json",
    )
    with pytest.raises(ValueError, match="no declaradas"):
        matrix_module.evaluate(toy_matrix(tmp_path, study.config_path), stray, "US+CN")
    undeclared = sources(
        tmp_path,
        [("walk_forward_comparison_report", path), ("session_metric_table", table, "diagnostics")],
        name="undeclared.json",
    )
    with pytest.raises(ValueError, match="sin métricas declaradas"):
        matrix_module.evaluate(toy_matrix(tmp_path, study.config_path), undeclared, "US+CN")


def test_hours_accumulate_each_ancestor_once_and_reject_cycles(tmp_path):
    document = dict(
        schema_version=1,
        kind="comparison_arm_hours",
        basis="projected",
        source="Prueba",
        scope="US",
        arms=dict(
            root=dict(gpu_hours=1.0, parents=[]),
            left=dict(gpu_hours=2.0, parents=["root"]),
            right=dict(gpu_hours=4.0, parents=["root"]),
            leaf=dict(gpu_hours=8.0, parents=["left", "right"]),
        ),
    )
    save(tmp_path / "hours.json", document)
    hours = tables.load_hours(tmp_path / "hours.json", "US")
    assert hours["cumulative"] == dict(root=1.0, left=3.0, right=5.0, leaf=15.0)
    with pytest.raises(ValueError, match="otro ámbito"):
        tables.load_hours(tmp_path / "hours.json", "CN")
    looped = copy.deepcopy(document)
    looped["arms"]["root"]["parents"] = ["leaf"]
    save(tmp_path / "looped.json", looped)
    with pytest.raises(ValueError, match="ciclo"):
        tables.load_hours(tmp_path / "looped.json", "US")


def test_cost_adjustment_needs_an_effect_more_hours_and_a_direction():
    hours = dict(basis="projected", cumulative=dict(a=2.0, b=6.0))
    row = dict(coefficients=dict(b=1.0, a=-1.0), estimate=-0.5, simultaneous_interval=[-0.9, -0.1])
    gain = tables.cost_adjusted(row, -1.0, hours)
    assert gain["improvement"] == 0.5 and gain["improvement_per_gpu_hour"] == 0.125
    assert gain["improvement_per_gpu_hour_interval"] == pytest.approx([0.025, 0.225])
    cheaper = tables.cost_adjusted(dict(row, coefficients=dict(a=1.0, b=-1.0)), -1.0, hours)
    assert cheaper["improvement_per_gpu_hour"] is None
    assert cheaper["gpu_hours_difference"] == -4.0 and "no consume" in cheaper["reason"]
    same = tables.cost_adjusted(row, -1.0, dict(hours, cumulative=dict(a=2.0, b=2.0)))
    assert same["gpu_hours_difference"] == 0.0 and same["improvement_per_gpu_hour"] is None
    level = tables.cost_adjusted(dict(row, coefficients=dict(a=1.0)), -1.0, hours)
    assert level["reason"] == "Es un nivel, no un efecto entre brazos"
    assert tables.cost_adjusted(row, None, hours)["reason"].startswith("La métrica no tiene")


# Equivalencia con la cartera larga y corta


@pytest.fixture(scope="module")
def portfolio(tmp_path_factory):
    from tests.evaluation.test_long_short_comparison import (
        PortfolioStudy,
        edition_assets,
        write_edition,
    )

    root = tmp_path_factory.mktemp("portfolio")
    write_edition(root / "edition", edition_assets())
    study = PortfolioStudy(root / "study", root / "edition")
    study.declare()
    output = root / "long-short"
    report = long_short_comparison.write_long_short(
        study.config_path, study.sources_path, "US+CN", root / "edition", output
    )
    return root, study, report, output / "long_short.json"


def test_portfolio_contrasts_equal_the_long_short_family(portfolio):
    root, study, long_short_report, path = portfolio
    matrix = toy_matrix(root, study.config_path)
    manifest = sources(root, [("long_short_comparison_report", path)])
    report = matrix_module.evaluate(matrix, manifest, "US+CN")
    for market in ("US", "CN"):
        for cost in ("0", "10"):
            for statistic in ("mean_net_return", "sharpe"):
                expected = long_short_report["views"][market]["contrasts"][cost][statistic]
                for ours, theirs in (
                    ("pair", "quantile_models"),
                    ("versus_zero", "references_vs_zero"),
                ):
                    actual = report["views"]["portfolio"][market][cost][statistic][ours]
                    assert actual["contrasts"] == expected[theirs]["contrasts"]
                    assert actual["critical_value"] == expected[theirs]["critical_value"]
    # Sin informes walk-forward ninguna familia tiene series de predicción.
    assert all("reason" in entry for entry in report["views"]["forecast"]["US+CN"]["mae"].values())
