"""Comparación walk-forward de la edición con máscaras sobre predicciones sintéticas.

Las predicciones se escriben en cada prueba con valores conocidos. No proceden de
ningún modelo ajustado ni de la edición real, y no se ejecuta ningún paso de
optimizador.
"""

import copy
import json
import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED, STRICT_INPUTS, policy_identity
from mars_titan.data.storage import sha256
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.forecast_scores import score_sessions
from mars_titan.evaluation.splits import build_folds
from mars_titan.models.quantile_head import LEVELS, QUANTILE_COLUMNS
from tests.evaluation.test_comparison_sources import masked_protocol, masked_view, save

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/evaluation/historical-masked-2000-comparison.json"
# Siete activos dan 21 filas por mercado y tramo, el mínimo para el orden del 95 %.
ASSETS = ("A", "B", "C", "D", "E", "F", "G")
OFFSETS = np.array([-1.96, -1.28, 0.0, 1.28, 1.96])
ARMS = dict(
    zero=dict(family="control", output="zero_control", seeds=[]),
    ridge=dict(family="reference", output="point", seeds=[42]),
    gru=dict(family="reference", output="quantile_head_v1", seeds=[42, 43]),
    titans=dict(family="titans_mac", output="quantile_head_v1", seeds=[42, 43]),
)


def decisions(month):
    """Tres instantes de decisión al cierre de US y de CN en el mes indicado de 2023."""
    first = datetime(2023, month, 6, tzinfo=UTC)
    return {
        market: [first + timedelta(days=day, hours=21 if market == "US" else 7) for day in range(3)]
        for market in ("US", "CN")
    }


def rows(markets, month, window):
    """Filas comunes de una ventana: activo, mercado, instante y objetivo conocido."""
    rng = np.random.default_rng(1000 * month + int(window[-3:]))
    keys = [
        (market, f"{market}/{asset}", moment)
        for market in markets
        for moment in decisions(month)[market]
        for asset in ASSETS
    ]
    target = np.round(rng.normal(0, 1, len(keys)), 2)
    return keys, target


class Study:
    """Configuración, vistas y predicciones sintéticas de un ámbito con dos ventanas."""

    def __init__(self, root, scope="US+CN", policy=HISTORICAL_MASKED):
        self.root, self.scope, self.policy = root, scope, policy
        self.markets = walk.SCOPES[scope]
        self.protocol = masked_protocol("US", "2023-09-01")
        self.folds = build_folds(self.protocol)
        assert [fold["evaluation"] for fold in self.folds] == [
            ["2023-11-01", "2023-12-01"],
            ["2023-12-01", "2024-01-01"],
        ]
        protocols = {}
        for market in self.markets:
            protocols[market] = f"protocol-{market}.json"
            save(root / "config" / protocols[market], dict(self.protocol, market=market))
        self.config = dict(
            schema_version=1,
            kind="walk_forward_comparison",
            name="fixture-2000",
            status="declared_before_evaluation",
            input_policy=policy,
            partition="evaluation",
            scopes={scope: dict(protocols=protocols, windows="all")},
            arms=copy.deepcopy(ARMS),
            metrics=dict(
                primary="mae",
                market_weighting="session",
                rank_ic_min_assets=3,
                quantile_head="quantile_head_v1",
            ),
            calibration=dict(
                method="cqr_symmetric_score_v1",
                partition="calibration",
                nominals=[0.8, 0.95],
                groups="market",
                min_rows=10,
                order_rule="widen_to_median_and_inner_interval",
            ),
            comparison=dict(
                metrics=["mae", "mse", "direction_accuracy", "rank_ic", "pinball"],
                block_length=1,
                sensitivity_block_lengths=[2],
                replicates=50,
                seed=7,
                confidence=0.95,
                families=dict(
                    references_vs_zero=dict(
                        kind="delta", base="zero", variants=["ridge", "gru", "titans"]
                    ),
                    quantile_models=dict(kind="delta", base="gru", variants=["titans"]),
                    levels=dict(kind="level", arms=["ridge", "gru", "titans"]),
                ),
            ),
        )
        self.changes = []
        self.publish()

    @property
    def config_path(self):
        return self.root / "config" / "comparison.json"

    @property
    def sources_path(self):
        return self.root / "sources" / "sources.json"

    def view(self, fold):
        views = {
            market: masked_view(dict(self.protocol, market=market), fold) for market in self.markets
        }
        manifest = dict(
            kind="corpus_supervision",
            cohort_complete=True,
            final_test_opened=False,
            **policy_identity(self.policy),
        )
        if len(self.markets) == 1:
            manifest["temporal_view"] = views[self.markets[0]]
        else:
            manifest.update(
                temporal_views=views,
                markets=list(self.markets),
                assets=[dict(market=market) for market in self.markets],
            )
        return manifest

    def table(self, arm, seed, fold, partition):
        """Predicciones de un brazo: mitad del objetivo más ruido, cuantiles estrechos."""
        month = int(fold[partition][0][5:7])
        keys, target = rows(self.markets, month, fold["id"])
        rng = np.random.default_rng(zlib.crc32(f"{arm}/{seed}/{fold['id']}/{partition}".encode()))
        prediction = np.round(0.5 * target + rng.normal(0, 0.5, len(target)), 3)
        values = dict(
            sample_id=[f"{asset}/{moment.isoformat()}" for _, asset, moment in keys],
            asset_id=[asset for _, asset, _ in keys],
            market=[market for market, _, _ in keys],
            prediction_at=pa.array([m for _, _, m in keys], pa.timestamp("us", tz="UTC")),
            target=target,
            prediction=prediction,
        )
        if ARMS[arm]["output"] == "quantile_head_v1":
            # Intervalos sobreconfiados: escala 0,2 frente a errores de desviación cercana a 0,7.
            quantiles = prediction[:, None] + 0.2 * OFFSETS
            values.update(zip(QUANTILE_COLUMNS, quantiles.T, strict=True))
        for change in self.changes:
            change(arm, seed, fold["id"], partition, values)
        return pa.table(values)

    def publish(self):
        save(self.config_path, self.config)
        windows, arms = {}, {}
        for fold in self.folds:
            path = self.root / "views" / f"{fold['id']}.json"
            windows[fold["id"]] = dict(
                view=dict(path=str(path), sha256=save(path, self.view(fold)))
            )
        for arm, declared in ARMS.items():
            if declared["output"] == "zero_control":
                continue
            arms[arm] = {}
            for seed in declared["seeds"]:
                arms[arm][str(seed)] = {}
                for fold in self.folds:
                    entry = dict(
                        input_policy=self.policy,
                        view_sha256=windows[fold["id"]]["view"]["sha256"],
                    )
                    parts = ["evaluation"]
                    if declared["output"] == "quantile_head_v1":
                        parts.append("calibration")
                    for part in parts:
                        relative = f"{arm}/{seed}/{fold['id']}/{part}-predictions.parquet"
                        path = self.sources_path.parent / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        pq.write_table(self.table(arm, seed, fold, part), path)
                        entry[part] = dict(path=relative, sha256=sha256(path))
                    arms[arm][str(seed)][fold["id"]] = entry
        self.sources = dict(
            schema_version=1,
            kind="walk_forward_prediction_sources",
            scope=self.scope,
            input_policy=self.policy,
            windows=windows,
            arms=arms,
        )
        save(self.sources_path, self.sources)

    def run(self):
        return walk.evaluate_walk_forward(self.config_path, self.sources_path, self.scope)


@pytest.fixture(scope="module")
def joint(tmp_path_factory):
    study = Study(tmp_path_factory.mktemp("joint"))
    return study, *study.run()


def by_hand_session_mae(study, arm, seed):
    """MAE por sesión calculado sin el panel: media por (mercado, instante) y media simple."""
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
    means = [sum(errors) / len(errors) for errors in sessions.values()]
    return sum(means) / len(means), len(sessions)


def test_reports_windows_markets_and_the_pooled_session_mae_by_hand(joint):
    study, report, sessions = joint
    assert report["final_test_opened"] is False
    assert report["input_policy"] == HISTORICAL_MASKED and report["mask_contract"]["version"] == 1
    assert list(report["windows"]) == ["fold-000", "fold-001"]
    gru = report["arms"]["gru"]["seeds"]["42"]
    assert list(gru["overall"]) == ["US+CN", "US", "CN"]
    expected, count = by_hand_session_mae(study, "gru", 42)
    point = gru["overall"]["US+CN"]["summary"]["point"]
    assert point["mae"] == pytest.approx(expected, rel=1e-12)
    assert gru["overall"]["US+CN"]["summary"]["sessions"] == count == 12
    assert gru["overall"]["US"]["summary"]["sessions"] == 6
    for window in ("fold-000", "fold-001"):
        assert gru["windows"][window]["summary"]["sessions"] == 6
        assert gru["windows"][window]["summary"]["rows"] == 42
    assert (
        report["arms"]["zero"]["seeds"]["deterministic"]["overall"]["US+CN"]["summary"]["point"][
            "direction_accuracy"
        ]
        == 0.0
    )
    # Por sesión: 12 sesiones por brazo y semilla, en bruto y calibradas para los cuantiles.
    assert sessions.num_rows == 12 * (1 + 1 + 2 * 2 + 2 * 2)
    assert set(sessions["quantiles"].to_pylist()) == {"raw", "calibrated"}


def test_pooled_windows_equal_one_panel_with_all_evaluation_rows(joint):
    study, report, _ = joint
    tables = [study.table("titans", 43, fold, "evaluation") for fold in study.folds]
    table = pa.concat_tables(tables)
    panel = ForecastPanel.from_columns(
        [
            f"{m}/{a}/{t}"
            for m, a, t in zip(
                table["market"].to_pylist(),
                table["asset_id"].to_pylist(),
                table["prediction_at"].cast(pa.int64()).to_pylist(),
                strict=True,
            )
        ],
        table["market"].to_numpy(zero_copy_only=False),
        table["prediction_at"].cast(pa.int64()).to_numpy(),
        table["target"].to_numpy(),
        table["prediction"].to_numpy(),
        quantiles=np.column_stack([table[c].to_numpy() for c in QUANTILE_COLUMNS]),
        levels=LEVELS,
    )
    expected = score_sessions(panel, rank_ic_min_assets=3).summary()
    actual = report["arms"]["titans"]["seeds"]["43"]["overall"]["US+CN"]["summary"]
    for key in ("point", "by_market", "direction", "rank_ic", "quantiles", "rows", "sessions"):
        assert actual[key] == expected[key], key


def test_point_and_zero_arms_keep_interval_metrics_absent(joint):
    _, report, _ = joint
    for arm, seed in (("ridge", "42"), ("zero", "deterministic")):
        summary = report["arms"][arm]["seeds"][seed]["overall"]["US+CN"]["summary"]
        assert summary["quantiles"] is None and summary["quantiles_reason"]
        assert report["arms"][arm]["seeds"][seed]["windows"]["fold-000"]["calibrator"] is None
    family = report["contrasts"]["US+CN"]["references_vs_zero"]
    assert "cuantiles" in family["pinball"]["reason"]
    pinball = report["contrasts"]["US+CN"]["quantile_models"]["pinball"]
    assert pinball["contrasts"][0]["name"] == "titans-gru"
    assert "ridge" not in report["interval_calibration"]


def test_contrasts_average_seeds_and_use_the_declared_resampling(joint):
    study, report, _ = joint
    mae = report["contrasts"]["US+CN"]["references_vs_zero"]["mae"]
    assert mae["resampling"]["seed"] == 7 and mae["resampling"]["block_length"] == 1
    assert mae["resampling"]["periods"] == 6  # seis días UTC con sesiones US y CN
    row = next(row for row in mae["contrasts"] if row["name"] == "gru-zero")
    gru = np.mean([by_hand_session_mae(study, "gru", seed)[0] for seed in (42, 43)])
    zero = report["arms"]["zero"]["seeds"]["deterministic"]["overall"]["US+CN"]["summary"]
    assert row["estimate"] == pytest.approx(gru - zero["point"]["mae"], rel=1e-12)
    assert row["interval"] is not None and row["simultaneous_interval"] is not None
    rank = report["contrasts"]["US+CN"]["references_vs_zero"]["rank_ic"]
    # El control cero es constante: su Rank IC no está definido y el contraste tampoco.
    assert all(row["estimate"] is None for row in rank["contrasts"])


def test_calibration_widens_overconfident_intervals_and_reports_both(joint):
    _, report, _ = joint
    calibration = report["interval_calibration"]["gru"]["US+CN"]
    for nominal in ("0.8", "0.95"):
        entry = calibration[nominal]
        assert entry["raw_coverage_error"] < -0.3
        assert abs(entry["calibrated_coverage_error"]) < abs(entry["raw_coverage_error"])
        assert entry["raw_status"] in {"undercovers", "inconclusive"}
    window = report["arms"]["gru"]["seeds"]["42"]["windows"]["fold-000"]
    record = window["calibrator"]["record"]
    assert record["partition"] == "calibration" and record["coverage_guaranteed"] is False
    assert set(record["groups"]) == {"US", "CN"}
    assert window["calibrator"]["sha256"] == walk.cqr.calibrator_sha256(record)
    raw = window["summary"]["quantiles"]["intervals"][0]
    calibrated = window["calibrated"]["quantiles"]["intervals"][0]
    assert calibrated["width"] > raw["width"]
    # La mediana no cambia, así que el MAE calibrado es el mismo.
    assert window["calibrated"]["point"] == window["summary"]["point"]


def test_calibrator_reads_calibration_before_evaluation_and_ignores_its_targets(
    tmp_path, monkeypatch
):
    study = Study(tmp_path / "a", scope="US")
    reads = []
    original = walk._read_predictions

    def spy(file, columns):
        reads.append(Path(file["path"]).name)
        return original(file, columns)

    monkeypatch.setattr(walk, "_read_predictions", spy)
    report, _ = study.run()
    for index, name in enumerate(reads):
        if name.startswith("calibration"):
            assert reads[index + 1].startswith("evaluation")
    changed = Study(tmp_path / "b", scope="US")
    changed.changes.append(
        lambda arm, seed, fold, part, values: (
            values.update(target=values["target"] * 3) if part == "evaluation" else None
        )
    )
    changed.publish()
    other, _ = changed.run()
    for arm in ("gru", "titans"):
        for seed in ("42", "43"):
            for window in ("fold-000", "fold-001"):
                first = report["arms"][arm]["seeds"][seed]["windows"][window]["calibrator"]
                second = other["arms"][arm]["seeds"][seed]["windows"][window]["calibrator"]
                assert first == second


def test_calibration_rows_from_the_evaluation_segment_are_rejected(tmp_path):
    study = Study(tmp_path, scope="US")

    def leak(arm, seed, fold, part, values):
        if arm == "titans" and part == "calibration" and fold == "fold-000":
            values["prediction_at"] = pa.array(
                [datetime(2023, 11, 7, 21, tzinfo=UTC)] + values["prediction_at"].to_pylist()[1:],
                pa.timestamp("us", tz="UTC"),
            )

    study.changes.append(leak)
    study.publish()
    with pytest.raises(ValueError, match="1 filas fuera del tramo de calibration"):
        study.run()


def test_rows_outside_the_evaluation_segment_or_in_2024_are_rejected(tmp_path):
    study = Study(tmp_path, scope="US")

    def move(day):
        def change(arm, seed, fold, part, values):
            if arm == "ridge" and part == "evaluation" and fold == "fold-001":
                moments = values["prediction_at"].to_pylist()
                moments[-1] = day
                values["prediction_at"] = pa.array(moments, pa.timestamp("us", tz="UTC"))

        return change

    study.changes.append(move(datetime(2024, 1, 2, 21, tzinfo=UTC)))
    study.publish()
    with pytest.raises(ValueError, match="1 filas pertenecen a la reserva final de 2024"):
        study.run()
    study.changes[:] = [move(datetime(2023, 11, 30, 21, tzinfo=UTC))]
    study.publish()
    with pytest.raises(ValueError, match="1 filas fuera del tramo de evaluation"):
        study.run()


def test_declared_windows_never_reach_the_sealed_year():
    config = walk.load_config(CONFIG)
    for scope in config["resolved_scopes"].values():
        assert all(window["evaluation"][1] <= "2024-01-01" for window in scope["windows"].values())


@pytest.mark.parametrize(
    "change,message",
    [
        (
            lambda a, s, f, p, v: v.update(target=v["target"] + (a == "titans")),
            "0 filas solo en este brazo, 0 filas solo en la referencia y 21 filas con otro",
        ),
        (
            lambda a, s, f, p, v: (
                [v.__setitem__(k, v[k][:-2]) for k in list(v)]
                if a == "gru" and s == 43 and p == "evaluation" and f == "fold-001"
                else None
            ),
            "0 filas solo en este brazo, 2 filas solo en la referencia y 0 filas",
        ),
        (
            lambda a, s, f, p, v: (
                v.__setitem__("target", np.r_[v["target"][:-3], 9, 9, 9])
                if a == "ridge" and p == "evaluation"
                else None
            ),
            "0 filas solo en la referencia y 3 filas con otro objetivo",
        ),
    ],
    ids=["other_targets", "missing_rows", "changed_targets"],
)
def test_arms_must_share_rows_and_targets(tmp_path, change, message):
    study = Study(tmp_path, scope="US")
    study.changes.append(change)
    study.publish()
    with pytest.raises(ValueError, match="no evalúa las mismas filas") as error:
        study.run()
    assert message in str(error.value)


def rewrite_sources(study, edit):
    sources = json.loads(study.sources_path.read_text())
    edit(sources)
    save(study.sources_path, sources)


@pytest.mark.parametrize(
    "edit,message",
    [
        (lambda s: s.update(input_policy=STRICT_INPUTS), "otra política de entradas"),
        (
            lambda s: s["arms"]["gru"]["43"]["fold-001"].update(input_policy=STRICT_INPUTS),
            "gru semilla 43 en fold-001 declara otra política",
        ),
        (
            lambda s: s["arms"]["gru"]["43"]["fold-001"].update(view_sha256="a" * 64),
            "usa otra vista",
        ),
        (lambda s: s["arms"]["titans"]["42"].pop("fold-001"), r"faltan \['fold-001'\]"),
        (lambda s: s["arms"]["titans"].pop("43"), "semillas declaradas"),
        (lambda s: s["arms"].pop("ridge"), "brazos declarados"),
        (
            lambda s: s["arms"]["ridge"]["42"]["fold-000"].update(
                calibration=s["arms"]["ridge"]["42"]["fold-000"]["evaluation"]
            ),
            "política, vista y predicciones",
        ),
    ],
)
def test_sources_that_mix_policies_views_windows_or_arms_are_rejected(tmp_path, edit, message):
    study = Study(tmp_path, scope="US")
    rewrite_sources(study, edit)
    with pytest.raises(ValueError, match=message):
        study.run()


@pytest.mark.parametrize(
    "defect,message",
    [
        ("strict_view", "política de entradas"),
        ("other_protocol", "protocolo o la ventana"),
        ("other_edition", "misma edición"),
    ],
)
def test_views_must_follow_the_policy_protocol_and_edition(tmp_path, defect, message):
    study = Study(tmp_path, scope="US")
    path = study.root / "views" / "fold-001.json"
    view = json.loads(path.read_text())
    if defect == "strict_view":
        for key in ("input_policy", "mask_contract"):
            view.pop(key)
    elif defect == "other_protocol":
        view["temporal_view"]["protocol"]["selection"]["patience"] = 6
    else:
        view["temporal_view"]["parent_sha256"] = "f" * 64
    digest = save(path, view)

    def edit(sources):
        sources["windows"]["fold-001"]["view"]["sha256"] = digest
        for seeds in sources["arms"].values():
            for windows in seeds.values():
                windows["fold-001"]["view_sha256"] = digest

    rewrite_sources(study, edit)
    with pytest.raises(ValueError, match=message):
        study.run()


def test_cli_writes_a_new_output_with_the_session_table(tmp_path, capsys):
    study = Study(tmp_path / "study", scope="US")
    output = tmp_path / "result"
    arguments = ["--config", str(study.config_path), "--sources", str(study.sources_path)]
    assert walk.main([*arguments, "--scope", "US", "--output", str(output)]) == 0
    assert "Reserva final cerrada" in capsys.readouterr().out
    report = json.loads((output / "comparison.json").read_text())
    assert report["artifacts"]["sessions.parquet"] == sha256(output / "sessions.parquet")
    assert pq.read_table(output / "sessions.parquet").num_rows == 6 * (1 + 1 + 2 * 2 + 2 * 2)
    with pytest.raises(ValueError, match="nueva"):
        walk.main([*arguments, "--scope", "US", "--output", str(output)])
    with pytest.raises(ValueError, match="fuera"):
        walk.main([*arguments, "--scope", "US", "--output", str(study.sources_path.parent / "x")])


def test_declared_campaign_configuration_covers_the_2000_protocol():
    config = walk.load_config(CONFIG)
    assert config["input_policy"] == HISTORICAL_MASKED
    scopes = config["resolved_scopes"]
    first = {name: scope["windows"]["fold-000"]["evaluation"] for name, scope in scopes.items()}
    assert {name: len(scope["windows"]) for name, scope in scopes.items()} == {
        "US": 19,
        "CN": 13,
        "US+CN": 13,
    }
    assert first == {
        "US": ["2005-01-01", "2006-01-01"],
        "CN": ["2011-01-01", "2012-01-01"],
        "US+CN": ["2011-01-01", "2012-01-01"],
    }
    assert config["metrics"]["primary"] == "mae"
    assert config["calibration"]["nominals"] == [0.8, 0.95]
    outputs = {arm["output"] for arm in config["arms"].values()}
    assert outputs == {"zero_control", "point", "quantile_head_v1"}
    assert config["comparison"]["seed"] == 20261009


def test_insufficient_calibration_leaves_calibrated_metrics_absent_with_reason(tmp_path):
    study = Study(tmp_path, scope="US")
    study.config["calibration"]["min_rows"] = 22
    study.publish()
    report, sessions = study.run()
    gru = report["arms"]["gru"]["seeds"]["42"]
    assert gru["missing_calibration_windows"] == ["fold-000", "fold-001"]
    window = gru["windows"]["fold-000"]
    assert window["calibrated"] is None and "US" in window["calibrated_reason"]
    entry = window["calibrator"]["record"]["groups"]["US"]["intervals"]["0.8"]
    assert entry["correction"] is None and "mínimo" in entry["reason"]
    assert gru["overall"]["US"]["calibrated"] is None
    calibration = report["interval_calibration"]["gru"]["US"]["0.8"]
    assert calibration["calibrated_coverage_error"] is None and calibration["calibrated_reason"]
    assert calibration["raw_coverage_error"] is not None
    assert set(sessions["quantiles"].to_pylist()) == {"raw"}


def test_session_table_keeps_window_market_and_metric_columns(joint):
    _, _, sessions = joint
    gru = sessions.filter(pc.and_(pc.equal(sessions["arm"], "gru"), pc.equal(sessions["seed"], 42)))
    raw = gru.filter(pc.equal(gru["quantiles"], "raw"))
    assert raw.num_rows == 12
    assert sorted(set(raw["window"].to_pylist())) == ["fold-000", "fold-001"]
    assert sorted(set(raw["market"].to_pylist())) == ["CN", "US"]
    assert set(raw["samples"].to_pylist()) == {7}
    zero = sessions.filter(pc.equal(sessions["arm"], "zero"))
    assert zero["seed"].null_count == zero.num_rows
    assert zero["coverage_0.8"].null_count == zero.num_rows
