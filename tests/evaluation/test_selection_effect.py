"""Efecto de elegir entre los casos de búsqueda con predicciones escritas en cada prueba.

Las predicciones y los recibos son dobles con valores conocidos. Ningún modelo se ajusta,
no se ejecutan pasos de optimizador y no se abre la reserva final.
"""

import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from mars_titan.evaluation import selection_effect as effect
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.training import masked_campaign as engine
from tests.evaluation.test_walk_forward_comparison import Study

ROOT = Path(__file__).resolve().parents[2]
DECLARATION = ROOT / "configs/evaluation/historical-masked-2000-selection-effect.json"


def test_the_chosen_case_is_placed_among_the_evaluation_maes():
    scores = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    best, worst = effect.window_effect(scores, "a"), effect.window_effect(scores, "d")
    assert best["rank"] == 4 and best["relative_rank"] == pytest.approx(0.8)
    assert best["logit"] == pytest.approx(math.log(4))
    assert best["gain_over_median"] == pytest.approx(1.5)
    assert best["gap_to_best_in_hindsight"] == 0
    assert worst["rank"] == 1 and worst["logit"] == pytest.approx(-best["logit"])
    assert worst["gain_over_median"] == pytest.approx(-1.5)
    assert worst["gap_to_best_in_hindsight"] == pytest.approx(3.0)


def test_the_median_case_and_ties_share_their_rank():
    middle = effect.window_effect({"a": 1.0, "b": 2.0, "c": 3.0}, "b")
    assert middle["rank"] == 2 and middle["logit"] == 0
    tied = effect.window_effect({"a": 1.0, "b": 1.0, "c": 2.0}, "a")
    assert tied["rank"] == 2.5 and tied["relative_rank"] == pytest.approx(2.5 / 4)


def test_a_single_case_has_no_choice_and_bad_inputs_are_rejected():
    assert effect.window_effect({"a": 1.0}, "a") == dict(
        cases=1, applicable=False, reason="Un solo caso: no hay elección"
    )
    with pytest.raises(ValueError, match="no finito"):
        effect.window_effect({"a": 1.0, "b": float("nan")}, "a")
    with pytest.raises(ValueError, match="no está entre"):
        effect.window_effect({"a": 1.0, "b": 2.0}, "c")


def test_the_summary_counts_windows_at_or_below_the_median():
    windows = {
        "w1": effect.window_effect({"a": 1.0, "b": 2.0, "c": 3.0}, "a"),
        "w2": effect.window_effect({"a": 1.0, "b": 2.0, "c": 3.0}, "b"),
        "w3": effect.window_effect({"a": 1.0, "b": 2.0, "c": 3.0}, "c"),
        "w4": effect.window_effect({"a": 1.0}, "a"),
    }
    summary = effect.arm_summary(windows)
    assert summary["windows"] == 3 and summary["at_or_below_median"] == 2
    assert summary["share_at_or_below_median"] == pytest.approx(2 / 3)
    assert summary["median_logit"] == 0
    assert summary["mean_gain_over_median"] == pytest.approx(0.0)
    assert summary["mean_gap_to_best_in_hindsight"] == pytest.approx(1.0)
    assert effect.arm_summary({"w4": windows["w4"]})["windows"] == 0


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.update(minimum_cases=3),
        lambda d: d.update(selection="min_validation_session_mae"),
        lambda d: d.update(final_test_opened=True),
        lambda d: d["not_applied"].update(cscv_pbo=""),
        lambda d: d.pop("not_applied"),
    ],
)
def test_the_declaration_must_describe_what_is_measured(tmp_path, edit):
    declaration, _ = effect.load_declaration(DECLARATION)
    assert declaration["status"] == "declared_not_executed"
    edit(declaration)
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(declaration))
    with pytest.raises(ValueError):
        effect.load_declaration(path)


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    study = Study(tmp_path_factory.mktemp("selection"), scope="US")
    return study, walk.load_config(study.config_path)


def write_case(study, folder, name, fold, arm="ridge", shift=None):
    """Predicciones de evaluación de un caso. Con `shift`, el objetivo desplazado."""
    table = study.table(arm, 42, fold, "evaluation")
    if shift is not None:
        target = table["target"].to_numpy()
        table = table.set_column(
            table.schema.get_field_index("prediction"), "prediction", pa.array(target + shift)
        )
    path = folder / f"{name}.parquet"
    pq.write_table(table, path)
    return dict(score=None, evaluation=dict(path=path, sha256=sha256(path)))


def test_each_case_is_scored_like_the_comparison(study, tmp_path):
    study, config = study
    fold = study.folds[0]
    cases = {
        "same": write_case(study, tmp_path, "same", fold),
        "quantile": write_case(study, tmp_path, "quantile", fold, arm="gru"),
    }
    report, _ = walk.evaluate_walk_forward(study.config_path, study.sources_path, "US")
    expected = report["arms"]["ridge"]["seeds"]["42"]["windows"][fold["id"]]["summary"]
    scores = effect.case_scores(config, "US", fold["id"], "ridge", {"same": cases["same"]})
    assert scores["same"] == expected["point"]["mae"]
    # Un modelo con cuantiles se puntúa con su predicción central.
    gru = effect.case_scores(config, "US", fold["id"], "gru", {"quantile": cases["quantile"]})
    expected = report["arms"]["gru"]["seeds"]["42"]["windows"][fold["id"]]["summary"]
    assert gru["quantile"] == expected["point"]["mae"]


def two_windows(study, folder):
    """Casos de dos ventanas: b gana la validación, evalúa peor en la primera y mejor después."""
    records = {}
    for fold in study.folds:
        shifts = dict(a=0.2, b=0.3, c=0.1) if fold is study.folds[0] else dict(a=0.2, b=0.05)
        (folder / fold["id"]).mkdir(parents=True)
        cases = {
            key: write_case(study, folder / fold["id"], key, fold, shift=s)
            for key, s in shifts.items()
        }
        for key, case in cases.items():
            case["score"] = 0.1 if key == "b" else 0.5
        records[fold["id"]] = dict(ridge=dict(selected="b", cases=cases))
    return records


def test_the_report_places_the_validation_winner_in_each_window(study, tmp_path):
    study, config = study
    records = two_windows(study, tmp_path)
    windows = {w: effect.window_effects(config, "US", w, models) for w, models in records.items()}
    declaration, digest = effect.load_declaration(DECLARATION)
    report = effect.selection_report("US", windows, declaration, digest)
    first, second = (report["arms"]["ridge"]["windows"][f["id"]] for f in study.folds)
    assert first["evaluation"] == pytest.approx(dict(a=0.2, b=0.3, c=0.1))
    assert first["rank"] == 1 and first["logit"] < 0
    assert second["rank"] == 2 and second["logit"] > 0
    assert first["validation"] == dict(a=0.5, b=0.1, c=0.5)
    assert (
        first["predictions"]["b"]
        == records[study.folds[0]["id"]]["ridge"]["cases"]["b"]["evaluation"]["sha256"]
    )
    assert report["arms"]["ridge"]["summary"]["at_or_below_median"] == 1
    assert report["final_test_opened"] is False and report["declaration_sha256"] == digest
    assert report["excluded"] == {}


def test_a_model_with_searches_outside_the_comparison_is_excluded_with_its_reason(study, tmp_path):
    # En A v2, las variantes de CM-v1 tienen búsquedas pero la comparación no las declara.
    study, config = study
    records = two_windows(study, tmp_path)
    for models in records.values():
        models["cm_v1_core_b"] = models["ridge"]
    assert "cm_v1_core_b" not in config["resolved_scopes"]["US"]["arms"]
    windows = {w: effect.window_effects(config, "US", w, models) for w, models in records.items()}
    for found in windows.values():
        assert list(found["arms"]) == ["ridge"]
        assert found["excluded"] == {"cm_v1_core_b": effect.NOT_DECLARED}
    declaration, digest = effect.load_declaration(DECLARATION)
    report = effect.selection_report("US", windows, declaration, digest)
    assert list(report["arms"]) == ["ridge"]
    assert report["excluded"] == {
        "cm_v1_core_b": dict(reason=effect.NOT_DECLARED, windows=[f["id"] for f in study.folds])
    }


def test_a_saved_window_is_read_back_only_with_its_comparison_and_code(
    study, tmp_path, monkeypatch
):
    study, config = study
    fold = study.folds[0]["id"]
    models = two_windows(study, tmp_path / "cases")[fold]
    record = effect.write_window(tmp_path / "aggregates", config, "US", fold, models)
    assert Path(record["path"]) == tmp_path / "aggregates" / "US" / f"{fold}.selection.json"
    assert record["sha256"] == sha256(Path(record["path"]))
    saved = effect.read_window(tmp_path / "aggregates", config, "US", fold)
    assert saved == json.loads(json.dumps(effect.window_effects(config, "US", fold, models)))
    # Una ventana sin búsquedas también deja su registro, vacío.
    effect.write_window(tmp_path / "aggregates", config, "US", "otra", {})
    assert effect.read_window(tmp_path / "aggregates", config, "US", "otra") == dict(
        arms={}, excluded={}
    )
    with pytest.raises(ValueError, match="no procede"):
        effect.read_window(tmp_path / "aggregates", dict(config, sha256="0" * 64), "US", fold)
    monkeypatch.setattr(effect, "CODE", effect.CODE[:1])
    with pytest.raises(ValueError, match="no procede"):
        effect.read_window(tmp_path / "aggregates", config, "US", fold)


def test_cases_with_other_rows_or_targets_are_rejected(study, tmp_path):
    study, config = study
    fold = study.folds[0]
    cases = {"a": write_case(study, tmp_path, "a", fold)}
    table = study.table("ridge", 42, fold, "evaluation")
    target = table["target"].to_numpy()
    changed = table.set_column(
        table.schema.get_field_index("target"), "target", pa.array(target + 1)
    )
    pq.write_table(changed, tmp_path / "b.parquet")
    cases["b"] = dict(
        evaluation=dict(path=tmp_path / "b.parquet", sha256=sha256(tmp_path / "b.parquet"))
    )
    with pytest.raises(ValueError):
        effect.case_scores(config, "US", fold["id"], "ridge", cases)


class State:
    """Doble del estado confirmado de la campaña: devuelve los recibos que se le dan."""

    def __init__(self, receipts):
        self.found, self.receipts = receipts, {}

    def resolve(self, job):
        return job.get("case"), None, {}

    def job_identity(self, job, case, sources):
        return dict(id=job["id"])

    def confirmed(self, job, identity):
        return self.found.get(identity["id"])


def search_job(window, arm, name, *, stage="search", scope="US", depends=(), **fields):
    return dict(
        depends=list(depends),
        id=f"{scope}/{window}/{arm}/{name}",
        scope=scope,
        window=window,
        arm=arm,
        stage=stage,
        **fields,
    )


def receipt(score):
    return dict(
        score=score, predictions=dict(evaluation=dict(path=f"p/{score}.parquet", sha256="0" * 64))
    )


def test_search_records_keep_every_confirmed_case_and_the_campaign_winner(monkeypatch, tmp_path):
    jobs = [
        search_job("w1", "ridge", "search-0"),
        search_job("w1", "ridge", "search-1"),
        search_job("w1", "ridge", "search-2"),
        search_job("w1", "ridge", "search-1-plateau", phase=engine.PLATEAU),
        search_job("w1", "ridge", "finalist-s43", stage="finalist"),
        search_job("w1", "xgboost", "search-0"),
        search_job("w1", "ridge", "search-0", scope="CN"),
    ]
    found = {
        job["id"]: receipt(score)
        for job, score in zip(jobs, [0.5, 0.4, 0.4, None, 0.3, 0.2, 0.1], strict=True)
    }
    campaign = dict(scopes=["US", "CN"], comparison_config=dict(name="comparación"))
    monkeypatch.setattr(engine, "_confirmed_state", lambda *a: (campaign, State(found)))
    monkeypatch.setattr(engine, "plan_campaign", lambda c: jobs)
    config, records = engine.search_records("campaña", {}, tmp_path, "US")
    assert config == dict(name="comparación")
    ridge = records["w1"]["ridge"]
    # Empate a 0,4: decide el identificador, como en la campaña.
    assert ridge["selected"] == "US/w1/ridge/search-1"
    assert set(ridge["cases"]) == {f"US/w1/ridge/search-{i}" for i in range(3)}
    assert ridge["cases"]["US/w1/ridge/search-0"]["score"] == 0.5
    assert (
        ridge["cases"]["US/w1/ridge/search-0"]["evaluation"]["path"] == tmp_path / "p/0.5.parquet"
    )
    assert list(records["w1"]["xgboost"]["cases"]) == ["US/w1/xgboost/search-0"]
    del found["US/w1/ridge/search-2"]
    with pytest.raises(ValueError, match="Falta confirmar US/w1/ridge/search-2"):
        engine.search_records("campaña", {}, tmp_path, "US")
    with pytest.raises(ValueError, match="no pertenece"):
        engine.search_records("campaña", {}, tmp_path, "US+CN")


def test_one_window_keeps_its_searches_and_confirms_their_dependencies(monkeypatch, tmp_path):
    # En la variante B el traslado de w2 parte del finalista del ancla w1, que depende de las
    # búsquedas de w1. Esas búsquedas se confirman, pero no son casos de w2.
    anchor = [f"US/w1/ridge/search-{i}" for i in range(2)]
    finalist = "US/w1/ridge/finalist-s43"
    jobs = [
        search_job("w1", "ridge", "search-0"),
        search_job("w1", "ridge", "search-1"),
        search_job("w1", "ridge", "finalist-s43", stage="finalist", depends=anchor),
        search_job("w2", "ridge", "carry-s43", stage="carry", depends=[finalist]),
        search_job("w2", "mars", "search-0"),
        search_job("w2", "mars", "search-1"),
        search_job("w3", "ridge", "search-0"),
    ]
    found = {job["id"]: receipt(0.1 * i) for i, job in enumerate(jobs)}
    campaign = dict(scopes=["US"], comparison_config={})
    monkeypatch.setattr(engine, "_confirmed_state", lambda *a: (campaign, State(found)))
    monkeypatch.setattr(engine, "plan_campaign", lambda c: jobs)
    _, records = engine.search_records("campaña", {}, tmp_path, "US", window="w2")
    assert list(records) == ["w2"] and list(records["w2"]) == ["mars"]
    assert records["w2"]["mars"]["selected"] == "US/w2/mars/search-0"
    del found["US/w3/ridge/search-0"]
    assert list(engine.search_records("campaña", {}, tmp_path, "US", window="w2")[1]) == ["w2"]
    del found[anchor[0]]
    with pytest.raises(ValueError, match=f"Falta confirmar {anchor[0]}"):
        engine.search_records("campaña", {}, tmp_path, "US", window="w2")
    with pytest.raises(ValueError, match="no tiene la ventana w9"):
        engine.search_records("campaña", {}, tmp_path, "US", window="w9")


def test_the_command_reports_from_receipts_or_from_the_saved_windows(study, tmp_path, monkeypatch):
    study, config = study
    records = two_windows(study, tmp_path / "cases")
    monkeypatch.setattr(engine, "search_records", lambda *a, **k: (config, records))
    common = ["--campaign", "c", "--views", "US=v", "--output", str(tmp_path), "--scope", "US"]
    common += ["--declaration", str(DECLARATION)]
    assert engine.main(["selection", *common, "--report", str(tmp_path / "a.json")]) == 0
    for window, models in records.items():
        effect.write_window(tmp_path / "aggregates", config, "US", window, models)
    monkeypatch.setattr(engine, "load_campaign", lambda path: dict(comparison_config=config))
    saved = ["--aggregates", str(tmp_path / "aggregates"), "--report", str(tmp_path / "b.json")]
    assert engine.main(["selection", *common, *saved]) == 0
    first, second = (json.loads((tmp_path / name).read_text()) for name in ("a.json", "b.json"))
    assert first == second and first["arms"]["ridge"]["summary"]["windows"] == 2
    (tmp_path / "aggregates" / "US" / f"{study.folds[1]['id']}.selection.json").unlink()
    with pytest.raises(ValueError, match="Falta el efecto de la selección de US"):
        engine.main(["selection", *common, *saved])


def test_the_campaign_rule_breaks_ties_by_identifier():
    searches = [("b", dict(score=0.4)), ("a", dict(score=0.4)), ("c", dict(score=0.3))]
    assert engine.best_search(searches)[0] == "c"
    assert engine.best_search(searches[:2])[0] == "a"
