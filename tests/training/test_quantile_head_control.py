"""Pruebas del control de la cabeza, que compara el Transformer compacto con salida L1 o cuantiles.

Se comprueban la declaración, el ejecutor con un sustituto que escribe predicciones y la
regla de retroceso. No se ejecuta ningún ajuste ni paso de optimizador.
"""

import copy
import hashlib
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data import input_policy
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.forecast_scores import score_sessions
from mars_titan.models import quantile_head
from mars_titan.training import masked_campaign as engine
from mars_titan.training import quantile_head_control as control
from mars_titan.training import reference_design
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.learning_hold import LearningHoldError
from mars_titan.training.reference_design import head_control_cases
from tests.training.test_walk_forward_v2_views import fixture

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs/baselines/quantile-head-control-us.json"
SEARCH = ROOT / "configs/baselines/historical-masked-reference-search-us.json"


def plan():
    return json.loads(PLAN.read_text())


def test_declared_pairs_differ_only_in_output_and_loss():
    cases = head_control_cases(plan())
    assert len(cases) == 3 * 2 * 2
    assert len({case["id"] for case in cases}) == len(cases)
    pairs = {}
    for item in cases:
        pairs.setdefault(item["pair"], {})[item["arm"]] = item["case"]
    assert len(pairs) == 6
    for pair in pairs.values():
        scalar, quantile = pair["scalar_l1"], pair["quantile_head_v1"]
        assert scalar["kind"] == quantile["kind"] == "transformer"
        assert scalar["loss"] == "mae" and "head" not in scalar
        assert quantile["loss"] == "pinball" and quantile["head"] == "quantile_head_v1"
        shared = {k: v for k, v in scalar.items() if k != "loss"}
        assert {k: v for k, v in quantile.items() if k not in {"loss", "head"}} == shared
        assert scalar["selection"]["stopping"] == "fixed_budget"
        assert scalar["selection"]["metric"] == "session_mae"
    assert {case["case"]["seed"] for case in cases} == {42, 43, 44}


def test_design_matches_the_reference_search_indices_seeds_and_budget():
    declared, search = plan(), json.loads(SEARCH.read_text())
    assert declared["case_indices"] == search["case_indices"]
    assert declared["seeds"] == search["finalist_seeds"]
    for key in ("patience", "min_delta", "stopping", "batch_size", "input_policy"):
        assert declared[key] == search[key]
    assert declared["max_epochs"] == search["max_epochs"]
    assert declared["prediction_retention"] == search["prediction_retention"]
    assert (ROOT / declared["walk_forward"]).is_file()
    base = reference_design.design_cases(
        ["transformer"],
        seed=42,
        epochs=search["max_epochs"],
        patience=search["patience"],
        min_delta=search["min_delta"],
        stopping=search["stopping"],
    )
    first = next(c for c in head_control_cases(declared) if c["arm"] == "scalar_l1")
    assert first["pair"] == base[declared["case_indices"][0]]["id"]
    assert {k: v for k, v in first["case"].items() if k != "loss"} == {
        k: v for k, v in base[0]["case"].items() if k != "loss"
    }


def test_each_declared_case_passes_the_runner_validation(monkeypatch):
    from mars_titan.training import reference_run

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    declared = plan()
    for item in head_control_cases(declared):
        reference_run._options(
            item["case"],
            declared["batch_size"],
            300,
            0,
            input_policy=declared["input_policy"],
            prediction_retention=declared["prediction_retention"],
        )


def test_names_repeat_the_head_and_policy_constants():
    assert reference_design.QUANTILE_HEAD == quantile_head.QUANTILE_HEAD
    assert reference_design.PINBALL == quantile_head.PINBALL
    assert reference_design.HISTORICAL_MASKED == input_policy.HISTORICAL_MASKED


def test_design_module_still_loads_without_pytorch():
    code = (
        "import sys, mars_titan.training.reference_design as d; "
        "assert 'torch' not in sys.modules; print(len(d.HEAD_CONTROL_ARMS))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "2"


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(status="executed"),
        lambda p: p.update(final_test_opened=True),
        lambda p: p.update(model="gru"),
        lambda p: p.update(arms=["quantile_head_v1", "scalar_l1"]),
        lambda p: p.update(arms=["scalar_mse", "quantile_head_v1"]),
        lambda p: p.update(stopping="validation_plateau"),
        lambda p: p.update(input_policy="strict_inputs_v1"),
        lambda p: p.update(case_indices=[10, 0]),
        lambda p: p.update(case_indices=[0, 0]),
        lambda p: p.update(case_indices=[12]),
        lambda p: p.update(case_indices=[]),
        lambda p: p.update(seeds=[42, 42]),
        lambda p: p.update(seeds=[]),
        lambda p: p.update(batch_size=512),
        lambda p: p["comparison"].update(partition="evaluation"),
        lambda p: p["comparison"].update(metric="session_mse"),
        lambda p: p["comparison"].update(fallback="keep_quantile_head"),
        lambda p: p["comparison"]["contrast"].update(base="quantile_head_v1"),
        lambda p: p.update(extra=True),
        lambda p: p.pop("decision"),
        lambda p: p.update(walk_forward=None),
        lambda p: p.update(patience=0),
        lambda p: p.update(max_epochs=1),
    ],
)
def test_any_change_to_the_declared_contrast_is_rejected(change):
    declared = copy.deepcopy(plan())
    change(declared)
    with pytest.raises(ValueError):
        head_control_cases(declared)


# Estas pruebas cubren el ejecutor del control. Las vistas se preparan sobre el corpus técnico y
# un ejecutor sustituto escribe predicciones de validación con las filas exactas de la vista. No
# se ajusta ningún modelo, no se aplican pasos de optimizador y la GPU no se usa.

CONFIGS = ROOT / "configs"
WINDOWS = ["fold-000", "fold-001", "fold-002", "fold-003", "fold-004", "fold-005"]


def write_comparison(folder, **changes):
    """Escribe una comparación principal reducida, con protocolos absolutos y pocas réplicas."""
    declared = json.loads(
        (CONFIGS / "evaluation/historical-masked-2000-comparison.json").read_text()
    )
    for scope in declared["scopes"].values():
        scope["protocols"] = {
            market: str((CONFIGS / "evaluation" / name).resolve())
            for market, name in scope["protocols"].items()
        }
    declared["arms"] = {k: v for k, v in declared["arms"].items() if k in {"zero", "gru", "ridge"}}
    # Los pares de retención nombran modelos que esta comparación reducida no conserva.
    declared.pop("retention_interference")
    declared["comparison"].update(
        block_length=2,
        sensitivity_block_lengths=[],
        replicates=200,
        families=dict(
            references_vs_zero=dict(kind="delta", base="zero", variants=["gru", "ridge"])
        ),
    )
    declared["comparison"].update(changes)
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "comparison.json", declared)
    return folder / "comparison.json"


def write_campaign(folder, comparison):
    """Declara una campaña mínima que solo prepara las vistas US en su orden habitual."""
    value = json.loads((CONFIGS / "baselines/historical-masked-campaign-a.json").read_text())
    tabular = json.loads((CONFIGS / "baselines/tabular-historical-masked.json").read_text())
    tabular.update(ridge_alphas=[1.0], depths=[3], bins=[64], rates=[0.1])
    atomic_json(folder / "tabular.json", tabular)
    value.pop("titans_mac")
    value.update(
        comparison=str(comparison),
        scopes=["US"],
        neural=value["neural"] | dict(arms={"gru": "gru"}),
        tabular=value["tabular"] | dict(config="tabular.json", arms={"ridge": "ridge"}),
        limits=dict(max_training_jobs=10_000, max_prediction_jobs=10_000),
    )
    atomic_json(folder / "campaign.json", value)
    return folder / "campaign.json"


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("head-control")
    data = fixture(root / "data", ("US",))
    comparison = write_comparison(root / "config")
    campaign = write_campaign(root / "config", comparison)
    engine.prepare_views(campaign, data.parent, root / "views")
    return SimpleNamespace(root=root, comparison=comparison, views=root / "views" / "US")


def error(job_case, moment, *, worse=0.0):
    """Devuelve un error determinista por fila, distinto para cada semilla, índice y brazo."""
    key = f"{job_case['seed']}/{job_case['architecture']['hidden_size']}/{moment}"
    value = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 2**32
    sign = 1.0 if value < 0.5 else -1.0
    quantile = job_case.get("head") == QUANTILE
    return sign * (0.01 + 0.02 * value + (worse if quantile else 0.0))


QUANTILE = quantile_head.QUANTILE_HEAD


class FakeFit:
    """Sustituye a `run_reference_case`, registra la llamada y escribe la validación."""

    def __init__(self, *, worse=None, pause_at=None, stop=None, mutate=None, report=None):
        self.calls, self.cache = [], {}
        self.worse = worse or {}
        self.pause_at, self.stop, self.mutate, self.report = pause_at, stop, mutate, report

    def rows(self, view):
        if view not in self.cache:
            dataset = CorpusDataset(view, input_policy=input_policy.HISTORICAL_MASKED)
            parts = dict(sample_id=[], market=[], prediction_at=[], target=[])
            for batch in dataset.batches(partition="validation", batch_size=64, epoch=0, seed=0):
                parts["sample_id"] += list(batch["sample_ids"])
                parts["market"] += list(batch["market"])
                parts["prediction_at"].append(np.asarray(batch["prediction_at"]))
                parts["target"].append(np.asarray(batch["target"], dtype=np.float64))
            for name in ("prediction_at", "target"):
                parts[name] = np.concatenate(parts[name])
            self.cache[view] = parts
        return self.cache[view]

    def table(self, view, case):
        rows = self.rows(view)
        moments = rows["prediction_at"].astype("datetime64[us]").astype(np.int64)
        index = 10 if case["architecture"]["hidden_size"] == 64 else 0
        worse = self.worse.get(index, 0.0)
        errors = np.array([error(case, int(m), worse=worse) for m in moments])
        median = (rows["target"] + errors).astype(np.float32)
        columns = dict(
            sample_id=rows["sample_id"],
            asset_id=["/".join(key.split("/")[:2]) for key in rows["sample_id"]],
            market=rows["market"],
            prediction_at=pa.array(moments, type=pa.timestamp("us", tz="UTC")),
            target=rows["target"],
            prediction=median,
            zero=np.zeros(len(median)),
        )
        if case.get("head") == QUANTILE:
            offsets = (-0.05, -0.02, 0.0, 0.02, 0.05)
            columns.update(
                {
                    name: (median + np.float32(offset)).astype(np.float32)
                    for name, offset in zip(quantile_head.QUANTILE_COLUMNS, offsets, strict=True)
                }
            )
        table = pa.table(columns)
        return self.mutate(case, table) if self.mutate else table

    def __call__(self, view, folder, case, **options):
        self.calls.append(dict(view=view, folder=folder, case=case, **options))
        folder.mkdir(parents=True, exist_ok=True)
        if len(self.calls) == self.pause_at:
            (folder / "partial.bin").write_bytes(b"incompleto")
            self.stop.requested = True
            return dict(status="paused")
        path = folder / "validation-predictions.parquet"
        table = self.table(view, case)
        pq.write_table(table, path)
        errors = np.abs(table["prediction"].to_numpy() - table["target"].to_numpy())
        report = dict(
            status="completed",
            final_test_opened=False,
            identity=dict(manifest_sha256=sha256(view), case=case),
            predictions=dict(
                validation=dict(
                    path=path.name,
                    sha256=sha256(path),
                    metrics=dict(session_mae=float(errors.mean())),
                )
            ),
        )
        if self.report:
            self.report(report)
        atomic_json(folder / "run.json", report)
        return report


def run_control(prepared, output, fit, *, windows=WINDOWS, stop=None):
    return control.run_control(
        PLAN,
        prepared.views,
        output,
        windows=windows,
        comparison_path=prepared.comparison,
        executor=fit,
        lease=nullcontext,
        stop=stop or SimpleNamespace(requested=False),
    )


def test_check_counts_the_default_window_and_declared_windows():
    checked = control.check_control(PLAN)
    assert checked["windows"] == ["fold-000"] and checked["jobs"] == 12
    assert checked["window_rule"] == control.WINDOW_RULE
    assert checked["contrasts"] == ["index_00", "index_10"]
    assert checked["bootstrap"]["block_length"] == 16
    assert checked["scientific_training_started"] is False
    both = control.check_control(PLAN, windows=["fold-000", "fold-018"])
    assert both["jobs"] == 24 and both["window_rule"] == "declared_before_execution"


@pytest.mark.parametrize(
    "windows", [[], ["fold-099"], ["fold-001", "fold-000"], ["fold-000", "fold-000"], [0]]
)
def test_windows_must_be_distinct_ordered_protocol_windows(windows):
    with pytest.raises(ValueError, match="ventanas"):
        control.load_control(PLAN, windows=windows)


def test_comparison_must_declare_the_control_protocol(tmp_path):
    protocol = json.loads((ROOT / plan()["walk_forward"]).read_text())
    protocol["seeds"] = [7, 8, 9]
    atomic_json(tmp_path / "us.json", protocol)
    declared = json.loads(
        (CONFIGS / "evaluation/historical-masked-2000-comparison.json").read_text()
    )
    declared["scopes"] = {"US": {"protocols": {"US": str(tmp_path / "us.json")}, "windows": "all"}}
    atomic_json(tmp_path / "comparison.json", declared)
    with pytest.raises(ValueError, match="protocolo"):
        control.load_control(PLAN, comparison_path=tmp_path / "comparison.json")


def test_jobs_pair_both_arms_with_the_same_window_seed_and_index():
    loaded = control.load_control(PLAN, windows=["fold-000", "fold-003"])
    pairs = {}
    for job in loaded["jobs"]:
        pairs.setdefault((job["window"], job["pair"]), []).append(job)
    assert len(pairs) == 2 * 6
    for (window, _), jobs in pairs.items():
        assert [job["arm"] for job in jobs] == ["scalar_l1", "quantile_head_v1"]
        assert len({(job["seed"], job["index"], job["window"]) for job in jobs}) == 1
        assert jobs[0]["window"] == window
    assert {job["index"] for job in loaded["jobs"]} == {0, 10}


def test_run_is_blocked_before_creating_any_output(learning_hold, tmp_path):
    learning_hold(False)
    fit = FakeFit()
    with pytest.raises(LearningHoldError):
        control.run_control(PLAN, tmp_path / "views", tmp_path / "out", executor=fit)
    assert not (tmp_path / "out").exists() and fit.calls == []


@pytest.mark.usefixtures("learning_doubles")
def test_run_confirms_each_job_once_with_its_view_and_case(prepared, tmp_path):
    fit = FakeFit()
    summary = run_control(prepared, tmp_path / "out", fit)
    assert summary["status"] == "completed" and summary["completed"] == summary["planned"] == 72
    loaded = control.load_control(PLAN, windows=WINDOWS, comparison_path=prepared.comparison)
    views = engine.scope_views(
        prepared.views,
        "US",
        dict(
            input_policy=loaded["plan"]["input_policy"],
            comparison_config=loaded["comparison_config"],
        ),
    )
    assert len(fit.calls) == 72
    for call, job in zip(fit.calls, loaded["jobs"], strict=True):
        assert call["case"] == job["case"]
        assert call["view"] == Path(views["windows"][job["window"]]["path"])
        assert call["folder"] == tmp_path / "out" / "jobs" / job["id"] / "run"
        assert call["batch_size"] == 256 and call["checkpoint_seconds"] == 300
        assert call["prediction_retention"] == "heldout_full_train_sessions_v1"
        assert call["input_policy"] == "historical_masked_2000_v1"
        assert call["resume"] is False
        receipt = json.loads((tmp_path / "out/jobs" / job["id"] / "receipt.json").read_text())
        assert receipt["final_test_opened"] is False
        assert (
            receipt["validation"]["rows"] == views["windows"][job["window"]]["counts"]["validation"]
        )
    again = FakeFit()
    assert run_control(prepared, tmp_path / "out", again)["status"] == "completed"
    assert again.calls == []


@pytest.mark.usefixtures("learning_doubles")
def test_a_paused_job_resumes_in_its_folder_and_the_rest_continue(prepared, tmp_path):
    stop = SimpleNamespace(requested=False)
    first = FakeFit(pause_at=3, stop=stop)
    summary = run_control(prepared, tmp_path / "out", first, windows=["fold-000"], stop=stop)
    assert summary["status"] == "paused" and summary["completed"] == 2
    second = FakeFit()
    summary = run_control(prepared, tmp_path / "out", second, windows=["fold-000"])
    assert summary["status"] == "completed" and summary["completed"] == 12
    assert [call["resume"] for call in second.calls] == [True] + [False] * 9
    assert second.calls[0]["case"] == first.calls[2]["case"]


@pytest.mark.usefixtures("learning_doubles")
def test_changed_artifacts_windows_or_receipts_are_rejected(prepared, tmp_path):
    output = tmp_path / "out"
    run_control(prepared, output, FakeFit(), windows=["fold-000"])
    with pytest.raises(ValueError, match="otro control"):
        run_control(prepared, output, FakeFit(), windows=["fold-000", "fold-001"])
    receipt = next(output.glob("jobs/**/receipt.json"))
    value = json.loads(receipt.read_text())
    value["identity"]["case"]["learning_rate"] = 0.5
    atomic_json(receipt, value)
    with pytest.raises(ValueError, match="cambió de identidad"):
        run_control(prepared, output, FakeFit(), windows=["fold-000"])
    value["identity"]["case"]["learning_rate"] = 0.0001
    atomic_json(receipt, value)
    path = output / value["validation"]["path"]
    table = pq.read_table(path)
    pq.write_table(table.slice(0, table.num_rows), path, compression="gzip")
    with pytest.raises(ValueError, match="ha cambiado"):
        run_control(prepared, output, FakeFit(), windows=["fold-000"])


def repeat_first_row(case, table):
    return pa.concat_tables([table, table.slice(0, 1)]) if case["seed"] == 43 else table


def shift_target(case, table):
    if case.get("head") != QUANTILE or case["seed"] != 44:
        return table
    target = table["target"].to_numpy().copy()
    target[0] += 1.0
    return table.set_column(table.schema.get_field_index("target"), "target", pa.array(target))


def into_reserve(case, table):
    if case["seed"] != 44:
        return table
    moments = table["prediction_at"].cast(pa.int64()).to_numpy().copy()
    moments[-1] = 1_704_153_600_000_000  # Es el 2024-01-02, ya en la reserva final.
    column = pa.array(moments, type=pa.timestamp("us", tz="UTC"))
    return table.set_column(table.schema.get_field_index("prediction_at"), "prediction_at", column)


@pytest.mark.usefixtures("learning_doubles")
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (repeat_first_row, "filas frente a"),
        (shift_target, "no evalúa las mismas filas"),
        (into_reserve, "reserva final"),
    ],
)
def test_rows_and_reserve_are_checked_before_confirming(prepared, tmp_path, mutate, message):
    with pytest.raises(ValueError, match=message):
        run_control(prepared, tmp_path / "out", FakeFit(mutate=mutate), windows=["fold-000"])
    receipts = list((tmp_path / "out").glob("jobs/**/receipt.json"))
    assert len(receipts) < 12


@pytest.mark.usefixtures("learning_doubles")
@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r["identity"].update(manifest_sha256="0" * 64), "no confirma"),
        (lambda r: r["identity"]["case"].update(seed=7), "no confirma"),
        (lambda r: r.update(final_test_opened=True), "no confirma"),
        (lambda r: r["predictions"]["validation"]["metrics"].update(session_mae=-0.1), "MAE"),
        (lambda r: r["predictions"]["validation"]["metrics"].update(session_mae="0.1"), "MAE"),
    ],
)
def test_reports_must_confirm_view_case_reserve_and_score(prepared, tmp_path, change, message):
    def report(value):
        value["identity"] = dict(value["identity"], case=dict(value["identity"]["case"]))
        change(value)

    with pytest.raises(ValueError, match=message):
        run_control(prepared, tmp_path / "out", FakeFit(report=report), windows=["fold-000"])
    assert not list((tmp_path / "out").glob("jobs/**/receipt.json"))


@pytest.mark.usefixtures("learning_doubles")
def test_decision_falls_back_to_option_a_when_the_head_worsens_one_index(prepared, tmp_path):
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="no corresponde"):
        control.decide_control(
            PLAN, prepared.views, output, windows=WINDOWS, comparison_path=prepared.comparison
        )
    run_control(prepared, output, FakeFit(worse={10: 0.05}))
    decision = control.decide_control(
        PLAN, prepared.views, output, windows=WINDOWS, comparison_path=prepared.comparison
    )
    assert decision["decision"]["option"] == control.OPTION_A
    assert decision["decision"]["against_quantile_head"] == ["index_10"]
    rows = {row["name"]: row for row in decision["contrasts"]["contrasts"]}
    assert rows["index_10"]["estimate"] == pytest.approx(0.05, abs=1e-6)
    assert rows["index_00"]["estimate"] == pytest.approx(0.0, abs=1e-6)
    assert not rows["index_00"]["simultaneous_excludes_zero"]
    assert len(decision["receipts"]) == 72 and decision["final_test_opened"] is False
    first = (output / "decision.json").read_bytes()
    control.decide_control(
        PLAN, prepared.views, output, windows=WINDOWS, comparison_path=prepared.comparison
    )
    assert (output / "decision.json").read_bytes() == first
    value = json.loads(first)
    value["decision"]["option"] = control.OPTION_B
    atomic_json(output / "decision.json", value)
    with pytest.raises(ValueError, match="distinta"):
        control.decide_control(
            PLAN, prepared.views, output, windows=WINDOWS, comparison_path=prepared.comparison
        )


@pytest.mark.usefixtures("learning_doubles")
def test_decision_keeps_the_head_when_it_does_not_worsen_the_mae(prepared, tmp_path):
    output = tmp_path / "out"
    run_control(prepared, output, FakeFit(worse={0: -0.005, 10: 0.0}))
    decision = control.decide_control(
        PLAN, prepared.views, output, windows=WINDOWS, comparison_path=prepared.comparison
    )
    assert decision["decision"]["option"] == control.OPTION_B
    assert decision["decision"]["against_quantile_head"] == []


@pytest.mark.usefixtures("learning_doubles")
def test_decision_needs_every_confirmed_job(prepared, tmp_path):
    stop = SimpleNamespace(requested=False)
    output = tmp_path / "out"
    run_control(prepared, output, FakeFit(pause_at=5, stop=stop), windows=["fold-000"], stop=stop)
    with pytest.raises(ValueError, match="Falta confirmar"):
        control.decide_control(
            PLAN, prepared.views, output, windows=["fold-000"], comparison_path=prepared.comparison
        )


def synthetic_scores(shifts, *, days=120, assets=12, seed=0):
    """Construye puntuaciones de validación con errores fijados por brazo, índice y semilla."""
    rng = np.random.default_rng(seed)
    moments = np.repeat(
        np.datetime64("2010-01-04T21:00", "us") + np.arange(days) * np.timedelta64(1, "D"), assets
    )
    target = rng.normal(0.0, 0.02, size=len(moments))
    scores = {}
    for (arm, index, seed_value), shift in shifts.items():
        noise = np.abs(rng.normal(0.0, 0.01, size=len(moments))) + shift
        prediction = target + np.where(rng.random(len(moments)) < 0.5, noise, -noise)
        panel = ForecastPanel.from_columns(
            pa.array([f"US/A{i % assets}/{i}" for i in range(len(moments))]),
            np.full(len(moments), "US"),
            moments,
            target,
            prediction,
            markets=("US",),
        )
        scores[arm, index, seed_value] = score_sessions(panel)
    return scores


def shifts(quantile_index_10):
    return {
        (arm, index, seed): (quantile_index_10 if (arm, index) == (QUANTILE, 10) else 0.0)
        for arm in ("scalar_l1", QUANTILE)
        for index in (0, 10)
        for seed in (42, 43, 44)
    }


BOOTSTRAP = dict(
    block_length=5,
    replicates=500,
    seed=20261009,
    confidence=0.95,
    market_weighting="session",
    sensitivity_block_lengths=(),
)


@pytest.mark.parametrize(
    ("shift", "option"),
    [(0.01, control.OPTION_A), (0.0, control.OPTION_B), (-0.01, control.OPTION_B)],
)
def test_contrast_and_rule_on_synthetic_scores(shift, option):
    result = control.contrast_scores(synthetic_scores(shifts(shift)), plan(), BOOTSTRAP)
    rows = {row["name"]: row for row in result["contrasts"]}
    assert set(rows) == {"index_00", "index_10"}
    assert result["multiplicity"]["family_size"] == 2
    assert rows["index_10"]["coefficients"] == {
        "quantile_head_v1@10": 1.0,
        "scalar_l1@10": -1.0,
    }
    scores = synthetic_scores(shifts(shift))
    per_seed = [
        scores[QUANTILE, 10, seed].series("mae").values.mean()
        - scores["scalar_l1", 10, seed].series("mae").values.mean()
        for seed in (42, 43, 44)
    ]
    assert rows["index_10"]["estimate"] == pytest.approx(np.mean(per_seed), rel=1e-9)
    decided = control.apply_fallback(result)
    assert decided["option"] == option
    if shift < 0:
        assert rows["index_10"]["simultaneous_interval"][1] < 0


def test_too_few_days_leave_the_decision_undetermined():
    options = dict(BOOTSTRAP, block_length=200)
    result = control.contrast_scores(synthetic_scores(shifts(0.0)), plan(), options)
    assert control.apply_fallback(result)["option"] == control.UNDETERMINED


@pytest.mark.parametrize(
    ("intervals", "option"),
    [
        ([[0.001, 0.01], None], control.OPTION_A),
        ([[0.0, 0.01], [-0.01, 0.02]], control.OPTION_B),
        ([[-0.02, -0.001], [-0.01, 0.0]], control.OPTION_B),
        ([[-0.01, 0.02], None], control.UNDETERMINED),
        ([[-0.01, 0.02], [1e-9, 0.02]], control.OPTION_A),
    ],
)
def test_fallback_rule_reads_the_lower_simultaneous_bound(intervals, option):
    result = dict(
        contrasts=[
            dict(name=f"index_{i:02d}", simultaneous_interval=value)
            for i, value in zip((0, 10), intervals, strict=True)
        ]
    )
    decided = control.apply_fallback(result)
    assert decided["option"] == option
    assert decided["consequence"] == control.CONSEQUENCES[option]


def test_command_line_check_prints_the_count(capsys):
    assert control.main(["check", "--plan", str(PLAN), "--windows", "fold-000", "fold-001"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["jobs"] == 24 and printed["status"] == "checked"
