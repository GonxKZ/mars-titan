"""Ventanas walk-forward de la GRU candidata sobre vistas v2 técnicas, sin pasos de optimizador.

Los ajustes calculan pérdidas y gradientes con el módulo nativo, pero el optimizador solo
registra llamadas. El de las ventanas ancla desplaza además los pesos una única vez al
construirse, sin gradientes, para que su estado elegido se distinga de una inicialización
nueva con la misma semilla. Las vistas salen del corpus técnico con filas en todos los años
y la GPU no se usa.
"""

import json
import os
import shutil
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.environments.walk_forward_receipt import read_window_receipt
from mars_titan.evaluation import walk_forward_comparison as comparison
from mars_titan.evaluation.splits import PARTITIONS
from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from mars_titan.training import campaign_plan as plan
from mars_titan.training import candidate_run
from mars_titan.training import candidate_walk_forward as walk
from mars_titan.training import masked_campaign as engine
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.learning_hold import HOLD_ENV, LearningHoldError
from tests.training.test_financial_run import RecordingOptimizer
from tests.training.test_masked_campaign import Recorder, doubles, write_campaign
from tests.training.test_walk_forward_v2_views import fixture

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("MARS_TITAN_EPISODIC_NATIVE"), reason="Falta el enlace nativo compilado"
    ),
    pytest.mark.usefixtures("learning_doubles"),
]

ROOT = Path(__file__).parents[2]
RECIPE = ROOT / "configs/candidate/chronological-training.json"
COMPARISON = ROOT / "configs/evaluation/historical-masked-2000-comparison.json"
SELECTION = dict(metric="session_mae", patience=5, min_delta=1e-05, stopping="fixed_budget")
# Receta reducida de las pruebas: una época y bloques pequeños sobre el corpus técnico.
SMALL = candidate_run.CandidateRecipe(
    update_instants=2, epochs=1, block_rows=2, bank_capacity=4, selection=SELECTION
)
MODEL = dict(feature_seed=43, key_seed=44, dtype="float64")
SHIFT = 0.01


class ShiftedStart(RecordingOptimizer):
    """Registrar llamadas y desplazar una vez los pesos al construirse, sin gradientes."""

    def __init__(self, groups):
        super().__init__(groups)
        with torch.no_grad():
            for group in self.param_groups:
                for value in group["params"]:
                    value.add_(SHIFT)


@pytest.fixture(scope="module")
def allowed(tmp_path_factory):
    """Protección temporal permitida para los ajustes compartidos del módulo."""
    path = tmp_path_factory.mktemp("module-hold") / "training-hold.json"
    path.write_text(json.dumps({"training_allowed": True}), encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(HOLD_ENV, str(path))
        yield path


@pytest.fixture(scope="module")
def views(tmp_path_factory):
    root = tmp_path_factory.mktemp("candidate-walk-forward")
    data = fixture(root / "data", ("US", "CN"))
    campaign = write_campaign(root / "config", scopes=("US", "US+CN"))
    prepared = engine.prepare_views(campaign, data.parent, root / "views")
    return SimpleNamespace(
        root=root,
        parent=data.parent,
        directories={scope: root / "views" / scope for scope in prepared},
        windows={
            scope: {name: Path(record["path"]) for name, record in value["windows"].items()}
            for scope, value in prepared.items()
        },
    )


def fit(
    view, output, *, seed=42, factory=ShiftedStart, parent="US/fold-000/gru_episodic", **options
):
    made = []

    def optimizer(groups):
        made.append(factory(groups))
        return made[-1]

    report = walk.fit_window(
        view,
        output,
        options.pop("recipe", SMALL),
        seed=seed,
        model=MODEL,
        parent_id=parent,
        device="cpu",
        optimizer_factory=optimizer,
        **options,
    )
    return report, made


@pytest.fixture(scope="module")
def anchor(views, allowed):
    output = views.root / "anchor"
    report, made = fit(views.windows["US"]["fold-000"], output)
    return SimpleNamespace(output=output, report=report, optimizer=made[0])


def shifted_parameters(specification, seed=42):
    adapter = candidate_run.CandidateInputAdapter(
        specification, dtype=torch.float64, parameter_seed=seed, feature_seed=43, key_seed=44
    )
    with torch.no_grad():
        for value in adapter.model.named_parameters().values():
            value.add_(SHIFT)
    return adapter.model.parameter_fingerprint()


def test_fit_window_writes_the_view_rows_receipts_and_selected_state(views, anchor):
    report, output = anchor.report, anchor.output
    view = views.windows["US"]["fold-000"]
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    assert report["status"] == "completed" and report["final_test_opened"] is False
    assert report["view"]["sha256"] == sha256(view) and report["markets"] == ["US"]
    assert report["bank_policy"] == walk.CARRY_POLICY
    assert set(report["predictions"]) == {"validation", "calibration", "evaluation"}
    columns = set(comparison.COLUMNS) | set(QUANTILE_COLUMNS) | {"sample_id", "zero"}
    for name, record in report["predictions"].items():
        path = output / record["path"]
        assert sha256(path) == record["sha256"]
        table = pq.read_table(path)
        assert set(table.column_names) == columns
        assert record["rows"] == table.num_rows == dataset.manifest["counts"][name]
        years = pa.compute.year(table["prediction_at"]).to_numpy()
        assert years.max() <= 2023
    checkpoint = report["checkpoint"]
    assert sha256(output / checkpoint["path"]) == checkpoint["sha256"]
    run = json.loads((output / report["run"]["path"]).read_text())
    assert run["best_checkpoint"]["sha256"] == checkpoint["sha256"]
    # Ningún paso cambia los pesos: el estado elegido es el inicial desplazado.
    train = report["sources"]["train"]
    assert train["phase"]["partition"] == "train"
    specification = walk.window_sources(dataset, output / "indices", ("train",))["train"]
    expected = shifted_parameters(specification.specification())
    assert checkpoint["parameters_sha256"] == expected
    assert anchor.optimizer.calls > 0
    assert any(
        value is not None and bool(value.abs().sum() > 0)
        for record in anchor.optimizer.records
        for value in record.values()
    )
    assert set(report["receipts"]) == {"US"}
    record = json.loads((output / report["receipts"]["US"]["path"]).read_text())
    checked = read_window_receipt(record)
    assert record["parent"] == dict(id="US/fold-000/gru_episodic", sha256=checkpoint["sha256"])
    assert checked.labels_used_until == checked.segment("evaluation")[0] - 1
    assert set(record["predictions"]) == {"calibration", "evaluation"}
    assert (
        record["predictions"]["evaluation"]["rows"] == report["predictions"]["evaluation"]["rows"]
    )


def reference_table(view, partition):
    """Filas que escriben los demás brazos: las del lector por lotes de la vista."""
    rows = Recorder().rows(view, partition)
    return pa.table(
        dict(
            sample_id=rows["sample_id"],
            asset_id=["/".join(key.split("/")[:2]) for key in rows["sample_id"]],
            market=rows["market"],
            prediction_at=pa.array(rows["prediction_at"], type=pa.timestamp("us", tz="UTC")),
            target=rows["target"],
        )
    )


def test_predicted_rows_are_the_rows_of_the_other_arms(views, anchor):
    view = views.windows["US"]["fold-000"]
    for name, record in anchor.report["predictions"].items():
        table = pq.read_table(anchor.output / record["path"])
        reference = reference_table(view, name)
        assert engine._rows_digest(table) == engine._rows_digest(reference)
        assert sorted(table["sample_id"].to_pylist()) == sorted(reference["sample_id"].to_pylist())


def test_row_check_counts_each_kind_of_difference(views, anchor):
    view = views.windows["US"]["fold-000"]
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    table = pq.read_table(anchor.output / anchor.report["predictions"]["evaluation"]["path"])
    assert table.num_rows >= 4
    assert walk.check_view_rows(table, dataset, "evaluation") == table.num_rows
    target = table["target"].to_numpy().copy()
    target[1] += 0.5
    changed = table.set_column(table.schema.get_field_index("target"), "target", pa.array(target))
    moments = table["prediction_at"].cast(pa.int64()).to_numpy().copy()
    moments[2] += 1
    foreign = table.slice(2, 1).set_column(
        table.schema.get_field_index("prediction_at"),
        "prediction_at",
        pa.array(moments[2:3], type=pa.timestamp("us", tz="UTC")),
    )
    mutated = pa.concat_tables([changed.slice(1), foreign, table.slice(3, 1)])
    message = (
        r"evaluation: 4 filas difieren de la vista \(1 sin predicción, 1 ajenas, "
        r"1 repetidas y 1 con otro objetivo\)"
    )
    with pytest.raises(ValueError, match=message):
        walk.check_view_rows(mutated, dataset, "evaluation")
    other = table.set_column(
        table.schema.get_field_index("market"), "market", pa.array(["CN"] * table.num_rows)
    )
    rows = table.num_rows
    with pytest.raises(ValueError, match=f"{2 * rows} filas difieren .*{rows} sin predicción"):
        walk.check_view_rows(other, dataset, "evaluation")


class StopAfter:
    """Pedir la parada después de un número de consultas en las barreras."""

    def __init__(self, checks):
        self.left = checks

    @property
    def requested(self):
        self.left -= 1
        return self.left < 0


def test_fit_window_resumes_after_a_pause_with_the_same_state(views, anchor, tmp_path):
    view = views.windows["US"]["fold-000"]
    output = tmp_path / "window"
    paused, _ = fit(view, output, stop=StopAfter(8))
    assert paused == dict(status="paused", final_test_opened=False)
    assert not (output / walk.WINDOW_REPORT).exists()
    run = json.loads((output / "run/run.json").read_text())
    assert run["status"] == "paused" and run["cursor"]["phase"] == "train"
    resumed, _ = fit(view, output)
    assert resumed["status"] == "completed"
    assert (
        resumed["checkpoint"]["parameters_sha256"]
        == anchor.report["checkpoint"]["parameters_sha256"]
    )
    for name, record in resumed["predictions"].items():
        assert record["sha256"] == anchor.report["predictions"][name]["sha256"]
    assert resumed["run"]["run_id"] == anchor.report["run"]["run_id"]
    run_sha256 = sha256(output / "run/run.json")
    assert len(json.loads((output / "run/run.json").read_text())["attempts"]) == 2
    again, made = fit(view, output)
    assert again["checkpoint"] == resumed["checkpoint"] and made[0].calls == 0
    assert sha256(output / "run/run.json") == run_sha256


def test_window_entries_need_the_learning_permission(views, anchor, tmp_path, learning_hold):
    learning_hold(False)
    with pytest.raises(LearningHoldError, match="ventana walk-forward"):
        fit(views.windows["US"]["fold-000"], tmp_path / "fit")
    with pytest.raises(LearningHoldError, match="predicción trasladada"):
        carry(views, anchor, "fold-001", tmp_path / "carry")
    assert not (tmp_path / "fit").exists() and not (tmp_path / "carry").exists()


def carry(views, anchor, window, output, *, scope="US", source="fold-000", stop=None):
    return walk.carry_window(
        anchor.output,
        views.windows[scope][source],
        views.windows[scope][window],
        output,
        parent_id=f"{scope}/{source}/gru_episodic",
        device="cpu",
        stop=stop,
    )


@pytest.fixture(scope="module")
def carried(views, anchor, allowed):
    output = views.root / "carried"
    return SimpleNamespace(output=output, report=carry(views, anchor, "fold-001", output))


def predictor(views, anchor, window, folder, *, adapter=None, audit=False):
    """Recorrido congelado independiente sobre los tramos trasladados de una ventana."""
    dataset = CorpusDataset(views.windows["US"][window], input_policy=HISTORICAL_MASKED)
    sources = walk.window_sources(dataset, folder / "indices", ("calibration", "evaluation"))
    specification = sources["calibration"].specification()
    if adapter is None:
        adapter, recipe, _, _ = walk.anchor_adapter(
            anchor.output, views.windows["US"]["fold-000"], specification, device="cpu"
        )
    else:
        adapter, recipe = adapter(specification), SMALL
    return candidate_run.CandidateChronologicalPredictor(
        adapter,
        recipe,
        sources=sources,
        output=folder / "predictions",
        world=walk.WORLD,
        fold=window,
        audit=audit,
    ), sources


def test_carry_predicts_from_the_anchor_state_without_fitting(views, anchor, carried, tmp_path):
    record, output = carried.report, carried.output
    checkpoint = anchor.report["checkpoint"]
    assert record["kind"] == "carried_predictions" and record["status"] == "completed"
    assert record["final_test_opened"] is False and record["model"] == walk.MODEL
    assert record["anchor"]["checkpoint_sha256"] == checkpoint["sha256"]
    assert record["anchor"]["parameters_sha256"] == checkpoint["parameters_sha256"]
    assert record["adapter"]["parameters_sha256"] == checkpoint["parameters_sha256"]
    assert record["anchor"]["fold"]["id"] == "fold-000" and record["fold"]["id"] == "fold-001"
    # La información del ancla termina en octubre de 2004 y la evaluación empieza en 2006.
    assert record["months_since_anchor_information"] == 15
    assert record["bank_policy"] == walk.CARRY_POLICY
    assert record["bank_scope"] == dict(world=walk.WORLD, fold="fold-001")
    assert record["recipe"] == SMALL.identity()
    dataset = CorpusDataset(views.windows["US"]["fold-001"], input_policy=HISTORICAL_MASKED)
    assert set(record["predictions"]) == {"calibration", "evaluation"}
    for name, value in record["predictions"].items():
        assert sha256(output / value["path"]) == value["sha256"]
        assert value["rows"] == dataset.manifest["counts"][name]
    receipt = json.loads((output / record["receipts"]["US"]["path"]).read_text())
    assert receipt["parent"] == dict(id="US/fold-000/gru_episodic", sha256=checkpoint["sha256"])
    assert read_window_receipt(receipt).fold == "fold-001"
    # Un recorrido independiente con el estado del ancla da los mismos bits.
    engine_, sources = predictor(views, anchor, "fold-001", tmp_path / "same")
    path = tmp_path / "same.parquet"
    engine_.evaluate(sources["evaluation"], destination=path)
    assert sha256(path) == record["predictions"]["evaluation"]["sha256"]
    # La misma semilla sin el estado del ancla predice otros valores.
    fresh = partial(
        candidate_run.CandidateInputAdapter,
        dtype=torch.float64,
        parameter_seed=42,
        feature_seed=43,
        key_seed=44,
    )
    other, sources = predictor(views, anchor, "fold-001", tmp_path / "fresh", adapter=fresh)
    other.evaluate(sources["evaluation"], destination=tmp_path / "fresh.parquet")
    carried_values = pq.read_table(output / record["predictions"]["evaluation"]["path"])
    fresh_values = pq.read_table(tmp_path / "fresh.parquet")
    assert not np.array_equal(
        carried_values["prediction"].to_numpy(), fresh_values["prediction"].to_numpy()
    )


def test_carried_bank_starts_empty_and_reads_only_matured_labels(views, anchor, carried, tmp_path):
    engine_, sources = predictor(views, anchor, "fold-001", tmp_path / "audit", audit=True)
    # El ancla sí admitió episodios en su propia evaluación.
    assert anchor.report["predictions"]["evaluation"]["metrics"]["admitted"] > 0
    for name in ("calibration", "evaluation"):
        engine_.audit.clear()
        path = tmp_path / f"{name}.parquet"
        engine_.evaluate(sources[name], destination=path)
        assert sha256(path) == carried.report["predictions"][name]["sha256"]
        phase = sources[name].phase
        audit = engine_.audit
        predictions = [entry for entry in audit if entry[0] == "prediction"]
        admits = [entry for entry in audit if entry[0] == "admit"]
        labels = [entry for entry in audit if entry[0] == "label"]
        assert predictions and labels
        assert {entry[1] for entry in audit} == {name}
        # Nada del ancla ni de tramos anteriores: la primera predicción lee un banco vacío.
        assert predictions[0][5] == 0
        for _, _, _, at, _, seen in predictions:
            assert phase.decision_start <= at < phase.decision_end
            assert seen == sum(len(ids) for _, _, when, ids, _ in admits if when < at)
        for _, _, _, decision, matured in labels:
            assert phase.decision_start <= decision < matured < phase.decision_end
        assert all(phase.decision_start < when < phase.decision_end for _, _, when, _, _ in admits)
    assert engine_.fold == "fold-001" and engine_.world == walk.WORLD


def test_carry_rejects_an_earlier_window_or_a_changed_anchor(views, anchor, tmp_path):
    with pytest.raises(ValueError, match="no es posterior"):
        carry(views, anchor, "fold-000", tmp_path / "same")
    with pytest.raises(ValueError, match="ventana completa"):
        walk.carry_window(
            anchor.output,
            views.windows["US"]["fold-001"],
            views.windows["US"]["fold-002"],
            tmp_path / "view",
            parent_id="x",
            device="cpu",
        )
    copy = tmp_path / "anchor"
    shutil.copytree(anchor.output, copy)
    window = json.loads((copy / walk.WINDOW_REPORT).read_text())
    changed = SimpleNamespace(output=copy)
    atomic_json(
        copy / walk.WINDOW_REPORT,
        window | dict(checkpoint=window["checkpoint"] | dict(parameters_sha256="0" * 64)),
    )
    with pytest.raises(ValueError, match="no son los elegidos"):
        carry(views, changed, "fold-001", tmp_path / "parameters")
    atomic_json(copy / walk.WINDOW_REPORT, window)
    state = copy / window["checkpoint"]["path"]
    state.write_bytes(state.read_bytes() + b"\0")
    with pytest.raises(ValueError):
        carry(views, changed, "fold-001", tmp_path / "checkpoint")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="directorio nuevo"):
        carry(views, anchor, "fold-001", existing)


def test_joint_window_fits_and_carries_with_one_receipt_per_market(views, tmp_path):
    view = views.windows["US+CN"]["fold-000"]
    report, _ = fit(view, tmp_path / "fit", parent="US+CN/fold-000/gru_episodic")
    assert report["markets"] == ["CN", "US"] and set(report["receipts"]) == {"CN", "US"}
    joint = SimpleNamespace(output=tmp_path / "fit")
    record = carry(views, joint, "fold-001", tmp_path / "carry", scope="US+CN")
    for value in (report, record):
        folder = tmp_path / ("fit" if value is report else "carry")
        table = pq.read_table(folder / value["predictions"]["evaluation"]["path"])
        assert set(table["market"].to_pylist()) == {"CN", "US"}
        rows = 0
        for market in ("CN", "US"):
            receipt = json.loads((folder / value["receipts"][market]["path"]).read_text())
            assert read_window_receipt(receipt).market == market
            rows += receipt["predictions"]["evaluation"]["rows"]
        assert rows == table.num_rows


def test_last_window_predicts_no_row_of_2024(views, tmp_path):
    view = views.windows["US"]["fold-018"]
    report, _ = fit(view, tmp_path / "last", parent="US/fold-018/gru_episodic")
    manifest = json.loads(view.read_text())
    fold = manifest["temporal_view"]["fold"]
    assert fold["evaluation"][1] == "2024-01-01"
    for name in PARTITIONS[1:]:
        table = pq.read_table(tmp_path / "last" / report["predictions"][name]["path"])
        assert pa.compute.max(pa.compute.year(table["prediction_at"])).as_py() <= 2023
        comparison._check_segment(table, fold, name, ["US"], name)
    receipt = json.loads((tmp_path / "last" / report["receipts"]["US"]["path"]).read_text())
    assert read_window_receipt(receipt).segment("evaluation")[1] <= 1_704_067_200_000_000


def section(**changes):
    value = dict(recipe=str(RECIPE), arms={"gru_episodic": "m1_k1"}, search_seed=42)
    return value | changes


def campaign_with_candidate(folder, *, scopes, seeds=(42,)):
    """Campaña reducida con una referencia GRU y la candidata declarada en su sección."""
    path = write_campaign(folder, scopes=scopes, arms=("gru",))
    declared = json.loads((folder / "comparison.json").read_text())
    full = json.loads(COMPARISON.read_text())["arms"]["gru_episodic"]
    declared["arms"]["gru_episodic"] = full | dict(seeds=list(seeds))
    declared["comparison"]["families"]["references_vs_zero"]["variants"].append("gru_episodic")
    atomic_json(folder / "comparison.json", declared)
    atomic_json(path, json.loads(path.read_text()) | {"episodic_gru": section()})
    return path


def test_campaign_runs_the_candidate_with_the_rows_of_the_other_arms(views, tmp_path, monkeypatch):
    scope = "US+CN"
    campaign = campaign_with_candidate(tmp_path / "config", scopes=(scope,))
    declared = walk.campaign_case

    def reduced(case):
        # Comprueba receta, huella y semilla del caso y conserva su precisión declarada.
        _, model = declared(case)
        return SMALL, model

    monkeypatch.setattr(walk, "campaign_case", reduced)
    executors = doubles(Recorder())
    candidate = partial(engine._episodic_gru, device="cpu", optimizer_factory=RecordingOptimizer)
    for kind in (plan.FIT, plan.CARRY):
        key = (plan.EPISODIC, kind)
        executors[key] = dict(engine.EXECUTORS[key], run=candidate, device="cpu")
    output = tmp_path / "out"
    views_ = {scope: views.directories[scope]}
    summary = engine.run_campaign(
        campaign,
        views_,
        output,
        executors=executors,
        lease=nullcontext,
        stop=SimpleNamespace(requested=False),
    )
    assert summary["status"] == "completed"
    # Cinco ventanas reentrenadas y ocho trasladadas: GRU con 2 + 2 ajustes y 3 traslados,
    # candidata con una búsqueda y un traslado por ventana con su única semilla.
    assert summary["planned"] == dict(training_jobs=5 * 4 + 5, prediction_jobs=8 * 3 + 8)
    jobs = plan.plan_campaign(plan.load_campaign(campaign))
    receipts = {
        job["id"]: json.loads((output / "jobs" / job["id"] / "receipt.json").read_text())
        for job in jobs
    }
    for job in jobs:
        if job["arm"] != "gru_episodic":
            continue
        receipt = receipts[job["id"]]
        name = "carry-s42" if job["kind"] == plan.CARRY else "search-gru-00"
        reference = receipts[f"{scope}/{job['window']}/gru/{name}"]
        for partition in ("calibration", "evaluation"):
            digests = {
                item["predictions"][partition]["rows_sha256"] for item in (receipt, reference)
            }
            assert len(digests) == 1, (job["id"], partition)
        if job["kind"] == plan.CARRY:
            anchor = receipts[job["depends"][0]]
            report = json.loads((output / receipt["report"]["path"]).read_text())
            assert report["anchor"]["checkpoint_sha256"] == anchor["parent"]["sha256"]
            assert receipt["parent"] == anchor["parent"]
        # Con un único candidato, cada trabajo es el predictor elegido de su semilla.
        folder = output / "windows" / scope / job["window"] / "gru_episodic" / f"seed-{job['seed']}"
        for market in ("CN", "US"):
            own = output / receipt["attempt"] / "receipts" / f"{market}.json"
            assert json.loads((folder / f"{market}.json").read_text()) == json.loads(
                own.read_text()
            )
    path = engine.write_sources(campaign, views_, output, scope)
    manifest = json.loads(path.read_text())
    assert set(manifest["arms"]["gru_episodic"]) == {"42"}
    assert set(manifest["arms"]["gru_episodic"]["42"]["fold-001"]) == {
        "input_policy",
        "view_sha256",
        "calibration",
        "evaluation",
    }
    report, sessions = comparison.evaluate_walk_forward(
        tmp_path / "config/comparison.json", path, scope
    )
    assert report["status"] == "completed" and set(report["arms"]) == {
        "zero",
        "gru",
        "gru_episodic",
    }
    assert pa.compute.max(pa.compute.year(sessions["prediction_at"])).as_py() <= 2023


def test_planner_declares_candidate_jobs_only_with_its_section(tmp_path):
    from tests.training.test_campaign_plan import write_variant

    declared = plan.check_campaign(ROOT / "configs/baselines/historical-masked-campaign-b.json")
    assert "accumulation_rows" in declared["pending_families"]["episodic_gru"]["pending"]
    assert "gru_episodic" not in declared["counts"]["scopes"]["US"]["arms"]
    path = write_variant(
        tmp_path,
        "B",
        episodic_gru=section(),
        titans_mac=None,
        limits=dict(max_training_jobs=629 + 17 * 3, max_prediction_jobs=532 + 28 * 3),
    )
    report = plan.check_campaign(path)
    assert "episodic_gru" not in report["pending_families"]
    counts = report["counts"]
    assert (counts["training_jobs"], counts["prediction_jobs"]) == (680, 616)
    for scope, (trained, carried) in dict(US=(7, 12), CN=(5, 8)).items():
        assert counts["scopes"][scope]["arms"]["gru_episodic"] == {
            str(seed): dict(fit=trained, carry=carried) for seed in (42, 43, 44)
        }
    jobs = [
        job for job in plan.plan_campaign(plan.load_campaign(path)) if job["arm"] == "gru_episodic"
    ]
    first = next(job for job in jobs if job["stage"] == "search")
    assert first["id"] == "US/fold-000/gru_episodic/search-m1_k1"
    assert (first["family"], first["model"], first["seed"]) == ("episodic_gru", "episodic_gru", 42)
    assert first["case"] == dict(
        recipe=str(RECIPE.resolve()), recipe_sha256=sha256(RECIPE), variant="m1_k1", seed=42
    )
    finalists = [
        job for job in jobs if job["stage"] == "finalist" and job["id"].startswith("US/fold-000/")
    ]
    assert [job["depends"] for job in finalists] == [[first["id"]]] * 2
    carry = next(job for job in jobs if job["id"] == "US/fold-001/gru_episodic/carry-s43")
    assert carry["anchor"] == "fold-000"
    assert carry["depends"] == ["US/fold-000/gru_episodic/finalist-s43"]
    # Con los límites declarados, que ya cuentan Titans-MAC, la sección no cabe.
    with pytest.raises(ValueError, match="952 trabajos"):
        plan.check_campaign(write_variant(tmp_path, "B", episodic_gru=section()))


def edited_recipe(tmp_path, **changes):
    document = json.loads(RECIPE.read_text())
    document["protocol"] = str((RECIPE.parent / document["protocol"]).resolve())
    for key, value in changes.items():
        document[key] = document[key] | value if isinstance(value, dict) else value
    path = tmp_path / "recipe.json"
    atomic_json(path, document)
    return str(path)


@pytest.mark.parametrize(
    "changes",
    [
        dict(arms={"gru_episodic": "m9_k1"}),
        dict(arms={}),
        dict(search_seed=7),
        dict(variant="m1_k1"),
        dict(recipe=lambda tmp: edited_recipe(tmp, recipe=dict(epochs=29))),
        dict(recipe=lambda tmp: edited_recipe(tmp, recipe_name="other")),
        dict(recipe=lambda tmp: edited_recipe(tmp, model=dict(output_head="point"))),
        dict(recipe=lambda tmp: edited_recipe(tmp, model=dict(seeds=[42, 43]))),
        dict(recipe=lambda tmp: edited_recipe(tmp, arm="gru")),
    ],
    ids=[
        "unknown_variant",
        "missing_arm",
        "foreign_search_seed",
        "extra_field",
        "other_budget",
        "other_recipe",
        "point_head",
        "missing_seed",
        "other_arm",
    ],
)
def test_planner_rejects_a_candidate_section_that_changes_the_design(tmp_path, changes):
    from tests.training.test_campaign_plan import write_variant

    changes = {k: v(tmp_path) if callable(v) else v for k, v in changes.items()}
    path = write_variant(tmp_path, "B", episodic_gru=section(**changes))
    with pytest.raises(ValueError):
        plan.load_campaign(path)


def test_campaign_case_reads_the_planned_recipe(tmp_path):
    case = dict(
        recipe=str(RECIPE.resolve()), recipe_sha256=sha256(RECIPE), variant="m1_k1", seed=43
    )
    recipe, model = walk.campaign_case(case)
    assert recipe == candidate_run.load_recipe(RECIPE, variant="m1_k1")[0]
    assert model == dict(feature_seed=43, key_seed=44, dtype="float32")
    with pytest.raises(ValueError, match="cambió"):
        walk.campaign_case(case | dict(recipe_sha256="0" * 64))
    with pytest.raises(ValueError, match="no pertenece"):
        walk.campaign_case(case | dict(seed=7))
    assert plan.CANDIDATE_RECIPE == candidate_run.RECIPE
    for kind, report in ((plan.FIT, walk.WINDOW_REPORT), (plan.CARRY, "carry.json")):
        executor = engine.EXECUTORS[plan.EPISODIC, kind]
        assert executor["report"] == report and executor["device"] == "cuda"
        assert executor["resumable"] is (kind == plan.FIT)


def test_campaign_reads_the_quantiles_declared_for_the_candidate_arm(views, tmp_path):
    # Los dobles solo escriben cuantiles para las referencias neuronales.
    campaign = campaign_with_candidate(tmp_path / "config", scopes=("US+CN",))
    with pytest.raises(ValueError, match="Faltan columnas"):
        engine.run_campaign(
            campaign,
            {"US+CN": views.directories["US+CN"]},
            tmp_path / "out",
            executors=doubles(Recorder()),
            lease=nullcontext,
            stop=SimpleNamespace(requested=False),
        )
