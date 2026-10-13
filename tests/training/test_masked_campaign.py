"""Ejecución de la campaña con máscaras mediante dobles que registran trabajos sin ajustar.

Las vistas se preparan sobre el corpus técnico con filas en todos los años. Los
ejecutores sustitutos anotan trabajo, vista, ventana, semilla y caso, y escriben
predicciones nulas con las filas exactas de la vista. Ningún modelo se ajusta, no se
aplican pasos de optimizador y la GPU no se usa. Como los ejecutores son dobles, las
pruebas admiten la campaña con una protección temporal permitida (`learning_doubles`) y
las pruebas del bloqueo declaran la suya.
"""

import hashlib
import json
import shutil
import threading
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.environments.walk_forward_receipt import prediction_fingerprint, read_window_receipt
from mars_titan.evaluation import walk_forward_comparison as comparison
from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from mars_titan.training import masked_campaign as engine
from mars_titan.training.campaign_plan import load_campaign, plan_campaign
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.learning_hold import LearningHoldError
from tests.hardware.platform_doubles import gb10, laptop
from tests.training.test_walk_forward_v2_views import fixture

pytestmark = pytest.mark.usefixtures("learning_doubles")

CONFIGS = Path("configs")
ARMS = ("gru", "ridge", "xgboost")


def write_campaign(folder, *, variant="B", scopes=("US",), arms=ARMS, **changes):
    """Campaña reducida: tres brazos, un candidato tabular y protocolos versionados."""
    folder.mkdir(parents=True, exist_ok=True)
    declared = json.loads(
        (CONFIGS / "evaluation/historical-masked-2000-comparison.json").read_text()
    )
    for scope in declared["scopes"].values():
        scope["protocols"] = {
            market: str((CONFIGS / "evaluation" / name).resolve())
            for market, name in scope["protocols"].items()
        }
    declared["arms"] = {k: v for k, v in declared["arms"].items() if k in {"zero", *arms}}
    declared["comparison"].update(
        replicates=20,
        families=dict(references_vs_zero=dict(kind="delta", base="zero", variants=list(arms))),
    )
    atomic_json(folder / "comparison.json", declared)
    tabular = json.loads((CONFIGS / "baselines/tabular-historical-masked.json").read_text())
    tabular.update(ridge_alphas=[1.0], depths=[3], bins=[64], rates=[0.1])
    atomic_json(folder / "tabular.json", tabular)
    value = json.loads(
        (CONFIGS / f"baselines/historical-masked-campaign-{variant.lower()}.json").read_text()
    )
    value.update(comparison="comparison.json", scopes=list(scopes))
    # La comparación reducida no declara los brazos de Titans-MAC. test_titans_campaign los añade.
    value.pop("titans_mac")
    value["neural"]["arms"] = {arm: arm for arm in arms if arm not in {"ridge", "xgboost"}}
    value["tabular"].update(
        config="tabular.json", arms={arm: arm for arm in arms if arm in {"ridge", "xgboost"}}
    )
    value["limits"] = dict(max_training_jobs=10_000, max_prediction_jobs=10_000)
    for key, item in changes.items():
        value[key] = value[key] | item if isinstance(item, dict) else item
    atomic_json(folder / "campaign.json", value)
    return folder / "campaign.json"


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("masked-campaign")
    data = fixture(root / "data", ("US", "CN"))
    campaign = write_campaign(root / "config", scopes=("US", "CN", "US+CN"))
    views = engine.prepare_views(campaign, data.parent, root / "views")
    return SimpleNamespace(
        root=root,
        parent=data.parent,
        views={scope: root / "views" / scope for scope in views},
        checked=views,
    )


def scores(job):
    """MAE de validación fijado para que el ganador no dependa del orden del plan."""
    if job["stage"] != "search":
        return 0.5
    return {"gru-00": 0.02, "gru-10": 0.01}.get(job["candidate"], 0.03)


class Recorder:
    """Ejecutor sustituto: registra el trabajo y escribe filas nulas de la vista."""

    def __init__(self, *, stop=None, interrupt_at=None, mutate=None, delay=0.0):
        self.calls, self.cache, self.times = [], {}, {}
        self.stop, self.interrupt_at, self.mutate, self.delay = stop, interrupt_at, mutate, delay
        self.active, self.peak, self.lock = 0, 0, threading.Lock()

    def rows(self, view, partition):
        key = (str(view), partition)
        if key not in self.cache:
            dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
            parts = dict(sample_id=[], market=[], prediction_at=[], target=[])
            for batch in dataset.batches(partition=partition, batch_size=64, epoch=0, seed=0):
                parts["sample_id"] += list(batch["sample_ids"])
                parts["market"] += list(batch["market"])
                parts["prediction_at"].append(np.asarray(batch["prediction_at"]))
                parts["target"].append(np.asarray(batch["target"], dtype=np.float64))
            for name in ("prediction_at", "target"):
                parts[name] = np.concatenate(parts[name])
            self.cache[key] = parts
        return self.cache[key]

    def table(self, run, partition):
        rows = self.rows(run.view, partition)
        count = len(rows["target"])
        columns = dict(
            sample_id=rows["sample_id"],
            asset_id=["/".join(key.split("/")[:2]) for key in rows["sample_id"]],
            market=rows["market"],
            prediction_at=pa.array(rows["prediction_at"], type=pa.timestamp("us", tz="UTC")),
            target=rows["target"],
            prediction=np.zeros(count, dtype=np.float32),
            zero=np.zeros(count, dtype=np.float64),
        )
        if run.job["family"] == "neural_reference":
            columns.update({name: np.zeros(count, dtype=np.float32) for name in QUANTILE_COLUMNS})
        table = pa.table(columns)
        return self.mutate(run, partition, table) if self.mutate else table

    def __call__(self, run):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.calls.append(
                dict(
                    id=run.job["id"],
                    view=run.view,
                    view_sha256=run.view_sha256,
                    window=run.job["window"],
                    seed=run.job["seed"],
                    case=run.case,
                    anchor=run.anchor,
                    folder=run.folder,
                    batch_size=run.batch_size,
                )
            )
            position = len(self.calls)
        started = time.perf_counter()
        try:
            time.sleep(self.delay)
            run.folder.mkdir(parents=True, exist_ok=True)
            if position == self.interrupt_at:
                (run.folder / "partial.bin").write_bytes(b"incompleto")
                self.stop.requested = True
                raise engine.Paused
            predictions = {}
            for partition in ("calibration", "evaluation"):
                path = run.folder / f"{partition}-predictions.parquet"
                pq.write_table(self.table(run, partition), path)
                predictions[partition] = dict(path=path.name, sha256=sha256(path))
            predictions["validation"] = dict(metrics=dict(session_mae=scores(run.job)))
            report = dict(status="completed", final_test_opened=False, predictions=predictions)
            if run.job["kind"] == "carry":
                anchor = json.loads((run.anchor["folder"] / "run.json").read_text())
                report["anchor"] = dict(checkpoint_sha256=anchor["checkpoint"]["sha256"])
                atomic_json(run.folder / "carry.json", report)
                return report
            # Estado elegido ficticio: su huella identifica el padre del recibo de ventana.
            (run.folder / "model.bin").write_text(run.job["id"])
            report["checkpoint"] = dict(path="model.bin", sha256=sha256(run.folder / "model.bin"))
            atomic_json(run.folder / "run.json", report)
            return report
        finally:
            with self.lock:
                self.active -= 1
                self.times[run.job["id"]] = (started, time.perf_counter())


def doubles(recorder, cpu=()):
    return {
        key: dict(entry, run=recorder, device="cpu" if key in cpu else entry["device"])
        for key, entry in engine.EXECUTORS.items()
    }


def run(campaign, views, output, recorder, *, stop=None, cpu=()):
    return engine.run_campaign(
        campaign,
        views,
        output,
        executors=doubles(recorder, cpu),
        lease=nullcontext,
        stop=stop or SimpleNamespace(requested=False),
    )


def test_run_launches_each_planned_job_once_with_its_view_window_and_seed(prepared, tmp_path):
    campaign = write_campaign(tmp_path / "config")
    views = {"US": prepared.views["US"]}
    recorder = Recorder()
    summary = run(campaign, views, tmp_path / "out", recorder)
    assert summary["status"] == "completed"
    # B sobre US: 7 ventanas reentrenadas con 4 + 1 + 3 ajustes y 12 trasladadas con 7 traslados.
    assert summary["planned"] == dict(training_jobs=56, prediction_jobs=84)
    assert summary["completed"] == summary["planned"]
    jobs = {job["id"]: job for job in plan_campaign(load_campaign(campaign))}
    assert [call["id"] for call in recorder.calls] == list(jobs)
    windows = prepared.checked["US"]["windows"]
    for call in recorder.calls:
        job = jobs[call["id"]]
        assert call["view"] == Path(windows[job["window"]]["path"])
        assert call["view_sha256"] == windows[job["window"]]["sha256"]
        assert call["seed"] == job["seed"] and call["window"] == job["window"]
        assert call["batch_size"] == (256 if job["arm"] == "gru" else 1024)
        if job["stage"] == "search":
            assert call["case"] == job["case"] and call["anchor"] is None
        elif job["stage"] == "finalist":
            # El finalista repite el candidato ganador de su ventana con otra semilla.
            assert call["case"]["seed"] == job["seed"]
            winner = "gru-10" if job["arm"] == "gru" else None
            if winner:
                search = jobs[f"US/{job['window']}/gru/search-{winner}"]["case"]
                assert call["case"] == search | dict(seed=job["seed"])
        else:
            assert call["case"] is None
            assert call["anchor"]["view"] == Path(windows[job["anchor"]]["path"])
            expected = (
                f"US/{job['anchor']}/gru/search-gru-10"
                if job["arm"] == "gru" and job["seed"] == 42
                else f"US/{job['anchor']}/{job['arm']}/"
                + ("finalist" if job["seed"] != 42 else "search")
            )
            assert call["anchor"]["job"].startswith(expected)
            assert call["anchor"]["folder"].is_dir()
    receipts = list((tmp_path / "out/jobs").rglob("receipt.json"))
    assert len(receipts) == 140
    for path in receipts:
        receipt = json.loads(path.read_text())
        assert receipt["final_test_opened"] is False
        assert set(receipt["predictions"]) == {"calibration", "evaluation"}


@pytest.mark.parametrize(("interrupt_at", "attempt"), [(1, "attempt-0001"), (5, "attempt-0002")])
def test_resume_after_interruption_redoes_only_the_incomplete_job(
    prepared, tmp_path, interrupt_at, attempt
):
    campaign = write_campaign(tmp_path / "config")
    views = {"US": prepared.views["US"]}
    stop = SimpleNamespace(requested=False)
    first = Recorder(stop=stop, interrupt_at=interrupt_at)
    summary = run(campaign, views, tmp_path / "out", first, stop=stop)
    assert summary["status"] == "paused"
    assert summary["completed"]["training_jobs"] == interrupt_at - 1
    interrupted = first.calls[-1]
    assert (interrupted["folder"] / "partial.bin").is_file()
    second = Recorder()
    summary = run(campaign, views, tmp_path / "out", second)
    assert summary["status"] == "completed"
    done = {call["id"] for call in first.calls[:-1]}
    again = [call["id"] for call in second.calls]
    assert again[0] == interrupted["id"] and not done & set(again)
    assert len(again) == 140 - len(done)
    # Neuronal y XGBoost reanudan su intento. Ridge y los traslados empiezan otro.
    assert second.calls[0]["folder"].name == attempt
    third = Recorder()
    assert run(campaign, views, tmp_path / "out", third)["status"] == "completed"
    assert third.calls == []


def test_resume_rejects_changed_artifacts_and_another_campaign(prepared, tmp_path):
    campaign = write_campaign(tmp_path / "config")
    views = {"US": prepared.views["US"]}
    output = tmp_path / "out"
    run(campaign, views, output, Recorder())
    receipt = next((output / "jobs").rglob("receipt.json"))
    record = json.loads(receipt.read_text())["predictions"]["evaluation"]
    target = output / record["path"]
    table = pq.read_table(target)
    pq.write_table(
        table.set_column(5, "prediction", pa.array(np.ones(len(table), np.float32))), target
    )
    with pytest.raises(ValueError, match="artefacto confirmado"):
        run(campaign, views, output, Recorder())
    other = write_campaign(tmp_path / "other", tabular=dict(cpu_workers=2))
    with pytest.raises(ValueError, match="otra campaña"):
        run(other, views, output, Recorder())


def test_active_hold_rejects_the_campaign_before_creating_outputs(
    prepared, tmp_path, learning_hold
):
    learning_hold(False)
    recorder = Recorder()
    campaign = write_campaign(tmp_path / "config")
    with pytest.raises(LearningHoldError, match="la campaña con máscaras"):
        run(campaign, {"US": prepared.views["US"]}, tmp_path / "out", recorder)
    assert not (tmp_path / "out").exists() and recorder.calls == []


def test_hold_reinstated_during_the_campaign_stops_before_the_next_job(
    prepared, tmp_path, learning_hold
):
    hold = learning_hold(True)

    def reinstate(run, partition, table):
        hold.write_text(json.dumps(dict(training_allowed=False)))
        return table

    recorder = Recorder(mutate=reinstate)
    campaign = write_campaign(tmp_path / "config")
    with pytest.raises(LearningHoldError, match="el trabajo US/fold-000"):
        run(campaign, {"US": prepared.views["US"]}, tmp_path / "out", recorder)
    assert len(recorder.calls) == 1
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["status"] == "blocked" and summary["completed"]["training_jobs"] == 1


def test_mixed_views_scopes_and_policies_are_rejected(prepared, tmp_path):
    campaign = write_campaign(tmp_path / "config")
    with pytest.raises(ValueError, match="ventanas del protocolo"):
        run(campaign, {"US": prepared.views["CN"]}, tmp_path / "a", Recorder())
    with pytest.raises(ValueError, match="vistas preparadas"):
        run(campaign, {"US": prepared.views["US+CN"]}, tmp_path / "b", Recorder())
    with pytest.raises(ValueError, match="exactamente los ámbitos"):
        run(campaign, dict(prepared.views), tmp_path / "c", Recorder())
    strict = tmp_path / "strict"
    strict.mkdir()
    report = json.loads((prepared.views["US"] / "report.json").read_text())
    atomic_json(strict / "report.json", report | dict(input_policy="strict_inputs_v1"))
    with pytest.raises(ValueError, match="política de la campaña"):
        run(campaign, {"US": strict}, tmp_path / "d", Recorder())
    assert not any((tmp_path / name).exists() for name in "abcd")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda run, partition, table: table.slice(1), "filas frente a"),
        (
            lambda run, partition, table: table.set_column(
                4,
                "target",
                pa.array(table["target"].to_numpy() + (run.job["arm"] == "xgboost")),
            ),
            "mismas filas",
        ),
        (
            lambda run, partition, table: pa.concat_tables(
                [
                    table,
                    table.slice(0, 1).set_column(
                        3,
                        "prediction_at",
                        pa.array([1_704_200_000_000_000], type=pa.timestamp("us", tz="UTC")),
                    ),
                ]
            ),
            "reserva final",
        ),
    ],
    ids=["missing_row", "other_target", "row_in_2024"],
)
def test_every_job_must_evaluate_the_same_rows_without_2024(prepared, tmp_path, mutate, message):
    campaign = write_campaign(tmp_path / "config")
    with pytest.raises(ValueError, match=message):
        run(campaign, {"US": prepared.views["US"]}, tmp_path / "out", Recorder(mutate=mutate))


def test_cpu_jobs_respect_the_declared_concurrency_and_gpu_jobs_run_alone(prepared, tmp_path):
    campaign = write_campaign(tmp_path / "config", tabular=dict(cpu_workers=2))
    gpu = Recorder()
    # Cada trabajo CPU dura más que los trabajos CUDA que lo separan del siguiente en el plan.
    cpu = Recorder(delay=0.5)

    def dispatch(run_):
        return (cpu if run_.job["model"] == "ridge" else gpu)(run_)

    executors = doubles(dispatch, cpu={("ridge", "fit"), ("ridge", "carry")})
    summary = engine.run_campaign(
        campaign,
        {"US": prepared.views["US"]},
        tmp_path / "out",
        executors=executors,
        lease=nullcontext,
        stop=SimpleNamespace(requested=False),
    )
    assert summary["status"] == "completed"
    assert gpu.peak == 1 and 1 <= cpu.peak <= 2
    assert len(cpu.calls) == 7 + 12 and len(gpu.calls) == 140 - 19
    # La cola también está acotada: ningún trabajo CUDA empieza con más de dos trabajos CPU
    # anteriores en el plan todavía en curso.
    order = [job["id"] for job in plan_campaign(load_campaign(campaign))]
    for call in gpu.calls:
        start = gpu.times[call["id"]][0]
        earlier = order[: order.index(call["id"])]
        running = [key for key in earlier if key in cpu.times and cpu.times[key][1] > start]
        assert len(running) <= 2, call["id"]


def test_sources_feed_the_walk_forward_comparison_with_the_same_rows(prepared, tmp_path):
    scopes = ("US", "US+CN")
    campaign = write_campaign(tmp_path / "config", scopes=scopes)
    views = {scope: prepared.views[scope] for scope in scopes}
    output = tmp_path / "out"
    assert run(campaign, views, output, Recorder())["status"] == "completed"
    config = tmp_path / "config/comparison.json"
    for scope in scopes:
        path = engine.write_sources(campaign, views, output, scope)
        sources = comparison.load_sources(path, comparison.load_config(config), scope)
        assert set(sources["windows"]) == set(prepared.checked[scope]["windows"])
        manifest = json.loads(path.read_text())
        assert set(manifest["arms"]) == set(ARMS)
        assert set(manifest["arms"]["gru"]["42"]["fold-001"]) == {
            "input_policy",
            "view_sha256",
            "calibration",
            "evaluation",
        }
        assert "calibration" not in manifest["arms"]["ridge"]["42"]["fold-001"]
    report, sessions = comparison.evaluate_walk_forward(config, path, "US+CN")
    assert report["status"] == "completed" and report["final_test_opened"] is False
    assert set(report["arms"]) == {"zero", *ARMS}
    assert sessions.num_rows > 0
    years = pa.compute.year(sessions["prediction_at"]).to_numpy()
    assert years.max() <= 2023


def test_receipts_record_the_platform_and_sources_keep_one_machine(prepared, tmp_path):
    campaign = write_campaign(tmp_path / "config")
    views = {"US": prepared.views["US"]}
    output = tmp_path / "out"
    summary = engine.run_campaign(
        campaign,
        views,
        output,
        executors=doubles(Recorder()),
        lease=nullcontext,
        stop=SimpleNamespace(requested=False),
        platform=laptop(),
    )
    assert summary["status"] == "completed" and summary["platform"] == laptop()
    jobs = [job for job in plan_campaign(load_campaign(campaign)) if job["scope"] == "US"]
    receipts = [output / "jobs" / job["id"] / "receipt.json" for job in jobs]
    assert all(json.loads(path.read_text())["platform"] == laptop() for path in receipts)
    path = engine.write_sources(campaign, views, output, "US")
    manifest = json.loads(path.read_text())
    assert manifest["schema_version"] == 2
    assert manifest["platform"] == dict(
        platform_sha256=laptop()["platform_sha256"],
        recorded=len(jobs),
        unrecorded=0,
        unrecorded_profile=None,
    )
    config = comparison.load_config(tmp_path / "config/comparison.json")
    assert comparison.load_sources(path, config, "US")["platform"] == manifest["platform"]
    # Un traslado no es padre de ningún otro trabajo, así que se puede reescribir su recibo.
    carry = next(job for job in jobs if job["kind"] == engine.CARRY)
    receipt_path = output / "jobs" / carry["id"] / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    atomic_json(receipt_path, dict(receipt, platform=gb10()))
    with pytest.raises(ValueError, match="mezcla plataformas"):
        engine.write_sources(campaign, views, output, "US")
    # Un recibo anterior al registro solo entra con el perfil al que se atribuye.
    atomic_json(receipt_path, {k: v for k, v in receipt.items() if k != "platform"})
    with pytest.raises(ValueError, match="no registran su plataforma"):
        engine.write_sources(campaign, views, output, "US")
    with pytest.raises(ValueError, match="atribuido"):
        engine.write_sources(campaign, views, output, "US", unrecorded_platform="dgx-gb10")
    path = engine.write_sources(campaign, views, output, "US", unrecorded_platform="rtx4070-laptop")
    assert json.loads(path.read_text())["platform"] == dict(
        platform_sha256=laptop()["platform_sha256"],
        recorded=len(jobs) - 1,
        unrecorded=1,
        unrecorded_profile="rtx4070-laptop",
    )


def selected_fit(jobs, scope, window, arm, seed):
    """Ajuste elegido según las puntuaciones fijadas: gru-10 o el menor identificador."""
    if seed != 42:
        return f"{scope}/{window}/{arm}/finalist-s{seed}"
    searches = [
        job
        for job in jobs
        if (job["scope"], job["window"], job["arm"], job["stage"]) == (scope, window, arm, "search")
    ]
    return min(searches, key=lambda job: (scores(job), job["id"]))["id"]


def view_maturity(root, window, partitions):
    """Mayor maduración de las etiquetas de esos tramos, leída sin el código de la campaña."""
    latest = None
    for path in (root / window / "labels").rglob("labels.parquet"):
        table = pq.read_table(path, columns=["partition", "target_available_at"])
        table = table.filter(pa.compute.is_in(table["partition"], value_set=pa.array(partitions)))
        if table.num_rows:
            values = table["target_available_at"].cast(pa.timestamp("us", tz="UTC"))
            value = int(pa.compute.max(values.cast(pa.int64())).as_py())
            latest = value if latest is None else max(latest, value)
    return latest


def test_window_receipts_follow_the_selected_predictor_of_each_seed(prepared, tmp_path):
    scopes = ("US", "US+CN")
    campaign = write_campaign(tmp_path / "config", scopes=scopes)
    views = {scope: prepared.views[scope] for scope in scopes}
    output = tmp_path / "out"
    assert run(campaign, views, output, Recorder())["status"] == "completed"
    jobs = plan_campaign(load_campaign(campaign))
    groups = {}
    for job in jobs:
        groups.setdefault((job["scope"], job["window"], job["arm"], job["seed"]), []).append(job)
    written = sorted((output / "windows").rglob("*.json"))
    # 19 ventanas US y 13 conjuntas con dos mercados, por brazo y semilla (3 + 1 + 3).
    assert len(groups) == (19 + 13) * 7
    assert len(written) == (19 + 2 * 13) * 7
    for (scope, window, arm, seed), members in groups.items():
        anchor = members[0]["anchor"]
        parent = selected_fit(jobs, scope, anchor, arm, seed)
        carry = [job["id"] for job in members if job["kind"] == "carry"]
        source = carry[0] if carry else parent
        receipt = json.loads((output / "jobs" / source / "receipt.json").read_text())
        rows = 0
        for market in comparison.SCOPES[scope]:
            path = output / "windows" / scope / window / arm / f"seed-{seed}" / f"{market}.json"
            record = json.loads(path.read_text())
            checked = read_window_receipt(record)
            assert checked.market == market and checked.fold == window
            assert record["parent"] == dict(
                id=parent, sha256=hashlib.sha256(parent.encode()).hexdigest()
            )
            start = np.datetime64(record["fold"]["evaluation"][0], "us").astype(np.int64)
            # La última etiqueta leída por el predictor elegido o por su calibración común.
            expected = view_maturity(views[scope], anchor, ("train", "validation", "calibration"))
            if anchor != window:
                expected = max(expected, view_maturity(views[scope], window, ("calibration",)))
            assert record["labels_used_until"] == expected < int(start)
            assert set(record["predictions"]) == {"calibration", "evaluation"}
            for partition, declared in record["predictions"].items():
                table = pq.read_table(output / receipt["predictions"][partition]["path"])
                table = table.filter(pa.compute.equal(table["market"], market))
                expected = prediction_fingerprint(
                    table["prediction_at"].cast(pa.int64()).to_numpy(),
                    table["asset_id"].to_pylist(),
                    table["prediction"].to_numpy(),
                )
                assert (declared["rows"], declared["sha256"]) == expected
                rows += declared["rows"] * (partition == "evaluation")
        assert rows == receipt["predictions"]["evaluation"]["rows"]


def future_label_views(source, target, window):
    """Copia de vistas con una etiqueta de ajuste que madura al empezar la evaluación.

    Solo cambia el instante de maduración de una fila de ajuste. Las huellas del manifiesto
    y del informe se recalculan para que la vista siga pasando sus comprobaciones.
    """
    shutil.copytree(source, target)
    report = json.loads((target / "report.json").read_text())
    for row in report["folds"]:
        path = target / row["id"] / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["roots"]["labels"] = str((target / row["id"] / "labels").resolve())
        if row["id"] == window:
            asset = next(a for a in manifest["assets"] if a["counts"]["train"])
            labels = target / window / "labels" / asset["market"] / asset["symbol"]
            table = pq.read_table(labels / "labels.parquet")
            first = table["partition"].to_pylist().index("train")
            start = manifest["temporal_view"]["fold"]["evaluation"][0]
            times = table["target_available_at"].to_pylist()
            times[first] = datetime.fromisoformat(start).replace(tzinfo=UTC)
            column = pa.array(times, type=table["target_available_at"].type)
            index = table.schema.get_field_index("target_available_at")
            table = table.set_column(index, "target_available_at", column)
            pq.write_table(table, labels / "labels.parquet")
            asset["labels_sha256"] = sha256(labels / "labels.parquet")
        atomic_json(path, manifest)
        row["manifest_sha256"] = sha256(path)
    atomic_json(target / "report.json", report)
    return target


def test_a_future_label_in_the_fit_view_blocks_the_window_receipt(prepared, tmp_path):
    # Un ejecutor que no validase sus etiquetas leería las filas sin notar la maduración
    # adelantada. El recibo de ventana deriva su límite de la vista y la ventana no se publica.
    campaign = write_campaign(tmp_path / "config")
    clean = prepared.views["US"]
    tampered = future_label_views(clean, tmp_path / "views" / "US", "fold-000")
    with pytest.raises(ValueError, match="no concilia|no conserva"):
        # El lector del corpus rechaza por sí mismo la etiqueta que cruza la frontera.
        CorpusDataset(tampered / "fold-000" / "manifest.json", input_policy=HISTORICAL_MASKED)
        Recorder().rows(tampered / "fold-000" / "manifest.json", "train")

    class Unchecked(Recorder):
        def rows(self, view, partition):
            return super().rows(clean / Path(view).relative_to(tampered), partition)

    output = tmp_path / "out"
    with pytest.raises(ValueError, match="madurar antes de la evaluación"):
        run(campaign, {"US": tampered}, output, Unchecked())
    assert not (output / "windows/US/fold-000").exists()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "failed"


def test_window_receipts_are_checked_on_resume_and_need_the_anchor_state(prepared, tmp_path):
    campaign = write_campaign(tmp_path / "config")
    views = {"US": prepared.views["US"]}
    output = tmp_path / "out"
    run(campaign, views, output, Recorder())
    path = output / "windows/US/fold-001/gru/seed-43/US.json"
    record = json.loads(path.read_text())
    atomic_json(path, record | dict(labels_used_until=record["labels_used_until"] - 1))
    with pytest.raises(ValueError, match="no corresponde a sus trabajos"):
        run(campaign, views, output, Recorder())

    class OtherAnchor(Recorder):
        def __call__(self, run_):
            report = super().__call__(run_)
            if run_.job["kind"] == "carry":
                report["anchor"] = dict(checkpoint_sha256="0" * 64)
            return report

    with pytest.raises(ValueError, match="no parte del estado elegido"):
        run(campaign, views, tmp_path / "other", OtherAnchor())


def test_sources_list_the_arms_without_a_connected_trainer(prepared, tmp_path):
    campaign = write_campaign(tmp_path / "config")
    views = {"US": prepared.views["US"]}
    run(campaign, views, tmp_path / "out", Recorder())
    full = CONFIGS / "evaluation/historical-masked-2000-comparison.json"
    with pytest.raises(ValueError, match="Faltan productores para .*titans_mac_online"):
        engine.write_sources(campaign, views, tmp_path / "out", "US", comparison_path=full)
    assert not (tmp_path / "out/sources/US.json").exists()


def test_sources_require_every_job_confirmed(prepared, tmp_path):
    campaign = write_campaign(tmp_path / "config")
    views = {"US": prepared.views["US"]}
    stop = SimpleNamespace(requested=False)
    run(campaign, views, tmp_path / "out", Recorder(stop=stop, interrupt_at=3), stop=stop)
    with pytest.raises(ValueError, match="sin confirmar|Falta confirmar"):
        engine.write_sources(campaign, views, tmp_path / "out", "US")


def test_prepared_views_cover_every_window_without_2024_rows(prepared):
    for scope, record in prepared.checked.items():
        assert len(record["windows"]) == (19 if scope == "US" else 13)
        for window in record["windows"].values():
            manifest = json.loads(Path(window["path"]).read_text())
            assert manifest["final_test_opened"] is False
            labels = Path(manifest["roots"]["labels"])
            for path in labels.rglob("labels.parquet"):
                table = pq.read_table(path, columns=["prediction_at", "partition"])
                used = table.filter(pa.compute.is_valid(table["partition"]))
                assert pa.compute.max(pa.compute.year(used["prediction_at"])).as_py() <= 2023
    # Preparar otra vez reutiliza las vistas y la proyección confirmadas.
    again = engine.prepare_views(
        prepared.root / "config/campaign.json", prepared.parent, prepared.root / "views"
    )
    assert again == prepared.checked


def test_row_digest_ignores_order_but_not_targets():
    table = pa.table(
        dict(
            asset_id=["US/A", "US/B"],
            market=["US", "US"],
            prediction_at=pa.array([1, 2], type=pa.timestamp("us", tz="UTC")),
            target=[0.1, 0.2],
        )
    )
    assert engine._rows_digest(table) == engine._rows_digest(table.take([1, 0]))
    changed = table.set_column(3, "target", pa.array([0.1, 0.3]))
    assert engine._rows_digest(table) != engine._rows_digest(changed)
