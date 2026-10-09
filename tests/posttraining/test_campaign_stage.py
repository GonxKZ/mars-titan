"""Etapa de la matriz por ventana walk-forward, sin pasos de optimizador.

Los recuentos se comprueban con las configuraciones del repositorio. El recorrido completo
usa la campaña base reducida de `campaign_fixture`, con padres de cuantiles reales y pesos
iniciales, y el optimizador del fixture `recorder`, que no hereda de
`torch.optim.Optimizer`, registra gradientes y exige pesos sin cambios. La GPU no se usa.
"""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import atomic_json
from mars_titan.environments.walk_forward_receipt import read_window_receipt
from mars_titan.evaluation import walk_forward_comparison as comparison
from mars_titan.models.quantile_head import QUANTILE_COLUMNS, QUANTILE_HEAD
from mars_titan.posttraining import adapter_matrix, campaign_stage, matrix_runs
from mars_titan.training.learning_hold import LearningHoldError
from tests.posttraining.campaign_fixture import CpuLease, base_campaign

CONFIGS = Path("configs/posttraining")
FINAL_TEST = int(np.datetime64("2024-01-01", "us").astype(np.int64))


@pytest.mark.parametrize(
    ("variant", "fits", "carries", "scopes"),
    [
        ("a", 3915, 0, {"US": (19, 0), "CN": (13, 0), "US+CN": (13, 0)}),
        ("b", 1479, 2436, {"US": (7, 12), "CN": (5, 8), "US+CN": (5, 8)}),
    ],
)
def test_repository_stages_plan_every_window_arm_seed_and_case(variant, fits, carries, scopes):
    path = CONFIGS / f"historical-masked-adapter-stage-{variant}.json"
    result = campaign_stage.check_stage(path)
    counts = result["counts"]
    assert (counts["training_jobs"], counts["prediction_jobs"]) == (fits, carries)
    stage = campaign_stage.load_stage(path)
    assert stage["limits"] == dict(max_training_jobs=fits, max_prediction_jobs=carries)
    assert result["excluded_controls"].keys() == {"linear_residual"}
    assert result["objectives"]["adapters"] == "neural_pinball"
    # Por ventana y semilla: cuatro brazos y la continuación en cada red recurrente y
    # DLinear, ocho brazos y la continuación en el Transformer.
    per_window = 3 * (4 * 5 + 9)
    for scope, (trained, carried) in scopes.items():
        entry = counts["scopes"][scope]
        assert (len(entry["retrained_windows"]), entry["carried_windows"]) == (trained, carried)
        assert entry["training_jobs"] == trained * per_window
        assert entry["prediction_jobs"] == carried * per_window
        assert len(entry["arms"]) == 4 * 5 + 9
        for arm, seeds in entry["arms"].items():
            assert comparison._name(arm)
            assert seeds == {
                str(seed): dict(fit=trained, carry=carried) for seed in (42, 43, 44)
            }, arm


def test_plan_carries_each_case_from_its_anchor_with_the_same_case():
    stage = campaign_stage.load_stage(CONFIGS / "historical-masked-adapter-stage-b.json")
    jobs = {job["id"]: job for job in campaign_stage.plan_stage(stage)}
    matrix, digest = stage["matrix"], stage["matrix_sha256"]
    for job in jobs.values():
        assert job["case"]["mode"] == "neural_pinball" and job["control"] != "linear_residual"
        expected = {
            item["id"]: item["case"]
            for item in adapter_matrix.cases(matrix, digest, job["family"], head=QUANTILE_HEAD)
        }
        assert job["case"] == expected[f"seed-{job['seed']}/{job['point']}"]
        if job["kind"] == "carry":
            (anchor,) = job["depends"]
            fitted = jobs[anchor]
            assert fitted["kind"] == "fit" and fitted["window"] == job["anchor"]
            assert (fitted["arm"], fitted["seed"], fitted["case"]) == (
                job["arm"],
                job["seed"],
                job["case"],
            )
            assert list(jobs).index(anchor) < list(jobs).index(job["id"])
        else:
            assert job["depends"] == [] and job["anchor"] == job["window"]


def mutated(tmp_path, change):
    value = json.loads((CONFIGS / "historical-masked-adapter-stage-a.json").read_text())
    value.update(
        campaign=str(Path("configs/baselines/historical-masked-campaign-a.json").resolve()),
        matrix=str((CONFIGS / "adapter-matrix-v2.json").resolve()),
    )
    change(value)
    path = tmp_path / "stage.json"
    atomic_json(path, value)
    return path


INVALID = {
    "scalar_matrix": lambda v: v.update(matrix=str((CONFIGS / "adapter-matrix-v1.json").resolve())),
    "unknown_arm": lambda v: v.update(arms=["gru", "ridge"]),
    "duplicated_arm": lambda v: v.update(arms=["gru", "gru"]),
    "unordered_scopes": lambda v: v.update(scopes=["CN", "US"]),
    "unknown_scope": lambda v: v.update(scopes=["EU"]),
    "retention": lambda v: v.update(ordered_retention="forever"),
    "test_opened": lambda v: v.update(final_test_opened=True),
    "limit": lambda v: v["limits"].update(max_training_jobs=3914),
    "status": lambda v: v.update(status="executed"),
}


def test_the_unchanged_declaration_is_accepted(tmp_path):
    assert campaign_stage.check_stage(mutated(tmp_path, lambda _: None))["status"] == "checked"


@pytest.mark.parametrize("name", sorted(INVALID))
def test_stage_rejects_inconsistent_declarations(tmp_path, name):
    with pytest.raises(ValueError):
        campaign_stage.check_stage(mutated(tmp_path, INVALID[name]))


def test_loading_a_stage_requires_objectives_for_the_quantile_head(tmp_path):
    # La carga ya rechaza la matriz escalar, antes de planificar ningún trabajo.
    with pytest.raises(ValueError, match="escalar"):
        campaign_stage.load_stage(mutated(tmp_path, INVALID["scalar_matrix"]))


@pytest.fixture(scope="module")
def base_a(tmp_path_factory):
    return base_campaign(tmp_path_factory.mktemp("stage-a"), "A")


@pytest.fixture(scope="module")
def base_b(tmp_path_factory):
    return base_campaign(tmp_path_factory.mktemp("stage-b"), "B")


def run(base, output, stop=None):
    return campaign_stage.run_stage(
        base.stage,
        base.views,
        base.output,
        output,
        lease=CpuLease,
        stop=stop or SimpleNamespace(requested=False),
        device="cpu",
    )


def receipts(output):
    return {
        str(path.parent.relative_to(output / "jobs")): json.loads(path.read_text())
        for path in (output / "jobs").rglob("receipt.json")
    }


def base_predictions(base, job, partition):
    receipt = json.loads((base.output / "jobs" / job / "receipt.json").read_text())
    return pq.read_table(base.output / receipt["predictions"][partition]["path"])


def by_sample(table):
    order = np.argsort(np.asarray(table["sample_id"].to_pylist()))
    return {name: np.asarray(table[name].to_pylist())[order] for name in table.column_names}


def test_variant_a_fits_every_case_with_equal_updates_from_the_window_parent(
    base_a, tmp_path, recorder
):
    output = tmp_path / "stage"
    summary = run(base_a, output)
    assert summary["status"] == "completed"
    assert summary["planned"] == summary["completed"] == dict(training_jobs=10, prediction_jobs=0)
    found = receipts(output)
    assert len(found) == 10
    # Igualdad de actualizaciones: en cada ventana, cada brazo y la continuación aplican
    # las del plan de su padre. Entre ventanas cambian con las sesiones de ajuste.
    by_window = {}
    for job_id, receipt in found.items():
        by_window.setdefault(job_id.split("/")[1], set()).add(receipt["updates"])
    assert all(len(values) == 1 and min(values) > 0 for values in by_window.values())
    windows = {window: values.pop() for window, values in by_window.items()}
    assert sorted(len(item.calls) for item in recorder.optimizers) == sorted(
        [windows["fold-000"]] * 5 + [windows["fold-001"]] * 5
    )
    for name, budget in summary["budgets"].items():
        assert budget["head"] == QUANTILE_HEAD
        planned = {row["updates"] for row in budget["rows"][1:]}
        assert planned == {windows[name.split("/")[1]]}
        assert budget["updates_per_epoch"] == windows[name.split("/")[1]]
        assert [row["control"] for row in budget["rows"]].count("full_continuation") == 1
    stage_jobs = {
        job["id"]: job for job in campaign_stage.plan_stage(campaign_stage.load_stage(base_a.stage))
    }
    for job_id, receipt in found.items():
        job = stage_jobs[job_id]
        window = job["window"]
        # El padre de la ventana es el ganador de la búsqueda de esa ventana.
        assert receipt["identity"]["parent"]["job"] == f"US/{window}/gru/search-gru-10"
        assert receipt["selection"]["best_epoch"] == 0
        assert receipt["identity"]["case"]["mode"] == "neural_pinball"
        for partition in ("calibration", "evaluation"):
            table = pq.read_table(output / receipt["predictions"][partition]["path"])
            times = table["prediction_at"].cast(pa.int64()).to_numpy()
            assert (times < FINAL_TEST).all()
            levels = np.column_stack([table[name].to_numpy() for name in QUANTILE_COLUMNS])
            assert (np.diff(levels, axis=1) >= 0).all()
            # Sin pasos aplicados, cada brazo predice exactamente la salida del padre elegido.
            mine = by_sample(table)
            parent = by_sample(
                base_predictions(base_a, f"US/{window}/gru/search-gru-10", partition)
            )
            np.testing.assert_array_equal(mine["target"], parent["target"])
            for column in ("prediction", *QUANTILE_COLUMNS):
                np.testing.assert_array_equal(mine[column], parent[column].astype(np.float64))
        for market in ("US",):
            path = output / "windows/US" / window / job["arm"] / "seed-42" / f"{market}.json"
            record = read_manifest(path)[0]
            parsed = read_window_receipt(record)
            assert record["parent"] == receipt["parent"]
            assert receipt["parent"]["id"] == job_id
            assert parsed.labels_used_until == parsed.segment("evaluation")[0] - 1
            assert (
                dict(parsed.predictions)["evaluation"][0]
                == receipt["predictions"]["evaluation"]["rows"]
            )
    # La copia ordenada de cada ventana se retira al confirmar sus ajustes.
    for window in ("fold-000", "fold-001"):
        ordered = output / "windows-data/US" / window / "ordered"
        assert (ordered / "manifest.json").is_file()
        assert not list(ordered.glob("*.parquet"))
        assert summary["released_ordered_copies"][f"US/{window}"]
    # Repetir verifica los recibos confirmados sin abrir ventanas ni crear optimizadores.
    created = len(recorder.optimizers)
    again = run(base_a, output)
    assert again["status"] == "completed" and len(recorder.optimizers) == created
    assert receipts(output) == found


def test_variant_b_carries_the_anchor_cases_without_fitting(base_b, tmp_path, recorder):
    output = tmp_path / "stage"
    summary = run(base_b, output)
    assert summary["status"] == "completed"
    assert summary["planned"] == dict(training_jobs=5, prediction_jobs=5)
    found = receipts(output)
    fits = {key: value for key, value in found.items() if key.endswith("/fit-s42")}
    carries = {key: value for key, value in found.items() if key.endswith("/carry-s42")}
    assert len(fits) == len(carries) == 5
    # Solo los ajustes crean optimizadores. Los traslados no aplican ninguna actualización.
    assert len(recorder.optimizers) == 5
    for key, receipt in carries.items():
        anchor = key.replace("fold-001", "fold-000").replace("carry-s42", "fit-s42")
        assert receipt["updates"] == 0 and receipt["score"] is None
        assert receipt["parent"] == fits[anchor]["parent"]
        assert receipt["identity"]["anchor_fit"]["job"] == anchor
        assert receipt["identity"]["parent"]["job"] == "US/fold-000/gru/search-gru-10"
        for partition in ("calibration", "evaluation"):
            table = pq.read_table(output / receipt["predictions"][partition]["path"])
            assert (table["prediction_at"].cast(pa.int64()).to_numpy() < FINAL_TEST).all()
            np.testing.assert_array_equal(table["prediction"], table["parent"])
        arm = receipt["identity"]["arm"]
        record = read_manifest(output / "windows/US/fold-001" / arm / "seed-42/US.json")[0]
        assert read_window_receipt(record).parent == (
            fits[anchor]["parent"]["id"],
            fits[anchor]["parent"]["sha256"],
        )
    # El ancla retiró su copia ordenada antes de los traslados, que solo leen su manifiesto.
    assert not list((output / "windows-data/US/fold-000/ordered").glob("*.parquet"))


def test_a_paused_case_resumes_its_cursor_without_repeating_updates(base_a, tmp_path, recorder):
    output = tmp_path / "stage"
    stop = SimpleNamespace(requested=False)

    def pause(optimizer):
        if len(recorder.optimizers) == 2 and len(optimizer.calls) == 3:
            stop.requested = True

    recorder.on_step = pause
    paused = run(base_a, output, stop)
    assert paused["status"] == "paused"
    assert paused["completed"]["training_jobs"] == 1
    recorder.on_step = None
    stop.requested = False
    summary = run(base_a, output, stop)
    assert summary["status"] == "completed"
    found = receipts(output)
    assert len(found) == 10
    # El caso pausado suma sus pasos antes y después de reanudar, sin repetir ninguno.
    assert sum(len(item.calls) for item in recorder.optimizers) == sum(
        receipt["updates"] for receipt in found.values()
    )
    assert len(recorder.optimizers) == 11


def test_the_hold_blocks_before_reading_and_before_each_pending_job(
    base_a, tmp_path, recorder, learning_hold
):
    learning_hold(False)
    with pytest.raises(LearningHoldError):
        run(base_a, tmp_path / "never")
    assert not (tmp_path / "never").exists()
    allowed = learning_hold(True)

    def block(optimizer):
        allowed.write_text(json.dumps({"training_allowed": False}), encoding="utf-8")

    recorder.on_step = block
    output = tmp_path / "stage"
    with pytest.raises(LearningHoldError):
        run(base_a, output)
    summary = read_manifest(output / "summary.json")[0]
    assert summary["status"] == "blocked"
    assert summary["completed"]["training_jobs"] == 1
    # El trabajo siguiente no llega a crear su carpeta ni a abrir su ventana.
    assert len(list((output / "jobs").glob("*/*/*/*"))) == 1


def test_the_hold_also_stops_a_carry_that_would_not_fit(base_b, tmp_path, recorder, learning_hold):
    allowed = learning_hold(True)

    def block(optimizer):
        if len(recorder.optimizers) == 5:
            allowed.write_text(json.dumps({"training_allowed": False}), encoding="utf-8")

    recorder.on_step = block
    output = tmp_path / "stage"
    with pytest.raises(LearningHoldError):
        run(base_b, output)
    summary = read_manifest(output / "summary.json")[0]
    assert summary["status"] == "blocked"
    assert summary["completed"] == dict(training_jobs=5, prediction_jobs=0)
    assert not list((output / "jobs").rglob("carry-s42"))


def tamper(kind):
    def change(table):
        values = table.to_pydict()
        if kind == "reserved":
            values["prediction_at"][0] = np.datetime64("2024-01-02", "us").astype(object)
        elif kind == "target":
            values["target"][0] += 1.0
        elif kind == "median":
            values["prediction"][0] += 0.001
        else:
            values["quantile_0100"], values["quantile_0900"] = (
                values["quantile_0900"],
                values["quantile_0100"],
            )
        return pa.Table.from_pydict(values, schema=table.schema)

    return change


@pytest.mark.parametrize(
    ("kind", "message"),
    [("reserved", "2024"), ("target", "mismas filas"), ("order", "orden"), ("median", "mediana")],
)
def test_predictions_outside_the_contract_are_rejected(
    base_a, tmp_path, recorder, monkeypatch, kind, message
):
    original = matrix_runs.evaluate_partition

    def altered(dataset, parent, partition, destination, **options):
        metrics = original(dataset, parent, partition, destination, **options)
        if partition == "evaluation":
            pq.write_table(tamper(kind)(pq.read_table(destination)), destination)
        return metrics

    monkeypatch.setattr(matrix_runs, "evaluate_partition", altered)
    output = tmp_path / "stage"
    with pytest.raises(ValueError, match=message):
        run(base_a, output)
    assert read_manifest(output / "summary.json")[0]["status"] == "failed"
    assert not list((output / "jobs").rglob("receipt.json"))


def test_command_checks_the_declared_stage_without_reading_data(capsys):
    path = CONFIGS / "historical-masked-adapter-stage-a.json"
    assert campaign_stage.main(["check", "--stage", str(path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "checked" and printed["scientific_training_started"] is False
    assert printed["counts"]["training_jobs"] == 3915


def test_matrix_seeds_must_match_the_seeds_of_each_parent(tmp_path):
    matrix = json.loads((CONFIGS / "adapter-matrix-v2.json").read_text())
    matrix["budget"]["seeds"] = [42, 43]
    atomic_json(tmp_path / "matrix.json", matrix)
    path = mutated(tmp_path, lambda v: v.update(matrix=str(tmp_path / "matrix.json")))
    with pytest.raises(ValueError, match="semilla"):
        campaign_stage.check_stage(path)


def test_an_unconfirmed_base_job_stops_the_stage_before_any_fit(base_a, tmp_path, recorder):
    copy = tmp_path / "campaign"
    shutil.copytree(base_a.output, copy)
    (copy / "jobs/US/fold-001/gru/search-gru-00/receipt.json").unlink()
    base = SimpleNamespace(**dict(vars(base_a), output=copy))
    with pytest.raises(ValueError, match="Falta confirmar US/fold-001/gru/search-gru-00"):
        run(base, tmp_path / "stage")
    assert not (tmp_path / "stage").exists() and recorder.optimizers == []


def test_a_carry_requires_the_parent_that_the_base_campaign_carried(
    base_b, tmp_path, recorder, monkeypatch
):
    original = campaign_stage._Stage.base_parent

    def other(self, scope, window, base_arm, seed):
        key, receipt, report = original(self, scope, window, base_arm, seed)
        if window == "fold-001":
            receipt = dict(receipt, parent=dict(receipt["parent"], sha256="0" * 64))
        return key, receipt, report

    monkeypatch.setattr(campaign_stage._Stage, "base_parent", other)
    with pytest.raises(ValueError, match="padre del ancla"):
        run(base_b, tmp_path / "stage")
