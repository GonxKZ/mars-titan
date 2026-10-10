"""Walk-forward por etapas de la matriz de adaptadores, sin pasos de optimizador.

Los recuentos se comprueban con las configuraciones del repositorio. El recorrido completo
usa la campaña base reducida de `campaign_fixture` (dos ventanas US), con padres de
cuantiles reales y pesos iniciales, y el optimizador del fixture `recorder`, que no hereda
de `torch.optim.Optimizer`, registra gradientes y exige pesos sin cambios. La GPU no se usa.
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
from mars_titan.posttraining import adapter_matrix, campaign_stage, matrix_runs, staged_chain
from mars_titan.training.campaign_plan import plan_campaign
from mars_titan.training.learning_hold import LearningHoldError
from tests.posttraining.campaign_fixture import CpuLease, base_campaign
from tests.posttraining.real_only import real_data_only

CONFIGS = Path("configs/posttraining")
FINAL_TEST = int(np.datetime64("2024-01-01", "us").astype(np.int64))
# Por ventana y semilla: cuatro brazos y la continuación en cada red recurrente y DLinear,
# ocho brazos y la continuación en el Transformer, y el padre congelado de cada brazo base.
CASES = 4 * 5 + 9


def micros(day):
    return int(np.datetime64(day, "us").astype(np.int64))


def test_stage_a_plans_every_later_window_from_the_previous_parent():
    path = CONFIGS / "historical-masked-adapter-stage-a.json"
    result = campaign_stage.check_stage(path)
    assert (result["design"], result["executable"]) == ("staged_chain_v1", True)
    assert (result["chain_rule"], result["data_policy"]) == (
        "chain_validation_score_v1",
        "real_edition_only",
    )
    counts = result["counts"]
    assert (counts["training_jobs"], counts["prediction_jobs"]) == (3654, 630)
    assert counts["selection_jobs"] == 675
    stage = campaign_stage.load_stage(path)
    assert stage["limits"] == dict(max_training_jobs=3654, max_prediction_jobs=630)
    assert result["excluded_controls"].keys() == {"linear_residual"}
    assert result["objectives"]["adapters"] == "neural_pinball"
    for scope, windows in {"US": 19, "CN": 13, "US+CN": 13}.items():
        entry = counts["scopes"][scope]
        # La ventana 0 no tiene padre: no hay postentrenamiento, solo la cadena de la base.
        assert entry["windows"] == windows and len(entry["fitted_windows"]) == windows - 1
        assert "fold-000" not in entry["fitted_windows"]
        assert entry["training_jobs"] == (windows - 1) * 3 * CASES
        assert entry["prediction_jobs"] == (windows - 1) * 3 * 5
        assert entry["selection_jobs"] == windows * 3 * 5
        assert len(entry["arms"]) == CASES + 5
        for arm, seeds in entry["arms"].items():
            assert comparison._name(arm)
            kind = "frozen" if arm.endswith("__frozen_parent") else "fit"
            assert seeds == {str(seed): {kind: windows - 1} for seed in (42, 43, 44)}, arm


def test_stage_b_keeps_its_counts_and_is_not_executable():
    result = campaign_stage.check_stage(CONFIGS / "historical-masked-adapter-stage-b.json")
    assert (result["design"], result["executable"]) == ("anchored_not_executed", False)
    assert "9 de octubre" in result["not_executed_reason"]
    counts = result["counts"]
    assert (counts["training_jobs"], counts["prediction_jobs"], counts["selection_jobs"]) == (
        1479,
        2436,
        0,
    )


def test_jobs_depend_on_the_parent_selected_in_the_previous_window():
    stage = campaign_stage.load_stage(CONFIGS / "historical-masked-adapter-stage-a.json")
    jobs = campaign_stage.plan_stage(stage)
    base = plan_campaign(stage["campaign"])
    by_id = {job["id"]: job for job in base}
    matrix, digest = stage["matrix"], stage["matrix_sha256"]
    windows = {
        scope: [name for name, _ in staged_chain.scope_windows(stage["campaign"], scope)]
        for scope in stage["scopes"]
    }
    for job in jobs:
        order = windows[job["scope"]]
        assert order.index(job["parent_window"]) == order.index(job["window"]) - 1
        # Búsquedas de la semilla 42 o finalistas de 43 y 44 del brazo base en k-1.
        expected = staged_chain.parent_jobs(
            base, job["scope"], job["parent_window"], job["base_arm"], job["seed"]
        )
        assert job["depends"] == expected and expected
        for name in expected:
            parent = by_id[name]
            assert (parent["window"], parent["arm"]) == (job["parent_window"], job["base_arm"])
            assert parent["stage"] == "search" or parent["seed"] == job["seed"]
        if job["kind"] == "frozen":
            assert job["case"] is None and job["arm"] == f"{job['base_arm']}__frozen_parent"
            continue
        assert job["case"]["mode"] == "neural_pinball" and job["control"] != "linear_residual"
        cases = {
            item["id"]: item["case"]
            for item in adapter_matrix.cases(matrix, digest, job["family"], head=QUANTILE_HEAD)
        }
        assert job["case"] == cases[f"seed-{job['seed']}/{job['point']}"]
    chains = campaign_stage.plan_chain(stage, jobs)
    order = campaign_stage.ordered_jobs(jobs, chains)
    position = {job["id"]: index for index, job in enumerate(order)}
    for chain in chains:
        if chain["parent_window"] is None:
            assert chain["window"] == "fold-000"
            assert all(name in by_id for name in chain["depends"])
            continue
        own = [
            job["id"]
            for job in jobs
            if (job["scope"], job["window"], job["base_arm"], job["seed"])
            == (chain["scope"], chain["window"], chain["base_arm"], chain["seed"])
        ]
        assert chain["depends"] == own and len(own) == 1 + (
            9 if "transformer" in chain["arm"] else 5
        )
        assert all(position[name] < position[chain["id"]] for name in own)


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
    "retention": lambda v: v.update(
        cohort_reading=dict(source="ordered_corpus", retention="forever")
    ),
    "ordered_reading": lambda v: v.update(
        cohort_reading=dict(source="ordered_corpus", retention="keep")
    ),
    "small_block": lambda v: v.update(
        cohort_reading=dict(source="view_blocks", max_block_bytes=1024**2)
    ),
    "block_without_budget": lambda v: v.update(cohort_reading=dict(source="view_blocks")),
    "mixed_reading": lambda v: v["cohort_reading"].update(retention="keep"),
    "test_opened": lambda v: v.update(final_test_opened=True),
    "limit": lambda v: v["limits"].update(max_training_jobs=3653),
    "frozen_limit": lambda v: v["limits"].update(max_prediction_jobs=629),
    "status": lambda v: v.update(status="executed"),
    "chain_rule": lambda v: v.update(chain_rule="chain_validation_score_v2"),
    "data_policy": lambda v: v.update(data_policy="real_and_augmented"),
    "no_chain_rule": lambda v: v.pop("chain_rule"),
    "no_data_policy": lambda v: v.pop("data_policy"),
}


def test_the_unchanged_declaration_is_accepted(tmp_path):
    assert campaign_stage.check_stage(mutated(tmp_path, lambda _: None))["status"] == "checked"


@pytest.mark.parametrize("name", sorted(INVALID))
def test_stage_rejects_inconsistent_declarations(tmp_path, name):
    with pytest.raises(ValueError):
        campaign_stage.check_stage(mutated(tmp_path, INVALID[name]))


def test_variant_b_cannot_declare_the_chain(tmp_path):
    value = json.loads((CONFIGS / "historical-masked-adapter-stage-b.json").read_text())
    value.update(
        campaign=str(Path("configs/baselines/historical-masked-campaign-b.json").resolve()),
        matrix=str((CONFIGS / "adapter-matrix-v2.json").resolve()),
        chain_rule="chain_validation_score_v1",
        data_policy="real_edition_only",
    )
    atomic_json(tmp_path / "stage.json", value)
    with pytest.raises(ValueError, match="Solo la variante A"):
        campaign_stage.load_stage(tmp_path / "stage.json")


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


def run(base, output, stop=None, stage=None):
    # Sin cola, preparación, aumento ni mundos del postentrenamiento emparejado anterior.
    with real_data_only():
        return campaign_stage.run_stage(
            stage or base.stage,
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


def by_sample(table):
    order = np.argsort(np.asarray(table["sample_id"].to_pylist()))
    return {name: np.asarray(table[name].to_pylist())[order] for name in table.column_names}


PARENT = "US/fold-000/gru/search-gru-10"
FROZEN = "US/fold-001/gru__frozen_parent/frozen-s42"


def test_variant_a_adapts_the_previous_parent_with_new_rows_and_publishes_the_chain(
    base_a, tmp_path, recorder
):
    output = tmp_path / "stage"
    summary = run(base_a, output)
    assert summary["status"] == "completed"
    assert summary["planned"] == summary["completed"]
    assert summary["planned"] == dict(training_jobs=5, prediction_jobs=1, selection_jobs=2)
    found = receipts(output)
    assert set(found) == {FROZEN} | {
        f"US/fold-001/gru__{point}/fit-s42"
        for point in ("full_continuation", "head", "fusion", "head_fusion", "fusion_full_rank")
    }
    proof = json.loads((output / "windows-data/US/fold-001/fit-rows.json").read_text())
    assert proof["parent_window"] == "fold-000" and proof["window"] == "fold-001"
    assert (proof["start"], proof["end"]) == ("2022-01-01", "2022-04-01")
    assert micros("2022-01-01") <= proof["first_decision"] <= proof["last_decision"]
    assert proof["last_decision"] < micros("2022-04-01")
    assert proof["rows"] > 0 and proof["intersection"] == dict(train=0, validation=0, calibration=0)
    assert proof["parent_labels_mature_until"] < micros("2022-01-01")
    assert proof["labels_used_until"] < micros("2023-01-01")
    # El normalizador del padre solo ve las filas nuevas de la prueba.
    parent_folder = output / "windows-data/US/fold-001/parents/gru/seed-42"
    normalization = json.loads((parent_folder / "normalization.json").read_text())
    assert normalization["samples"] == proof["rows"]
    # Sin pasos aplicados, la época cero gana y cada caso emite las filas del padre congelado.
    updates = {receipt["updates"] for name, receipt in found.items() if name != FROZEN}
    assert len(updates) == 1 and min(updates) > 0
    assert len(recorder.optimizers) == 5
    assert all(len(item.calls) == min(updates) for item in recorder.optimizers)
    frozen = found[FROZEN]
    assert frozen["updates"] == 0 and frozen["reported_score"] is None
    assert frozen["identity"]["parent"]["job"] == PARENT and frozen["fit_rows"] is None
    tables = {
        partition: by_sample(pq.read_table(output / frozen["predictions"][partition]["path"]))
        for partition in campaign_stage.PREDICTED
    }
    for name, receipt in found.items():
        identity = receipt["identity"]
        assert (identity["window"], identity["parent_window"]) == ("fold-001", "fold-000")
        assert identity["parent"]["job"] == PARENT
        assert receipt["labels_used_until"] == proof["labels_used_until"]
        assert receipt["score"] == frozen["score"]
        if name != FROZEN:
            assert receipt["selection"]["best_epoch"] == 0
            assert receipt["fit_rows"]["rows"] == proof["rows"]
            assert receipt["fit_rows"]["sha256"] == proof["sha256"]
            assert receipt["fit_rows"]["intersection"] == proof["intersection"]
        for partition in campaign_stage.PREDICTED:
            table = pq.read_table(output / receipt["predictions"][partition]["path"])
            assert (table["prediction_at"].cast(pa.int64()).to_numpy() < FINAL_TEST).all()
            levels = np.column_stack([table[column].to_numpy() for column in QUANTILE_COLUMNS])
            assert (np.diff(levels, axis=1) >= 0).all()
            mine = by_sample(table)
            for column in ("target", "prediction", *QUANTILE_COLUMNS):
                np.testing.assert_array_equal(mine[column], tables[partition][column])
        record = read_manifest(
            output / "windows/US/fold-001" / identity["arm"] / "seed-42/US.json"
        )[0]
        parsed = read_window_receipt(record)
        assert record["parent"]["id"] == name
        assert parsed.labels_used_until == proof["labels_used_until"]
        assert (
            dict(parsed.predictions)["validation"][0]
            == (receipt["predictions"]["validation"]["rows"])
        )
    # La cadena: en la ventana 0 el estado de la base, en la 1 el padre congelado, porque
    # ningún caso mejora estrictamente su puntuación de validación.
    first = staged_chain.read_selection(output, "US", "fold-000", "gru", 42)
    assert first["selected"]["kind"] == "base" and first["selected"]["job"] == PARENT
    assert first["parent"] is None and first["candidates"] == [] and first["fit_rows"] is None
    base_validation = json.loads((base_a.output / "jobs" / PARENT / "receipt.json").read_text())
    assert first["state"]["sha256"] == base_validation["parent"]["sha256"]
    assert first["receipts"]["US"].labels_used_until < micros("2022-01-01")
    second = staged_chain.read_selection(output, "US", "fold-001", "gru", 42)
    assert second["selected"]["kind"] == "frozen_parent" and second["selected"]["job"] == FROZEN
    assert second["fit_rows"] is None and len(second["candidates"]) == 6
    assert second["parent"]["job"] == PARENT
    assert second["labels_used_until"] == proof["labels_used_until"]
    assert summary["chain"] == {
        "US/fold-000/gru__chain/select-s42": dict(kind="base", job=PARENT),
        "US/fold-001/gru__chain/select-s42": dict(kind="frozen_parent", job=FROZEN),
    }
    # La lectura por bloques solo deja índices: el de la vista del padre y el de la ventana.
    for window in ("fold-000", "fold-001"):
        assert (output / "windows-data/US" / window / "cohorts/manifest.json").is_file()
    assert not list((output / "windows-data").rglob("*.parquet"))
    # Repetir verifica recibos y selecciones sin abrir ventanas ni crear optimizadores.
    created = len(recorder.optimizers)
    again = run(base_a, output)
    assert again["status"] == "completed" and len(recorder.optimizers) == created
    assert receipts(output) == found
    assert again["chain"] == summary["chain"]


def test_a_strictly_better_candidate_replaces_the_frozen_parent(
    base_a, tmp_path, recorder, monkeypatch
):
    original, calls = staged_chain.validation_score, []

    def lower_head(table):
        calls.append(None)
        # Tercer trabajo de la ventana: el brazo head, tras el padre y la continuación.
        return original(table) - (1e-9 if len(calls) == 3 else 0.0)

    monkeypatch.setattr(staged_chain, "validation_score", lower_head)
    output = tmp_path / "stage"
    assert run(base_a, output)["status"] == "completed"
    chosen = staged_chain.read_selection(output, "US", "fold-001", "gru", 42)
    assert chosen["selected"]["kind"] == "adapter"
    assert chosen["selected"]["job"] == "US/fold-001/gru__head/fit-s42"
    proof = json.loads((output / "windows-data/US/fold-001/fit-rows.json").read_text())
    assert chosen["fit_rows"] == {
        name: proof[name] for name in ("first_decision", "last_decision", "rows", "sha256")
    }
    assert chosen["receipts"]["US"].parent[0] == "US/fold-001/gru__head/fit-s42"


def test_a_validation_score_that_differs_from_the_fit_is_rejected(
    base_a, tmp_path, recorder, monkeypatch
):
    original, calls = staged_chain.validation_score, []

    def shifted(table):
        calls.append(None)
        # El ajuste declara su puntuación. La recalculada difiere en una milésima relativa.
        return original(table) * (1.001 if len(calls) == 2 else 1.0)

    monkeypatch.setattr(staged_chain, "validation_score", shifted)
    with pytest.raises(ValueError, match="no reproduce la puntuación"):
        run(base_a, tmp_path / "stage")


@pytest.mark.parametrize("change", ["selected", "receipt", "proof"])
def test_a_tampered_chain_or_proof_stops_the_next_run(base_a, tmp_path, recorder, change):
    output = tmp_path / "stage"
    assert run(base_a, output)["status"] == "completed"
    folder = output / "windows/US/fold-001/gru__chain/seed-42"
    if change == "selected":
        path = folder / "selection.json"
        value = json.loads(path.read_text())
        value["selected"] = dict(value["candidates"][1], kind="continuation")
        value["selected"].pop("score")
        atomic_json(path, value)
    elif change == "receipt":
        path = folder / "US.json"
        value = json.loads(path.read_text())
        value["labels_used_until"] -= 1
        atomic_json(path, value)
    else:
        path = output / "windows-data/US/fold-001/fit-rows.json"
        value = json.loads(path.read_text())
        value["intersection"]["calibration"] = 1
        atomic_json(path, value)
    with pytest.raises(ValueError):
        run(base_a, output)


def test_variant_b_is_rejected_before_reading_anything(base_b, tmp_path, recorder):
    with pytest.raises(ValueError, match="variante B no se ejecuta"):
        run(base_b, tmp_path / "stage")
    assert not (tmp_path / "stage").exists() and recorder.optimizers == []


def test_a_paused_case_resumes_its_cursor_without_repeating_updates(base_a, tmp_path, recorder):
    output = tmp_path / "stage"
    stop = SimpleNamespace(requested=False)

    def pause(optimizer):
        if len(recorder.optimizers) == 2 and len(optimizer.calls) == 1:
            stop.requested = True

    recorder.on_step = pause
    paused = run(base_a, output, stop)
    assert paused["status"] == "paused"
    assert paused["completed"]["training_jobs"] == 1
    assert paused["completed"]["prediction_jobs"] == 1
    recorder.on_step = None
    stop.requested = False
    summary = run(base_a, output, stop)
    assert summary["status"] == "completed"
    found = receipts(output)
    assert len(found) == 6
    # El caso pausado suma sus pasos antes y después de reanudar, sin repetir ninguno.
    assert sum(len(item.calls) for item in recorder.optimizers) == sum(
        receipt["updates"] for receipt in found.values()
    )
    assert len(recorder.optimizers) == 6


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
    # El padre congelado y el primer caso terminan. El siguiente no crea su carpeta.
    assert len(list((output / "jobs").glob("*/*/*/*"))) == 2


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

    # El padre congelado y los casos escriben por el mismo recorrido.
    monkeypatch.setattr(matrix_runs, "evaluate_partition", altered)
    monkeypatch.setattr(campaign_stage, "evaluate_partition", altered)
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
    assert printed["counts"]["training_jobs"] == 3654


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


def test_the_b6_correction_has_no_adapter_arm():
    """La corrección B6 no tiene lector que adaptar, así que no figura como brazo cronológico.

    Los lectores conservan su banco. Sin esta exclusión la etapa no podría contar sus trabajos
    con la declaración ampliada, que incluye los dos brazos B6.
    """
    from mars_titan.training import campaign_plan as plan

    arms = campaign_stage.chronological_arms(
        {
            plan.MARS: dict(
                arms=dict(
                    mars_titan_m0=dict(episodic_bank="m0_no_bank"),
                    mars_titan_m1=dict(episodic_bank="m1"),
                    mars_titan_b6=dict(associative_memory=dict(rule="proximal", key="codec")),
                )
            )
        }
    )
    assert arms == dict(
        mars_titan_m0=dict(family=plan.MARS, variant=None, bank=False),
        mars_titan_m1=dict(family=plan.MARS, variant=None, bank=True),
    )
