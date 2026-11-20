"""Ablación de modalidades sobre el estado de una campaña comparativa.

El traslado real con ablación de cada familia lo prueba `test_modality_ablation_carry`. Aquí se
prueba la orden: qué trabajos entran, que una ablación con recibo comprobado no se repite, que
un recibo incoherente detiene la orden, que un intento interrumpido se descarta y que un fallo
se registra sin detener las demás tareas.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.modality_ablation import VARIANTS, ablation_identity
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training import campaign_ablation as ablation
from mars_titan.training import comparison_campaign as campaign
from tests.training.test_comparison_campaign import declaration, layout

MODELS = ["titans_mac", "mars_titan_m1"]


def finished(where, key):
    """Marca un trabajo como terminado con un informe propio."""
    atomic_json(where.record(key), dict(status=campaign.DONE, attempts=[]))
    where.output(key).mkdir(parents=True, exist_ok=True)
    atomic_json(where.output(key) / "run.json", dict(kind="window", key=key))


def receipt(where, key, variant):
    folder = where.state / ablation.FOLDER / variant / key
    path = folder / "evaluation-predictions.parquet"
    pq.write_table(pa.table({"prediction": [0.1, -0.2]}), path)
    document = dict(
        status="completed",
        final_test_opened=False,
        scientific_training_started=False,
        modality_ablation=ablation_identity(variant),
        anchor=dict(run_sha256=sha256(where.output(key) / "run.json")),
        predictions=dict(evaluation=dict(path=path.name, sha256=sha256(path))),
    )
    atomic_json(folder / ablation.RECEIPT, document)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Campaña con el primer origen de la semilla 42 terminado y tareas dentro del proceso."""
    where = layout(tmp_path)
    document = campaign.checked_declaration(declaration())
    for key in ("titans_mac/2019/seed-42", "mars_titan_m1/2019/seed-42"):
        finished(where, key)
    calls, outcomes = [], {}

    def task(state, views, declared, machine, key, variant, cuda_gib, device):
        calls.append((key, variant))
        folder = where.state / ablation.FOLDER / variant / key
        assert not folder.exists()
        folder.mkdir(parents=True)
        outcome = outcomes.get((key, variant), "completed")
        if outcome == "fail":
            raise RuntimeError("fallo del traslado")
        if outcome == "paused":
            (folder / "evaluation-predictions.parquet.part").write_bytes(b"incompleto")
            return dict(status="paused", seconds=1.0)
        receipt(where, key, variant)
        return dict(status="completed", seconds=2.0)

    monkeypatch.setattr(ablation, "run_task", task)
    monkeypatch.setattr(ablation, "_pool", lambda workers: ThreadPoolExecutor(1))
    return where, document, calls, outcomes


def run(world, **options):
    where, document, _, _ = world
    return ablation.run(document, where, "mars_titan", MODELS, list(VARIANTS), **options)


@pytest.mark.parametrize(
    ("models", "variants"),
    [
        ([], ["mask_news"]),
        (["unknown"], ["mask_news"]),
        (["rule"], ["mask_news"]),
        (["titans_mac", "titans_mac"], ["mask_news"]),
        (["titans_mac"], []),
        (["titans_mac"], ["mask_prices"]),
        (["titans_mac"], ["mask_news", "mask_news"]),
    ],
)
def test_selection_takes_only_declared_neural_jobs_and_declared_variants(models, variants):
    document = campaign.checked_declaration(declaration())
    with pytest.raises(ValueError):
        ablation.selected(document, "mars_titan", models, variants)


@pytest.mark.parametrize(
    "options", [dict(workers=0), dict(workers=9), dict(cuda_gib=0), dict(device="cuda:1")]
)
def test_run_rejects_resources_outside_their_contract(world, options):
    with pytest.raises(ValueError):
        run(world, **options)


def test_only_finished_jobs_are_ablated_and_a_verified_receipt_is_never_repeated(world):
    where, _, calls, _ = world
    first = run(world)
    expected = {
        f"{variant}/{key}"
        for key in ("titans_mac/2019/seed-42", "mars_titan_m1/2019/seed-42")
        for variant in VARIANTS
    }
    assert {item["task"] for item in first["completed"]} == expected
    assert len(calls) == len(expected) and not first["failed"] and not first["verified"]
    # Los demás trabajos de los modelos elegidos esperan a terminar en la campaña.
    assert set(first["waiting"]) == {
        job.key
        for job in campaign.plan(world[1], "mars_titan")
        if job.model in MODELS
        and job.key not in ("titans_mac/2019/seed-42", "mars_titan_m1/2019/seed-42")
    }
    second = run(world)
    assert set(second["verified"]) == expected and not second["completed"]
    assert len(calls) == len(expected)
    summary = json.loads((where.state / ablation.FOLDER / "summary.json").read_text())
    assert summary["kind"] == ablation.KIND and set(summary["verified"]) == expected


def test_an_interrupted_attempt_is_discarded_and_repeated(world):
    where, _, calls, outcomes = world
    outcomes[("titans_mac/2019/seed-42", "mask_news")] = "paused"
    first = run(world)
    assert [item["task"] for item in first["paused"]] == ["mask_news/titans_mac/2019/seed-42"]
    outcomes.clear()
    second = run(world)
    assert second["discarded"] == ["mask_news/titans_mac/2019/seed-42"]
    assert [item["task"] for item in second["completed"]] == ["mask_news/titans_mac/2019/seed-42"]
    folder = where.state / ablation.FOLDER / "mask_news/titans_mac/2019/seed-42"
    assert not (folder / "evaluation-predictions.parquet.part").exists()


def test_a_failed_task_is_recorded_and_the_others_complete(world):
    _, _, _, outcomes = world
    outcomes[("mars_titan_m1/2019/seed-42", "mask_fundamentals")] = "fail"
    summary = run(world)
    assert [item["task"] for item in summary["failed"]] == [
        "mask_fundamentals/mars_titan_m1/2019/seed-42"
    ]
    assert summary["failed"][0]["error_type"] == "RuntimeError"
    assert len(summary["completed"]) == 2 * len(VARIANTS) - 1


@pytest.mark.parametrize("tamper", ["variant", "predictions", "anchor", "partitions"])
def test_a_receipt_that_does_not_match_its_job_stops_the_command(world, tamper):
    where = world[0]
    run(world)
    key, variant = "titans_mac/2019/seed-42", "mask_news"
    folder = where.state / ablation.FOLDER / variant / key
    document = json.loads((folder / ablation.RECEIPT).read_text())
    if tamper == "variant":
        document["modality_ablation"] = ablation_identity("mask_fundamentals")
    elif tamper == "partitions":
        document["predictions"]["calibration"] = document["predictions"]["evaluation"]
    elif tamper == "predictions":
        pq.write_table(pa.table({"prediction": [0.3]}), folder / "evaluation-predictions.parquet")
    else:
        atomic_json(where.output(key) / "run.json", dict(kind="window", key=key, changed=True))
    if tamper in ("variant", "partitions"):
        atomic_json(folder / ablation.RECEIPT, document)
    with pytest.raises(ValueError, match="corresponde|Han cambiado"):
        run(world)


def test_the_command_needs_the_authorized_declaration(tmp_path):
    where = layout(tmp_path)
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(declaration()), encoding="utf-8")
    arguments = [
        "--declaration",
        str(path),
        "--machine",
        "mars_titan",
        "--state",
        str(where.state),
        "--views",
        str(where.views),
        "--models",
        "titans_mac",
    ]
    with pytest.raises(ValueError, match="autorizada"):
        ablation.main(arguments)
    where.state.mkdir(parents=True, exist_ok=True)
    atomic_json(where.authorization, dict(training_allowed=True, declaration_sha256="0" * 64))
    with pytest.raises(ValueError, match="autorizada"):
        ablation.main(arguments)
