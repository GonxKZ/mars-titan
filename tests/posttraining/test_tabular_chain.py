"""Cadena trivial de Ridge y XGBoost en el walk-forward por etapas, sin ajustar nada.

Ridge y XGBoost no tienen puntos de adaptación. En cada ventana k ≥ 1 su cadena solo tiene
el padre congelado de k-1, que se predice con el traslado tabular de la campaña base y se
confirma con el mismo recibo y la misma selección que los demás brazos. Así las políticas
leen de todas las familias el mismo tipo de predicción fuera de muestra. El recorrido usa
la campaña reducida de `campaign_fixture` con un Ridge sustituto sin parámetros ajustados.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import atomic_json
from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from mars_titan.posttraining import campaign_stage
from mars_titan.training import campaign_chain, carried_predictions
from tests.posttraining.campaign_fixture import CpuLease, base_campaign
from tests.posttraining.real_only import real_data_only
from tests.training.test_carried_predictions import PresenceCount

CONFIGS = Path("configs/posttraining")
BASE_FIT = "US/fold-001/ridge/search-ridge-a1.0"
FROZEN = "US/fold-001/ridge__frozen_parent/frozen-s42"


def test_stage_a_publishes_a_chain_for_every_predictor_that_the_policies_read():
    from mars_titan.simulation import policy_plan

    adapters = campaign_stage.load_stage(CONFIGS / "historical-masked-adapter-stage-a.json")
    jobs = campaign_stage.plan_stage(adapters)
    published = {chain["id"] for chain in campaign_stage.plan_chain(adapters, jobs)}
    policies = policy_plan.load_stage("configs/simulation/historical-masked-rl-stage-a.json")
    needed = {
        name
        for job in policy_plan.plan_stage(policies)
        for name in job["depends"]
        if "__chain/select-s" in name
    }
    assert needed and needed <= published
    predictors = {name.split("/")[2].removesuffix("__chain") for name in needed}
    assert {"ridge", "xgboost", "titans_mac_online", "titans_transformer_direct"} <= predictors


def test_tabular_arms_only_plan_the_frozen_parent_and_their_selection():
    stage = campaign_stage.load_stage(CONFIGS / "historical-masked-adapter-stage-a.json")
    jobs = campaign_stage.plan_stage(stage)
    chains = campaign_stage.plan_chain(stage, jobs)
    for arm, seeds in {"ridge": [42], "xgboost": [42, 43, 44]}.items():
        own = [job for job in jobs if job["base_arm"] == arm]
        assert {job["kind"] for job in own} == {campaign_stage.FROZEN}
        assert all(job["case"] is None and job["family"] == arm for job in own)
        assert sorted({job["seed"] for job in own}) == seeds
        later = [c for c in chains if c["base_arm"] == arm and c["parent_window"] is not None]
        assert later
        for chain in later:
            frozen = (
                f"{chain['scope']}/{chain['window']}/{arm}__frozen_parent/frozen-s{chain['seed']}"
            )
            assert chain["depends"] == [frozen]


def test_variant_b_has_no_trivial_chain(tmp_path):
    value = json.loads((CONFIGS / "historical-masked-adapter-stage-b.json").read_text())
    value.update(
        campaign=str(Path("configs/baselines/historical-masked-campaign-b.json").resolve()),
        matrix=str((CONFIGS / "adapter-matrix-v2.json").resolve()),
        arms=[*value["arms"], "ridge"],
    )
    atomic_json(tmp_path / "stage.json", value)
    with pytest.raises(ValueError, match="solo existe en el walk-forward por etapas"):
        campaign_stage.load_stage(tmp_path / "stage.json")


@pytest.fixture(scope="module")
def tabular_base(tmp_path_factory):
    return base_campaign(tmp_path_factory.mktemp("stage-tabular"), "A", tabular=True)


def by_sample(table):
    order = np.argsort(np.asarray(table["sample_id"].to_pylist()))
    return {name: np.asarray(table[name].to_pylist())[order] for name in table.column_names}


def test_the_tabular_chain_is_the_frozen_parent_carried_from_the_previous_window(
    tabular_base, tmp_path, recorder, monkeypatch
):
    # El traslado carga el modelo del ancla. El sustituto no tiene parámetros que ajustar.
    monkeypatch.setattr(carried_predictions, "_tabular_model", lambda *args: PresenceCount())
    output = tmp_path / "stage"
    with real_data_only():
        summary = campaign_stage.run_stage(
            tabular_base.stage,
            tabular_base.views,
            tabular_base.output,
            output,
            lease=CpuLease,
            stop=SimpleNamespace(requested=False),
            device="cpu",
        )
    assert summary["status"] == "completed"
    # La GRU ajusta sus cinco casos y Ridge solo predice con su padre congelado.
    assert summary["planned"] == dict(training_jobs=5, prediction_jobs=2, selection_jobs=4)
    assert summary["planned"] == summary["completed"]
    receipt = json.loads((output / "jobs" / FROZEN / "receipt.json").read_text())
    assert receipt["updates"] == 0 and receipt["fit_rows"] is None
    assert receipt["identity"]["parent"]["job"].startswith("US/fold-000/ridge/")
    carried = json.loads((output / receipt["run"]["path"]).read_text())
    assert carried["frozen_parent"] is True and carried["anchor"]["fold"] != carried["fold"]
    assert set(carried["predictions"]) == set(campaign_stage.PREDICTED)
    # El sustituto no depende de la ventana: el padre de fold-000 aplicado a fold-001 repite
    # las filas, objetivos y predicciones del estado de la base en fold-001.
    base_receipt = json.loads(
        (tabular_base.output / "jobs" / BASE_FIT / "receipt.json").read_text()
    )
    base_report = tabular_base.output / base_receipt["report"]["path"]
    for partition in campaign_stage.PREDICTED:
        mine = pq.read_table(output / receipt["predictions"][partition]["path"])
        assert not set(QUANTILE_COLUMNS) & set(mine.column_names)
        theirs = pq.read_table(base_report.parent / f"{partition}-predictions.parquet")
        mine, theirs = by_sample(mine), by_sample(theirs)
        for column in ("sample_id", "target", "prediction"):
            np.testing.assert_array_equal(mine[column], theirs[column])
    first = campaign_chain.read_selection(output, "US", "fold-000", "ridge", 42)
    assert first["selected"]["kind"] == "base" and first["candidates"] == []
    later = campaign_chain.read_selection(output, "US", "fold-001", "ridge", 42)
    assert later["selected"]["kind"] == "frozen_parent" and later["fit_rows"] is None
    assert [c["kind"] for c in later["candidates"]] == ["frozen_parent"]
    assert later["selected"]["job"] == FROZEN
    assert later["receipts"]["US"].labels_used_until == receipt["labels_used_until"]
