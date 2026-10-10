"""Ventanas de postentrenamiento de MARS-TITAN M1 sobre la campaña reducida, sin cambiar pesos.

La campaña base es la de `test_mars_titan_campaign`: B sobre US con `titans_mac_online` y
`mars_titan_m1`, una ventana reentrenada y dos trasladadas. Cada caso de la matriz de versión
3 parte del lector elegido en la ventana reentrenada y se ajusta en ella y, como en el
walk-forward por etapas, con las filas nuevas de la siguiente. El padre congelado predice
esa siguiente ventana sin ajustar. El optimizador registra gradientes sin modificar pesos,
así que todas las predicciones deben ser las del brazo base o las de su padre congelado.
Necesita el enlace nativo del banco episódico.
"""

import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from mars_titan.posttraining import chronological_matrix as cm
from mars_titan.posttraining import chronological_windows as cw
from mars_titan.posttraining.adapter_matrix import read_matrix
from mars_titan.training.titans_walk_forward import unfused_attention
from tests.training.test_mars_titan_campaign import ARM
from tests.training.test_mars_titan_campaign import campaign_run as campaign_run
from tests.training.test_mars_titan_campaign import explicit_fastpath as explicit_fastpath
from tests.training.test_titans_campaign import permitted as permitted
from tests.training.test_titans_walk_forward import Factory

MATRIX = Path("configs/posttraining/adapter-matrix-v3.json")
pytestmark = pytest.mark.skipif(
    not os.environ.get("MARS_TITAN_EPISODIC_NATIVE"), reason="Falta el enlace nativo"
)


def receipt(run, job_id):
    return json.loads((run.output / "jobs" / job_id / "receipt.json").read_text())


def rows(path):
    table = pq.read_table(path).sort_by("sample_id")
    return {name: table[name].to_pylist() for name in ("sample_id", "target", *QUANTILE_COLUMNS)}


@pytest.fixture(scope="module")
def windows(campaign_run, tmp_path_factory):  # noqa: F811
    matrix, digest = read_matrix(MATRIX)
    matrix["budget"].update(epochs=1)
    fitted, carried = campaign_run.windows[0], campaign_run.windows[1]
    searches = {
        name: receipt(campaign_run, f"US/{fitted}/{ARM}/search-{name}")
        for name in ("lr1e-4", "lr1e-3")
    }
    chosen = min(searches, key=lambda name: (searches[name]["score"], name))
    arm = (campaign_run.output / searches[chosen]["report"]["path"]).parent
    prepared = campaign_run.prepared["US"]["windows"]
    root = tmp_path_factory.mktemp("readout-windows")
    result = {}
    # La continuación anclada (#444) queda de reserva en los lectores. Se recorre igual para
    # comprobar que llega al entrenador con su ancla y el presupuesto de la completa.
    reserve = cm.cases(matrix, digest, "mars_titan", bank=True, reserve=True)
    items = cm.cases(matrix, digest, "mars_titan", bank=True)
    items += [item for item in reserve if item["control"] == cm.ANCHORED]
    for item in items:
        if item["case"]["seed"] != 42:
            continue
        name = item["id"].split("/", 1)[1]
        factory, staged_factory = Factory(), Factory()
        common = dict(case=item["case"], matrix=matrix, digest=digest, device="cpu")
        with unfused_attention():
            fit = cw.run_readout_posttraining(
                arm,
                Path(prepared[fitted]["path"]),
                root / name / "fit",
                optimizer_factory=factory,
                **common,
            )
            staged = cw.run_readout_posttraining(
                arm,
                Path(prepared[carried]["path"]),
                root / name / "staged",
                optimizer_factory=staged_factory,
                parent_view=Path(prepared[fitted]["path"]),
                **common,
            )
        result[name] = dict(
            fit=fit,
            staged=staged,
            factory=factory,
            staged_factory=staged_factory,
            root=root / name,
        )
    with unfused_attention():
        frozen = cw.frozen_readout(
            arm,
            Path(prepared[fitted]["path"]),
            Path(prepared[carried]["path"]),
            root / "frozen",
            device="cpu",
        )
    base = dict(
        fit=json.loads((arm / "run.json").read_text()),
        arm=arm,
        carry=receipt(campaign_run, f"US/{carried}/{ARM}/carry-s42"),
        frozen=frozen,
        frozen_root=root / "frozen",
    )
    return dict(runs=result, base=base, output=campaign_run.output)


def test_matrix_declares_the_core_the_reader_and_both_with_a_bank(windows):
    assert set(windows["runs"]) == {
        "full_continuation",
        "anchored_continuation",
        "core",
        "episodic_readout",
        "core+episodic_readout",
    }


def test_every_case_emits_the_rows_of_the_base_arm(windows):
    base = windows["base"]
    for name, run in windows["runs"].items():
        assert run["fit"]["status"] == "completed", name
        for partition in ("validation", "calibration", "evaluation"):
            mine = rows(run["root"] / "fit" / run["fit"]["predictions"][partition]["path"])
            theirs = rows(base["arm"] / base["fit"]["predictions"][partition]["path"])
            assert mine == theirs, (name, partition)
        # El ajuste por etapas sin cambios de pesos emite las filas del padre congelado.
        for partition in ("validation", "calibration", "evaluation"):
            mine = rows(run["root"] / "staged" / run["staged"]["predictions"][partition]["path"])
            theirs = rows(base["frozen_root"] / base["frozen"]["predictions"][partition]["path"])
            assert mine == theirs, (name, "staged", partition)


def test_the_frozen_parent_reproduces_the_carry_of_the_base_campaign(windows):
    base, output = windows["base"], windows["output"]
    frozen = base["frozen"]
    assert frozen["kind"] == cw.READOUT_FROZEN and frozen["status"] == "completed"
    assert frozen["months_since_parent_information"] > 0
    for partition in ("calibration", "evaluation"):
        mine = rows(base["frozen_root"] / frozen["predictions"][partition]["path"])
        theirs = rows(output / base["carry"]["predictions"][partition]["path"])
        assert mine == theirs, partition


def test_staged_cases_fit_only_the_new_rows_of_the_next_window(windows):
    for name, run in windows["runs"].items():
        report = json.loads((run["root"] / "staged" / "run.json").read_text())
        identity = report["identity"]
        placement = identity["posttraining"]["placement"]
        assert placement["design"] == cw.STAGED and placement["parent_window"] == "fold-000"
        # M1 no fija escalas. Con M3 serían las del ajuste del brazo padre.
        assert placement["scalers"] is None
        start = placement["fit_start"]
        since = int(np.datetime64(start, "us").astype(np.int64))
        assert identity["phases"]["train"]["decision_start"] == since, name
        assert identity["phases"]["train"]["warmup_start"] < since
        fit = json.loads((run["root"] / "fit" / "run.json").read_text())
        # Mismo padre y mismo caso, ajustes distintos: la ventana y sus filas cambian.
        assert identity["posttraining"]["case"] == fit["identity"]["posttraining"]["case"]
        assert identity["phases"]["train"] != fit["identity"]["phases"]["train"]


def test_cases_share_updates_and_train_only_their_declared_parameters(windows):
    updates, roles = set(), {}
    for name, run in windows["runs"].items():
        factory = run["factory"]
        assert factory.instances and all(i.calls > 0 for i in factory.instances), name
        updates.add(sum(i.calls for i in factory.instances))
        roles[name] = {group["role"] for i in factory.instances for group in i.param_groups}
        report = json.loads((run["root"] / "fit" / "fit" / "run.json").read_text())
        identity = report["identity"]
        adapter = identity["posttraining"]["adapter"]
        anchored = name == cm.ANCHORED
        if name in ("full_continuation", cm.ANCHORED):
            assert adapter is None
        else:
            assert set(adapter) == set(name.split("+")), name
        # Solo la continuación anclada declara el ancla del decaimiento y su código.
        assert identity["recipe"].get("weight_decay_anchor") == (
            "initial_parameters" if anchored else None
        ), name
        assert ("anchored_decay" in identity) is anchored, name
    assert len(updates) == 1
    assert roles["full_continuation"] != roles["core"]
    assert roles["full_continuation"] == roles[cm.ANCHORED]
    staged = {
        sum(i.calls for i in run["staged_factory"].instances) for run in windows["runs"].values()
    }
    assert len(staged) == 1 and min(staged) > 0
