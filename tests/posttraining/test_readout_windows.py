"""Ventanas de postentrenamiento de MARS-TITAN M1 sobre la campaña reducida, sin cambiar pesos.

La campaña base es la de `test_mars_titan_campaign`: B sobre US con `titans_mac_online` y
`mars_titan_m1`, una ventana reentrenada y dos trasladadas. Cada caso de la matriz de versión
3 parte del lector elegido en la ventana reentrenada y se traslada después a la siguiente.
El optimizador registra gradientes sin modificar pesos, así que todas las predicciones deben
ser las del brazo base. Necesita el enlace nativo del banco episódico.
"""

import json
import os
from pathlib import Path

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
    for item in cm.cases(matrix, digest, "mars_titan", bank=True):
        if item["case"]["seed"] != 42:
            continue
        name = item["id"].split("/", 1)[1]
        factory = Factory()
        with unfused_attention():
            fit = cw.run_readout_posttraining(
                arm,
                Path(prepared[fitted]["path"]),
                root / name / "fit",
                case=item["case"],
                matrix=matrix,
                digest=digest,
                device="cpu",
                optimizer_factory=factory,
            )
            carry = cw.carry_readout_posttraining(
                root / name / "fit",
                Path(prepared[fitted]["path"]),
                Path(prepared[carried]["path"]),
                root / name / "carry",
                device="cpu",
            )
        result[name] = dict(fit=fit, carry=carry, factory=factory, root=root / name)
    base = dict(
        fit=json.loads((arm / "run.json").read_text()),
        arm=arm,
        carry=receipt(campaign_run, f"US/{carried}/{ARM}/carry-s42"),
    )
    return dict(runs=result, base=base, output=campaign_run.output)


def test_matrix_declares_the_core_the_reader_and_both_with_a_bank(windows):
    assert set(windows["runs"]) == {
        "full_continuation",
        "core",
        "episodic_readout",
        "core+episodic_readout",
    }


def test_every_case_emits_the_rows_of_the_base_arm(windows):
    base, output = windows["base"], windows["output"]
    for name, run in windows["runs"].items():
        assert run["fit"]["status"] == "completed", name
        for partition in ("validation", "calibration", "evaluation"):
            mine = rows(run["root"] / "fit" / run["fit"]["predictions"][partition]["path"])
            theirs = rows(base["arm"] / base["fit"]["predictions"][partition]["path"])
            assert mine == theirs, (name, partition)
        for partition in ("calibration", "evaluation"):
            mine = rows(run["root"] / "carry" / run["carry"]["predictions"][partition]["path"])
            theirs = rows(output / base["carry"]["predictions"][partition]["path"])
            assert mine == theirs, (name, "carry", partition)


def test_cases_share_updates_and_train_only_their_declared_parameters(windows):
    updates, roles = set(), {}
    for name, run in windows["runs"].items():
        factory = run["factory"]
        assert factory.instances and all(i.calls > 0 for i in factory.instances), name
        updates.add(sum(i.calls for i in factory.instances))
        roles[name] = {group["role"] for i in factory.instances for group in i.param_groups}
        report = json.loads((run["root"] / "fit" / "fit" / "run.json").read_text())
        adapter = report["identity"]["posttraining"]["adapter"]
        if name == "full_continuation":
            assert adapter is None
        else:
            assert set(adapter) == set(name.split("+")), name
    assert len(updates) == 1
    assert roles["full_continuation"] != roles["core"]
