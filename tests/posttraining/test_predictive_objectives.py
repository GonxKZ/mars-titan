"""Objetivos de la corrección lineal de #128 en la etapa de adaptadores, sin pasos de optimizador.

Las pruebas puras fijan la sección de la matriz, los casos, el presupuesto y lo que queda
fuera. Las de la etapa A v2 derivan el plan, la cadena y la comparación de la configuración
del repositorio sin leer datos. El recorrido de la etapa usa la campaña reducida de
`campaign_fixture` (GRU, semilla 42, dos ventanas US) y el optimizador del fixture
`recorder`, que nunca cambia pesos. La comparación usa la etapa reducida de
`test_stage_comparison`, cuyas predicciones se escriben con valores conocidos.
"""

import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import atomic_json
from mars_titan.models.quantile_head import QUANTILE_COLUMNS, QUANTILE_HEAD
from mars_titan.posttraining import adapter_matrix, campaign_stage, stage_comparison
from mars_titan.posttraining import chronological_matrix as cm
from mars_titan.posttraining import predictive_objectives as predictive
from mars_titan.posttraining.run import validate_case
from mars_titan.training import campaign_chain
from tests.posttraining.campaign_fixture import CpuLease, base_campaign
from tests.posttraining.real_only import real_data_only
from tests.posttraining.test_adapter_matrix import parent

CONFIGS = "configs/posttraining"
SECTION = predictive.SECTION_NAME
STAGE_V2 = f"{CONFIGS}/historical-masked-adapter-stage-a-v2.json"
COMPARISON_V2 = f"{CONFIGS}/historical-masked-adapter-comparison-a-v2.json"
REFERENCES = ("rnn", "lstm", "gru", "dlinear", "transformer_compact")
MODES = ("reinforce", "expected", "mae", "klpo_full", "klpo_mc", "klpo_exact")


@pytest.fixture(scope="module")
def matrix():
    return adapter_matrix.read_matrix(f"{CONFIGS}/adapter-matrix-v3.json")


@pytest.fixture(scope="module")
def v2():
    stage = campaign_stage.load_stage(STAGE_V2)
    return stage, campaign_stage.plan_stage(stage)


def _mutated(document, change):
    value = copy.deepcopy(document)
    change(value)
    return value


def controls(items):
    return [item["control"] for item in items if item["control"] is not None]


def test_the_references_add_one_control_per_objective_with_the_continuation_budget(matrix):
    document, digest = matrix
    assert predictive.MODES == MODES
    assert document[SECTION]["modes"] == list(MODES)
    assert document[SECTION]["campaign"] == ["references"]
    for family in adapter_matrix.FAMILIES:
        items = adapter_matrix.cases(document, digest, family, head=QUANTILE_HEAD)
        for seed in document["budget"]["seeds"]:
            own = {item["control"]: item for item in items if item["case"]["seed"] == seed}
            full = own["full_continuation"]["case"]
            assert full["mode"] == "neural_pinball"
            for mode in MODES:
                item = own[f"objective_{mode}"]
                assert item["id"] == f"seed-{seed}/objective_{mode}"
                # Mismo presupuesto, semilla y selección que la continuación. Solo cambia el
                # objetivo, sin adaptador, ancla ni repaso.
                assert item["case"] == dict(full, mode=mode)
                validate_case(item["case"])


def test_a_scalar_parent_does_not_repeat_its_linear_correction(matrix):
    document, digest = matrix
    own = [
        item
        for item in adapter_matrix.cases(document, digest, "gru", head=adapter_matrix.SCALAR)
        if item["case"]["seed"] == 42
    ]
    linear = [item["case"]["mode"] for item in own if item["control"] == "linear_residual"]
    assert linear == ["mae"]
    found = [item["case"]["mode"] for item in own if item["control"] in predictive.CONTROLS]
    assert found == ["reinforce", "expected", "klpo_full", "klpo_mc", "klpo_exact"]
    assert len({json.dumps(item["case"], sort_keys=True) for item in own}) == len(own)


def test_the_chronological_families_have_no_objectives(matrix):
    document, digest = matrix
    for family, variant in [("titans_mac", "mac_online"), ("cm_v1", None), ("episodic_gru", None)]:
        found = cm.cases(document, digest, family, variant=variant, reserve=True)
        assert not set(controls(found)) & set(predictive.CONTROLS), family


def test_without_proposal_the_objectives_stay_in_reserve(matrix):
    document, digest = matrix
    value = adapter_matrix.validate_matrix(
        _mutated(document, lambda v: v[SECTION].update(campaign=[]))
    )
    plain = adapter_matrix.cases(value, digest, "gru", head=QUANTILE_HEAD)
    assert not set(controls(plain)) & set(predictive.CONTROLS)
    reserve = adapter_matrix.cases(value, digest, "gru", head=QUANTILE_HEAD, reserve=True)
    assert set(predictive.CONTROLS) <= set(controls(reserve))
    absent = adapter_matrix.validate_matrix(_mutated(document, lambda v: v.pop(SECTION)))
    reserve = adapter_matrix.cases(absent, digest, "gru", head=QUANTILE_HEAD, reserve=True)
    assert not set(controls(reserve)) & set(predictive.CONTROLS)


def test_the_excluded_objectives_and_parents_declare_why_and_when_to_revisit(matrix):
    excluded = matrix[0][SECTION]["excluded"]
    assert tuple(excluded) == ("neural_mae", "neural_mse", "chronological_parents")
    for name, item in excluded.items():
        assert set(item) == {"reason", "revisit_if"}, name
    # Las dos continuaciones quedan fuera porque el padre se continúa con su pinball.
    assert "pinball" in excluded["neural_mae"]["reason"]
    assert "MatrixParent" in excluded["chronological_parents"]["reason"]


def _excluded(name, **changes):
    return lambda v: v[SECTION]["excluded"][name].update(changes)


INVALID = {
    "missing_mode": lambda v: v[SECTION]["modes"].pop(),
    "reordered_modes": lambda v: v[SECTION]["modes"].reverse(),
    "neural_mode": lambda v: v[SECTION]["modes"].append("neural_mse"),
    "chronological_scope": lambda v: v[SECTION].update(campaign=["readers"]),
    "repeated_scope": lambda v: v[SECTION].update(campaign=["references", "references"]),
    "scope_string": lambda v: v[SECTION].update(campaign="references"),
    "extra_key": lambda v: v[SECTION].update(ratio=0.5),
    "missing_excluded": lambda v: v[SECTION].pop("excluded"),
    "missing_exclusion": lambda v: v[SECTION]["excluded"].pop("neural_mse"),
    "unknown_exclusion": lambda v: v[SECTION]["excluded"].update(
        neural_pinball=dict(reason="x" * 50, revisit_if="x" * 30)
    ),
    "short_reason": _excluded("neural_mae", reason="corto"),
    "short_revisit": _excluded("neural_mse", revisit_if="nunca"),
    "missing_revisit": lambda v: v[SECTION]["excluded"]["chronological_parents"].pop("revisit_if"),
    "reason_not_text": _excluded("neural_mae", reason=["x" * 50]),
}


@pytest.mark.parametrize("name", sorted(INVALID))
def test_invalid_sections_are_rejected(matrix, name):
    with pytest.raises(ValueError, match="#128|queda fuera"):
        adapter_matrix.validate_matrix(_mutated(matrix[0], INVALID[name]))


def test_only_version_three_admits_the_section(matrix):
    document, _ = adapter_matrix.read_matrix(f"{CONFIGS}/adapter-matrix-v2.json")
    with pytest.raises(ValueError, match="contrato"):
        adapter_matrix.validate_matrix(
            dict(copy.deepcopy(document), **{SECTION: matrix[0][SECTION]})
        )


def test_an_unknown_objective_has_no_control():
    assert predictive.control("klpo_mc") == "objective_klpo_mc"
    for mode in ("neural_mae", "neural_pinball", "mse"):
        with pytest.raises(ValueError, match="#128"):
            predictive.control(mode)


@pytest.mark.parametrize("family", ["gru", "transformer"])
def test_the_plan_sizes_each_objective_as_the_linear_correction(matrix, family):
    document, digest = matrix
    plan = adapter_matrix.plan(
        document, digest, family, parent(family), updates_per_epoch=13, linear_features=1734
    )
    rows = {row["id"]: row for row in plan}
    full = rows["seed-42/full_continuation"]
    for mode in MODES:
        row = rows.get(f"seed-42/objective_{mode}")
        if row is None:
            # Un padre escalar ya ajusta su MAE con la corrección lineal residual.
            assert rows["seed-42/linear_residual"]["case"]["mode"] == mode
            continue
        # Una capa lineal float32 con sesgo, pesos y dos momentos, y las mismas
        # actualizaciones que la continuación completa.
        assert row["trainable_parameters"] == 1735
        assert row["state_bytes"] == 3 * 4 * 1735
        assert row["updates"] == full["updates"] == document["budget"]["epochs"] * 13
        assert row["invalidates"] == []


def test_campaign_a_v2_fits_the_six_objectives_for_the_five_references(v2):
    stage, jobs = v2
    assert stage["additional_controls"] == ["anchored_continuation", SECTION]
    found = [job for job in jobs if job["control"] in predictive.CONTROLS]
    # Seis objetivos, cinco referencias, 18 ventanas con padre y tres semillas.
    assert len(found) == 6 * 5 * 18 * 3
    assert Counter(job["base_arm"] for job in found) == {arm: 6 * 18 * 3 for arm in REFERENCES}
    assert Counter(job["control"] for job in found) == {
        control: 5 * 18 * 3 for control in predictive.CONTROLS
    }
    counts = campaign_stage.count_stage(stage, jobs)
    totals = (counts["training_jobs"], counts["prediction_jobs"], counts["selection_jobs"])
    assert totals == (7074 + 1620, 1494, 1577)
    assert stage["limits"] == dict(max_training_jobs=8694, max_prediction_jobs=1494)
    # Cada objetivo parte del mismo padre de k-1, con las mismas dependencias y el mismo
    # presupuesto que la continuación completa de su ventana y semilla.
    full = {
        (job["scope"], job["window"], job["base_arm"], job["seed"]): job
        for job in jobs
        if job["control"] == "full_continuation"
    }
    for job in found:
        twin = full[job["scope"], job["window"], job["base_arm"], job["seed"]]
        assert job["depends"] == twin["depends"] and job["parent_window"] == twin["parent_window"]
        assert job["case"] == dict(twin["case"], mode=job["case"]["mode"])
        assert job["arm"] == f"{job['base_arm']}__{job['control']}"


def test_the_objectives_do_not_compete_in_the_chain(v2):
    stage, jobs = v2
    chains = campaign_stage.plan_chain(stage, jobs)
    objectives = {job["id"] for job in jobs if job["control"] in predictive.CONTROLS}
    read = {name for chain in chains for name in chain["depends"]}
    assert objectives and not objectives & read
    # Los demás ajustes y padres congelados de la etapa siguen siendo candidatos.
    assert {job["id"] for job in jobs} - objectives <= read
    assert not any(campaign_stage.chain_candidate(job) for job in jobs if job["id"] in objectives)
    # Se ejecutan en su ventana, antes de la selección, aunque no la alimenten.
    order = campaign_stage.ordered_jobs(jobs, chains)
    assert len(order) == len(jobs) + len(chains)


def test_a_stage_plans_the_objectives_only_when_it_names_the_section(tmp_path, v2):
    stage, jobs = v2
    folder = Path(STAGE_V2).parent.resolve()
    value = json.loads(open(STAGE_V2).read())
    value.update(
        campaign=str(folder / value["campaign"]),
        matrix=str(folder / value["matrix"]),
        additional_controls=["anchored_continuation"],
        limits=dict(max_training_jobs=7074, max_prediction_jobs=1494),
    )
    atomic_json(tmp_path / "stage.json", value)
    plain = campaign_stage.load_stage(tmp_path / "stage.json")
    planned = campaign_stage.plan_stage(plain)
    assert len(planned) == len(jobs) - 1620
    assert not any(job["control"] in predictive.CONTROLS for job in planned)
    # Una matriz sin la sección no admite que la etapa la nombre.
    matrix = json.loads(open(f"{CONFIGS}/adapter-matrix-v3.json").read())
    matrix.pop(SECTION)
    atomic_json(tmp_path / "matrix.json", matrix)
    value.update(matrix=str(tmp_path / "matrix.json"), additional_controls=[SECTION])
    atomic_json(tmp_path / "stage.json", value)
    with pytest.raises(ValueError, match="controles que su matriz declara"):
        campaign_stage.load_stage(tmp_path / "stage.json")


def test_the_comparison_contrasts_the_objectives_as_point_outputs():
    loaded = stage_comparison.load_declaration(COMPARISON_V2)
    families = loaded["families"]
    assert families["versus_expected_loss"] == dict(
        base="objective_expected",
        variants=[f"objective_{mode}" for mode in MODES if mode != "expected"],
    )
    assert families["objectives_versus_frozen_parent"] == dict(
        base="frozen_parent", variants=list(predictive.CONTROLS)
    )
    for base_arm, config in loaded["configs"].items():
        group = loaded["groups"][base_arm]
        arms = {control: group[control] for control in predictive.CONTROLS}
        resolved = config["resolved_families"]
        if base_arm not in REFERENCES:
            assert set(arms.values()) == {None}
            assert not {"versus_expected_loss", "objectives_versus_frozen_parent"} & set(resolved)
            continue
        assert arms == {control: f"{base_arm}__{control}" for control in predictive.CONTROLS}
        for arm in arms.values():
            # Corrigen la mediana: salida puntual, con las semillas del padre.
            assert config["arms"][arm] == dict(
                family="posttraining_objective",
                output="point",
                seeds=config["arms"][base_arm]["seeds"],
            )
        expected = arms["objective_expected"]
        assert set(resolved["versus_expected_loss"]) == {
            f"{arm}-{expected}" for arm in arms.values() if arm != expected
        }
        assert set(resolved["objectives_versus_frozen_parent"]) == {
            f"{arm}-{group['frozen_parent']}" for arm in arms.values()
        }
        # Las familias de los adaptadores y de la cadena no cambian.
        for name in ("versus_frozen_parent", "versus_full_continuation"):
            assert not any("objective_" in contrast for contrast in resolved[name])


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    base = base_campaign(tmp_path_factory.mktemp("objectives-stage"), "A")
    value = json.loads(base.stage.read_text())
    value["additional_controls"] = [*value["additional_controls"], SECTION]
    path = base.stage.parent / "stage-objectives.json"
    atomic_json(path, value)
    return SimpleNamespace(base=base, stage=path)


def test_the_stage_fits_each_objective_with_the_continuation_updates(staged, tmp_path, recorder):
    output = tmp_path / "stage"
    with real_data_only():
        summary = campaign_stage.run_stage(
            staged.stage,
            staged.base.views,
            staged.base.output,
            output,
            lease=CpuLease,
            stop=SimpleNamespace(requested=False),
            device="cpu",
        )
    assert summary["status"] == "completed"
    # Cuatro adaptadores, la continuación completa, la anclada y los seis objetivos.
    assert summary["planned"]["training_jobs"] == 12
    found = {
        str(path.parent.relative_to(output / "jobs")): json.loads(path.read_text())
        for path in (output / "jobs").rglob("receipt.json")
    }
    full = found["US/fold-001/gru__full_continuation/fit-s42"]
    for mode in MODES:
        receipt = found[f"US/fold-001/gru__objective_{mode}/fit-s42"]
        assert receipt["identity"]["case"] == dict(full["identity"]["case"], mode=mode)
        assert receipt["updates"] == full["updates"]
        assert receipt["fit_rows"] == full["fit_rows"]
        # Solo la mediana de la política, sin los cinco niveles de la cabeza.
        for partition in ("validation", "calibration", "evaluation"):
            record = receipt["predictions"][partition]
            names = pq.read_schema(output / record["path"]).names
            assert "prediction" in names and not set(QUANTILE_COLUMNS) & set(names)
            assert record["rows_sha256"] == full["predictions"][partition]["rows_sha256"]
    # Cada optimizador da exactamente los pasos de su plan, sin cambiar pesos.
    assert {len(item.calls) for item in recorder.optimizers} == {full["updates"]}
    assert len(recorder.optimizers) == 12
    # La cadena elige solo entre el padre congelado, los adaptadores y las continuaciones.
    selection = campaign_chain.read_selection(output, "US", "fold-001", "gru", 42)
    arms = {item["arm"] for item in selection["candidates"]}
    assert len(arms) == 7 and "gru__full_continuation" in arms
    assert not any("objective_" in arm for arm in arms)


def test_the_comparison_reads_only_the_evaluation_of_each_objective(tmp_path):
    from tests.posttraining.test_stage_comparison import CHAIN_FAMILIES, Stage

    stage = Stage(tmp_path, families=CHAIN_FAMILIES, controls=(SECTION,))
    selected = stage.publish_selection("gru__head")
    assert selected["arm"] == "gru__head"
    sources = stage.sources()
    manifest = json.loads(sources.read_text())
    for mode in MODES:
        entry = manifest["arms"][f"gru__objective_{mode}"]["42"]["fold-001"]
        assert set(entry) == {"input_policy", "view_sha256", "evaluation"}
    config, report, _, _ = stage_comparison.evaluate(stage.declaration, sources, "US", "gru")
    roles = report["posttraining"]["roles"]
    assert {control: roles[control] for control in predictive.CONTROLS} == {
        control: f"gru__{control}" for control in predictive.CONTROLS
    }
    contrasts = report["contrasts"]["US"]
    rows = {row["name"]: row for row in contrasts["versus_expected_loss"]["mae"]["contrasts"]}
    assert set(rows) == {
        f"gru__objective_{mode}-gru__objective_expected" for mode in MODES if mode != "expected"
    }
    # En el fixture todos los objetivos predicen lo mismo: ninguna diferencia.
    assert all(row["estimate"] == pytest.approx(0.0, abs=1e-12) for row in rows.values())
    rows = contrasts["objectives_versus_frozen_parent"]["mae"]["contrasts"]
    assert {row["name"] for row in rows} == {
        f"gru__{control}-gru__frozen_parent" for control in predictive.CONTROLS
    }
