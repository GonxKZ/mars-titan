"""Continuación anclada al padre (#444) en la matriz, los casos, la etapa y la comparación.

Ninguna prueba da pasos de optimizador. El recorrido de `run_case` usa el registrador del
postentrenamiento, que también sustituye a la versión anclada de AdamW y nunca cambia pesos.
"""

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import atomic_json
from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.posttraining import (
    adapter_matrix,
    anchored_continuation,
    campaign_stage,
    stage_comparison,
)
from mars_titan.posttraining import chronological_matrix as cm
from mars_titan.posttraining.run import case_code, run_case, validate_case
from mars_titan.training.anchored_decay import INITIAL
from mars_titan.training.candidate_run import CandidateRecipe
from mars_titan.training.financial_run import ChronologicalRecipe
from mars_titan.training.mars_titan_run import ReadoutRecipe
from tests.posttraining.test_adapter_matrix import parent
from tests.posttraining.test_quantile_adaptation import setup

CONFIGS = Path("configs/posttraining")
ANCHORED = anchored_continuation.CONTROL
CHRONOLOGICAL = [("titans_mac", variant) for variant in cm.TITANS_VARIANTS] + [
    ("mars_titan", None),
    ("cm_v1", None),
    ("episodic_gru", None),
]


@pytest.fixture(scope="module")
def matrix():
    return adapter_matrix.read_matrix(CONFIGS / "adapter-matrix-v3.json")


def controls(items):
    return [item["control"] for item in items if item["control"] is not None]


def _mutated(document, change):
    value = copy.deepcopy(document)
    change(value)
    return value


@pytest.mark.parametrize("head", [QUANTILE_HEAD, adapter_matrix.SCALAR])
def test_the_references_add_the_anchored_continuation_after_the_full_one(matrix, head):
    document, digest = matrix
    assert document[ANCHORED] == dict(anchor=INITIAL, campaign=["references"])
    expected = ["full_continuation", ANCHORED]
    if head == adapter_matrix.SCALAR:
        expected = ["linear_residual", *expected]
    for family in adapter_matrix.FAMILIES:
        items = adapter_matrix.cases(document, digest, family, head=head)
        for seed in document["budget"]["seeds"]:
            own = [item for item in items if item["case"]["seed"] == seed]
            assert controls(own) == expected
            full, anchored = (own[expected.index(name)] for name in expected[-2:])
            # Mismo objetivo, presupuesto, λ y selección. Solo cambia el ancla.
            assert anchored["case"] == dict(full["case"], weight_decay_anchor=INITIAL)
            # Presupuesto fijo sin paciencia: ninguna de las dos para antes que la otra, así
            # que el par conserva el mismo número de actualizaciones.
            assert anchored["case"]["selection"]["patience"] is None
            assert anchored["id"] == f"seed-{seed}/{ANCHORED}"
            validate_case(anchored["case"])
            assert "training/anchored_decay.py" in case_code(anchored["case"], masked=True)
            assert "training/anchored_decay.py" not in case_code(full["case"], masked=True)


def test_the_chronological_families_keep_the_control_in_reserve(matrix):
    document, digest = matrix
    for family, variant in CHRONOLOGICAL:
        proposed = cm.cases(document, digest, family, variant=variant)
        assert ANCHORED not in controls(proposed), family
        reserve = cm.cases(document, digest, family, variant=variant, reserve=True)
        anchored = [item for item in reserve if item["control"] == ANCHORED]
        assert len(anchored) == len(document["budget"]["seeds"])
        for item in anchored:
            case = cm.validate_case(item["case"])
            full = next(
                other["case"]
                for other in reserve
                if other["control"] == cm.CONTROL and other["case"]["seed"] == case["seed"]
            )
            assert case == dict(full, control=ANCHORED)
            assert cm.recipe_options(case)["weight_decay_anchor"] == INITIAL
            assert cm.recipe_options(full)["weight_decay_anchor"] is None


def test_the_scope_decides_where_the_control_is_proposed(matrix):
    document, digest = matrix
    value = adapter_matrix.validate_matrix(
        _mutated(
            document,
            lambda v: v[ANCHORED].update(
                campaign=["titans_mac:mac_online", "readers", "episodic_gru"]
            ),
        )
    )
    assert ANCHORED in controls(cm.cases(value, digest, "titans_mac", variant="mac_online"))
    assert ANCHORED not in controls(cm.cases(value, digest, "titans_mac", variant="mac_frozen"))
    assert ANCHORED in controls(cm.cases(value, digest, "cm_v1", bank=False))
    assert ANCHORED in controls(cm.cases(value, digest, "episodic_gru"))
    gru = adapter_matrix.cases(value, digest, "gru", head=QUANTILE_HEAD)
    assert ANCHORED not in controls(gru)
    gru = adapter_matrix.cases(value, digest, "gru", head=QUANTILE_HEAD, reserve=True)
    assert ANCHORED in controls(gru)


INVALID = {
    "missing_anchor": lambda v: v[ANCHORED].pop("anchor"),
    "extra_key": lambda v: v[ANCHORED].update(weight_decay=0.02),
    "zero_anchor": lambda v: v[ANCHORED].update(anchor="zero"),
    "unknown_scope": lambda v: v[ANCHORED].update(campaign=["titans_mac:mac_offline"]),
    "repeated_scope": lambda v: v[ANCHORED].update(campaign=["references", "references"]),
    "scope_string": lambda v: v[ANCHORED].update(campaign="references"),
    "not_a_section": lambda v: v.update({ANCHORED: True}),
    "without_decay": lambda v: v["budget"].update(weight_decay=0.0),
}


@pytest.mark.parametrize("name", sorted(INVALID))
def test_invalid_sections_are_rejected(matrix, name):
    with pytest.raises(ValueError):
        adapter_matrix.validate_matrix(_mutated(matrix[0], INVALID[name]))


def test_only_version_three_admits_the_section(matrix):
    document, _ = adapter_matrix.read_matrix(CONFIGS / "adapter-matrix-v2.json")
    with pytest.raises(ValueError, match="contrato"):
        adapter_matrix.validate_matrix(
            dict(copy.deepcopy(document), anchored_continuation=matrix[0][ANCHORED])
        )
    # Sin la sección, la v3 conserva los casos de antes.
    plain = _mutated(matrix[0], lambda v: v.pop(ANCHORED))
    adapter_matrix.validate_matrix(plain)
    items = adapter_matrix.cases(plain, matrix[1], "gru", head=QUANTILE_HEAD, reserve=True)
    assert ANCHORED not in controls(items)


def test_the_anchor_belongs_only_to_a_supervised_full_continuation(matrix):
    document, digest = matrix
    items = {
        item["id"].split("/", 1)[1]: item["case"]
        for item in adapter_matrix.cases(document, digest, "gru", head=QUANTILE_HEAD)
        if item["case"]["seed"] == 42
    }
    anchored = items[ANCHORED]
    for broken in (
        dict(items["fusion"], weight_decay_anchor=INITIAL),
        dict(anchored, weight_decay_anchor="zero"),
        dict(anchored, weight_decay=0.0),
        dict(anchored, mode="mae"),
        dict(anchored, condition="real_resampled"),
    ):
        with pytest.raises(ValueError, match="decaimiento anclado"):
            validate_case(broken)
    chronological = next(
        item["case"]
        for item in cm.cases(document, digest, "episodic_gru", reserve=True)
        if item["control"] == ANCHORED
    )
    for broken in (dict(chronological, weight_decay=0.0), dict(chronological, control="l2_sp")):
        with pytest.raises(ValueError, match="matriz declarada"):
            cm.validate_case(broken)


@pytest.mark.parametrize(
    "recipe",
    [
        ChronologicalRecipe,
        CandidateRecipe,
        lambda **options: ReadoutRecipe(**options),
    ],
    ids=["titans", "candidate", "readout"],
)
def test_recipes_keep_their_identity_without_anchor(recipe):
    plain = recipe()
    assert "weight_decay_anchor" not in plain.identity()
    anchored = replace(plain, weight_decay_anchor=INITIAL, weight_decay=0.01)
    assert anchored.identity()["weight_decay_anchor"] == INITIAL
    # El caso fija siempre el ancla: un caso sin ella no hereda la del padre.
    assert (
        replace(anchored, weight_decay_anchor=None).identity()
        == replace(plain, weight_decay=0.01).identity()
    )
    for value in ("parent", 0, True, ["initial_parameters"]):
        with pytest.raises(ValueError):
            replace(plain, weight_decay_anchor=value)


@pytest.mark.parametrize("family", ["gru", "transformer"])
def test_the_plan_counts_the_anchors_in_the_state(matrix, family):
    document, digest = matrix
    model = parent(family)
    rows = {
        row["id"]: row
        for row in adapter_matrix.plan(
            document, digest, family, model, updates_per_epoch=13, linear_features=1734
        )
    }
    full, anchored = rows["seed-42/full_continuation"], rows[f"seed-42/{ANCHORED}"]
    total = sum(value.numel() for value in model.parameters())
    assert anchored["trainable_parameters"] == full["trainable_parameters"] == total
    # Pesos y dos momentos en la continuación, más las anclas en la anclada (float32).
    assert (full["state_bytes"], anchored["state_bytes"]) == (12 * total, 16 * total)
    assert anchored["invalidates"] == full["invalidates"]
    assert anchored["updates"] == full["updates"]


def _stage(name):
    return campaign_stage.load_stage(CONFIGS / f"historical-masked-adapter-stage-{name}.json")


@pytest.mark.parametrize(("name", "expected"), [("a", 630), ("a-v2", 270)])
def test_only_the_stages_that_name_the_control_plan_it(name, expected):
    stage = _stage(name)
    # A y A v2 comparten la matriz v3 y las dos añaden el control. A tiene tres ámbitos y
    # 42 ventanas con padre y A v2 el ámbito conjunto con 18: 5 × 42 × 3 y 5 × 18 × 3.
    assert stage["matrix_sha256"] == _stage("a")["matrix_sha256"]
    assert stage["additional_controls"] == [ANCHORED]
    jobs = campaign_stage.plan_stage(stage)
    # Las cinco referencias neuronales de la etapa.
    references = set(stage["families"])
    assert len(references) == 5
    anchored = [job for job in jobs if job["control"] == ANCHORED]
    full = [
        job
        for job in jobs
        if job["control"] == "full_continuation" and job["base_arm"] in references
    ]
    assert {job["base_arm"] for job in anchored} == references

    def key(job):
        return job["scope"], job["window"], job["base_arm"], job["seed"]

    # Una por ventana, brazo de referencia y semilla, igual que la continuación completa.
    assert sorted(map(key, anchored)) == sorted(map(key, full)) and len(anchored) == expected
    for job in anchored:
        assert job["arm"] == f"{job['base_arm']}__{ANCHORED}" and job["point"] == ANCHORED
        assert job["case"]["weight_decay_anchor"] == INITIAL
        assert job["depends"] == next(j["depends"] for j in full if key(j) == key(job))
        assert campaign_stage.candidate_kind(job) == "continuation"
    # Sin nombrarlo, la misma etapa no lo planifica y conserva el resto del plan.
    without = {key: value for key, value in stage.items() if key != "additional_controls"}
    remaining = campaign_stage.plan_stage(without)
    assert not any(job["control"] == ANCHORED for job in remaining)
    assert len(remaining) == len(jobs) - expected


def test_a_stage_cannot_add_a_control_its_matrix_does_not_declare(tmp_path):
    value = json.loads((CONFIGS / "historical-masked-adapter-stage-a.json").read_text())
    value.update(
        campaign=str(Path("configs/baselines/historical-masked-campaign-a.json").resolve()),
        matrix=str((CONFIGS / "adapter-matrix-v2.json").resolve()),
    )
    atomic_json(tmp_path / "stage.json", value)
    with pytest.raises(ValueError, match="controles que su matriz declara"):
        campaign_stage.load_stage(tmp_path / "stage.json")


def test_the_comparison_contrasts_the_anchored_continuation_where_it_exists():
    declaration = CONFIGS / "historical-masked-adapter-comparison-a.json"
    loaded = stage_comparison.load_declaration(declaration)
    references = set(loaded["stage"]["families"])
    assert references
    for base_arm, config in loaded["configs"].items():
        families = config["comparison"]["families"]
        group = loaded["groups"][base_arm]
        if base_arm in references:
            anchored = f"{base_arm}__{ANCHORED}"
            assert group[stage_comparison.ANCHORED] == anchored
            assert config["arms"][anchored]["family"] == "posttraining_control"
            assert anchored in families["versus_frozen_parent"]["variants"]
            assert families["versus_anchored_continuation"] == dict(
                kind="delta",
                base=anchored,
                variants=[*group[stage_comparison.ADAPTED], group[stage_comparison.CONTINUATION]],
            )
            # La familia previa con la continuación como base no cambia.
            assert families["versus_full_continuation"]["variants"] == group["adapted"]
        else:
            assert group[stage_comparison.ANCHORED] is None
            assert "versus_anchored_continuation" not in families
            assert not any(ANCHORED in arm for arm in config["arms"])


def test_the_anchored_continuation_is_a_base_but_not_its_own_variant():
    valid = {"versus_anchored": dict(base=ANCHORED, variants=["adapted", "full_continuation"])}
    assert stage_comparison._families(valid) == valid
    with pytest.raises(ValueError, match="debe contrastar papeles declarados"):
        stage_comparison._families({"self": dict(base=ANCHORED, variants=[ANCHORED, "adapted"])})


@pytest.fixture(scope="module")
def edition(tmp_path_factory):
    from types import SimpleNamespace

    from tests.posttraining.masked_fixture import masked_ordered

    root = tmp_path_factory.mktemp("anchored-edition")
    view, ordered, report = masked_ordered(root / "data")
    return SimpleNamespace(view=view, ordered=ordered, report=report, root=root)


def test_run_case_builds_the_anchored_optimizer_with_the_continuation_budget(
    tmp_path, recorder, edition, matrix
):
    """La continuación anclada recorre `run_case` con las actualizaciones de la completa.

    El registrador no cambia pesos, así que la época cero es la elegida y las predicciones
    son las del padre en las dos continuaciones.
    """
    document, digest = matrix
    parent_model, data, grid, scale, handles = setup(edition, tmp_path, "gru")
    cases = {
        item["control"]: item["case"]
        for item in adapter_matrix.cases(document, digest, "gru", head=QUANTILE_HEAD)
        if item["case"]["seed"] == 42 and item["control"] is not None
    }
    reports = {}
    for control in ("full_continuation", ANCHORED):
        count = len(recorder.optimizers)
        reports[control] = run_case(
            data,
            tmp_path / control,
            cases[control],
            grid,
            scale,
            parent=parent_model,
            batch_size=2,
            device="cpu",
            diagnostic=True,
        )
        optimizer = recorder.optimizers[count]
        assert optimizer.anchored is (control == ANCHORED)
        assert (optimizer.lr, optimizer.weight_decay) == (
            document["budget"]["learning_rate"],
            document["budget"]["weight_decay"],
        )
        assert len(optimizer.parameters) == len(list(parent_model.model.parameters()))
        table = pq.read_table(tmp_path / control / "validation-predictions.parquet").to_pydict()
        np.testing.assert_array_equal(table["prediction"], table["parent"])
    full, anchored = reports["full_continuation"], reports[ANCHORED]
    assert anchored["status"] == full["status"] == "completed"
    assert anchored["global_step"] == full["global_step"] > 0
    assert anchored["selection"]["best_epoch"] == full["selection"]["best_epoch"] == 0
    assert anchored["identity"]["case"] == dict(
        full["identity"]["case"], weight_decay_anchor=INITIAL
    )
    assert set(anchored["identity"]["code"]) - set(full["identity"]["code"]) == {
        "training/anchored_decay.py"
    }
    for handle in handles:
        handle.close()
