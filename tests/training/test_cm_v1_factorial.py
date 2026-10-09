"""Factorial CM-v1 sobre Titans-MAC, comprobado sin pasos que ajusten pesos.

Los núcleos y los lectores se ajustan con el registrador de gradientes, así que sus
parámetros son los iniciales. Eso permite exigir que B y B+C emitan lo mismo cuando C no ha
modificado el núcleo, y aislar dónde actúa cada factor.
"""

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.memory.retention_bank import RetentionBank
from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from mars_titan.models.titans.config import MACConfig, MemoryConfig
from mars_titan.models.titans.local_control import MACProjectionConfig, MACProjectionControl
from mars_titan.training import cm_v1_factorial as cm
from mars_titan.training import mars_titan_run as mt
from mars_titan.training import mars_titan_walk_forward as mw
from tests.training.test_financial_run import RecordingOptimizer, entries
from tests.training.test_financial_run import shared as shared
from tests.training.test_mars_titan_run import gradients, parent, reader, recipe
from tests.training.test_mars_titan_run import native as native
from tests.training.test_mars_titan_walk_forward import readout_recipe
from tests.training.test_titans_walk_forward import Factory, views
from tests.training.test_titans_walk_forward import (
    learning_doubles_module as learning_doubles_module,
)
from tests.training.test_titans_walk_forward import recipe as core_recipe

DECLARED = Path("configs/titans/cm-v1-factorial.json")
CORE = {arm: "cm_v1_core_c" if c else "cm_v1_core_b" for arm, (c, _) in cm.ARMS.items()}


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def declaration(root, core, readout, **changes):
    """Declaración técnica: C mide un flujo cada dos observaciones y M actúa con capacidad 8."""
    document = json.loads(DECLARED.read_text())
    document["base"].update(core_recipe=str(core), readout_recipe=str(readout))
    document["control"].update(rank=2, frequency=2, grid_size=16, threshold=0.0, max_flows=1)
    document["control"]["weight"] = 0.5
    document["consolidation"].update(frontier=2, new_candidates=2)
    for key, value in changes.items():
        document[key] = value
    path = root / f"cm-{len(list(root.glob('cm-*')))}.json"
    path.write_text(json.dumps(document))
    return path


@pytest.fixture(scope="module")
def factorial(tmp_path_factory, learning_doubles_module):
    root = tmp_path_factory.mktemp("cm-v1-factorial")
    view, _ = views(root / "base")
    path = declaration(root, core_recipe(root), readout_recipe(root))
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    cores, arms = {}, {}
    try:
        for name in cm.CORES:
            factory = Factory()
            report = cm.run_cm_v1_core_window(
                view,
                "fold-000",
                core=name,
                seed=42,
                output=root / "runs" / name,
                search_case=None,
                declaration=path,
                device="cpu",
                indices=root / "runs" / "indices",
                optimizer_factory=factory,
            )
            cores[name] = (report, factory)
        for arm in cm.ARMS:
            factory = Factory()
            report = cm.run_cm_v1_window(
                view,
                root / "runs" / CORE[arm],
                arm=arm,
                seed=42,
                output=root / "runs" / arm,
                search_case="lr1e-4",
                declaration=path,
                device="cpu",
                indices=root / "runs" / "indices",
                optimizer_factory=factory,
            )
            arms[arm] = (report, factory)
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
    return dict(root=root, view=view, declaration=path, cores=cores, arms=arms)


def fit(folder):
    return json.loads((folder / "fit/run.json").read_text())


def columns(folder, partition, names):
    rows = pq.read_table(folder / f"{partition}-predictions.parquet").sort_by("sample_id")
    return {name: rows[name].to_pylist() for name in names}


def test_declared_factorial_fixes_b_its_recipes_and_its_four_arms():
    document = cm.load_declaration(DECLARED)
    root = Path("configs/titans").resolve()
    assert document["recipes"] == dict(
        core_recipe=str(root / "chronological-training-historical-masked.json"),
        readout_recipe=str(root / "episodic-readout-historical-masked.json"),
    )
    b, c = cm.control_contract(document, False), cm.control_contract(document, True)
    assert (b["mode"], b["weight"], c["mode"], c["weight"]) == ("disabled", 0.0, "penalty", 0.1)
    assert {k: v for k, v in b.items() if k not in ("mode", "weight")} == {
        k: v for k, v in c.items() if k not in ("mode", "weight")
    }
    plan = mt.case_recipe(mt.load_recipe(document["recipes"]["readout_recipe"]), "lr1e-4")
    reservoir, anchored = (cm.bank_retention(document, plan, m) for m in (False, True))
    assert (reservoir.policy, anchored.policy) == ("reservoir", "anchored")
    assert {**vars(reservoir), "policy": None} == {**vars(anchored), "policy": None}
    assert reservoir.capacity == plan.bank_capacity == 1024
    # Dos flujos medidos por evento caben en el presupuesto de C con dimensión 64 en FP32.
    titans = json.loads(Path(document["recipes"]["core_recipe"]).read_text())["predictor"]
    mac = MACConfig(
        memory=MemoryConfig(dim=titans["hidden_size"], max_tokens=1, max_batch=256),
        persistent_tokens=titans["persistent_tokens"],
        max_segment=1,
    )
    control = MACProjectionControl(MACProjectionConfig(**c), mac, dtype=torch.float32)
    assert control.estimated_bytes(c["max_flows"]) <= c["max_estimated_bytes"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d["base"].update(refinements=2), "B y sus cuatro brazos"),
        (lambda d: d["base"].update(retention="anchored"), "B y sus cuatro brazos"),
        (lambda d: d["base"].update(admission="m2"), "B y sus cuatro brazos"),
        (lambda d: d["arms"].pop("cm_v1_bcm"), "B y sus cuatro brazos"),
        (lambda d: d["arms"]["cm_v1_bm"].update(control=True), "B y sus cuatro brazos"),
        (lambda d: d["cores"].update(cm_v1_core_b=True), "B y sus cuatro brazos"),
        (lambda d: d.update(status="executed"), "B y sus cuatro brazos"),
        (lambda d: d["control"].update(mode="penalty"), "campos de su control"),
        (lambda d: d["consolidation"].update(policy="uniform"), "centros fijos"),
        (lambda d: d["control"].update(weight=0.0), "peso positivo"),
        (lambda d: d["control"].update(rank=5), "rank"),
    ],
)
def test_declaration_rejects_changes_to_b_the_arms_or_the_factors(tmp_path, change, message):
    document = json.loads(DECLARED.read_text())
    change(document)
    path = tmp_path / "cm.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=message):
        cm.load_declaration(path)


def test_both_cores_start_equal_keep_equal_updates_and_emit_the_same_rows(factorial):
    (b, b_factory), (c, c_factory) = (factorial["cores"][name] for name in cm.CORES)
    root = factorial["root"] / "runs"
    assert b["request"]["local_control"]["mode"] == "disabled"
    assert c["request"]["local_control"]["mode"] == "penalty"
    left, right = fit(root / "cm_v1_core_b"), fit(root / "cm_v1_core_c")
    assert (
        left["identity"]["initial_parameters_sha256"]
        == right["identity"]["initial_parameters_sha256"]
    )
    assert (
        left["identity"]["local_control"]["basis_sha256"]
        == right["identity"]["local_control"]["basis_sha256"]
    )
    assert (
        left["global_step"]
        == right["global_step"]
        == len(b_factory.records)
        == len(c_factory.records)
        > 0
    )
    assert right["history"][-1]["train"]["control_groups_kept"] > 0
    assert "control_groups" not in left["history"][-1]["train"]
    names = ["sample_id", "prediction", *QUANTILE_COLUMNS]
    for partition in ("validation", "calibration", "evaluation"):
        assert columns(root / "cm_v1_core_b", partition, names) == columns(
            root / "cm_v1_core_c", partition, names
        )
    # El término de C añade gradiente al núcleo en algún paso.
    assert any(
        not torch.equal(x, y)
        for left_step, right_step in zip(b_factory.records, c_factory.records, strict=True)
        for x, y in zip(left_step, right_step, strict=True)
        if x is not None
    )


def test_each_factor_acts_only_where_it_is_declared(factorial):
    root = factorial["root"] / "runs"
    reports = {arm: report for arm, (report, _) in factorial["arms"].items()}
    assert all(report["status"] == "completed" for report in reports.values())
    assert len({report["identity"]["variant_sha256"] for report in reports.values()}) == 4
    identities = {arm: fit(root / arm)["identity"] for arm in cm.ARMS}
    policies = {arm: identity["retention"]["policy"] for arm, identity in identities.items()}
    assert policies == dict(
        cm_v1_b="reservoir", cm_v1_bc="reservoir", cm_v1_bm="anchored", cm_v1_bcm="anchored"
    )
    # Mismo lector inicial en los cuatro brazos y mismo núcleo inicial en los dos núcleos.
    assert len({identity["initial_readout_sha256"] for identity in identities.values()}) == 1
    for arm, report in reports.items():
        parent = report["request"]["parent"]
        assert parent["path"] == str((root / CORE[arm]).resolve())
        assert report["identity"]["variant"]["base"]["recipe"]["local_control"]["mode"] == (
            "penalty" if cm.ARMS[arm][0] else "disabled"
        )
    names = ["sample_id", "prediction", *QUANTILE_COLUMNS]
    for partition in ("validation", "calibration", "evaluation"):
        rows = {arm: columns(root / arm, partition, names) for arm in cm.ARMS}
        # Sin pasos, el núcleo de C conserva los parámetros de B y C no cambia la emisión.
        assert rows["cm_v1_b"] == rows["cm_v1_bc"]
        assert rows["cm_v1_bm"] == rows["cm_v1_bcm"]
        assert rows["cm_v1_b"]["sample_id"] == rows["cm_v1_bm"]["sample_id"]


def test_arm_and_parent_control_must_agree(factorial, tmp_path):
    root = factorial["root"] / "runs"
    options = dict(
        seed=42,
        search_case="lr1e-4",
        declaration=factorial["declaration"],
        device="cpu",
        optimizer_factory=Factory(),
    )
    with pytest.raises(ValueError, match="control C del padre"):
        cm.run_cm_v1_window(
            factorial["view"],
            root / "cm_v1_core_b",
            arm="cm_v1_bc",
            output=tmp_path / "x",
            **options,
        )
    with pytest.raises(ValueError, match="no pertenece"):
        cm.run_cm_v1_window(
            factorial["view"],
            root / "cm_v1_core_b",
            arm="cm_v1_x",
            output=tmp_path / "y",
            **options,
        )
    # MARS-TITAN no parte de un núcleo con C, ni siquiera en modo disabled.
    with pytest.raises(ValueError, match="control C del padre"):
        mw.run_mars_titan_window(
            factorial["view"],
            root / "cm_v1_core_b",
            readout_recipe(tmp_path),
            components={"episodic_bank": "m1"},
            seed=42,
            output=tmp_path / "z",
            search_case="lr1e-4",
            device="cpu",
        )
    assert not any((tmp_path / name).exists() for name in ("x", "y", "z"))


def test_completed_arm_is_returned_and_a_changed_declaration_is_refused(factorial, tmp_path):
    root = factorial["root"] / "runs"
    report, _ = factorial["arms"]["cm_v1_bm"]
    again = cm.run_cm_v1_window(
        factorial["view"],
        root / "cm_v1_core_b",
        arm="cm_v1_bm",
        seed=42,
        output=root / "cm_v1_bm",
        search_case="lr1e-4",
        declaration=factorial["declaration"],
        device="cpu",
    )
    assert again == report
    document = json.loads(factorial["declaration"].read_text())
    document["consolidation"]["max_swaps"] = 9
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="petición"):
        cm.run_cm_v1_window(
            factorial["view"],
            root / "cm_v1_core_b",
            arm="cm_v1_bm",
            seed=42,
            output=root / "cm_v1_bm",
            search_case="lr1e-4",
            declaration=changed,
            device="cpu",
        )


def consolidating(streams, output, native, *, capacity, policy, **retention):
    plan = recipe(bank_capacity=capacity)
    readout, codec = reader(streams, "m1")
    return mt.ReadoutTrainer(
        parent(streams),
        readout,
        plan,
        admission="m1",
        retention=mt.retention_config(plan, "m1", policy=policy, **retention),
        native=native,
        codec=codec,
        train=streams["train"],
        validation=streams["validation"],
        output=output,
        world="fixture",
        fold="0",
        optimizer_factory=RecordingOptimizer,
        audit=True,
    )


def test_consolidation_keeps_real_admitted_episodes_within_its_capacity(
    shared, tmp_path, native, learning_doubles, monkeypatch
):
    """Cada selección parte del banco confirmado más las etiquetas maduras del evento."""
    _, streams = shared
    proposals, original = [], RetentionBank.propose

    def spy(self, incoming, *, confirmed_at):
        before = {r.id: (r.label, r.decision_at, tuple(r.key)) for r in self.records()}
        arrived = {r.id: (r.label, r.decision_at) for r in incoming}
        proposed = original(self, incoming, confirmed_at=confirmed_at)
        after = {r.id: (r.label, r.decision_at, tuple(r.key)) for r in proposed.records()}
        proposals.append((confirmed_at, before, arrived, proposed.receipt, after))
        return proposed

    monkeypatch.setattr(RetentionBank, "propose", spy)
    engine = consolidating(
        streams,
        tmp_path / "run",
        native,
        capacity=8,
        policy="anchored",
        frontier=2,
        new_candidates=2,
    )
    engine.run()
    restricted = 0
    for confirmed_at, before, arrived, receipt, after in proposals:
        clients = set(receipt["client_ids"])
        assert clients == set(before) | set(arrived) and not set(before) & set(arrived)
        assert set(after) == set(receipt["retained_ids"]) <= clients
        assert len(after) <= 8
        for identifier, (label, decision_at, key) in after.items():
            if identifier in before:
                assert before[identifier] == (label, decision_at, key)
            else:
                assert arrived[identifier] == (label, decision_at)
            # La etiqueta de cada episodio había madurado antes de confirmarse el banco.
            assert decision_at < confirmed_at
        if receipt["status"] == "one_swap_local_restricted" or len(clients) > 8:
            restricted += 1
            assert len(receipt["fixed_ids"]) == 8 - 2
    assert restricted > 0


def test_consolidation_only_acts_when_the_bank_overflows(
    shared, tmp_path, native, learning_doubles
):
    _, streams = shared
    results = {}
    for name, capacity, policy in (
        ("reservoir", 1024, "reservoir"),
        ("anchored", 1024, "anchored"),
        ("small-reservoir", 8, "reservoir"),
        ("small-anchored", 8, "anchored"),
    ):
        engine = consolidating(streams, tmp_path / name, native, capacity=capacity, policy=policy)
        engine.run()
        results[name] = engine
    wide, anchored = results["reservoir"], results["anchored"]
    assert entries(wide.audit, "prediction") == entries(anchored.audit, "prediction")
    for left, right in zip(gradients(wide), gradients(anchored), strict=True):
        for key, value in left.items():
            if value is None:
                assert right[key] is None
            else:
                torch.testing.assert_close(right[key], value, rtol=0, atol=0)
    assert entries(results["small-reservoir"].audit, "prediction") != entries(
        results["small-anchored"].audit, "prediction"
    )
