"""Componer los grafos anteriores y posteriores a MAC sobre los fixtures nativos."""

import hashlib
import importlib
import importlib.util
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_bridge_episodic_readout import configuration as base_readout_config
from test_financial_mechanisms import assert_same
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native
from test_regional_program_session import (
    DEVICES,
    RESOURCES,
    attempt,
    limits,
    observed_calls,
    predictions,
    session,
    work_index,
)

from mars_titan.memory.mechanism_collection_record import CollectedMechanisms
from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.models.titans.regional_execution import RegionalExecutionConfig
from mars_titan.models.titans.regional_fusion import RegionalFusionConfig
from mars_titan.models.titans.regional_hierarchy import RegionalHierarchyConfig

POST_CONTRACT = "shared_evidence_bridge_hierarchy_v1"
PRE_FAMILIES = ("program", "hierarchy")


def pre_config(family):
    if family == "program":
        return RegionalExecutionConfig(program="summary_interleaved_v1")
    return RegionalHierarchyConfig()


def predictor_config(spec, family):
    return FinancialConfig(
        spec,
        variant="mac_online",
        hidden_size=32,
        seed=42,
        max_batch=8,
        head=QUANTILE_HEAD,
        session_context="market_memory_v1",
        regional_fusion=RegionalFusionConfig(
            topology="self", routing="uniform", cycles=2, local_steps=2, summary=True
        ),
        regional_execution=pre_config(family),
    )


def post_config(*, codec_id=None, budget=988):
    name = "mars_titan.models.titans.bridge_hierarchy"
    assert importlib.util.find_spec(name) is not None, "Falta integrar el puente de #720"
    hierarchy = importlib.import_module(name).BridgeHierarchyConfig(
        topology="tree", local_steps=2, feedback=True, max_node_updates_per_row=budget
    )
    base = base_readout_config(refinements=1, episode_selection="first_read")
    codec_id = base.codec_id if codec_id is None else codec_id
    return replace(
        base,
        codec_id=codec_id,
        max_batch=2,
        evidence_bridge=replace(
            base.evidence_bridge,
            codec_id=codec_id,
            max_batch=2,
            regions=("view_1", "view_2", "view_3", "view_4"),
            topology="self",
            cycles=0,
            hierarchy=hierarchy,
        ),
    )


def composed_engines(source, family, device, *, budget=988):
    config = predictor_config(source["spec"], family)
    readout_config = post_config(codec_id=source["codec"].fingerprint(), budget=budget)
    catalog = MechanismCatalog(
        reader_contract=POST_CONTRACT, regional_execution=config.regional_execution
    )
    model = FinancialPredictor(config, device=device, dtype=torch.float64)
    weights = model.state_dict()
    weights["session_context.output"] = torch.eye(32, device=device, dtype=torch.float64) * 0.2
    model.load_state_dict(weights)
    model.eval().requires_grad_(False)
    reader = EpisodicReadout(readout_config, device=device, dtype=torch.float64)
    reader.eval().requires_grad_(False)
    consumer = FrozenMechanismConsumer(
        model,
        readout=reader,
        catalog=catalog,
        action=MechanismAction("parent", "k4cc", "none"),
    )
    return None, source["spec"], source["codec"], consumer, source["prefixes"]


def check_emission(event, family):
    payload = event["payload"]
    work = payload["work"]
    assert work["mac_online_observations"] == 4
    assert work["regional_local_node_evaluations"] == 80
    assert work["regional_summary_evaluations"] == (8 if family == "program" else 40)
    assert work["bridge_value_projections"] == 16
    assert work["bridge_node_updates"] == 416
    assert work["bridge_group_updates"] == 128
    assert work["bridge_root_updates"] == 32
    program = "summary_interleaved_v1" if family == "program" else "hierarchical_group_root_v1"
    for row in payload["rows"]:
        assert row["regional_execution"]["program"] == program
        trace = row["regional_readout"]
        assert trace["kind"] == "hierarchical_mechanism_trace_v1"
        assert len(trace["reads"]) == 4
        assert trace["reads"][0]["group_values"]
        assert trace["reads"][0]["root_values"]


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize("family", PRE_FAMILIES)
@pytest.mark.parametrize("device", DEVICES)
def test_composed_native_recovery_preserves_weights_states_and_the_next_emission(
    native, four_flow_source, tmp_path, family, device
):
    data = composed_engines(four_flow_source, family, device)
    consumer = data[3]
    assert consumer.max_batch == 2
    weights = deepcopy(consumer.predictor.state_dict()), deepcopy(consumer.readout.state_dict())
    frozen = consumer.identity()
    assert frozen["schema_version"] == 6
    assert frozen["recipe"] == "frozen_pre_mac_hierarchical_bridge_cohort_v1"
    assert frozen["catalog"]["reader_contract"] == POST_CONTRACT
    assert frozen["catalog"]["regional_execution"] == pre_config(family).identity()
    assert canonical(frozen["readout"]) == canonical(consumer.readout.get_extra_state())
    assert frozen["regional_execution"] == consumer.predictor.regional_execution.get_extra_state()
    sources = ["regional_execution", "bridge_hierarchy"]
    if family == "hierarchy":
        sources.append("regional_hierarchy")
    for name in sources:
        module = importlib.import_module(f"mars_titan.models.titans.{name}")
        assert (
            frozen["implementation"][module.__name__]
            == hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        )
    consumer.verify()
    with pytest.raises(ValueError):
        FrozenMechanismConsumer(
            consumer.predictor,
            readout=consumer.readout,
            catalog=MechanismCatalog(regional_execution=pre_config(family)),
        )
    options = {"regional_work_limits": limits()}
    with session(native, tmp_path / "reference", data, **options) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        first = run.step(four_flow_source["batches"][126], [])
        labels = [native.Feedback(row.id, 0, moment(127), 0.125) for row in first.predictions]
        run.step([], labels, kind="settlement", cutoff=moment(127))
        expected = predictions(run.step(four_flow_source["batches"][128], []))
        expected_fast, expected_event = run.fast_state(), run.mechanism_event(4)
        check_emission(expected_event, family)
    with session(native, tmp_path / "cut", data, **options) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        run.step(four_flow_source["batches"][126], [])

        def cut(boundary):
            if boundary == native.Boundary.committed:
                raise RuntimeError("Corte después de confirmar el feedback compuesto")

        with pytest.raises(RuntimeError, match="feedback compuesto"):
            run.step([], labels, kind="settlement", cutoff=moment(127), fault=cut)
    with observed_calls(consumer) as reopened:
        restored = session(native, tmp_path / "cut", data, resume=True, **options)
    with restored as run:
        assert all(count == 0 for count in reopened.values())
        actual = predictions(run.step(four_flow_source["batches"][128], []))
        assert actual == expected
        assert_same(run.fast_state(), expected_fast)
        assert_same(run.mechanism_event(4), expected_event)
        assert run.snapshot()["applied"] == 4
    assert_same(consumer.predictor.state_dict(), weights[0])
    assert_same(consumer.readout.state_dict(), weights[1])


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize("family", PRE_FAMILIES)
@pytest.mark.parametrize("budget", (987, 988))
def test_collection_checks_post_mac_budget_without_resetting_pre_mac_charges(
    native, four_flow_source, tmp_path, family, budget
):
    data = composed_engines(four_flow_source, family, "cpu", budget=budget)
    consumer = data[3]
    output = tmp_path / "composed_collection"
    options = {"regional_work_limits": limits(), "collection": True}
    rejected_head = rejected_fast = rejected_attempt = None
    with session(native, output, data, **options) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        issued = run.step(four_flow_source["batches"][126], [])
        run.step(
            [],
            [native.Feedback(row.id, 0, moment(127), 0.125) for row in issued.predictions],
            kind="settlement",
            cutoff=moment(127),
        )
        if budget == 987:
            rejected_head = (output / "latest.json").read_bytes()
            rejected_fast = run.fast_state()
            with observed_calls(consumer) as calls:
                with pytest.raises(ValueError, match="presupuesto|cuota|actualizaciones"):
                    run.step(four_flow_source["batches"][128], [])
            assert calls["predictor_prepare"] == calls["consumer_prepare"] == calls["mac"] == 0
            assert (output / "latest.json").read_bytes() == rejected_head
        else:
            run.step(four_flow_source["batches"][128], [])
            event = run.mechanism_event(4)
            check_emission(event, family)
            packet = CollectedMechanisms.restore(run.mechanism_collection(4)["collection"])
            body = packet.payload()
            assert body["catalog"] == consumer.catalog.identity()
            assert body["regional_execution"]["schema_version"] == (1 if family == "program" else 2)
            assert body["work"]["core_observations"] == 8
            assert body["work"]["reader_value_projections"] == 72
            assert consumer.readout.recurrence_work(88)["bridge_node_updates"] == 2288
        current = attempt(output)
        assert set(current["units"]) == set(RESOURCES)
        assert current["units"]["core_observations"] == 12
        assert current["units"]["local_node_evaluations"] == 240
        assert current["units"]["summary_evaluations"] == (24 if family == "program" else 120)
        if budget == 987:
            rejected_attempt = deepcopy(current)
    if budget == 987:
        with observed_calls(consumer) as reopened:
            restored = session(native, output, data, resume=True, **options)
        with restored as run:
            assert all(count == 0 for count in reopened.values())
            assert (output / "latest.json").read_bytes() == rejected_head
            assert_same(run.fast_state(), rejected_fast)
            current = attempt(output)
            assert current["reservation_id"] == rejected_attempt["reservation_id"]
            assert current["status"] == "abandoned"
            assert work_index(output)["cohorts"][-1]["charged_units"] == rejected_attempt["units"]
