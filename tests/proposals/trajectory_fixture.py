"""Corpus y parámetros técnicos compartidos, con objetivos manuales y sin ajuste."""

import copy
import importlib
import json
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import sha256
from mars_titan.data.temporal import MarketClock
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.memory.financial_observations import (
    FinancialObservationSource,
    prepare_observation_index,
    run_observation_source,
)
from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.memory.native_mechanism_collection import MechanismCollectionConfig
from mars_titan.memory.regional_work import RESOURCES
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.models.titans.bridge_recurrence import BridgeRecurrenceConfig
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.models.titans.regional_execution import RegionalExecutionConfig
from mars_titan.models.titans.regional_fusion import RegionalFusionConfig
from mars_titan.proposals.contracts import ProposalRegistry, RegisteredConfiguration
from mars_titan.proposals.error_proposer import propose_from_mature_errors
from mars_titan.proposals.error_rule import ErrorProposalPlan
from mars_titan.proposals.error_source import prepare_error_review
from mars_titan.proposals.ledger import ProposalLedger
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.prefix_eligibility import PrefixTargetVerifier

FLOWS = tuple(f"US/T{index:04d}" for index in range(4))
POSITIONS = (125, 126, 127, 130, 131, 132, 133, 136, 137)
MANUAL_TARGETS = {126: 0.125, 127: -0.25, 131: 0.125, 132: -0.25, 133: 0.375, 137: 0.5}


def regional_limits():
    return dict.fromkeys(sorted(RESOURCES), 1_000_000)


@contextmanager
def manual_corpus(root):
    """Reutilizar precios/calendario de fixture, no estimar etiquetas ni gráficos reales."""
    with pytest.MonkeyPatch.context() as imports:
        imports.syspath_prepend(str(Path(__file__).resolve().parents[1] / "memory"))
        controls = importlib.import_module("test_financial_session_controls")
        with contextmanager(controls.no_target_estimation.__wrapped__)():
            controls.flow_source(root, flows=FLOWS)
            parent_path = root / "materialized.json"
            parent = json.loads(parent_path.read_text())
            supervised = root / "supervised/manifest.json"
            metadata = json.loads(supervised.read_text())
            clock = MarketClock("US", "2021-01-01", "2023-12-31")
            for asset in metadata["assets"]:
                flow = f"{asset['market']}/{asset['symbol']}"
                sample_path = Path(metadata["roots"]["samples"]) / flow / "samples.parquet"
                original = pq.read_table(sample_path)
                prototype = original.to_pylist()[0]
                samples = []
                for position in POSITIONS:
                    row = copy.deepcopy(prototype)
                    at = clock.decisions[position]
                    row.update(prediction_at=at, price_end_index=position)
                    row["input_availability"] = {
                        name: at if row["presence"][index] else None
                        for index, name in enumerate(
                            ("prices", "news", "charts", "fundamentals", "macro")
                        )
                    }
                    samples.append(row)
                pq.write_table(
                    pa.Table.from_pylist(samples, schema=original.schema),
                    sample_path,
                    row_group_size=2,
                )
                sample_receipt = sample_path.parent / "manifest.json"
                receipt = json.loads(sample_receipt.read_text())
                receipt["samples_sha256"] = sha256(sample_path)
                sample_receipt.write_text(json.dumps(receipt))
                labels_path = Path(metadata["roots"]["labels"]) / flow / "labels.parquet"
                schema = pq.read_schema(labels_path)
                labels = [
                    dict(
                        sample_row=index,
                        prediction_at=clock.decisions[position],
                        target_available_at=clock.decisions[position + 1],
                        target=MANUAL_TARGETS.get(position, 0.0),
                        partition="train",
                        reason="accepted",
                        cohort_id=metadata["cohort_id"],
                    )
                    for index, position in enumerate(POSITIONS)
                ]
                pq.write_table(pa.Table.from_pylist(labels, schema=schema), labels_path)
                asset.update(
                    samples=len(POSITIONS),
                    samples_sha256=sha256(sample_path),
                    labels_sha256=sha256(labels_path),
                    counts=dict(train=len(POSITIONS), validation=0),
                )
            for document in (parent, metadata):
                document["samples"] = len(FLOWS) * len(POSITIONS)
                for row in document["coverage"]:
                    if row["state"] == "encoded":
                        row["samples"] = len(POSITIONS)
            parent_path.write_text(json.dumps(parent))
            metadata["counts"] = dict(train=len(FLOWS) * len(POSITIONS), validation=0)
            metadata["configuration"].update(
                backend="manual_trajectory_fixture_no_estimation",
                source_manifest_sha256=sha256(parent_path),
            )
            metadata["technical_fixture"] = True
            supervised.write_text(json.dumps(metadata))
            dataset = CorpusDataset(supervised, input_policy=HISTORICAL_MASKED)
            moment = importlib.import_module("test_financial_session").moment
            phases = dict(
                g=FinancialPhase("train", moment(125), moment(126), moment(129), moment(129)),
                s=FinancialPhase("train", moment(130), moment(131), moment(135), moment(135)),
                e=FinancialPhase("train", moment(136), moment(137), moment(139), moment(139)),
            )
            sources = {
                name: FinancialObservationSource(
                    dataset,
                    prepare_observation_index(dataset, root / "indices" / name, phase=phase),
                )
                for name, phase in phases.items()
            }
            prefixes = PrefixTargetVerifier(dataset, source_manifest=parent_path)
            yield dict(dataset=dataset, sources=sources, prefixes=prefixes, moment=moment)


@contextmanager
def trajectory_case(root, native):
    with ExitStack() as resources:
        corpus = resources.enter_context(manual_corpus(root / "corpus"))
        sources, moment = corpus["sources"], corpus["moment"]
        specification = sources["g"].specification()
        codec = FrozenEpisodeCodec(specification)
        program = RegionalExecutionConfig(program="summary_at_end_v1")
        configuration = FinancialConfig(
            specification,
            variant="mac_online",
            hidden_size=32,
            seed=42,
            max_batch=8,
            head="quantile_head_v1",
            regional_fusion=RegionalFusionConfig(topology="self", routing="uniform"),
            regional_execution=program,
        )
        bridge = EvidenceBridgeConfig(
            codec.fingerprint(),
            hidden_size=32,
            max_batch=8,
            cycles=0,
            recurrence=BridgeRecurrenceConfig(steps=1, max_node_updates_per_row=456),
        )
        reader_configuration = EpisodicReadoutConfig(
            codec.fingerprint(), hidden_size=32, max_batch=8, evidence_bridge=bridge
        )
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(42)
            model = FinancialPredictor(configuration, device="cuda:0", dtype=torch.float64)
            reader = EpisodicReadout(reader_configuration, device="cuda:0", dtype=torch.float64)
        model.eval().requires_grad_(False)
        reader.eval().requires_grad_(False)
        catalog = MechanismCatalog(
            reader_contract=bridge.identity()["kind"], regional_execution=program
        )
        parent = FrozenMechanismConsumer(
            model, readout=reader, action=MechanismAction(), catalog=catalog
        )
        baseline = RegisteredConfiguration(
            configuration, reader_configuration, parent.action, catalog, "torch.float64"
        )
        interleaved = RegionalExecutionConfig(program="summary_interleaved_v1")
        changed_program = replace(
            baseline,
            predictor=replace(configuration, regional_execution=interleaved),
            catalog=replace(catalog, regional_execution=interleaved),
        )
        changed_j = replace(
            baseline,
            readout=replace(
                reader_configuration,
                evidence_bridge=replace(
                    bridge, recurrence=BridgeRecurrenceConfig(2, max_node_updates_per_row=456)
                ),
            ),
        )
        registry = ProposalRegistry(
            (baseline, changed_program, changed_j), baseline.configuration_id
        )
        retention = RetentionConfig(capacity=8, frontier=2, new_candidates=4)
        session = resources.enter_context(
            FinancialSession(
                root / "generation",
                native=native,
                consumer=parent,
                codec=codec,
                prefixes=corpus["prefixes"],
                retention=retention,
                phase=sources["g"].phase,
                world="trajectory_fixture",
                fold="G",
                admission="m1",
                block_rows=2,
                mechanism_collection=MechanismCollectionConfig(),
                regional_work_limits=regional_limits(),
            )
        )
        run_observation_source(sources["g"], session)
        plan = ErrorProposalPlan(
            source_contract_id=session.contract_id,
            parent_consumer_id=parent.model_id,
            source_partition="train",
            declared_before=moment(124),
            parent_selected_before=moment(124),
            generation_start=moment(126),
            generation_end=moment(130),
            selection_start=moment(131),
            selection_end=moment(136),
            evaluation_start=moment(137),
            evaluation_end=moment(140),
            flow_ids=FLOWS,
            market_weights=(("US", 1.0),),
            min_sessions_per_market=2,
            min_excess_mae=0.0,
            candidate_ids=(changed_program.configuration_id, changed_j.configuration_id),
            control="enumerate",
        )
        evidence = prepare_error_review(session, registry=registry, plan=plan)
        limits = dict(
            max_candidates=3,
            max_attempts=2,
            max_review_rows=8192,
            max_review_bytes=1024**2,
            max_state_bytes=1024**2,
        )
        ledger = ProposalLedger(
            root / "proposal-ledger",
            registry=registry,
            review=evidence.review,
            limits=limits,
            generation_plan_id=plan.plan_id,
        )
        proposals = propose_from_mature_errors(ledger, plan=plan, evidence=evidence)
        assert proposals["proposed"] == list(plan.candidate_ids)
        yield dict(
            **corpus,
            native=native,
            parent=parent,
            codec=codec,
            registry=registry,
            retention=retention,
            generation=session,
            generation_head=session.snapshot(),
            plan=plan,
            evidence=evidence,
            ledger=ledger,
            limits=limits,
            proposals=proposals,
        )
