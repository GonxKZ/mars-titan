"""Emisión y recuperación nativas con vistas regionales y una evidencia común."""

import pytest
import torch
from test_financial_mechanisms import assert_same
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native

from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def engine(source, device):
    model = (
        FinancialPredictor(
            FinancialConfig(source["spec"], hidden_size=32, variant="mac_online"),
            device=device,
            dtype=torch.float64,
        )
        .eval()
        .requires_grad_(False)
    )
    common = dict(codec_id=source["codec"].fingerprint(), hidden_size=32, neighbors=4)
    reader = (
        EpisodicReadout(
            EpisodicReadoutConfig(
                **common, refinements=2, evidence_bridge=EvidenceBridgeConfig(**common)
            ),
            device=device,
            dtype=torch.float64,
        )
        .eval()
        .requires_grad_(False)
    )
    return FrozenFinancialConsumer(model, readout=reader)


def session(native, path, source, consumer, *, resume=False, block_rows=2):
    return FinancialSession(
        path,
        native=native,
        consumer=consumer,
        codec=source["codec"],
        prefixes=source["prefixes"],
        retention=RetentionConfig(capacity=4, frontier=1, new_candidates=1),
        phase=FinancialPhase("validation", moment(125), moment(126), moment(130), moment(131)),
        admission="m1",
        block_rows=block_rows,
        world="bridge_fixture",
        fold="0",
        resume=resume,
    )


def control(run):
    return run._read(run._bundle(run.snapshot()["state"])["control"], "control")


@pytest.mark.parametrize("device", DEVICES)
def test_native_recovery_preserves_emission_trace_and_single_maturation(
    native, four_flow_source, tmp_path, device
):
    source, consumer = four_flow_source, engine(four_flow_source, device)
    with session(native, tmp_path / "reference", source, consumer) as run:
        run.step(source["batches"][125], [], kind="warmup")
        first = run.step(source["batches"][126], [])
        assert control(run)["regional_reads"]["snapshot_sha256"]
        labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in first.predictions]
        run.step([], labels, kind="settlement", cutoff=moment(127))
        expected = run.step(source["batches"][128], [])
        expected_fast, expected_control = run.fast_state(), control(run)
        assert run.snapshot()["applied"] == 4
        assert expected_control["mac_updates"] == 4
        assert expected_control["bridge_candidate_queries"] == 8
        assert expected_control["bridge_projected_episode_rows"] == 32
        trace = expected_control["regional_reads"]
        assert trace["rows"] == 4 and len(trace["blocks"]) == 2
        for block in trace["blocks"]:
            assert len(block["reads"]) == 2
            for read in block["reads"]:
                assert read["evidence_ids"].shape == (2, 4)
                assert read["regional_weights"].shape == (2, 3, 4)
                assert set(read) == {
                    "evidence_ids",
                    "regional_weights",
                    "regional_values",
                    "read_values",
                    "presence",
                }
    with session(native, tmp_path / "cut", source, consumer) as run:
        run.step(source["batches"][125], [], kind="warmup")
        run.step(source["batches"][126], [])

        def cut(boundary):
            if boundary == native.Boundary.committed:
                raise RuntimeError("Corte tras confirmar feedback")

        with pytest.raises(RuntimeError, match="Corte tras"):
            run.step([], labels, kind="settlement", cutoff=moment(127), fault=cut)
    with session(native, tmp_path / "cut", source, consumer, resume=True) as run:
        actual = run.step(source["batches"][128], [])
        assert [(p.asset, p.value) for p in actual.predictions] == [
            (p.asset, p.value) for p in expected.predictions
        ]
        assert_same(run.fast_state(), expected_fast)
        assert_same(control(run), expected_control)
        assert run.snapshot()["applied"] == 4
