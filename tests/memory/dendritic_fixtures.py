"""Fixtures compartidos del recorrido N2, con supervisión y escalas manuales."""

import numpy as np
import torch

from mars_titan.data.price_windows import price_window_contract
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.memory.write_policy import CompositeScoreConfig, MatureErrorConfig
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import FinancialInputSpec, validated_cpu_batch
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from tests.memory.test_financial_session import moment
from tests.memory.test_financial_session_controls import FLOWS, flow_source
from tests.memory.test_mars_titan_session_parity import SCALERS


def build_source(root, *, flows=FLOWS):
    base = flow_source(root, flows=flows)
    old = base["spec"]
    representation = old.representation
    representation["price_window"] = price_window_contract(
        {"US": dict(start="2021-01-01", end="2023-12-31", decisions_sha256="f" * 64)},
        {"US": []},
    )
    spec = FinancialInputSpec(
        source_sha256=old.source_sha256,
        view_sha256=old.view_sha256,
        representation=representation,
        dimensions=old.dimensions | dict(prices=6),
        input_policy=old.input_policy,
    )
    batches = {}
    for at, blocks in base["batches"].items():
        batches[at] = []
        for block in blocks:
            inputs = {name: np.array(value, copy=True) for name, value in block.inputs.items()}
            inputs["prices"] = np.concatenate(
                [inputs["prices"], np.ones((1, 64, 1), np.float32)], axis=-1
            )
            batches[at].append(
                validated_cpu_batch(
                    dict(
                        inputs=inputs,
                        presence=block.presence,
                        sample_ids=block.sample_ids,
                        prediction_at=np.array(block.prediction_at, dtype="datetime64[us]"),
                        input_available_at=np.array(
                            block.input_available_at, dtype="datetime64[us]"
                        ),
                    ),
                    spec,
                )
            )
    return dict(base, spec=spec, codec=FrozenEpisodeCodec(spec), batches=batches)


def engine(
    source,
    mode,
    *,
    admission="m1",
    gain=1.0,
    device="cpu",
    dtype=torch.float64,
    min_assets=2,
    head=None,
):
    config = FinancialConfig(
        source["spec"],
        variant="mac_online",
        hidden_size=32,
        active_dendrites=None
        if mode is None
        else dict(mode=mode, segments=4, min_assets=min_assets, recent_returns=21, min_returns=42),
        **({"head": head} if head is not None else {}),
    )
    model = FinancialPredictor(config, dtype=dtype, device=device)
    if mode is not None:
        with torch.no_grad():
            model.active_dendrites.gain.fill_(gain)
    model._seal_parameters()
    readout = (
        EpisodicReadout(
            EpisodicReadoutConfig(
                source["codec"].fingerprint(),
                hidden_size=32,
                mode="no_bank" if admission == "m0" else "bank",
            ),
            dtype=dtype,
        )
        .to(device=device)
        .eval()
        .requires_grad_(False)
    )
    return FrozenFinancialConsumer(model.eval().requires_grad_(False), readout=readout)


def run(
    native,
    path,
    source,
    consumer,
    *,
    block_rows=1,
    resume=False,
    decision_start=125,
    admission="m1",
    capacity=8,
    retention=None,
    mechanism_collection=None,
):
    retention = (
        retention
        if retention is not None
        else (
            MatureErrorConfig(capacity=capacity)
            if admission == "m2"
            else CompositeScoreConfig(SCALERS, capacity=capacity)
            if admission == "m3"
            else RetentionConfig(capacity=capacity, frontier=2, new_candidates=4)
        )
    )
    return FinancialSession(
        path,
        native=native,
        consumer=consumer,
        codec=source["codec"],
        prefixes=source["prefixes"],
        retention=retention,
        phase=FinancialPhase(
            "validation", moment(125), moment(decision_start), moment(200), moment(201)
        ),
        admission=admission,
        world="dendritic_fixture",
        fold="0",
        block_rows=block_rows,
        resume=resume,
        mechanism_collection=mechanism_collection,
    )
