"""Exclusiones demostradas por precios del prefijo, sin usar la etiqueta futura."""

import importlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import sha256
from mars_titan.data.temporal import MarketClock
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.training.test_historical_corpus_inputs import supervised


def verifier(path, **kwargs):
    api = importlib.import_module("mars_titan.training.prefix_eligibility")
    return api.PrefixTargetVerifier(
        CorpusDataset(path, input_policy=HISTORICAL_MASKED),
        source_manifest=path.parent.parent / "materialized.json",
        **kwargs,
    )


def moment(index):
    return int(
        pd.Timestamp(MarketClock("US", "2021-01-01", "2023-12-31").decisions[index]).value // 1000
    )


@pytest.mark.parametrize(
    "index,count,reason",
    [
        (63, 64, "insufficient_pairs"),
        (124, 125, "insufficient_pairs"),
        (125, 126, None),
        (300, 252, None),
    ],
)
def test_exact_past_window_and_minimum(tmp_path, index, count, reason):
    proof = verifier(supervised(tmp_path)).evidence("US/AAA", moment(index))
    assert proof.history_pairs == count
    assert proof.reason == reason


@pytest.mark.parametrize(
    "field,value",
    [
        ("flow_id", "CN/AAA"),
        ("history_pairs", 0),
        ("decision_at", 1),
        ("policy_id", "f" * 64),
        ("records_sha256", "f" * 64),
    ],
)
def test_forged_evidence_is_rejected(tmp_path, field, value):
    source = verifier(supervised(tmp_path))
    proof = source.evidence("US/AAA", moment(63))
    with pytest.raises(ValueError):
        source.verify(replace(proof, **{field: value}), flow_id="US/AAA", decision_at=moment(63))


def test_mutated_source_is_rejected_after_cache_hit(tmp_path):
    source = verifier(supervised(tmp_path))
    proof = source.evidence("US/AAA", moment(63))
    path = tmp_path / "prepared/US/AAA/prices.parquet"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        source.verify(proof, flow_id="US/AAA", decision_at=moment(63))


def rewrite_factor(path, transform):
    meta = json.loads(path.read_text())
    factor = meta["market_factors"]["US"]
    filename = factor["prices_path"]
    frame = pq.read_table(filename).to_pandas()
    transform(frame)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), filename)
    factor["prices_sha256"] = sha256(Path(filename))
    parent_path = path.parent.parent / "materialized.json"
    parent = json.loads(parent_path.read_text())
    parent["market_factors"]["US"] = dict(factor)
    parent_path.write_text(json.dumps(parent))
    meta["configuration"]["source_manifest_sha256"] = sha256(parent_path)
    path.write_text(json.dumps(meta))


def test_factor_cannot_be_substituted_without_a_new_parent_or_revision(tmp_path):
    path = supervised(tmp_path)
    metadata = json.loads(path.read_text())
    factor = metadata["market_factors"]["US"]
    factor["prices_path"] = str(tmp_path / "prepared/US/AAA/prices.parquet")
    factor["prices_sha256"] = sha256(Path(factor["prices_path"]))
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        verifier(path)


def test_future_suffix_cannot_change_prefix_arithmetic_or_records(tmp_path):
    path = supervised(tmp_path)
    first = verifier(path).evidence("US/AAA", moment(125))
    rewrite_factor(path, lambda f: f.loc.__setitem__((f.index > 125, "close"), 100.0))
    second_source = verifier(path)
    second = second_source.evidence("US/AAA", moment(125))
    assert (first.history_pairs, first.market_variance, first.reason, first.records_sha256) == (
        second.history_pairs,
        second.market_variance,
        second.reason,
        second.records_sha256,
    )
    assert first.policy_id != second.policy_id
    with pytest.raises(ValueError):
        second_source.verify(first, flow_id="US/AAA", decision_at=moment(125))


def test_zero_variance_uses_original_epsilon_with_at_least_126_pairs(tmp_path):
    path = supervised(tmp_path)
    rewrite_factor(path, lambda f: f.__setitem__("close", f["open"]))
    source = verifier(path)
    assert source.evidence("US/AAA", moment(124)).reason == "insufficient_pairs"
    proof = source.evidence("US/AAA", moment(125))
    assert proof.reason == "zero_market_variance" and proof.market_variance == 0
    assert proof.market_variance <= np.finfo(float).eps


def test_reserved_or_non_session_dates_are_rejected(tmp_path):
    source = verifier(supervised(tmp_path))
    for stamp in (1_704_067_200_000_000, moment(63) + 1, True):
        with pytest.raises(ValueError):
            source.evidence("US/AAA", stamp)
