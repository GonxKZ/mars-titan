"""Contextos S2 desde las filas y codecs reales del recorrido, con fixtures sin etiquetas."""

import csv
import hashlib
import importlib
import io
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import test_native_episode_backend as native_fixture
import torch
from test_frozen_financial import frozen_backend as frozen_backend

from mars_titan.memory.candidate_bank import CandidateBankConfig
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.memory.episodic_session import EpisodicSession
from mars_titan.memory.financial_consumers import CandidateBinding, TitansBinding
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.models.candidate.episode_codec import FrozenCandidateCodec
from mars_titan.models.candidate.frozen_consumer import FrozenCandidateConsumer
from mars_titan.models.candidate.input_adapter import CandidateInputAdapter
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import validated_cpu_batch
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.holographic_network import CATALOG
from tests.models.titans.test_holographic_network import (
    MACRO_FAMILIES,
    configs,
    raw_batch,
    specification,
)

native = native_fixture.native
pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
CATALOG_SHA = hashlib.sha256(CATALOG.read_bytes()).hexdigest()


def api():
    return importlib.import_module("mars_titan.memory.failure_producer")


@pytest.fixture(params=["titans", "gru"])
def source(request, native):
    return make_source(native, request.param)


def make_source(native, kind="titans", network_config=None):
    spec = specification()
    adapter = None
    if kind == "titans":
        codec = FrozenEpisodeCodec(spec)
        model = (
            FinancialPredictor(
                FinancialConfig(
                    spec,
                    variant="mac_online",
                    hidden_size=32,
                    layers=1,
                    holographic_network=network_config,
                ),
                dtype=torch.float64,
            )
            .eval()
            .requires_grad_(False)
        )
        binding = TitansBinding(
            FrozenFinancialConsumer(model),
            codec,
            RetentionConfig(capacity=4, frontier=2, new_candidates=4),
            "m0",
        )
    else:
        adapter = CandidateInputAdapter(spec, dtype=torch.float64)
        adapter.model.eval()
        for parameter in adapter.model.named_parameters().values():
            parameter.requires_grad_(False)
        codec = FrozenCandidateCodec(adapter)
        binding = CandidateBinding(
            FrozenCandidateConsumer(adapter), codec, CandidateBankConfig(capacity=4), "m0"
        )
    batch = validated_cpu_batch(raw_batch(), spec)
    rows = EpisodicSession._rows(SimpleNamespace(codec=codec, native=native), [batch])
    return dict(spec=spec, codec=codec, rows=rows, adapter=adapter, binding=binding)


def producer(source, **changes):
    return api().FailureContextProducer(
        source["binding"], macro_catalog_sha256=CATALOG_SHA, **changes
    )


def test_source_contracts_are_checked_before_capturing_rows(native):
    source = make_source(native)
    spec = source["spec"]
    result = producer(source)
    assert result.layout.input_id == spec.fingerprint()
    other = type(spec)(
        source_sha256=spec.source_sha256,
        view_sha256="d" * 64,
        representation=spec.representation,
        dimensions=spec.dimensions,
        input_policy=spec.input_policy,
    )
    mismatched = TitansBinding(
        source["binding"].consumer,
        FrozenEpisodeCodec(other),
        RetentionConfig(capacity=4, frontier=2, new_candidates=4),
        "m0",
    )
    with pytest.raises(ValueError):
        api().FailureContextProducer(mismatched, macro_catalog_sha256=CATALOG_SHA)


def test_both_real_codecs_capture_raw_macro_masks_clocks_and_their_own_key_width(source):
    result = producer(source)
    assert result.layout.codec_id == source["codec"].fingerprint()
    assert result.layout.key_width == source["codec"].identity()["key_width"]
    assert result.layout.family_names == MACRO_FAMILIES
    for row in source["rows"]:
        record = result.capture(row, action="fixture_fixed_point/no_tool")
        assert record.action == "fixture_fixed_point/no_tool"
        assert record.input_sha256 == row.input_sha256
        assert record.prediction_at == row.prediction_at
        np.testing.assert_array_equal(record.macro, row.inputs["macro"])
        np.testing.assert_array_equal(record.presence, row.presence)
        np.testing.assert_array_equal(record.key_inputs, row.key_inputs)
        assert record.area_activity.size == 0
        assert np.all(record.macro_available_at == np.iinfo(np.int64).min)


@pytest.mark.parametrize("config", configs(), ids=lambda c: f"{c.kind}/{c.areas}")
def test_area_activity_matches_the_actual_h1_h2_and_h3_structural_buffers(native, config):
    source = make_source(native, network_config=config)
    net = source["binding"].consumer.predictor.holographic_network
    result = producer(source)
    for row in source["rows"]:
        record = result.capture(row, action="fixture_fixed_point/no_tool")
        present = torch.tensor(row.presence, dtype=torch.float64) @ net.block_modality.T > 0
        parts = []
        if config.catalog_areas:
            parts.append(present[net.area_owner])
        if config.hidden_groups:
            parts.append(present.any().expand(config.hidden_groups))
        np.testing.assert_array_equal(record.area_activity, torch.cat(parts).numpy())
        assert len(result.layout.area_names) == config.nodes


def test_a_valid_zero_key_is_allowed_only_by_the_codec_that_declares_it(source):
    result = producer(source)
    row = source["rows"][0]
    zero = replace(row, key_inputs=np.zeros(result.layout.key_width))
    if type(source["codec"]) is FrozenCandidateCodec:
        record = result.capture(zero, action="fixture_fixed_point/no_tool")
        assert not record.normalized_key().any()
    else:
        with pytest.raises(ValueError):
            result.capture(zero, action="fixture_fixed_point/no_tool")


def test_changed_input_bytes_are_detected_even_when_they_still_have_a_valid_shape(source):
    result = producer(source)
    row = source["rows"][0]
    inputs = dict(row.inputs)
    inputs["prices"] = inputs["prices"].copy()
    inputs["prices"][0, 0] += 1
    with pytest.raises(ValueError, match="huella"):
        result.capture(replace(row, inputs=inputs), action="fixture_fixed_point/no_tool")


def test_a_catalogue_cannot_change_between_versioned_captures(source, tmp_path):
    path = tmp_path / "catalog.csv"
    path.write_bytes(CATALOG.read_bytes())
    assert producer(source, catalog=path).layout.macro_catalog_id == CATALOG_SHA
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="catálogo"):
        producer(source, catalog=path)


def test_a_forged_area_owner_buffer_is_rejected(native):
    source = make_source(native, network_config=configs()[0])
    net = source["binding"].consumer.predictor.holographic_network
    net.area_owner.fill_(0)
    with pytest.raises(ValueError, match="estructura"):
        producer(source)


def test_the_producer_does_not_accept_future_labels(source):
    result = producer(source)
    with pytest.raises(TypeError):
        result.capture(source["rows"][0], action="fixture_fixed_point/no_tool", target=0.1)


def test_changed_shared_weights_cannot_be_described_as_the_original_source(source):
    consumer = source["binding"].consumer
    if type(source["codec"]) is FrozenCandidateCodec:
        parameter = next(iter(consumer.model.named_parameters().values()))
    else:
        parameter = next(consumer.predictor.parameters())
    parameter.data.add_(0.01)
    with pytest.raises(ValueError):
        producer(source)


@pytest.mark.parametrize("edit", ["missing", "target", "dtype", "shape", "presence"])
def test_malformed_row_contracts_are_rejected_before_capture(source, edit):
    row = source["rows"][0]
    inputs = dict(row.inputs)
    if edit == "missing":
        inputs.pop("news")
    elif edit == "target":
        inputs["target"] = np.array([0.1], np.float32)
    elif edit == "dtype":
        inputs["news"] = inputs["news"].astype(np.float64)
    elif edit == "shape":
        inputs["prices"] = inputs["prices"][:-1]
    else:
        row = replace(row, presence=row.presence.astype(np.int64))
    with pytest.raises(ValueError, match="forma y el tipo" if edit in {"dtype", "shape"} else None):
        producer(source).capture(replace(row, inputs=inputs), action="fixture_fixed_point/no_tool")


@pytest.mark.parametrize("edit", ["duplicate", "category", "missing", "large", "header"])
def test_catalogue_validation_uses_the_actual_bounded_bytes(source, tmp_path, edit):
    table = list(csv.DictReader(io.StringIO(CATALOG.read_text())))
    fields = list(table[0])
    if edit == "duplicate":
        table.append(dict(table[0]))
    elif edit == "category":
        table[0]["category"] = "not_a_macro_family"
    elif edit == "missing":
        table.pop(0)
    path = tmp_path / "invalid-catalog.csv"
    if edit == "large":
        path.write_bytes(b"x" * (2 * 1024**2 + 1))
    elif edit == "header":
        path.write_text("wrong,headers\nfixture,inflation\n")
    else:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(table)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="límite" if edit == "large" else "catálogo"):
        api().FailureContextProducer(source["binding"], catalog=path, macro_catalog_sha256=expected)


def test_unknown_links_and_rows_are_rejected(source):
    with pytest.raises(ValueError):
        api().FailureContextProducer(object(), macro_catalog_sha256=CATALOG_SHA)
    with pytest.raises(ValueError):
        producer(source).capture(object(), action="fixture_fixed_point/no_tool")
