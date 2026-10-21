"""Rasgos de cohortes admitidas, sin etiquetas ni ajustes de parámetros."""

import hashlib
import importlib
import math
import weakref
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest
import torch
from test_episodic_readout import CODEC, CONTEXT, source
from test_financial_adapter import raw_batch, setup
from test_financial_mechanisms import assert_same, engines, session
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_mechanism_consumer import calibration, consumer
from test_native_episode_backend import native as native

from mars_titan.calibration.cohort_feedback import CohortOutcome
from mars_titan.data.input_policy import MODALITIES
from mars_titan.data.price_windows import price_window_contract
from mars_titan.memory.episodic_session import DecisionRow
from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import FinancialInputSpec, validated_cpu_batch
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer

AT = 1_609_459_200_000_000
DAY = 86_400_000_000


def api():
    return importlib.import_module("mars_titan.memory.mechanism_observations")


class ForbiddenEmbedding:
    def __array__(self, *args, **kwargs):
        raise AssertionError("El extractor no consulta valores episódicos ni supervisión")


def row_sha(row, spec):
    metadata = {k: v for k, v in row.metadata().items() if k != "input_sha256"}
    metadata["input_contract_id"] = spec.fingerprint()
    digest = hashlib.sha256(canonical(metadata).encode())
    for name in MODALITIES:
        digest.update(row.inputs[name].tobytes())
    digest.update(row.presence.tobytes())
    return digest.hexdigest()


def observation_rows(engine, raw=None):
    raw = raw_batch() if raw is None else raw
    batch = validated_cpu_batch(raw, engine.predictor.config.inputs)
    rows = []
    for i, flow in enumerate(batch.flow_ids):
        metadata = dict(
            flow_id=flow,
            sample_id=batch.sample_ids[i],
            prediction_at=batch.prediction_at[i],
            input_available_at=batch.input_available_at[i],
            input_contract_id=batch.input_contract_id,
        )
        row = DecisionRow(
            flow,
            metadata["sample_id"],
            metadata["prediction_at"],
            metadata["input_available_at"],
            "",
            MappingProxyType({k: v[i] for k, v in batch.inputs.items()}),
            batch.presence[i],
            ForbiddenEmbedding(),
            ForbiddenEmbedding(),
        )
        rows.append(replace(row, input_sha256=row_sha(row, engine.predictor.config.inputs)))
    return tuple(rows)


def extract(engine, rows, *, state=None, flows=None, snapshot=None, cutoff=AT):
    return api().extract_observation(
        rows,
        consumer=engine,
        state=engine.initial_state() if state is None else state,
        flow_ids=tuple(r.flow_id for r in rows) if flows is None else flows,
        cutoff=cutoff,
        context_id=CONTEXT,
        snapshot=snapshot,
    )


def hand_inputs():
    raw = raw_batch()
    raw["inputs"]["prices"][:] = 2
    raw["inputs"]["news"][0] = 3
    raw["inputs"]["charts"][:] = 4
    raw["input_available_at"][0] = np.datetime64(AT - 2_000_000, "us")
    return raw


def test_values_match_the_hand_computed_masked_means_and_ages():
    engine = consumer()
    rows = observation_rows(engine, hand_inputs())
    result = extract(engine, rows)
    numeric = math.log1p(1 / math.sqrt(3)) / 2
    expected = [
        math.log(3),
        1,
        0.5,
        1,
        0.5,
        0.5,
        math.log(3),
        math.log(4) / 2,
        math.log(5),
        numeric,
        numeric,
        math.log(3),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    np.testing.assert_allclose(
        result.tensor(device="cpu", dtype=torch.float64).numpy()[0],
        expected,
        rtol=1e-13,
        atol=1e-13,
    )
    body = result.record()["observation"]
    assert body["groups"][0]["market"] == "US"
    assert body["groups"][0]["rows"] == 2
    assert body["groups"][0]["oldest_input_available_at"] == AT - 2_000_000
    assert body["normalization"] == "none_raw_features"


def test_permutation_and_physical_partition_preserve_the_complete_observation():
    engine = consumer()
    rows = observation_rows(engine, hand_inputs())
    expected = extract(engine, rows)
    changed = tuple(r for chunk in ((rows[1],), (rows[0],)) for r in chunk)
    assert extract(engine, changed).record() == expected.record()
    with pytest.raises(ValueError, match="cohorte"):
        extract(engine, rows[:1], flows=tuple(r.flow_id for r in rows))


def test_future_metadata_is_rejected_before_inspecting_its_arrays():
    engine = consumer()
    row = observation_rows(engine)[0]
    future = replace(row, input_available_at=AT + 1, inputs=ForbiddenEmbedding())
    with pytest.raises(ValueError, match="futuro"):
        extract(engine, (future,))


def test_historical_observations_cannot_move_the_prediction_before_year_2000():
    engine = consumer()
    past = 946_684_799_999_999
    changed = []
    for row in observation_rows(engine):
        row = replace(
            row,
            prediction_at=past,
            input_available_at=past - 1,
            sample_id=f"{row.flow_id}/{past}",
        )
        changed.append(replace(row, input_sha256=row_sha(row, engine.predictor.config.inputs)))
    with pytest.raises(ValueError, match="históric|2000"):
        extract(engine, tuple(changed), cutoff=past)


def test_historical_cutoff_is_checked_before_accessing_input_arrays():
    engine = consumer()
    past = 946_684_799_999_999
    row = observation_rows(engine)[0]
    row = replace(
        row,
        prediction_at=past,
        input_available_at=past - 1,
        sample_id=f"{row.flow_id}/{past}",
        inputs=ForbiddenEmbedding(),
    )
    with pytest.raises(ValueError, match="históric|2000"):
        extract(engine, (row,), cutoff=past)


def test_snapshot_count_uses_only_the_identified_eligible_bank():
    engine = consumer(reader=True, populated=True)
    rows = observation_rows(engine)
    snapshot = EpisodeSnapshot.create(
        **source(3),
        cutoff=AT,
        codec_id=CODEC,
        context_id=CONTEXT,
        device="cpu",
        dtype=torch.float64,
    )
    result = extract(engine, rows, snapshot=snapshot)
    assert result.record()["observation"]["snapshot_sha256"] == snapshot._digest
    assert result.tensor(device="cpu", dtype=torch.float64)[0, 12].item() == math.log(4)
    other = EpisodeSnapshot.create(
        **source(3),
        cutoff=AT - 1,
        codec_id=CODEC,
        context_id=CONTEXT,
        device="cpu",
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="snapshot"):
        extract(engine, rows, snapshot=other)


def test_previous_context_age_and_calibration_are_read_without_changing_the_snapshot():
    engine = consumer(kind="market_memory_v1", calibrator=calibration())
    state = engine.initial_state()
    previous = AT - DAY
    sample = f"US/AAA/{previous}"
    emitted = state.calibration.emit(
        "US", previous, (sample,), np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]]), (previous + 10,)
    )
    received = emitted.state.feedback(
        (CohortOutcome("US", previous, sample, previous + 10, 0.0),), at=previous + 10
    )
    state = replace(
        state,
        cutoff=previous + 10,
        context=replace(state.context, committed_at=(previous, -1)),
        calibration=received.state,
    )
    before = engine.export_state(state)
    result = extract(engine, observation_rows(engine), state=state)
    vector = result.tensor(device="cpu", dtype=torch.float64)[0].tolist()
    assert vector[13:17] == [1.0, math.log(2), 1.0, 0.0]
    np.testing.assert_allclose(vector[17:], [7.0, 7.0, 6.75625, 6.9125], rtol=1e-13)
    assert_same(engine.export_state(state), before)


def test_maximum_float32_inputs_have_finite_rms_without_overflow():
    engine = consumer()
    raw = hand_inputs()
    raw["inputs"]["news"][0] = np.finfo(np.float32).max
    result = extract(engine, observation_rows(engine, raw))
    actual = result.tensor(device="cpu", dtype=torch.float64)[0, 7].item()
    assert math.isclose(actual, math.log1p(float(np.finfo(np.float32).max)) / 2, rel_tol=1e-14)


def test_correction_summary_does_not_overflow_when_its_mean_is_representable():
    assert api()._corrections({"0.8": 1e308, "0.95": 1e308}) == [1e308, 1e308]
    assert api()._corrections({"0.8": -1e308, "0.95": 1e308}) == [0.0, 1e308]


@pytest.mark.parametrize("values", [{}, {"0.8": 10**1000}], ids=["missing", "overflow"])
def test_missing_or_unrepresentable_corrections_are_not_replaced_by_zero(values):
    with pytest.raises(ValueError):
        api()._corrections(values)


@pytest.mark.parametrize("value", [None, True, float("nan"), "1.0"])
def test_invalid_correction_values_are_contract_errors(value):
    with pytest.raises(ValueError):
        api()._corrections({"0.8": value})


def test_logical_input_budget_counts_shared_arrays_before_accepting_a_large_cohort():
    old = consumer().predictor.config.inputs
    spec = FinancialInputSpec(
        source_sha256=old.source_sha256,
        view_sha256=old.view_sha256,
        representation=old.representation,
        dimensions=old.dimensions | dict(news=2048, charts=2048),
        input_policy=old.input_policy,
    )
    config = FinancialConfig(
        spec,
        variant="transformer_direct",
        hidden_size=32,
        max_batch=4096,
        max_state_bytes=1024**3,
    )
    engine = FrozenMechanismConsumer(
        FinancialPredictor(config, dtype=torch.float64).eval().requires_grad_(False)
    )
    raw = raw_batch()
    for name in ("news", "charts"):
        raw["inputs"][name] = np.tile(raw["inputs"][name], (1, 1024))
    base = observation_rows(engine, raw)[0]
    rows = []
    for i in range(4000):
        row = replace(base, flow_id=f"US/T{i:04d}", sample_id=f"US/T{i:04d}/{AT}")
        rows.append(replace(row, input_sha256=row_sha(row, spec)))
    with pytest.raises(ValueError, match="64 MiB"):
        extract(engine, tuple(rows))


@pytest.mark.parametrize("fault", ["consumer", "cutoff", "empty", "row", "snapshot", "reader"])
def test_wrong_producer_state_or_cohort_is_rejected(fault):
    engine = consumer()
    rows = observation_rows(engine)
    arguments = dict(
        consumer=engine,
        state=engine.initial_state(),
        cutoff=AT,
        flow_ids=tuple(r.flow_id for r in rows),
        context_id=CONTEXT,
    )
    if fault == "consumer":
        arguments["consumer"] = object()
    elif fault == "cutoff":
        arguments["state"] = replace(arguments["state"], cutoff=AT)
    elif fault == "empty":
        rows = ()
    elif fault == "row":
        rows = (object(),)
    elif fault == "snapshot":
        arguments["snapshot"] = object()
    else:
        arguments["snapshot"] = EpisodeSnapshot.create(
            **source(1),
            cutoff=AT,
            codec_id=CODEC,
            context_id=CONTEXT,
            device="cpu",
            dtype=torch.float64,
        )
    with pytest.raises(ValueError):
        api().extract_observation(rows, **arguments)


@pytest.mark.parametrize(
    "fault", ["digest", "duplicate", "dtype", "writeable", "nonfinite", "late_cutoff"]
)
def test_invalid_rows_cannot_be_summarised_as_valid_evidence(fault):
    engine = consumer()
    rows = observation_rows(engine)
    first = rows[0]
    if fault == "digest":
        rows = (replace(first, input_sha256="f" * 64), rows[1])
    elif fault == "duplicate":
        rows = (first, first)
    elif fault == "late_cutoff":
        with pytest.raises(ValueError):
            extract(engine, rows, cutoff=1_704_067_200_000_000)
        return
    else:
        values = dict(first.inputs)
        changed = values["news"].copy()
        if fault == "dtype":
            changed = changed.astype(np.float64)
        elif fault == "nonfinite":
            changed[0] = float("nan")
        if fault != "writeable":
            changed = np.frombuffer(changed.tobytes(), dtype=changed.dtype)
        values["news"] = changed
        rows = (replace(first, inputs=MappingProxyType(values)), rows[1])
    with pytest.raises(ValueError):
        extract(engine, rows)


def test_strict_inputs_have_explicit_complete_presence_in_the_observation():
    model, _ = setup(historical=False)
    engine = FrozenMechanismConsumer(model.eval().requires_grad_(False))
    raw = raw_batch(absent=False)
    raw.pop("presence")
    result = extract(engine, observation_rows(engine, raw))
    assert result.tensor(device="cpu", dtype=torch.float64)[0, 1:6].tolist() == [1] * 5


def test_price_presence_channel_is_not_counted_as_a_numeric_price_magnitude():
    old = consumer().predictor.config.inputs
    representation = old.representation
    representation["price_window"] = price_window_contract(
        {"US": dict(start="2020-01-01", end="2023-12-31", decisions_sha256="f" * 64)},
        {"US": []},
    )
    spec = FinancialInputSpec(
        source_sha256=old.source_sha256,
        view_sha256=old.view_sha256,
        representation=representation,
        dimensions=old.dimensions | dict(prices=6),
        input_policy=old.input_policy,
    )
    model = FinancialPredictor(FinancialConfig(spec, hidden_size=32), dtype=torch.float64)
    engine = FrozenMechanismConsumer(model.eval().requires_grad_(False))
    raw = hand_inputs()
    raw["inputs"]["prices"] = np.concatenate(
        [raw["inputs"]["prices"], np.ones((2, 64, 1), dtype=np.float32)], axis=-1
    )
    raw["inputs"]["prices"][0, 5] = 0
    observation = extract(engine, observation_rows(engine, raw))
    actual = observation.tensor(device="cpu", dtype=torch.float64)[0, 6].item()
    expected = (math.log1p(2 * math.sqrt(63 / 64)) + math.log(3)) / 2
    assert math.isclose(actual, expected, rel_tol=1e-13)


def test_record_is_copied_and_round_trip_recovers_the_exact_vector():
    engine = consumer()
    observation = extract(engine, observation_rows(engine))
    record = observation.record()
    original = record["observation"]["groups"][0]["values"][8]
    restored = api().MechanismObservation.restore(record)
    assert restored.record() == record
    record["observation"]["groups"][0]["values"][8] = original + 1
    assert observation.record()["observation"]["groups"][0]["values"][8] == original
    with pytest.raises(ValueError, match="huella"):
        api().MechanismObservation.restore(record)


def test_incomplete_record_envelopes_are_rejected():
    with pytest.raises(ValueError):
        api().MechanismObservation.restore({"observation": {}})


def test_unsupported_precision_does_not_convert_the_vector():
    engine = consumer()
    observation = extract(engine, observation_rows(engine))
    with pytest.raises(ValueError, match="FP32 o FP64"):
        observation.tensor(device="cpu", dtype=torch.float16)


def test_markets_keep_their_own_means_and_row_counts():
    engine = consumer()
    raw = hand_inputs()
    raw["sample_ids"][0] = f"CN/000001.SZ/{AT}"
    observation = extract(engine, observation_rows(engine, raw))
    groups = observation.record()["observation"]["groups"]
    assert [g["market"] for g in groups] == ["CN", "US"]
    assert [g["rows"] for g in groups] == [1, 1]
    assert groups[0]["values"][2] == 1 and groups[1]["values"][2] == 0
    assert math.isclose(groups[0]["values"][7], math.log(4), rel_tol=1e-14)
    assert groups[1]["values"][7] == 0


def test_sealing_does_not_retain_the_source_arrays():
    engine = consumer()
    rows = observation_rows(engine)
    reference = weakref.ref(rows[0].inputs["news"])
    observation = extract(engine, rows)
    del rows
    assert reference() is None
    assert observation.tensor(device="cpu", dtype=torch.float64).shape == (1, 21)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
def test_conversion_cannot_publish_infinity_when_fp64_features_do_not_fit_fp32(device):
    if device == "cuda:0":
        assert torch.cuda.is_available()
        torch.cuda.set_device(device)
    engine = consumer()
    record = extract(engine, observation_rows(engine)).record()
    vector = record["observation"]["groups"][0]["values"]
    vector[15], vector[17:] = 1.0, [1e308] * 4
    record["sha256"] = hashlib.sha256(canonical(record["observation"]).encode()).hexdigest()
    observation = api().MechanismObservation.restore(record)
    assert observation.tensor(device=device, dtype=torch.float64)[0, 17].item() == 1e308
    with pytest.raises(ValueError, match="finit|precisión"):
        observation.tensor(device=device, dtype=torch.float32)


@pytest.mark.parametrize(
    "fault",
    [
        "version",
        "empty",
        "duplicate",
        "row_count",
        "bank",
        "producer",
        "age",
        "fraction",
        "absent",
        "context",
        "context_age",
        "calibration",
        "correction",
    ],
)
def test_resealed_metadata_cannot_contradict_its_own_values_or_identity(fault):
    engine = consumer()
    record = extract(engine, observation_rows(engine)).record()
    body = record["observation"]
    group = body["groups"][0]
    vector = group["values"]
    if fault == "version":
        body["schema_version"] = True
    elif fault == "empty":
        body["groups"] = []
    elif fault == "duplicate":
        body["groups"].append(dict(group))
    elif fault == "row_count":
        group["rows"], vector[0] = 0, 0.0
    elif fault == "bank":
        body["bank_entries"], vector[12] = 1, math.log(2)
    elif fault == "producer":
        body["producer_sha256"] = "x" * 64
    elif fault == "age":
        vector[11] = 999.0
    elif fault == "fraction":
        vector[2] = 0.1
    elif fault == "absent":
        vector[2], vector[7] = 0.0, 1.0
    elif fault == "context":
        group["context_committed_at"] = AT
        vector[13] = 1.0
    elif fault == "context_age":
        body["previous_cutoff"] = group["context_committed_at"] = AT - DAY
        vector[13], vector[14] = 1.0, 999.0
    elif fault == "calibration":
        vector[15] = 2.0
    else:
        vector[15], vector[17], vector[18] = 1.0, 3.0, 2.0
    record["sha256"] = hashlib.sha256(canonical(body).encode()).hexdigest()
    with pytest.raises(ValueError):
        api().MechanismObservation.restore(record)


@pytest.mark.parametrize("fault", ["declared_type", "row_type", "presence_stride"])
def test_malformed_identities_and_masks_fail_with_a_contract_error(fault):
    engine = consumer()
    rows = observation_rows(engine)
    flows = tuple(row.flow_id for row in rows)
    if fault == "declared_type":
        flows = (["US/AAA"], "US/BBB")
    elif fault == "row_type":
        rows = (replace(rows[0], flow_id=["US/AAA"]), rows[1])
    else:
        rows = (replace(rows[0], presence=rows[0].presence[::-1]), rows[1])
    with pytest.raises(ValueError):
        extract(engine, rows, flows=flows)


@pytest.mark.parametrize(
    "fault",
    ["future", "features", "negative_presence", "supervision", "nan", "mean_without_calibrator"],
)
def test_even_a_resealed_record_must_keep_the_temporal_and_feature_contract(fault):
    engine = consumer()
    record = extract(engine, observation_rows(engine)).record()
    body = record["observation"]
    if fault == "future":
        body["groups"][0]["newest_input_available_at"] = AT + 1
    elif fault == "features":
        body["feature_names"][0] = "future_label"
    elif fault == "negative_presence":
        body["groups"][0]["values"][1] = -0.5
    elif fault == "supervision":
        body["target"] = 0.0
    elif fault == "nan":
        body["groups"][0]["values"][0] = float("nan")
        with pytest.raises(ValueError):
            api().MechanismObservation.restore(record)
        return
    else:
        body["groups"][0]["values"][17] = 1.0
    record["sha256"] = hashlib.sha256(canonical(body).encode()).hexdigest()
    with pytest.raises(ValueError):
        api().MechanismObservation.restore(record)


def test_restored_historical_observation_keeps_the_year_2000_boundary():
    engine = consumer()
    record = extract(engine, observation_rows(engine)).record()
    body = record["observation"]
    past = 946_684_799_999_999
    body["cutoff"] = past
    for group in body["groups"]:
        group["oldest_input_available_at"] = group["newest_input_available_at"] = past - 1
        group["values"][11] = math.log1p(1e-6)
    record["sha256"] = hashlib.sha256(canonical(body).encode()).hexdigest()
    with pytest.raises(ValueError, match="históric|2000"):
        api().MechanismObservation.restore(record)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_observation_conversion_to_cuda_is_explicit_and_does_not_consume_rng(dtype):
    assert torch.cuda.is_available()
    torch.cuda.set_device("cuda:0")
    engine = consumer()
    observation = extract(engine, observation_rows(engine))
    rng = torch.cuda.get_rng_state(0).clone()
    actual = observation.tensor(device="cuda:0", dtype=dtype)
    expected = observation.tensor(device="cpu", dtype=dtype)
    assert actual.device == torch.device("cuda:0") and actual.dtype == dtype
    assert torch.equal(actual.cpu(), expected)
    assert torch.equal(torch.cuda.get_rng_state(0), rng)


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
def test_real_fixture_rows_and_recovery_preserve_the_next_pre_mac_vector(
    native, four_flow_source, tmp_path, device
):
    if device == "cuda:0":
        assert torch.cuda.is_available()
        torch.cuda.set_device(device)
    data = engines(four_flow_source, device, online=True)
    batches = four_flow_source["batches"]
    output = tmp_path / "observation-session"
    with session(native, output, data) as run:
        run.step(batches[125], (), kind="warmup")
        run.step(batches[126], ())
        rows = run._rows(batches[127])
        cutoff = rows[0].prediction_at
        before = run.snapshot()
        observed = api().extract_observation(
            rows,
            consumer=data[3],
            state=data[3].restore_state(run.mechanism_state()),
            flow_ids=tuple(r.flow_id for r in rows),
            cutoff=cutoff,
            context_id=CONTEXT,
        )
        assert observed.record()["observation"]["groups"][0]["rows"] == 4
        assert observed.tensor(device=device, dtype=torch.float64)[0, 13].item() == 1
        assert run.snapshot() == before
    with session(native, output, data, resume=True) as recovered:
        rows = recovered._rows(batches[127])
        actual = api().extract_observation(
            rows,
            consumer=data[3],
            state=data[3].restore_state(recovered.mechanism_state()),
            flow_ids=tuple(r.flow_id for r in rows),
            cutoff=cutoff,
            context_id=CONTEXT,
        )
        assert actual.record() == observed.record()
