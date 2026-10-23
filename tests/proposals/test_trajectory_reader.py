"""Lectura de emisiones maduras de una trayectoria nativa completa, sin alternativas."""

import copy
import hashlib
import importlib
import importlib.util
import json
import random
import sys
from dataclasses import asdict

import numpy as np
import pytest
import torch

from tests.proposals.test_error_source import _archive_hashes
from tests.proposals.test_error_source import native_error_sources as native_error_sources
from tests.proposals.test_mature_proposals import no_optimizer_steps as no_optimizer_steps


def test_mature_trajectory_reader_is_available_in_call():
    name = "mars_titan.proposals.trajectory_reader"
    assert importlib.util.find_spec(name) is not None, "Falta el lector de trayectorias maduras"
    module = importlib.import_module(name)
    assert callable(getattr(module, "read_mature_trajectory", None))
    assert callable(getattr(module, "MatureTrajectory", None))


def _api():
    return importlib.import_module("mars_titan.proposals.trajectory_reader")


def _body(trajectory):
    return {name: value for name, value in trajectory.record().items() if name != "trajectory_id"}


def _read(source, **changes):
    options = dict(before=source["moment"](132))
    return _api().read_mature_trajectory(source["session"], **dict(options, **changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"before": True},
        {"before": 1_704_067_200_000_001},
        {"max_records": True},
        {"max_records": 0},
        {"max_records": 257},
        {"max_bytes": True},
        {"max_bytes": 0},
        {"max_bytes": 64 * 1024**2 + 1},
    ],
)
def test_invalid_bounds_fail_before_inspecting_the_source(changes):
    class Forbidden:
        def __getattr__(self, name):
            pytest.fail("Se inspeccionó la fuente antes de validar sus límites")

    options = dict(before=1_640_995_200_000_000)
    with pytest.raises(ValueError):
        _api().read_mature_trajectory(Forbidden(), **dict(options, **changes))


@pytest.mark.cuda
@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
class TestMatureTrajectory:
    def test_reads_both_real_decisions_without_a_collection_and_matches_native_ids(
        self, native_error_sources
    ):
        source = native_error_sources(collected=False)
        record = _read(source).record()
        run = source["session"]
        expected = {}
        for generation in (2, 4):
            native, _, _ = run._binding._read_record(run, generation, maximum=64 * 1024**2)
            for prediction in native["predictions"]:
                expected[prediction["id"]] = prediction
        assert run._binding.collections is None
        assert set(record) == {
            "schema_version",
            "kind",
            "phase",
            "consumer_id",
            "parameter_id",
            "source_contract_id",
            "through_generation",
            "head_sha256",
            "before",
            "native_records",
            "logical_source_bytes",
            "rows",
            "trajectory_id",
        }
        assert record["schema_version"] == 1
        assert record["kind"] == "native_mature_trajectory_v1"
        assert record["phase"] == asdict(run.phase)
        assert record["consumer_id"] == source["consumer"].model_id
        assert record["parameter_id"] == source["consumer"].predictor._parameter_id
        assert record["source_contract_id"] == run.contract_id
        assert record["through_generation"] == 6
        assert record["head_sha256"] == source["head"]["record_sha256"]
        assert [item["generation"] for item in record["native_records"]] == [1, 2, 3, 4, 5, 6]
        assert len(record["rows"]) == len(expected) == 8
        assert {row["prediction_id"] for row in record["rows"]} == set(expected)
        for row in record["rows"]:
            prediction = expected[row["prediction_id"]]
            assert set(row) == {
                "flow_id",
                "sample_id",
                "prediction_at",
                "input_available_at",
                "input_sha256",
                "matures_at",
                "snapshot_sha256",
                "bank_entries",
                "prediction_id",
                "emitted_quantiles",
                "status",
                "target",
                "available_at",
                "native_received_at",
                "receipt_generation",
                "native_record_sha256",
                "emitted_point_loss",
            }
            assert row["flow_id"] == prediction["asset"]
            assert row["prediction_at"] == prediction["decision_at"]
            assert row["emitted_quantiles"][2] == prediction["value"]
            assert row["status"] == "label"
            assert row["snapshot_sha256"] is None and row["bank_entries"] == 0
            first = row["prediction_at"] == source["moment"](126)
            assert row["target"] == (0.125 if first else -0.25)
            assert (
                row["available_at"]
                == row["native_received_at"]
                == source["moment"](127 if first else 129)
            )
            assert row["receipt_generation"] == (3 if first else 5)
            assert row["emitted_point_loss"] == abs(prediction["value"] - row["target"])
        saved = _body(_read(source))
        canonical = json.dumps(
            saved, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        assert record["trajectory_id"] == hashlib.sha256(canonical).hexdigest()
        assert run.snapshot() == source["head"]

    def test_censored_second_decision_stays_in_the_full_frame(self, native_error_sources):
        source = native_error_sources(collected=False, complete=False)
        record = _read(source).record()
        assert len(record["rows"]) == 8
        assert [row["status"] for row in record["rows"]].count("label") == 4
        assert [row["status"] for row in record["rows"]].count("censored") == 4
        for row in record["rows"]:
            if row["prediction_at"] == source["moment"](128):
                assert row["status"] == "censored"
                assert row["target"] is row["emitted_point_loss"] is row["available_at"] is None
                assert row["native_received_at"] == source["moment"](131)
                assert row["receipt_generation"] == 6

    def test_native_prefix_exclusion_keeps_the_emitted_prediction_and_unknown_loss(
        self, native_error_sources, tmp_path
    ):
        from mars_titan.memory.financial_session import FinancialPhase
        from mars_titan.models.titans.financial_inputs import validated_cpu_batch
        from mars_titan.training.corpus_inputs import CorpusDataset, _price_contexts

        source = native_error_sources(collected=False)
        run = source["session"]
        dataset = CorpusDataset(
            run.prefixes._manifest_paths[0], input_policy="historical_masked_2000_v1"
        )
        asset = dataset.assets[0]
        flow = f"{asset['market']}/{asset['symbol']}"
        prices, _ = dataset._prices(asset)
        moment = source["moment"]
        at = moment(64)
        batch = validated_cpu_batch(
            dict(
                inputs=dict(
                    prices=_price_contexts(prices, np.array([64]), 64),
                    news=np.zeros((1, 2), np.float32),
                    charts=np.array([[3.0]], np.float32),
                    fundamentals=np.zeros((1, 3), np.float32),
                    macro=np.zeros((1, 3), np.float32),
                ),
                presence=np.array([[True, False, True, False, False]]),
                sample_ids=[f"{flow}/{at}"],
                prediction_at=np.array([at], dtype="datetime64[us]"),
                input_available_at=np.array([at], dtype="datetime64[us]"),
            ),
            source["inputs"]["spec"],
        )
        session = importlib.import_module("test_native_mechanism_collection").session
        phase = FinancialPhase("validation", moment(63), moment(64), moment(65), moment(66))
        with session(
            run.native, tmp_path / "excluded", source["data"], enabled=False, phase=phase
        ) as excluded:
            issued = excluded.step([batch], [])
            assert len(issued.predictions) == len(issued.excluded) == 1
            excluded.step([], [], kind="settlement", cutoff=moment(66), close_phase=True)
            body = _api().read_mature_trajectory(excluded, before=moment(67)).record()
        assert len(body["rows"]) == 1
        row = body["rows"][0]
        assert row["prediction_id"] == issued.predictions[0].id
        assert row["emitted_quantiles"][2] == issued.predictions[0].value
        assert row["status"] == "excluded"
        assert row["target"] is row["emitted_point_loss"] is None
        assert row["native_received_at"] == row["available_at"] == at
        assert row["receipt_generation"] == 1

    @pytest.mark.parametrize("fault", ("id", "asset", "value", "duplicate", "error"))
    def test_native_result_interpreter_rejects_changed_or_repeated_predictions(
        self, native_error_sources, fault
    ):
        source = native_error_sources(collected=False)
        run = source["session"]
        native, digest, _ = run._binding._read_record(run, 3, maximum=64 * 1024**2)
        emission, _, _ = run._binding._read_record(run, 2, maximum=64 * 1024**2)
        rows = {
            row["prediction_id"]: copy.deepcopy(row)
            for row in _read(source).record()["rows"]
            if row["prediction_at"] == source["moment"](126)
        }
        lookup = {}
        for prediction in emission["predictions"]:
            row = rows[prediction["id"]]
            row.update(
                status="pending",
                target=None,
                available_at=None,
                native_received_at=None,
                receipt_generation=None,
                native_record_sha256=None,
                emitted_point_loss=None,
            )
            lookup[prediction["id"]] = (row, prediction)
        # Se comprueba el intérprete de un registro ya decodificado, sin resellar el archivo.
        native = copy.deepcopy(native)
        received = native["feedback"][0]
        if fault == "id":
            received["prediction"]["id"] = "f" * 64
        elif fault == "asset":
            received["prediction"]["asset"] = "US/OTHER"
        elif fault == "value":
            received["prediction"]["value"] += 0.125
        elif fault == "duplicate":
            native["feedback"].append(copy.deepcopy(received))
        else:
            received["error"] += 0.125
        with pytest.raises(ValueError):
            _api()._receive(native, digest, run, lookup)

    @pytest.mark.parametrize("partition", ("calibration", "evaluation"))
    def test_reserved_partitions_are_not_trajectory_sources(self, native_error_sources, partition):
        source = native_error_sources(partition, collected=False)
        with pytest.raises(ValueError):
            _read(source)
        assert source["session"].snapshot() == source["head"]

    def test_open_phase_is_not_accepted_as_a_completed_trajectory(self, native_error_sources):
        source = native_error_sources(collected=False, closed=False)
        with pytest.raises(ValueError):
            _read(source)

    def test_close_and_source_budgets_have_exact_boundaries(self, native_error_sources):
        source = native_error_sources(collected=False)
        close_at = source["session"].phase.close_at
        with pytest.raises(ValueError):
            _read(source, before=close_at)
        assert len(_read(source, before=close_at + 1).record()["rows"]) == 8
        with pytest.raises(ValueError):
            _read(source, max_records=5)
        trajectory = _read(source, max_records=6)
        size = trajectory.record()["logical_source_bytes"]
        assert _read(source, max_bytes=size).record() == trajectory.record()
        with pytest.raises(ValueError):
            _read(source, max_bytes=size - 1)

    @pytest.mark.parametrize(
        "fault",
        [
            "header_extra",
            "row_extra",
            "duplicate_sample",
            "duplicate_prediction",
            "native_head",
            "native_count",
            "logical_bytes",
            "input_future",
            "maturity_future",
            "received_future",
            "received_sha",
            "loss",
            "target_nan",
            "quantiles",
            "boolean_id",
            "censored_loss",
            "missing_snapshot_count",
        ],
    )
    def test_constructor_rejects_incoherent_trajectory_bodies(self, native_error_sources, fault):
        source = native_error_sources(collected=False, complete=fault != "censored_loss")
        body = _body(_read(source))
        row = body["rows"][4 if fault == "censored_loss" else 0]
        if fault == "header_extra":
            body["candidate_losses"] = []
        elif fault == "row_extra":
            row["requested"] = "off/k1/none"
        elif fault == "duplicate_sample":
            body["rows"].append(copy.deepcopy(row))
        elif fault == "duplicate_prediction":
            body["rows"][1]["prediction_id"] = row["prediction_id"]
        elif fault == "native_head":
            body["head_sha256"] = "e" * 64
        elif fault == "native_count":
            body["native_records"].pop(0)
        elif fault == "logical_bytes":
            body["logical_source_bytes"] = 1
        elif fault == "input_future":
            row["input_available_at"] = row["prediction_at"] + 1
        elif fault == "maturity_future":
            row["matures_at"] = body["before"]
        elif fault == "received_future":
            row["native_received_at"] = body["before"]
        elif fault == "received_sha":
            row["native_record_sha256"] = "f" * 64
        elif fault == "loss":
            row["emitted_point_loss"] += 0.125
        elif fault == "target_nan":
            row["target"] = float("nan")
        elif fault == "quantiles":
            row["emitted_quantiles"].reverse()
            row["emitted_quantiles"][0] += 1.0
        elif fault == "boolean_id":
            row["prediction_id"] = True
        elif fault == "censored_loss":
            row.update(target=0.125, emitted_point_loss=1.0)
        else:
            row["bank_entries"] = None
        with pytest.raises(ValueError):
            _api().MatureTrajectory(body)

    def test_constructor_caps_the_complete_frame_at_8192_rows(self, native_error_sources):
        body = _body(_read(native_error_sources(collected=False)))
        body["rows"] *= 1025
        assert len(body["rows"]) == 8200
        with pytest.raises(ValueError):
            _api().MatureTrajectory(body)

    def test_record_checks_its_own_eight_mib_budget_before_decoding(self, native_error_sources):
        trajectory = _read(native_error_sources(collected=False))
        content = trajectory._json + " " * (8 * 1024**2 - len(trajectory._json.encode()) + 1)
        object.__setattr__(trajectory, "_json", content)
        object.__setattr__(trajectory, "_identity", hashlib.sha256(content.encode()).hexdigest())
        with pytest.raises(ValueError):
            trajectory.record()

    def test_unknown_snapshot_count_is_not_deduced_from_its_hash(self, native_error_sources):
        body = _body(_read(native_error_sources(collected=False)))
        # El constructor comprueba coherencia, sin autenticar esta procedencia declarada.
        body["rows"][0].update(snapshot_sha256="a" * 64, bank_entries=None)
        row = _api().MatureTrajectory(body).record()["rows"][0]
        assert row["snapshot_sha256"] == "a" * 64 and row["bank_entries"] is None

    def test_constructor_preserves_excluded_as_unknown_loss(self, native_error_sources):
        body = _body(_read(native_error_sources(collected=False)))
        row = body["rows"][0]
        row.update(status="excluded", target=None, emitted_point_loss=None)
        restored = _api().MatureTrajectory(body).record()["rows"][0]
        assert restored["status"] == "excluded"
        assert restored["target"] is restored["emitted_point_loss"] is None

    def test_record_returns_copies_and_checks_its_seal(self, native_error_sources):
        trajectory = _read(native_error_sources(collected=False))
        original = trajectory.record()
        changed = trajectory.record()
        changed["rows"][0]["emitted_quantiles"][2] += 1.0
        changed["native_records"].clear()
        assert trajectory.record() == original
        object.__setattr__(trajectory, "_json", trajectory._json + " ")
        with pytest.raises(ValueError):
            trajectory.record()

    def test_reading_preserves_files_parameters_and_rng_without_model_execution(
        self, native_error_sources
    ):
        from mars_titan.models.titans.financial import FinancialPredictor

        source = native_error_sources(collected=False)
        before_files = _archive_hashes(source["output"])
        identity = source["consumer"].identity()
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        cpu_rng, cuda_rng = (
            torch.random.get_rng_state().clone(),
            torch.cuda.get_rng_state(0).clone(),
        )
        forbidden_codes = {
            FinancialPredictor.__init__.__code__,
            torch.nn.Module._call_impl.__code__,
        }

        def forbid_execution(frame, event, arg):
            if event == "call" and frame.f_code in forbidden_codes:
                pytest.fail("El lector construyó o ejecutó un modelo")

        previous = sys.getprofile()
        sys.setprofile(forbid_execution)
        try:
            trajectory = _read(source)
        finally:
            sys.setprofile(previous)
        assert len(trajectory.record()["rows"]) == 8
        assert source["consumer"].identity() == identity
        assert source["session"].snapshot() == source["head"]
        assert _archive_hashes(source["output"]) == before_files
        assert random.getstate() == python_rng
        for a, b in zip(numpy_rng, np.random.get_state(), strict=True):
            assert np.array_equal(a, b)
        assert torch.equal(torch.random.get_rng_state(), cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state(0), cuda_rng)
