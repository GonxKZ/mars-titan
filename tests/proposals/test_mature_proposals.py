"""Revisión madura y propuestas registradas, sin ajustar ni desplegar modelos."""

import copy
import hashlib
import importlib
import importlib.util
import json
import multiprocessing
import os
import random
import signal
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from mars_titan.memory.mechanism_collection import MechanismCollector
from mars_titan.memory.mechanism_collection_record import CollectedMechanisms
from mars_titan.memory.native_backend import load_native
from mars_titan.models.titans.financial_inputs import FINAL_TEST_US
from mars_titan.models.titans.mechanism_catalog import MechanismAction
from mars_titan.models.titans.mechanism_selection import digest
from mars_titan.proposals.contracts import ProposalRegistry, RegisteredConfiguration

REQUEST_A, REQUEST_B, REQUEST_C = (character * 64 for character in "123")
LIMITS = dict(
    max_candidates=2,
    max_attempts=2,
    max_review_rows=8192,
    max_review_bytes=1024**2,
    max_state_bytes=1024**2,
)


def api(name):
    """La ausencia del módulo debe fallar en CALL, sin romper la colección."""
    module = f"mars_titan.proposals.{name}"
    try:
        found = importlib.util.find_spec(module)
    except ModuleNotFoundError:
        found = None
    assert found is not None, f"Falta el módulo {module}"
    return importlib.import_module(module)


def _fifo_metadata_probe(ledger, channel):
    """Leer solo metadatos en spawn, sin inicializar CUDA en el hijo."""

    class Deadline(Exception):
        pass

    def expired(*args):
        raise Deadline("La apertura del estado se bloqueó")

    def forbidden_cuda(*args, **kwargs):
        raise AssertionError("El proceso de metadatos intentó inicializar CUDA")

    torch.cuda._lazy_init = forbidden_cuda
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 2.0)
    try:
        ledger.snapshot()
    except Deadline:
        channel.send(("timeout", None))
    except ValueError as error:
        channel.send(("ValueError", str(error)))
    except BaseException as error:
        channel.send(("unexpected", type(error).__name__))
    else:
        channel.send(("accepted", None))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        channel.close()


def test_mature_proposal_apis_are_available_in_call():
    assert callable(getattr(api("mature_review"), "build_mature_review", None))
    assert callable(getattr(api("ledger"), "ProposalLedger", None))


@pytest.mark.parametrize("review_at", [FINAL_TEST_US + 1, FINAL_TEST_US + 365 * 86400_000_000])
def test_final_horizon_is_rejected_before_source_inspection(review_at):
    class Forbidden:
        def __iter__(self):
            pytest.fail("Se inspeccionó una fuente fuera del horizonte admitido")

    with pytest.raises(ValueError):
        api("mature_review").build_mature_review(
            Forbidden(),
            Forbidden(),
            review_at=review_at,
            parent_consumer_id="a" * 64,
            expected_window_sha256="b" * 64,
            expected_events_sha256="c" * 64,
        )


def moment(index):
    return importlib.import_module("test_financial_session").moment(index)


@pytest.fixture(scope="module", autouse=True)
def no_optimizer_steps():
    from torch.optim.optimizer import register_optimizer_step_pre_hook

    def forbidden(*args, **kwargs):
        pytest.fail("Esta selección no admite pasos de optimizador")

    handle = register_optimizer_step_pre_hook(forbidden)
    try:
        yield
    finally:
        handle.remove()


@pytest.fixture(scope="module")
def native_archive(tmp_path_factory):
    """Archivar entradas sintéticas por las APIs nativas reales, solo en CUDA."""
    assert torch.cuda.is_available(), "La fixture requiere CUDA, sin ruta CPU alternativa"
    torch.cuda.set_device("cuda:0")
    library = os.environ.get("MARS_TITAN_EPISODIC_NATIVE")
    assert library, "La fixture necesita MARS_TITAN_EPISODIC_NATIVE"
    native = load_native(library)
    output = tmp_path_factory.mktemp("mature-proposal-native") / "source"
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    original_finish = MechanismCollector.finish
    helpers = ExitStack()

    def shifted_collection(collector):
        # El ancla admite esta diferencia FP64. La emisión nativa queda intacta.
        packet = original_finish(collector)
        body = packet.payload()
        for group in body["groups"]:
            for candidate in group["candidates"]:
                if candidate["quantiles"] is not None:
                    candidate["quantiles"] = [
                        [value + 5e-13 for value in row] for row in candidate["quantiles"]
                    ]
        return CollectedMechanisms(body)

    try:
        # Los helpers antiguos se importan solo aquí y se restaura su ruta al salir.
        imports = helpers.enter_context(pytest.MonkeyPatch.context())
        imports.syspath_prepend(str(Path(__file__).resolve().parents[1] / "memory"))
        controls = importlib.import_module("test_financial_session_controls")
        engines = importlib.import_module("test_financial_mechanisms").engines
        session = importlib.import_module("test_native_mechanism_collection").session
        helpers.enter_context(contextmanager(controls.no_target_estimation.__wrapped__)())
        four_flow_source = controls.four_flow_source.__wrapped__(tmp_path_factory)
        data = engines(four_flow_source, "cuda:0")
        baseline = RegisteredConfiguration(
            predictor=data[3].predictor.config,
            readout=None,
            action=MechanismAction(),
            catalog=data[3].catalog,
            dtype="torch.float64",
        )
        alternative = replace(
            baseline, predictor=replace(baseline.predictor, seed=baseline.predictor.seed + 1)
        )
        registry = ProposalRegistry(
            entries=(baseline, alternative), baseline_id=baseline.configuration_id
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(MechanismCollector, "finish", shifted_collection)
            with session(native, output, data) as run:
                first = run.step(four_flow_source["batches"][126], [])
                run.step(four_flow_source["batches"][127], [])
                assert len(first.predictions) == 4
                decisions = (copy.deepcopy(run.mechanism_event(1)),)
                empty = run.mechanism_collections_at((1,), through_generation=2, before=moment(128))
                early_labels = [
                    native.Feedback(point.id, 0, moment(127), point.value + 0.125)
                    for point in first.predictions[:2]
                ]
                run.step([], early_labels, kind="settlement", cutoff=moment(127))
                early = run.mechanism_collections_at((1,), through_generation=3, before=moment(128))
                late_labels = [
                    native.Feedback(point.id, 0, moment(127), point.value + 0.125)
                    for point in first.predictions[2:]
                ]
                run.step([], late_labels, kind="settlement", cutoff=moment(129))
                full = run.mechanism_collections_at((1,), through_generation=4, before=moment(130))
                after_future = run.mechanism_collections_at(
                    (1,), through_generation=3, before=moment(128)
                )
                assert after_future == early
                run.step([], [], kind="settlement", cutoff=moment(131), close_phase=True)
        return dict(
            output=output,
            parent=data[3].model_id,
            registry=registry,
            candidate=alternative.configuration_id,
            decisions=decisions,
            empty=empty,
            early=early,
            after_future=after_future,
            full=full,
        )
    finally:
        helpers.close()
        torch.backends.mha.set_fastpath_enabled(fastpath)


def build_review(source, which="full", *, window=None, events=None, **changes):
    window = copy.deepcopy(source[which] if window is None else window)
    arguments = dict(
        review_at=window["as_of"]["before"],
        parent_consumer_id=source["parent"],
        expected_window_sha256=window["as_of"]["window_sha256"],
        expected_events_sha256=digest(source["decisions"]),
    )
    arguments.update(changes)
    return api("mature_review").build_mature_review(
        window,
        copy.deepcopy(source["decisions"] if events is None else events),
        **arguments,
    )


def rows(review):
    values = review.record()["rows"]
    assert len(values) == 4
    assert len({row["sample_id"] for row in values}) == 4
    return {row["sample_id"]: row for row in values}


def reseal_window(window):
    """Resellar un falsificador coherente, sin atribuirle autenticidad nativa."""
    for item in window["collections"]:
        state = item["supervision"]
        state["state_sha256"] = digest(
            {name: value for name, value in state.items() if name != "state_sha256"}
        )
    stamp = window["as_of"]
    stamp["window_sha256"] = digest(
        dict(
            collections=[
                dict(
                    collection_sha256=item["collection"]["sha256"],
                    **{name: value for name, value in item.items() if name != "collection"},
                )
                for item in window["collections"]
            ],
            as_of={name: value for name, value in stamp.items() if name != "window_sha256"},
        )
    )


def ledger_case(source, directory, *, which="full", limits=None, resume=False):
    review = build_review(source, which)
    ledger = api("ledger").ProposalLedger(
        directory,
        registry=source["registry"],
        review=review,
        limits=dict(LIMITS if limits is None else limits),
        resume=resume,
    )
    return ledger, review


def directory_bytes(directory):
    if not directory.exists():
        return None
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir()) if path.is_file()}


@pytest.mark.cuda
@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
class TestMatureReview:
    def test_receipt_at_final_test_boundary_is_not_eligible(self, native_archive):
        changed = copy.deepcopy(native_archive["full"])
        changed["as_of"]["before"] = FINAL_TEST_US
        changed["as_of"]["native_records"][-1]["cutoff"] = FINAL_TEST_US
        reseal_window(changed)
        with pytest.raises(ValueError):
            build_review(native_archive, window=changed)

    def test_native_fixture_has_exact_emissions_shifted_anchors_and_late_receipts(
        self, native_archive
    ):
        source = native_archive["full"]["collections"][0]
        packet = CollectedMechanisms.restore(source["collection"]).payload()
        native_rows = native_archive["decisions"][0]["payload"]["rows"]
        approximate = packet["groups"][0]["candidates"][0]["quantiles"]
        assert source["supervision"]["clock"] == moment(127)
        assert [row["cutoff"] for row in native_archive["full"]["as_of"]["native_records"]] == [
            moment(126),
            moment(127),
            moment(127),
            moment(129),
        ]
        for index, row in enumerate(native_rows):
            assert len(row["quantiles"]) == 5 and row["point"] == row["quantiles"][2]
            assert 0 < approximate[index][2] - row["quantiles"][2] < 1e-12
        assert source["resolutions"][-1]["generation"] == 4
        assert native_archive["early"] == native_archive["after_future"]

    def test_native_median_and_negative_gain_are_preserved(self, native_archive):
        review = build_review(native_archive)
        actual = rows(review)
        source = native_archive["full"]["collections"][0]
        packet = CollectedMechanisms.restore(source["collection"]).payload()
        emitted = native_archive["decisions"][0]["payload"]["rows"]
        reference = packet["groups"][0]["candidates"][0]["quantiles"]
        for index, row in enumerate(emitted):
            outcome = source["supervision"]["outcomes"][index]
            observed = actual[row["sample_id"]]
            assert len(row["quantiles"]) == 5
            assert reference[index][2] != row["quantiles"][2]
            assert observed["emitted_quantiles"] == row["quantiles"]
            loss = abs(row["quantiles"][2] - outcome["value"])
            reference_loss = abs(reference[index][2] - outcome["value"])
            assert observed["emitted_point_loss"] == loss
            assert observed["reference_point_loss"] == reference_loss
            assert observed["emitted_point_gain"] == reference_loss - loss < 0
            assert (
                observed["snapshot_sha256"]
                == packet["observation"]["observation"]["snapshot_sha256"]
            )
        assert review.record()["parent_consumer_id"] == native_archive["parent"]
        assert review.record()["window_sha256"] == native_archive["full"]["as_of"]["window_sha256"]
        assert review.record()["events_sha256"] == digest(native_archive["decisions"])

    def test_native_event_digest_rejects_joint_point_and_median_change_with_intact_window(
        self, native_archive
    ):
        events = copy.deepcopy(native_archive["decisions"])
        row = events[0]["payload"]["rows"][0]
        row["point"] += 2.5e-13
        row["quantiles"][2] += 2.5e-13
        assert row["point"] == row["quantiles"][2]
        assert digest(events) != digest(native_archive["decisions"])
        before = copy.deepcopy(native_archive["full"])
        with pytest.raises(ValueError, match="huella.*eventos|eventos.*huella|events_sha256"):
            build_review(native_archive, events=events)
        assert native_archive["full"] == before

    def test_received_at_uses_native_resolution_instead_of_asof_clock(self, native_archive):
        source = native_archive["full"]["collections"][0]
        assert source["supervision"]["clock"] == moment(127)
        actual = rows(build_review(native_archive))
        for index, metadata in enumerate(
            CollectedMechanisms.restore(source["collection"]).payload()["rows"]
        ):
            row = actual[metadata["sample_id"]]
            assert row["native_received_at"] == moment(127 if index < 2 else 129)
            assert row["receipt_generation"] == (3 if index < 2 else 4)
            assert row["available_at"] == moment(127)

    def test_pending_rows_and_unobserved_losses_remain_unknown(self, native_archive):
        actual = list(rows(build_review(native_archive, "early")).values())
        assert sum(row["emitted_point_loss"] is not None for row in actual) == 2
        for row in actual[2:]:
            assert row["target"] is row["emitted_point_loss"] is row["emitted_point_gain"] is None
            assert row["native_received_at"] is None
        for row in actual:
            assert any(value is None for value in row["candidate_point_losses"])
            assert [value is None for value in row["candidate_point_losses"]] == [
                value is None for value in row["candidate_point_gains"]
            ]

    def test_later_native_suffix_cannot_change_an_earlier_review(self, native_archive):
        first = build_review(native_archive, "early")
        later = build_review(native_archive, "after_future")
        assert later.record() == first.record() and later.review_id == first.review_id

    def test_review_record_does_not_expose_mutable_aliases(self, native_archive):
        review = build_review(native_archive)
        before, identity = review.record(), review.review_id
        changed = review.record()
        changed["rows"][0]["emitted_point_loss"] = 0.0
        changed["rows"][0]["emitted_quantiles"][2] += 1.0
        assert review.record() == before and review.review_id == identity

    @pytest.mark.parametrize("clock", [127, 128, 130])
    def test_clock_cannot_backdate_native_reception(self, native_archive, clock):
        changed = copy.deepcopy(native_archive["full"])
        changed["as_of"]["before"] = moment(129)
        changed["collections"][0]["supervision"]["clock"] = moment(clock)
        reseal_window(changed)
        with pytest.raises(ValueError):
            build_review(native_archive, window=changed)

    @pytest.mark.parametrize(
        "fault", ["parent", "window_sha", "events_sha", "native_sha", "snapshot", "sample"]
    )
    def test_incoherent_native_links_are_rejected(self, native_archive, fault):
        changed = copy.deepcopy(native_archive["full"])
        events = copy.deepcopy(native_archive["decisions"])
        options = {}
        if fault == "parent":
            options["parent_consumer_id"] = "e" * 64
        elif fault == "window_sha":
            options["expected_window_sha256"] = "e" * 64
        elif fault == "events_sha":
            options["expected_events_sha256"] = "e" * 64
        elif fault == "native_sha":
            changed["collections"][0]["resolutions"][0]["native_record_sha256"] = "e" * 64
            reseal_window(changed)
        elif fault == "snapshot":
            events[0]["payload"]["snapshot_sha256"] = "e" * 64
        else:
            events[0]["payload"]["rows"][0]["sample_id"] += "/foreign"
        with pytest.raises(ValueError):
            build_review(native_archive, window=changed, events=events, **options)

    @pytest.mark.parametrize("boundary", ["decision", "maturity", "availability", "reception"])
    def test_review_cutoff_is_strict_for_all_usable_timestamps(self, native_archive, boundary):
        source = native_archive["full"]["collections"][0]
        metadata = CollectedMechanisms.restore(source["collection"]).payload()["rows"][0]
        cutoff = dict(
            decision=metadata["prediction_at"],
            maturity=metadata["matures_at"],
            availability=source["supervision"]["outcomes"][0]["available_at"],
            reception=moment(129),
        )[boundary]
        changed = copy.deepcopy(native_archive["full"])
        changed["as_of"]["before"] = cutoff
        reseal_window(changed)
        with pytest.raises(ValueError):
            build_review(native_archive, window=changed)

    def test_build_and_persistence_do_not_construct_predictors_or_fit(
        self, native_archive, tmp_path, monkeypatch, learning_hold
    ):
        from mars_titan.models.titans.financial import FinancialPredictor
        from mars_titan.training import mechanism_policy_fit

        learning_hold(False)
        calls = []

        def forbidden(*args, **kwargs):
            calls.append(True)
            pytest.fail("La revisión o la propuesta intentó ejecutar un modelo o ajuste")

        before_source = {
            str(path.relative_to(native_archive["output"])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in native_archive["output"].rglob("*")
            if path.is_file()
        }
        cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        monkeypatch.setattr(FinancialPredictor, "__init__", forbidden)
        monkeypatch.setattr(torch.nn.Module, "__call__", forbidden)
        monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
        monkeypatch.setattr(torch.optim.SGD, "step", forbidden)
        monkeypatch.setattr(mechanism_policy_fit, "fit_policy", forbidden)
        ledger, _ = ledger_case(native_archive, tmp_path / "pure")
        reserved = ledger.reserve(native_archive["candidate"], request_id=REQUEST_A)
        ledger.finish(
            reserved.record()["proposal_id"],
            disposition="proposed",
            reason="registered_configuration",
        )
        assert calls == []
        assert torch.equal(torch.random.get_rng_state(), cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
        assert random.getstate() == python_rng
        for before, after in zip(numpy_rng, np.random.get_state(), strict=True):
            assert np.array_equal(before, after)
        after_source = {
            str(path.relative_to(native_archive["output"])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in native_archive["output"].rglob("*")
            if path.is_file()
        }
        assert after_source == before_source


@pytest.mark.cuda
@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
class TestProposalLedger:
    def test_fifo_state_is_rejected_without_blocking_a_metadata_child(
        self, native_archive, tmp_path
    ):
        output = tmp_path / "fifo"
        ledger, _ = ledger_case(native_archive, output)
        state = output / "state.json"
        confirmed = state.read_bytes()
        state.unlink()
        os.mkfifo(state)
        context = multiprocessing.get_context("spawn")
        received, sent = context.Pipe(duplex=False)
        process = context.Process(target=_fifo_metadata_probe, args=(ledger, sent))
        try:
            process.start()
            sent.close()
            process.join(timeout=10.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
                result = ("parent_watchdog", None)
            else:
                result = received.recv() if received.poll() else ("missing", process.exitcode)
            assert result[0] == "ValueError", result
            assert process.exitcode == 0
        finally:
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
            sent.close()
            received.close()
            state.unlink(missing_ok=True)
            state.write_bytes(confirmed)

    def test_incoherent_review_body_cannot_create_an_output(self, native_archive, tmp_path):
        review = build_review(native_archive)
        body = review.record()
        body.pop("review_id")
        body.update(rows=[], mature_rows=1)
        output = tmp_path / "incoherent-review"
        with pytest.raises(ValueError):
            forged = api("mature_review").MatureReview(body)
            api("ledger").ProposalLedger(
                output,
                registry=native_archive["registry"],
                review=forged,
                limits=dict(LIMITS),
            )
        assert not output.exists()

    def test_proposed_requires_mature_evidence_but_not_positive_gain(
        self, native_archive, tmp_path
    ):
        ledger, review = ledger_case(native_archive, tmp_path / "negative")
        assert all(row["emitted_point_gain"] < 0 for row in rows(review).values())
        reserved = ledger.reserve(native_archive["candidate"], request_id=REQUEST_A)
        assert reserved.record()["window_sha256"] == review.record()["window_sha256"]
        assert reserved.record()["events_sha256"] == review.record()["events_sha256"]
        receipt = ledger.finish(
            reserved.record()["proposal_id"],
            disposition="proposed",
            reason="registered_configuration",
        ).record()
        assert receipt["status"] == "proposed"
        assert receipt["selection_performed"] is receipt["deployment_performed"] is False
        assert receipt["window_sha256"] == review.record()["window_sha256"]
        assert receipt["events_sha256"] == review.record()["events_sha256"]
        assert ledger.snapshot()["charged_attempts"] == 1

        empty, _ = ledger_case(native_archive, tmp_path / "empty", which="empty")
        attempt = empty.reserve(native_archive["candidate"], request_id=REQUEST_A)
        before = empty.snapshot()
        with pytest.raises(ValueError):
            empty.finish(
                attempt.record()["proposal_id"],
                disposition="proposed",
                reason="registered_configuration",
            )
        assert empty.snapshot() == before
        rejected = empty.finish(
            attempt.record()["proposal_id"],
            disposition="rejected",
            reason="insufficient_mature_evidence",
        )
        assert rejected.record()["status"] == "rejected"
        assert empty.snapshot()["charged_attempts"] == 1

    def test_requests_are_idempotent_and_do_not_reset_the_quota(self, native_archive, tmp_path):
        ledger, _ = ledger_case(native_archive, tmp_path / "requests")
        first = ledger.reserve(native_archive["candidate"], request_id=REQUEST_A)
        assert first.record()["ordinal"] == 1
        assert (
            ledger.reserve(native_archive["candidate"], request_id=REQUEST_A).record()
            == first.record()
        )
        before = ledger.snapshot()
        with pytest.raises(ValueError):
            ledger.reserve(native_archive["registry"].baseline_id, request_id=REQUEST_A)
        with pytest.raises(ValueError):
            ledger.reserve(native_archive["candidate"], request_id=REQUEST_B)
        assert ledger.snapshot() == before
        terminal = ledger.finish(
            first.record()["proposal_id"], disposition="abandoned", reason="discarded"
        )
        assert (
            ledger.reserve(native_archive["candidate"], request_id=REQUEST_A).record()
            == terminal.record()
        )
        second = ledger.reserve(native_archive["candidate"], request_id=REQUEST_B)
        assert second.record()["ordinal"] == 2
        ledger.finish(second.record()["proposal_id"], disposition="rejected", reason="discarded")
        before = ledger.snapshot()
        with pytest.raises(ValueError):
            ledger.reserve(native_archive["candidate"], request_id=REQUEST_C)
        assert ledger.snapshot() == before and before["charged_attempts"] == 2

    def test_reopen_keeps_reserved_and_terminal_receipts(self, native_archive, tmp_path):
        output = tmp_path / "reopen"
        ledger, _ = ledger_case(native_archive, output)
        first = ledger.reserve(native_archive["candidate"], request_id=REQUEST_A)
        restored, _ = ledger_case(native_archive, output, resume=True)
        assert restored.snapshot() == ledger.snapshot()
        assert (
            restored.reserve(native_archive["candidate"], request_id=REQUEST_A).record()
            == first.record()
        )
        final = restored.finish(
            first.record()["proposal_id"], disposition="proposed", reason="registered_configuration"
        )
        assert (
            restored.finish(
                first.record()["proposal_id"],
                disposition="proposed",
                reason="registered_configuration",
            ).record()
            == final.record()
        )
        reopened, _ = ledger_case(native_archive, output, resume=True)
        assert (
            reopened.reserve(native_archive["candidate"], request_id=REQUEST_A).record()
            == final.record()
        )
        assert reopened.snapshot()["charged_attempts"] == 1

    def test_interrupted_terminal_replace_preserves_the_charge_and_pending_attempt(
        self, native_archive, tmp_path, monkeypatch
    ):
        output = tmp_path / "interrupted"
        ledger, _ = ledger_case(native_archive, output)
        reserved = ledger.reserve(native_archive["candidate"], request_id=REQUEST_A)
        before, files = ledger.snapshot(), directory_bytes(output)
        replace_file = os.replace

        def failed(source, destination):
            if Path(destination) == output / "state.json":
                raise OSError("Interrupción de fixture antes de sustituir el estado")
            return replace_file(source, destination)

        with monkeypatch.context() as patch:
            patch.setattr(os, "replace", failed)
            with pytest.raises(OSError):
                ledger.finish(
                    reserved.record()["proposal_id"],
                    disposition="proposed",
                    reason="registered_configuration",
                )
        assert directory_bytes(output) == files
        assert ledger.snapshot() == before
        recovered, _ = ledger_case(native_archive, output, resume=True)
        assert recovered.snapshot() == before and before["charged_attempts"] == 1
        result = recovered.finish(
            reserved.record()["proposal_id"],
            disposition="proposed",
            reason="registered_configuration",
        )
        assert result.record()["ordinal"] == 1

    @pytest.mark.parametrize("fault", ["counter", "json", "oversized", "symlink"])
    def test_corrupt_state_is_rejected_without_erasing_the_receipts(
        self, native_archive, tmp_path, fault
    ):
        output = tmp_path / "corrupt"
        ledger, _ = ledger_case(native_archive, output)
        ledger.reserve(native_archive["candidate"], request_id=REQUEST_A)
        state = output / "state.json"
        if fault == "counter":
            changed = json.loads(state.read_text())
            changed["charged_attempts"] += 1
            changed["state_sha256"] = digest(
                {name: value for name, value in changed.items() if name != "state_sha256"}
            )
            state.write_text(json.dumps(changed))
        elif fault == "json":
            state.write_text("{truncated")
        elif fault == "oversized":
            state.write_bytes(state.read_bytes() + b" " * LIMITS["max_state_bytes"])
        else:
            retained = tmp_path / "retained.json"
            retained.write_bytes(state.read_bytes())
            state.unlink()
            state.symlink_to(retained)
        before = state.read_bytes()
        with pytest.raises(ValueError):
            ledger_case(native_archive, output, resume=True)
        assert state.read_bytes() == before

    @pytest.mark.parametrize("name", list(LIMITS))
    def test_boolean_limits_are_rejected_before_creating_output(
        self, native_archive, tmp_path, name
    ):
        output = tmp_path / "invalid-limit"
        chosen = dict(LIMITS, **{name: True})
        with pytest.raises(ValueError):
            ledger_case(native_archive, output, limits=chosen)
        assert not output.exists()

    def test_unknown_configuration_has_no_state_effect(self, native_archive, tmp_path):
        output = tmp_path / "unknown"
        ledger, _ = ledger_case(native_archive, output)
        before, files = ledger.snapshot(), directory_bytes(output)
        with pytest.raises(ValueError):
            ledger.reserve("f" * 64, request_id=REQUEST_A)
        assert ledger.snapshot() == before and directory_bytes(output) == files

    @pytest.mark.parametrize(
        "name,value",
        [
            ("max_candidates", 9),
            ("max_attempts", 33),
            ("max_review_rows", 8193),
            ("max_review_bytes", 1024**2 + 1),
            ("max_state_bytes", 1024**2 + 1),
            ("unregistered_limit", 1),
        ],
    )
    def test_limits_are_closed_and_cannot_expand_the_contract(
        self, native_archive, tmp_path, name, value
    ):
        output = tmp_path / "expanded-limit"
        with pytest.raises(ValueError):
            ledger_case(native_archive, output, limits=dict(LIMITS, **{name: value}))
        assert not output.exists()

    @pytest.mark.parametrize("name", ["max_review_rows", "max_review_bytes", "max_state_bytes"])
    def test_admitted_review_must_fit_before_creating_output(self, native_archive, tmp_path, name):
        output = tmp_path / "small-limit"
        with pytest.raises(ValueError):
            ledger_case(native_archive, output, limits=dict(LIMITS, **{name: 1}))
        assert not output.exists()

    def test_symlink_output_is_rejected_without_changing_its_target(self, native_archive, tmp_path):
        target = tmp_path / "target"
        target.mkdir()
        output = tmp_path / "linked"
        output.symlink_to(target, target_is_directory=True)
        with pytest.raises(ValueError):
            ledger_case(native_archive, output)
        assert directory_bytes(target) == {}

    def test_two_writers_share_one_idempotent_charge(self, native_archive, tmp_path):
        output = tmp_path / "writers"
        first, _ = ledger_case(native_archive, output)

        def reserve_again(_):
            independent, _ = ledger_case(native_archive, output, resume=True)
            return independent.reserve(native_archive["candidate"], request_id=REQUEST_A).record()

        with ThreadPoolExecutor(max_workers=2) as pool:
            left, right = tuple(pool.map(reserve_again, range(2)))
        assert left == right
        reopened, _ = ledger_case(native_archive, output, resume=True)
        assert reopened.snapshot()["charged_attempts"] == 1
        assert len(reopened.snapshot()["attempts"]) == 1
        assert first.snapshot() == reopened.snapshot()

    def test_snapshot_copies_and_foreign_identity_cannot_replace_the_state(
        self, native_archive, tmp_path
    ):
        output = tmp_path / "identity"
        ledger, _ = ledger_case(native_archive, output)
        original = ledger.snapshot()
        changed = ledger.snapshot()
        changed["charged_attempts"] = 99
        assert ledger.snapshot() == original
        other = dict(native_archive)
        baseline = native_archive["registry"].entries[1]
        other["registry"] = ProposalRegistry(
            entries=(baseline,), baseline_id=baseline.configuration_id
        )
        files = directory_bytes(output)
        with pytest.raises(ValueError):
            ledger_case(other, output, resume=True)
        assert directory_bytes(output) == files
