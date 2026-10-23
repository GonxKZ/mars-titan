"""Reservas durables con archivos y bloqueo nativos reales, sin modelos ni optimizadores."""

import hashlib
import importlib
import importlib.util
import json
import os
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")

IDENTITY = dict(contract_id="a" * 64, model_id="b" * 64, program_id="c" * 64)
LIMITS = dict(mac_observations=3, local_node_evaluations=10, context_commits=1)
UNITS = dict(mac_observations=1, local_node_evaluations=4, context_commits=0)
ROWS = "d" * 64
INDEX_BYTES = 65536
COMPACT_INDEX_BYTES = 16384


def ledger_type():
    name = "mars_titan.memory.regional_work"
    assert importlib.util.find_spec(name) is not None, "Falta implementar RegionalWorkLedger"
    module = importlib.import_module(name)
    assert hasattr(module, "RegionalWorkLedger"), "Falta implementar RegionalWorkLedger"
    return module.RegionalWorkLedger


@contextmanager
def workspace(directory, *, resume=False, financial=False, history=False):
    # La ausencia de producción debe ser un fallo en call, antes de cargar el enlace nativo.
    ledger_type()
    from mars_titan.memory.native_backend import load_native
    from mars_titan.memory.session_artifacts import SessionArtifacts

    path = os.environ.get("MARS_TITAN_EPISODIC_NATIVE")
    assert path, "Falta MARS_TITAN_EPISODIC_NATIVE para comprobar persistencia real"
    native = load_native(path)
    if financial:
        from test_native_financial_lifecycle import callbacks, contract

        definition = contract(native)
        if history:
            definition.phase = native.PhaseContract("validation", 1, 10, 3010, 3020, "e" * 64)
        prepare_base, resolve_base = callbacks(native, [])
    else:
        from test_native_episode_backend import callbacks, definition

        definition = definition(native)
        prepare_base, resolve_base = callbacks()
    link = dict(reservation_id=None)

    def prepare(*args):
        values, proposal = prepare_base(*args)
        proposal = json.loads(proposal)
        proposal["regional_reservation_id"] = link["reservation_id"]
        return values, json.dumps(proposal)

    def resolve(*args):
        proposal = json.loads(resolve_base(*args))
        proposal["regional_reservation_id"] = link["reservation_id"]
        if history:
            work = link["archive_work"]
            cohort = work["cohort"]
            generation = cohort["prefix"]["generation"] + 1
            record = (directory / f"record-{generation}.json").read_bytes()
            proposal["regional_work"] = work
            proposal["regional_work_history"] = link["history"].append(
                proposal.get("regional_work_history", link["history"].empty()),
                dict(
                    schema_version=1,
                    model_id=IDENTITY["model_id"],
                    generation=generation,
                    cutoff=cohort["cutoff"],
                    kind=cohort["event_kind"],
                    native_record_sha256=hashlib.sha256(record).hexdigest(),
                    payload=dict(regional_work=work),
                ),
            )
            link["staged_history"] = proposal["regional_work_history"]
        return json.dumps(proposal)

    executor = native.Executor(str(directory), definition, prepare, resolve, resume)
    try:
        artifacts = SessionArtifacts(
            native,
            directory / "artifacts",
            max_bytes=256 * 1024,
            max_files=64,
            max_total_bytes=8 * 1024**2,
        )
        if history:
            from mars_titan.memory.mechanism_history import MechanismHistory, MechanismHistoryConfig

            link["history"] = MechanismHistory(
                artifacts,
                identity=IDENTITY["contract_id"],
                model_id=IDENTITY["model_id"],
                config=MechanismHistoryConfig(
                    target_chunk_bytes=128 * 1024, max_event_bytes=192 * 1024
                ),
            )
        yield SimpleNamespace(
            directory=directory,
            artifacts=artifacts,
            executor=executor,
            native=native,
            link=link,
            history=link.get("history"),
        )
    finally:
        executor.close()


def ledger(env, **options):
    settings = dict(
        artifacts=env.artifacts,
        **IDENTITY,
        limits=dict(LIMITS),
        max_attempts=2,
        max_index_bytes=INDEX_BYTES,
    )
    directory = options.pop("directory", env.directory)
    return ledger_type()(directory, **(settings | options))


def latest(env):
    head = json.loads((env.directory / "latest.json").read_text())
    content = (env.directory / f"checkpoint-{head['generation']}.json").read_bytes()
    assert hashlib.sha256(content).hexdigest() == head["checkpoint_sha256"]
    checkpoint = json.loads(content)
    return dict(
        generation=head["generation"],
        cursor=checkpoint["cursor"],
        checkpoint_sha256=head["checkpoint_sha256"],
        reservation_id=checkpoint["state"].get("regional_reservation_id"),
    )


def prefix(env):
    return {key: value for key, value in latest(env).items() if key != "reservation_id"}


def reserve(book, before, *, cutoff=10, units=None, rows=ROWS):
    return book.reserve(
        cutoff=cutoff,
        prefix=before,
        rows_digest=rows,
        units=dict(UNITS if units is None else units),
    )


def only_cohort(book):
    cohorts = book.snapshot()["cohorts"]
    assert len(cohorts) == 1
    return cohorts[0]


def index_bytes(env):
    return (env.directory / "regional-work.json").read_bytes()


def artifact_bytes(env):
    folder = env.artifacts.directory
    return (
        {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()}
        if folder.exists()
        else {}
    )


def commit(env, reservation, *, fault=None):
    from test_native_episode_backend import cohort

    env.link["reservation_id"] = reservation["reservation_id"]
    return env.executor.step(cohort(env.native, env.executor.cursor), [], 1, fault)


def settle(env, reservation, *, fault=None):
    from test_native_financial_lifecycle import event

    env.link["reservation_id"] = reservation["reservation_id"]
    return env.executor.step(event(env.native, env.executor.cursor, 10, "settlement"), [], 1, fault)


def archive_step(env, book, reservation, *, fault=None):
    from test_native_financial_lifecycle import event

    work = book.archive_payload(reservation["reservation_id"])
    cohort = work["cohort"]
    env.link.update(reservation_id=work["reservation_id"], archive_work=work)
    return env.executor.step(
        event(env.native, env.executor.cursor, cohort["cutoff"], cohort["event_kind"]),
        [],
        1,
        fault,
    )


def confirmed_history(env):
    head = json.loads((env.directory / "latest.json").read_bytes())
    content = (env.directory / f"checkpoint-{head['generation']}.json").read_bytes()
    assert hashlib.sha256(content).hexdigest() == head["checkpoint_sha256"]
    checkpoint = json.loads(content)
    manifest = checkpoint["state"]["regional_work_history"]
    assert env.history.verify(manifest)["events"] == checkpoint["cursor"]
    current = env.history.read_event(manifest, checkpoint["cursor"])
    record = (env.directory / f"record-{checkpoint['cursor']}.json").read_bytes()
    assert current["native_record_sha256"] == hashlib.sha256(record).hexdigest()
    assert current["payload"]["regional_work"] == checkpoint["state"]["regional_work"]
    assert current["payload"]["regional_work"]["reservation_id"] == latest(env)["reservation_id"]
    return manifest


def archive_proof(env):
    manifest = confirmed_history(env)
    return dict(
        reference=manifest["chunks"][-1]["reference"],
        generation=manifest["events"],
        reservation_id=latest(env)["reservation_id"],
    )


def prune_archived(env, book):
    head = json.loads((env.directory / "latest.json").read_bytes())
    live = book.references()
    for generation, digest in (
        (head["generation"], head["checkpoint_sha256"]),
        (head["generation"] - 1, head["previous_sha256"]),
    ):
        if generation <= 0:
            continue
        content = (env.directory / f"checkpoint-{generation}.json").read_bytes()
        assert hashlib.sha256(content).hexdigest() == digest
        state = json.loads(content)["state"]
        live.extend(env.history.live_references(state["regional_work_history"]))
    env.artifacts.prune_unreferenced(live)


def archive_event(env, book, *, cutoff, event_kind="decision", units=None):
    receipt = book.reserve(
        cutoff=cutoff,
        prefix=prefix(env),
        rows_digest=ROWS,
        units=dict(UNITS if units is None else units),
        event_kind=event_kind,
    )
    archive_step(env, book, receipt)
    book.reconcile(latest(env))
    return receipt


def test_reservation_is_sealed_and_durable_before_any_native_observation(tmp_path):
    with workspace(tmp_path / "run") as env:
        book = ledger(env)
        before = prefix(env)
        book.initialize(before)
        receipt = reserve(book, before)
        assert env.executor.cursor == 0
        assert latest(env)["reservation_id"] is None
        assert receipt["status"] == "reserved"
        assert receipt["ordinal"] == 1
        assert receipt["units"] == UNITS and receipt["observed_units"] is None
        assert len(receipt["reservation_id"]) == 64
        int(receipt["reservation_id"], 16)
        reference = receipt["reference"]
        payload = (env.artifacts.directory / reference["name"]).read_bytes()
        assert len(payload) == reference["bytes"]
        assert hashlib.sha256(payload).hexdigest() == reference["sha256"]
        assert reference in book.references()
        assert only_cohort(book)["charged_units"] == UNITS
        assert 0 < len(index_bytes(env)) <= INDEX_BYTES


def test_reopen_after_before_predictions_failure_keeps_charge_and_unknown_work(tmp_path):
    root = tmp_path / "run"
    with workspace(root) as env:
        book = ledger(env)
        before = prefix(env)
        book.initialize(before)
        receipt = reserve(book, before)

        def fail(boundary):
            if boundary == env.native.Boundary.before_predictions:
                raise RuntimeError("Corte después de reservar y antes de predecir")

        with pytest.raises(RuntimeError, match="antes de predecir"):
            commit(env, receipt, fault=fail)
    with workspace(root, resume=True) as env:
        book = ledger(env)
        book.resume(latest(env))
        state = only_cohort(book)
        assert state["charged_units"] == UNITS
        attempt = state["attempts"][0]
        assert attempt["reservation_id"] == receipt["reservation_id"]
        assert attempt["status"] == "abandoned" and attempt["ordinal"] == 1
        assert attempt["observed_units"] is None
        second = reserve(book, before)
        assert second["ordinal"] == 2
        assert second["reservation_id"] != receipt["reservation_id"]
        assert only_cohort(book)["charged_units"] == {
            key: 2 * value for key, value in UNITS.items()
        }


def test_crash_after_native_commit_reconciles_without_replaying_prediction(tmp_path):
    root = tmp_path / "run"
    with workspace(root) as env:
        book = ledger(env)
        book.initialize(prefix(env))
        receipt = reserve(book, prefix(env))

        def fail(boundary):
            if boundary == env.native.Boundary.committed:
                raise RuntimeError("Corte tras publicar el checkpoint nativo")

        with pytest.raises(RuntimeError, match="checkpoint nativo"):
            commit(env, receipt, fault=fail)
    with workspace(root, resume=True) as env:
        assert env.executor.cursor == 1
        native_before = latest(env)
        book = ledger(env)
        book.resume(native_before)
        assert only_cohort(book)["attempts"][0]["status"] == "confirmed"
        assert only_cohort(book)["charged_units"] == UNITS
        snapshot = deepcopy(book.snapshot())
        book.reconcile(native_before)
        assert book.snapshot() == snapshot
        assert latest(env) == native_before
        assert json.loads(env.executor.snapshot_json())["state"]["fast"] == 1


def test_confirm_alone_does_not_authorize_a_later_cohort_or_refund(tmp_path):
    with workspace(tmp_path / "run") as env:
        book = ledger(env)
        before = prefix(env)
        book.initialize(before)
        receipt = reserve(book, before)
        book.confirm(receipt["reservation_id"], bundle_generation=1)
        with pytest.raises(ValueError):
            reserve(book, dict(before, generation=1, cursor=1), cutoff=20)
        book.reconcile(latest(env))
        assert only_cohort(book)["charged_units"] == UNITS
        assert only_cohort(book)["attempts"][0]["status"] == "abandoned"


def test_only_linked_native_confirmation_allows_next_cohort_to_start_at_one(tmp_path):
    with workspace(tmp_path / "run") as env:
        book = ledger(env, max_attempts=1)
        before = prefix(env)
        book.initialize(before)
        first = reserve(book, before)
        commit(env, first)
        book.confirm(first["reservation_id"], bundle_generation=1)
        book.reconcile(latest(env))
        second = reserve(book, prefix(env), cutoff=20)
        assert second["ordinal"] == 1
        cohorts = book.snapshot()["cohorts"]
        assert len(cohorts) == 2
        assert [item["charged_units"] for item in cohorts] == [UNITS, UNITS]
        assert cohorts[0]["attempts"][0]["status"] == "confirmed"
        assert cohorts[1]["attempts"][0]["status"] == "reserved"


def test_same_cutoff_cannot_reset_ordinal_after_native_confirmation(tmp_path):
    with workspace(tmp_path / "run") as env:
        book = ledger(env)
        book.initialize(prefix(env))
        first = reserve(book, prefix(env))
        commit(env, first)
        book.confirm(first["reservation_id"], bundle_generation=1)
        book.reconcile(latest(env))
        before = index_bytes(env)
        with pytest.raises(ValueError):
            reserve(book, prefix(env), cutoff=10)
        assert index_bytes(env) == before


def test_settlement_can_share_cutoff_after_a_linked_native_decision(tmp_path):
    with workspace(tmp_path / "run", financial=True) as env:
        book = ledger(env)
        book.initialize(prefix(env))
        first = reserve(book, prefix(env))
        commit(env, first)
        book.confirm(first["reservation_id"], bundle_generation=1)
        book.reconcile(latest(env))
        receipt = book.reserve(
            cutoff=10,
            prefix=prefix(env),
            rows_digest=ROWS,
            units=dict.fromkeys(LIMITS, 0),
            event_kind="settlement",
        )
        assert receipt["ordinal"] == 1
        settle(env, receipt)
        book.confirm(receipt["reservation_id"], bundle_generation=2)
        book.reconcile(latest(env))
        cohorts = book.snapshot()["cohorts"]
        assert [item["cutoff"] for item in cohorts] == [10, 10]
        assert [item["event_kind"] for item in cohorts] == ["decision", "settlement"]
        assert len({item["attempts"][0]["reservation_id"] for item in cohorts}) == 2
        assert json.loads(env.executor.snapshot_json())["state"]["fast"] == 2
        assert env.executor.cursor == 2
        with pytest.raises(ValueError):
            reserve(book, prefix(env), cutoff=10)
        with pytest.raises(ValueError, match="orden de la fase"):
            settle(env, receipt)
        assert env.executor.cursor == 2


def test_failed_settlement_at_same_cutoff_keeps_charge_and_ordinal_after_resume(tmp_path):
    root = tmp_path / "run"
    units = dict(mac_observations=0, local_node_evaluations=0, context_commits=1)
    with workspace(root, financial=True) as env:
        book = ledger(env)
        book.initialize(prefix(env))
        first = reserve(book, prefix(env))
        commit(env, first)
        book.confirm(first["reservation_id"], bundle_generation=1)
        book.reconcile(latest(env))
        before = prefix(env)
        # Es una cota reservada, no una escritura de contexto observada.
        receipt = book.reserve(
            cutoff=10, prefix=before, rows_digest=ROWS, units=units, event_kind="settlement"
        )

        def fail(boundary):
            if boundary == env.native.Boundary.before_predictions:
                raise RuntimeError("Corte antes de resolver settlement")

        with pytest.raises(RuntimeError, match="resolver settlement"):
            settle(env, receipt, fault=fail)
    with workspace(root, resume=True, financial=True) as env:
        book = ledger(env)
        book.resume(latest(env))
        state = book.snapshot()["cohorts"][-1]
        assert state["event_kind"] == "settlement" and state["cutoff"] == 10
        assert state["prefix"] == before and state["charged_units"] == units
        assert state["attempts"][0]["status"] == "abandoned"
        assert state["attempts"][0]["observed_units"] is None
        following = book.reserve(
            cutoff=10,
            prefix=before,
            rows_digest=ROWS,
            units=dict.fromkeys(LIMITS, 0),
            event_kind="settlement",
        )
        assert following["ordinal"] == 2
        assert book.snapshot()["cohorts"][-1]["charged_units"] == units
        assert env.executor.cursor == 1


@pytest.mark.parametrize("event_kind", ["unknown", None, True])
def test_unknown_event_kind_is_rejected_before_changing_index_or_blobs(tmp_path, event_kind):
    with workspace(tmp_path / "run") as env:
        book = ledger(env)
        book.initialize(prefix(env))
        before, blobs = index_bytes(env), artifact_bytes(env)
        with pytest.raises(ValueError):
            book.reserve(
                cutoff=10,
                prefix=prefix(env),
                rows_digest=ROWS,
                units=UNITS,
                event_kind=event_kind,
            )
        assert index_bytes(env) == before and artifact_bytes(env) == blobs


def test_active_reservation_cannot_be_replaced_by_a_second_one(tmp_path):
    with workspace(tmp_path / "run") as env:
        book = ledger(env)
        book.initialize(prefix(env))
        reserve(book, prefix(env))
        before = index_bytes(env)
        with pytest.raises(ValueError):
            reserve(book, prefix(env))
        assert index_bytes(env) == before


def test_one_resource_cannot_borrow_unused_units_from_another(tmp_path):
    with workspace(tmp_path / "run") as env:
        book = ledger(env)
        before = prefix(env)
        book.initialize(before)
        reserve(book, before, units=dict(UNITS, local_node_evaluations=10))
        book.reconcile(latest(env))
        index = index_bytes(env)
        with pytest.raises(ValueError):
            reserve(book, before, units=dict(UNITS, local_node_evaluations=1))
        assert index_bytes(env) == index
        assert only_cohort(book)["charged_units"]["local_node_evaluations"] == 10


def test_core_observations_survive_resume_without_borrowing_unused_mac_budget(tmp_path):
    root = tmp_path / "run"
    limits = dict(core_observations=2, mac_observations=8)
    units = dict(core_observations=2, mac_observations=0)
    with workspace(root) as env:
        book = ledger(env, limits=limits)
        before = prefix(env)
        book.initialize(before)
        receipt = reserve(book, before, units=units)
        assert receipt["units"] == units
        assert env.executor.cursor == 0
    with workspace(root, resume=True) as env:
        book = ledger(env, limits=limits)
        book.resume(latest(env))
        state = only_cohort(book)
        assert state["charged_units"] == units
        assert state["attempts"][0]["status"] == "abandoned"
        assert state["attempts"][0]["observed_units"] is None
        index = index_bytes(env)
        with pytest.raises(ValueError):
            reserve(book, before, units=dict(core_observations=1, mac_observations=0))
        assert index_bytes(env) == index
        assert only_cohort(book)["charged_units"] == units


def test_exact_vector_boundary_is_admitted_but_reopen_does_not_renew_it(tmp_path):
    root = tmp_path / "run"
    with workspace(root) as env:
        book = ledger(env)
        before = prefix(env)
        book.initialize(before)
        reserve(book, before)
        book.reconcile(latest(env))
        reserve(
            book,
            before,
            units=dict(mac_observations=2, local_node_evaluations=6, context_commits=1),
        )
        assert only_cohort(book)["charged_units"] == LIMITS
    with workspace(root, resume=True) as env:
        book = ledger(env)
        book.resume(latest(env))
        index = index_bytes(env)
        with pytest.raises(ValueError):
            reserve(book, before)
        assert index_bytes(env) == index
        assert only_cohort(book)["charged_units"] == LIMITS


def test_attempt_limit_also_applies_to_zero_unit_reservations(tmp_path):
    with workspace(tmp_path / "run") as env:
        book = ledger(env, max_attempts=1)
        before = prefix(env)
        book.initialize(before)
        reserve(book, before, units=dict.fromkeys(LIMITS, 0))
        book.reconcile(latest(env))
        with pytest.raises(ValueError):
            reserve(book, before, units=dict.fromkeys(LIMITS, 0))
        assert only_cohort(book)["attempts"][0]["ordinal"] == 1


@pytest.mark.parametrize(
    "units",
    [
        dict(UNITS, mac_observations=True),
        dict(UNITS, mac_observations=1.0),
        dict(UNITS, mac_observations=-1),
        dict(UNITS, arbitrary_resource=1),
        {key: value for key, value in UNITS.items() if key != "context_commits"},
    ],
)
def test_invalid_units_fail_before_changing_index_or_artifacts(tmp_path, units):
    with workspace(tmp_path / "run") as env:
        book = ledger(env)
        book.initialize(prefix(env))
        before, blobs = index_bytes(env), artifact_bytes(env)
        with pytest.raises(ValueError):
            reserve(book, prefix(env), units=units)
        assert index_bytes(env) == before and artifact_bytes(env) == blobs


@pytest.mark.parametrize("limits", [{}, {"arbitrary_resource": 1}, {"mac_observations": True}])
def test_only_a_fixed_subset_of_registered_resource_names_can_be_declared(tmp_path, limits):
    with workspace(tmp_path / "run") as env:
        with pytest.raises(ValueError):
            ledger(env, limits=limits)
        assert not (env.directory / "regional-work.json").exists()


@pytest.mark.parametrize("change", ["rows", "checkpoint", "cursor", "generation"])
def test_same_cohort_cannot_be_renamed_to_escape_its_charge(tmp_path, change):
    with workspace(tmp_path / "run") as env:
        book = ledger(env)
        before = prefix(env)
        book.initialize(before)
        reserve(book, before)
        book.reconcile(latest(env))
        changed = dict(before)
        rows = ROWS
        if change == "rows":
            rows = "e" * 64
        elif change == "checkpoint":
            changed["checkpoint_sha256"] = "e" * 64
        else:
            changed[change] += 1
        index = index_bytes(env)
        with pytest.raises(ValueError):
            reserve(book, changed, rows=rows)
        assert index_bytes(env) == index


def test_unconfirmed_cohort_cannot_be_evaded_by_using_a_later_cutoff(tmp_path):
    with workspace(tmp_path / "run") as env:
        book = ledger(env)
        before = prefix(env)
        book.initialize(before)
        reserve(book, before)
        book.reconcile(latest(env))
        with pytest.raises(ValueError):
            reserve(book, before, cutoff=20)
        assert len(book.snapshot()["cohorts"]) == 1


@pytest.mark.parametrize("field", ["contract_id", "model_id", "program_id", "limits"])
def test_resume_rejects_changed_identity_or_limits_without_reinitializing(tmp_path, field):
    root = tmp_path / "run"
    with workspace(root) as env:
        book = ledger(env)
        book.initialize(prefix(env))
        reserve(book, prefix(env))
    with workspace(root, resume=True) as env:
        before = index_bytes(env)
        value = dict(LIMITS, mac_observations=4) if field == "limits" else "e" * 64
        with pytest.raises(ValueError):
            ledger(env, **{field: value}).resume(latest(env))
        assert index_bytes(env) == before


def test_initialize_never_overwrites_an_existing_reservation(tmp_path):
    with workspace(tmp_path / "run") as env:
        book = ledger(env)
        book.initialize(prefix(env))
        reserve(book, prefix(env))
        before = index_bytes(env)
        with pytest.raises(ValueError):
            ledger(env).initialize(prefix(env))
        assert index_bytes(env) == before


def test_resume_with_missing_index_rejects_even_at_native_cursor_zero(tmp_path):
    with workspace(tmp_path / "run") as env:
        assert env.executor.cursor == 0
        with pytest.raises((ValueError, FileNotFoundError)):
            ledger(env).resume(latest(env))
        assert not (env.directory / "regional-work.json").exists()


@pytest.mark.parametrize("fault", ["truncated", "oversized_valid_json", "empty_state"])
def test_damaged_or_oversized_index_never_creates_a_fresh_balance(tmp_path, fault):
    with workspace(tmp_path / "run") as env:
        book = ledger(env)
        book.initialize(prefix(env))
        reserve(book, prefix(env))
        path = env.directory / "regional-work.json"
        content = index_bytes(env)
        if fault == "truncated":
            damaged = content[: len(content) // 2]
        elif fault == "oversized_valid_json":
            damaged = content + b" " * (INDEX_BYTES + 1 - len(content))
            json.loads(damaged)
        else:
            damaged = b"{}\n"
        path.write_bytes(damaged)
        with pytest.raises(ValueError):
            ledger(env).resume(latest(env))
        with pytest.raises(ValueError):
            ledger(env).initialize(prefix(env))
        assert path.read_bytes() == damaged


def test_missing_or_corrupt_reserved_blob_is_not_reinterpreted_as_zero_work(tmp_path):
    with workspace(tmp_path / "run") as env:
        book = ledger(env)
        book.initialize(prefix(env))
        receipt = reserve(book, prefix(env))
        path = env.artifacts.directory / receipt["reference"]["name"]
        content = path.read_bytes()
        for corrupt in (True, False):
            if corrupt:
                path.write_bytes(b"contenido alterado")
            else:
                path.unlink()
            with pytest.raises((ValueError, FileNotFoundError)):
                ledger(env).resume(latest(env))
            path.write_bytes(content)
        restored = ledger(env)
        restored.resume(latest(env))
        assert only_cohort(restored)["charged_units"] == UNITS


def test_references_keep_abandoned_attempts_alive_before_pruning(tmp_path):
    with workspace(tmp_path / "run") as env:
        book = ledger(env)
        before = prefix(env)
        book.initialize(before)
        first = reserve(book, before)
        book.reconcile(latest(env))
        second = reserve(book, before)
        orphan = env.artifacts.stage(
            dict(value=0), identity=IDENTITY["contract_id"], kind="fixture"
        )
        references = book.references()
        assert first["reference"] in references and second["reference"] in references
        env.artifacts.prune_unreferenced(references)
        assert not (env.artifacts.directory / orphan["name"]).exists()
        recovered = ledger(env)
        recovered.resume(latest(env))
        assert only_cohort(recovered)["charged_units"] == {
            key: 2 * value for key, value in UNITS.items()
        }


def test_lexical_path_alias_reopens_the_same_charge_and_ordinal(tmp_path):
    with workspace(tmp_path / "run") as env:
        (env.directory / "unused").mkdir()
        book = ledger(env)
        before = prefix(env)
        book.initialize(before)
        reserve(book, before)
        alias = ledger(env, directory=env.directory / "unused" / "..")
        alias.resume(latest(env))
        assert only_cohort(alias)["charged_units"] == UNITS
        assert reserve(alias, before)["ordinal"] == 2
        assert (env.directory / "regional-work.json").is_file()


def test_symbolic_path_alias_is_rejected_without_erasing_the_original_charge(tmp_path):
    with workspace(tmp_path / "run") as env:
        book = ledger(env)
        book.initialize(prefix(env))
        reserve(book, prefix(env))
        alias = tmp_path / "alias"
        alias.symlink_to(env.directory, target_is_directory=True)
        before = index_bytes(env)
        with pytest.raises(ValueError):
            ledger(env, directory=alias).resume(latest(env))
        assert index_bytes(env) == before


@pytest.mark.parametrize("field", ["reservation_id", "generation", "cursor"])
def test_latest_must_link_the_reserved_attempt_and_the_expected_generation(tmp_path, field):
    with workspace(tmp_path / "run") as env:
        book = ledger(env)
        book.initialize(prefix(env))
        receipt = reserve(book, prefix(env))
        commit(env, receipt)
        changed = latest(env)
        changed[field] = "f" * 64 if field == "reservation_id" else changed[field] + 1
        before = index_bytes(env)
        with pytest.raises(ValueError):
            book.reconcile(changed)
        assert index_bytes(env) == before


@pytest.mark.parametrize("generation", [0, 2, True])
def test_confirmation_rejects_a_wrong_generation_without_changing_charge(tmp_path, generation):
    with workspace(tmp_path / "run") as env:
        book = ledger(env)
        book.initialize(prefix(env))
        receipt = reserve(book, prefix(env))
        before = index_bytes(env)
        with pytest.raises(ValueError):
            book.confirm(receipt["reservation_id"], bundle_generation=generation)
        assert index_bytes(env) == before


def test_ledger_reuses_the_existing_output_lock_without_acquiring_another(tmp_path):
    with workspace(tmp_path / "run") as env:
        book = ledger(env)
        book.initialize(prefix(env))
        reserve(book, prefix(env))
        with pytest.raises(RuntimeError, match="otra ejecución activa"):
            with workspace(env.directory, resume=True):
                pytest.fail("El ejecutor nativo permitió dos propietarios del mismo directorio")
        assert only_cohort(book)["charged_units"] == UNITS


def test_archive_payload_keeps_sealed_receipts_without_confirming_the_attempt(tmp_path):
    with workspace(tmp_path / "run", financial=True, history=True) as env:
        book = ledger(env)
        book.initialize(prefix(env))
        receipt = reserve(book, prefix(env))
        before = index_bytes(env)
        payload = book.archive_payload(receipt["reservation_id"])
        assert index_bytes(env) == before
        assert payload["reservation_id"] == receipt["reservation_id"]
        assert payload["reference"] == receipt["reference"]
        assert "confirmed_prefix" not in payload["cohort"]
        assert payload["cohort"]["attempts"][-1]["status"] == "confirmed"
        assert payload["cohort"]["attempts"][-1]["bundle_generation"] == 1
        assert only_cohort(book)["attempts"][-1]["status"] == "reserved"
        assert payload["receipts"] == [
            env.artifacts.read(
                receipt["reference"],
                identity=IDENTITY["contract_id"],
                kind="regional_work_reservation",
            )
        ]
        assert payload["receipts"][0]["history_before"]["cohorts"] == 0
        assert payload["cohort"]["history_before"] == payload["receipts"][0]["history_before"]
        assert (
            book.verify_archive_payload(
                payload, generation=1, cutoff=10, rows_digest=ROWS, event_kind="decision"
            )
            == payload
        )


@pytest.mark.parametrize("change", ["units", "rows", "previous", "status", "summary"])
def test_archive_payload_rejects_changed_inline_evidence_without_reading_old_blobs(
    tmp_path, change
):
    with workspace(tmp_path / "run", financial=True, history=True) as env:
        book = ledger(env)
        book.initialize(prefix(env))
        receipt = reserve(book, prefix(env))
        payload = book.archive_payload(receipt["reservation_id"])
        if change == "units":
            payload["receipts"][0]["units"]["mac_observations"] += 1
        elif change == "rows":
            payload["cohort"]["rows_digest"] = "e" * 64
        elif change == "previous":
            payload["receipts"][0]["previous_reservation_id"] = "f" * 64
        elif change == "status":
            payload["cohort"]["attempts"][-1]["status"] = "abandoned"
        else:
            payload["receipts"][0]["history_before"]["attempts"] += 1
        (env.artifacts.directory / receipt["reference"]["name"]).unlink()
        before = index_bytes(env)
        with pytest.raises(ValueError):
            book.verify_archive_payload(
                payload, generation=1, cutoff=10, rows_digest=ROWS, event_kind="decision"
            )
        assert index_bytes(env) == before


def test_compaction_keeps_600_events_recoverable_with_a_bounded_active_index(tmp_path, monkeypatch):
    root = tmp_path / "run"
    reservations, abandoned, receipt_reads = [], [], []
    for segment in range(6):
        with workspace(root, resume=segment > 0, financial=True, history=True) as env:
            book = ledger(env, max_index_bytes=COMPACT_INDEX_BYTES)
            assert hasattr(book, "compact"), "Falta compactar el prefijo archivado del ledger"
            if segment:
                book.resume(latest(env))
            else:
                book.initialize(prefix(env))
            original_read = env.artifacts.read
            reads = []

            def counted_read(reference, *, identity, kind, _reads=reads, _read=original_read):
                if kind == "regional_work_reservation":
                    _reads.append(reference["bytes"])
                return _read(reference, identity=identity, kind=kind)

            monkeypatch.setattr(env.artifacts, "read", counted_read)
            for generation in range(100 * segment + 1, 100 * (segment + 1) + 1):
                reads.clear()
                cutoff = 10 * ((generation + 1) // 2)
                event_kind = "decision" if generation % 2 else "settlement"
                units = dict(UNITS) if generation % 2 else dict.fromkeys(LIMITS, 0)
                if generation in (101, 201, 301, 401, 501):
                    failed = book.reserve(
                        cutoff=cutoff,
                        prefix=prefix(env),
                        rows_digest=ROWS,
                        units=units,
                        event_kind=event_kind,
                    )
                    abandoned.append(failed["reservation_id"])
                    book.reconcile(latest(env))
                receipt = archive_event(
                    env, book, cutoff=cutoff, event_kind=event_kind, units=units
                )
                reservations.append(receipt["reservation_id"])
                assert receipt["ordinal"] == (2 if generation in (101, 201, 301, 401, 501) else 1)
                proof = archive_proof(env)
                book.compact(latest(env), archive=proof)
                state = book.snapshot()
                assert len(state["cohorts"]) <= 2
                assert len(index_bytes(env)) <= COMPACT_INDEX_BYTES
                assert len(book.references()) <= 5
                prune_archived(env, book)
                assert len(list(env.artifacts.directory.iterdir())) <= 24
                receipt_reads.append(len(reads))
            assert env.executor.cursor == 100 * (segment + 1)
            monkeypatch.setattr(env.artifacts, "read", original_read)

    with workspace(root, resume=True, financial=True, history=True) as env:
        book = ledger(env, max_index_bytes=COMPACT_INDEX_BYTES)
        book.resume(latest(env))
        state = book.snapshot()
        assert state["archived"]["cohorts"] == 598
        assert state["archived"]["attempts"] == 603
        assert state["archived"]["charged_units"] == dict(
            mac_observations=304, local_node_evaluations=1216, context_commits=0
        )
        assert state["archived"]["last_reservation_id"] == reservations[597]
        assert state["archived"]["frontier"]["cutoff"] == 2990
        assert state["archived"]["frontier"]["event_kind"] == "settlement"
        assert [cohort["cutoff"] for cohort in state["cohorts"]] == [3000, 3000]
        assert len(state["archived"]["sha256"]) == 64
        manifest = confirmed_history(env)
        assert env.history.verify(manifest, strong=True)["events"] == 600
        confirmed_ids, abandoned_ids, archived_receipts = [], [], []
        for event in env.history.iter_events(manifest):
            payload = event["payload"]["regional_work"]
            assert (
                book.verify_archive_payload(
                    payload,
                    generation=event["generation"],
                    cutoff=event["cutoff"],
                    rows_digest=ROWS,
                    event_kind=event["kind"],
                )
                == payload
            )
            confirmed_ids.append(payload["reservation_id"])
            abandoned_ids.extend(
                attempt["reservation_id"]
                for attempt in payload["cohort"]["attempts"]
                if attempt["status"] == "abandoned"
            )
            archived_receipts.extend(payload["receipts"])
        assert confirmed_ids == reservations and abandoned_ids == abandoned
        assert len(archived_receipts) == 605
        assert all(
            attempt["observed_units"] is None
            for event in env.history.iter_events(manifest)
            for attempt in event["payload"]["regional_work"]["cohort"]["attempts"]
        )
        first = env.history.read_event(manifest, 1)["payload"]["regional_work"]
        assert not (env.artifacts.directory / first["reference"]["name"]).exists()
        assert env.executor.cursor == 600
    assert len(receipt_reads) == 600 and max(receipt_reads) <= 96


@pytest.mark.parametrize("field", ["generation", "reservation_id", "reference"])
def test_compaction_rejects_a_foreign_archive_without_removing_receipts(tmp_path, field):
    with workspace(tmp_path / "run", financial=True, history=True) as env:
        book = ledger(env)
        book.initialize(prefix(env))
        archive_event(env, book, cutoff=10)
        old_proof = archive_proof(env)
        archive_event(env, book, cutoff=20)
        archive_event(env, book, cutoff=30)
        proof = archive_proof(env)
        proof[field] = old_proof[field]
        before, blobs = index_bytes(env), artifact_bytes(env)
        with pytest.raises(ValueError):
            book.compact(latest(env), archive=proof)
        assert index_bytes(env) == before and artifact_bytes(env) == blobs


def test_staged_history_without_native_commit_cannot_authorize_compaction(tmp_path):
    with workspace(tmp_path / "run", financial=True, history=True) as env:
        book = ledger(env)
        book.initialize(prefix(env))
        archive_event(env, book, cutoff=10)
        archive_event(env, book, cutoff=20)
        receipt = reserve(book, prefix(env), cutoff=30)

        def fail(boundary):
            if boundary == env.native.Boundary.before_commit:
                raise RuntimeError("Corte con historial preparado sin confirmar")

        with pytest.raises(RuntimeError, match="sin confirmar"):
            archive_step(env, book, receipt, fault=fail)
        proof = dict(
            reference=env.link["staged_history"]["chunks"][-1]["reference"],
            generation=3,
            reservation_id=receipt["reservation_id"],
        )
        before, blobs = index_bytes(env), artifact_bytes(env)
        with pytest.raises(ValueError):
            book.compact(latest(env), archive=proof)
        assert index_bytes(env) == before and artifact_bytes(env) == blobs


def test_compaction_preserves_an_abandoned_active_cohort_and_its_quota(tmp_path):
    root = tmp_path / "run"
    with workspace(root, financial=True, history=True) as env:
        book = ledger(env)
        book.initialize(prefix(env))
        for cutoff in (10, 20, 30):
            archive_event(env, book, cutoff=cutoff)
        proof = archive_proof(env)
        failed = reserve(book, prefix(env), cutoff=40)
        book.reconcile(latest(env))
        book.compact(latest(env), archive=proof)
        assert len(book.snapshot()["cohorts"]) == 3
        assert book.snapshot()["archived"]["cohorts"] == 1
        prune_archived(env, book)
    with workspace(root, resume=True, financial=True, history=True) as env:
        book = ledger(env)
        book.resume(latest(env))
        active = book.snapshot()["cohorts"][-1]
        assert active["attempts"][0]["reservation_id"] == failed["reservation_id"]
        assert active["attempts"][0]["status"] == "abandoned"
        assert active["charged_units"] == UNITS
        receipt = reserve(book, prefix(env), cutoff=40)
        assert receipt["ordinal"] == 2
        archive_step(env, book, receipt)
        book.reconcile(latest(env))
        book.compact(latest(env), archive=archive_proof(env))
        assert book.snapshot()["cohorts"][-1]["charged_units"] == {
            name: value * 2 for name, value in UNITS.items()
        }


@pytest.mark.parametrize("field", ["cohorts", "attempts", "charged_units", "sha256", "frontier"])
def test_archived_accumulators_must_match_the_retained_sealed_receipt(tmp_path, field):
    with workspace(tmp_path / "run", financial=True, history=True) as env:
        book = ledger(env)
        book.initialize(prefix(env))
        for cutoff in (10, 20, 30):
            archive_event(env, book, cutoff=cutoff)
        book.compact(latest(env), archive=archive_proof(env))
        state = json.loads(index_bytes(env))
        if field in ("cohorts", "attempts"):
            state["archived"][field] += 1
        elif field == "charged_units":
            state["archived"][field]["mac_observations"] += 1
        elif field == "sha256":
            state["archived"][field] = "0" * 64
        else:
            state["archived"][field]["cutoff"] += 1
        (env.directory / "regional-work.json").write_text(json.dumps(state))
        with pytest.raises(ValueError):
            book.snapshot()


@pytest.mark.parametrize("operation", ["snapshot", "resume"])
def test_compacted_archive_must_contain_its_declared_native_event(tmp_path, operation):
    with workspace(tmp_path / "run", financial=True, history=True) as env:
        book = ledger(env)
        book.initialize(prefix(env))
        archive_event(env, book, cutoff=10)
        previous = archive_proof(env)
        archive_event(env, book, cutoff=20)
        archive_event(env, book, cutoff=30)
        book.compact(latest(env), archive=archive_proof(env))
        state = json.loads(index_bytes(env))
        assert previous["reference"] != state["archive"]["reference"]
        assert (env.artifacts.directory / previous["reference"]["name"]).is_file()
        state["archive"]["reference"] = previous["reference"]
        (env.directory / "regional-work.json").write_text(json.dumps(state))
        before, blobs = index_bytes(env), artifact_bytes(env)
        with pytest.raises(ValueError, match="generación enlazada"):
            if operation == "snapshot":
                book.snapshot()
            else:
                book.resume(latest(env))
        assert index_bytes(env) == before and artifact_bytes(env) == blobs


def test_failure_before_index_replacement_keeps_the_uncompacted_charge(tmp_path, monkeypatch):
    with workspace(tmp_path / "run", financial=True, history=True) as env:
        book = ledger(env)
        book.initialize(prefix(env))
        for cutoff in (10, 20, 30):
            archive_event(env, book, cutoff=cutoff)
        proof = archive_proof(env)
        before, blobs = index_bytes(env), artifact_bytes(env)
        module = importlib.import_module("mars_titan.memory.regional_work")

        def fail_replace(source, destination):
            raise OSError("Corte antes del reemplazo del índice")

        with monkeypatch.context() as patch:
            patch.setattr(module.os, "replace", fail_replace)
            with pytest.raises(OSError, match="antes del reemplazo"):
                book.compact(latest(env), archive=proof)
        assert index_bytes(env) == before and artifact_bytes(env) == blobs
        assert len(book.snapshot()["cohorts"]) == 3
        book.compact(latest(env), archive=proof)
        assert len(book.snapshot()["cohorts"]) == 2


def test_reopen_after_compaction_before_pruning_preserves_receipt_ids(tmp_path):
    root = tmp_path / "run"
    with workspace(root, financial=True, history=True) as env:
        book = ledger(env)
        book.initialize(prefix(env))
        receipts = [archive_event(env, book, cutoff=cutoff) for cutoff in (10, 20, 30)]
        book.compact(latest(env), archive=archive_proof(env))
        assert (env.artifacts.directory / receipts[0]["reference"]["name"]).is_file()
    with workspace(root, resume=True, financial=True, history=True) as env:
        book = ledger(env)
        book.resume(latest(env))
        before = deepcopy(book.snapshot())
        assert [cohort["attempts"][-1]["reservation_id"] for cohort in before["cohorts"]] == [
            receipt["reservation_id"] for receipt in receipts[1:]
        ]
        book.compact(latest(env), archive=archive_proof(env))
        assert book.snapshot() == before
        prune_archived(env, book)
        assert not (env.artifacts.directory / receipts[0]["reference"]["name"]).exists()
        assert (
            env.history.read_event(confirmed_history(env), 1)["payload"]["regional_work"][
                "reservation_id"
            ]
            == receipts[0]["reservation_id"]
        )
        assert env.executor.cursor == 3
