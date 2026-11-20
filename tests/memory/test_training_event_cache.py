"""La caché opcional alimenta al lector real sin adelantar etiquetas ni alterar el cursor."""

import json

import pytest

from benchmarks.comparison_input_archive import fingerprint
from mars_titan.memory.training_event_cache import (
    STORED_BLOCK,
    PackedPhaseSource,
    source_implementation,
)
from tests.training.test_financial_run import corpus

ENV = "MARS_TITAN_TRAIN_ARCHIVE"


def test_training_uses_verified_copy_on_repeat_and_resume(tmp_path, monkeypatch):
    _, sources = corpus(tmp_path / "src")
    src = sources["train"]
    spec = src.specification()
    ref = {i: fingerprint(src.batched_events(start_cursor=i, block_rows=4), spec) for i in (0, 1)}
    stored = fingerprint(src.batched_events(block_rows=STORED_BLOCK), spec)
    root = tmp_path / "cache"
    monkeypatch.setenv(ENV, str(root))
    src = PackedPhaseSource(src.dataset, src.path, root)
    assert fingerprint(src.batched_events(block_rows=4), spec) == ref[0]
    manifest = json.loads((root / "events.json").read_text())
    assert manifest["verified"] == stored and manifest["index"] == src.identity
    assert fingerprint(src.batched_events(start_cursor=1, block_rows=4), spec) == ref[1]
    assert source_implementation(src)
    assert source_implementation(src).keys() == {
        "mars_titan.memory.event_archive",
        "mars_titan.memory.training_event_cache",
    }
    assert not source_implementation(sources["validation"])


def test_measured_phases_replay_their_own_verified_copy(tmp_path, monkeypatch):
    from mars_titan.memory import training_event_cache as cache
    from mars_titan.memory.financial_observations import FinancialObservationSource
    from mars_titan.training.window_indices import phase_sources

    data, original = corpus(tmp_path / "source")
    phases = {k: s.phase for k, s in original.items()}
    monkeypatch.setenv(ENV, str(tmp_path / "cache"))
    packed = phase_sources(data, phases, tmp_path / "index", train_cache=True, predicted_cache=True)
    plain = phase_sources(data, phases, tmp_path / "index")
    assert all(type(source) is PackedPhaseSource for source in packed.values())
    assert all(type(source) is FinancialObservationSource for source in plain.values())
    for name, source in packed.items():
        spec = plain[name].specification()
        reference = fingerprint(plain[name].batched_events(block_rows=4), spec)
        assert source.identity == plain[name].identity
        cache.reset_usage()
        assert fingerprint(source.batched_events(block_rows=4), spec) == reference
        assert fingerprint(source.batched_events(block_rows=4), spec) == reference
        report = cache.receipt()
        assert report["builds"] == 1 and report["packed_passes"] == 2
        folder = tmp_path / "cache" / source.phase.partition
        manifest = json.loads((folder / "events.json").read_text())
        stored = fingerprint(plain[name].batched_events(block_rows=STORED_BLOCK), spec)
        assert manifest["index"] == source.identity and manifest["verified"] == stored


def test_specification_is_stored_next_to_the_index_and_reused_without_reading(
    tmp_path, monkeypatch
):
    from mars_titan.memory import training_event_cache as cache
    from mars_titan.memory.financial_observations import FinancialObservationSource

    _, sources = corpus(tmp_path / "src")
    for name, plain in sources.items():
        first = PackedPhaseSource(plain.dataset, plain.path, tmp_path / "cache" / name)
        expected = plain.specification()
        assert first.specification().fingerprint() == expected.fingerprint()
        stored = json.loads((plain.path.parent / cache.SPECIFICATION).read_text())
        assert stored["identity"]["view_sha256"] == plain.identity
        assert stored["dimensions"] == expected.dimensions
        second = PackedPhaseSource(plain.dataset, plain.path, tmp_path / "cache" / name)
        with monkeypatch.context() as unread:

            def refuse(self):
                raise AssertionError("La especificación guardada no debe leer eventos")

            unread.setattr(FinancialObservationSource, "events", refuse)
            assert second.specification().fingerprint() == expected.fingerprint()
            assert second.specification() is second.specification()
        stored["identity"]["view_sha256"] = "0" * 64
        (plain.path.parent / cache.SPECIFICATION).write_text(json.dumps(stored))
        third = PackedPhaseSource(plain.dataset, plain.path, tmp_path / "cache" / name)
        assert third.specification().fingerprint() == expected.fingerprint()


def test_damaged_optional_copy_is_rebuilt_without_touching_the_source(tmp_path, monkeypatch):
    _, sources = corpus(tmp_path / "src")
    source = sources["train"]
    spec = source.specification()
    root = tmp_path / "cache"
    monkeypatch.setenv(ENV, str(root))
    source = PackedPhaseSource(source.dataset, source.path, root)
    original = fingerprint(source.batched_events(block_rows=4), spec)
    path = root / "events.arrow"
    with path.open("r+b") as stream:
        stream.write(b"invalid")
    assert fingerprint(source.batched_events(block_rows=4), spec) == original
    assert fingerprint(source.batched_events(block_rows=4), spec) == original


def test_space_or_another_builder_keeps_the_original_cursor(tmp_path, monkeypatch):
    import fcntl

    from mars_titan.memory import training_event_cache as cache

    _, sources = corpus(tmp_path / "src")
    source = sources["train"]
    spec = source.specification()
    root = tmp_path / "cache"
    original = fingerprint(source.batched_events(block_rows=4), spec)
    monkeypatch.setenv(ENV, str(root))
    source = PackedPhaseSource(source.dataset, source.path, root)
    with monkeypatch.context() as constrained:
        constrained.setattr(cache, "_budget", lambda folder: 0)
        assert fingerprint(source.batched_events(block_rows=4), spec) == original
    with (root / "events.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert fingerprint(source.batched_events(block_rows=4), spec) == original
    assert not (root / "events.arrow").exists()
    with pytest.raises(ValueError, match="cursor"):
        list(source.batched_events(start_cursor=-1, block_rows=4))


def test_factory_preserves_the_original_index_and_records_the_actual_backend(tmp_path, monkeypatch):
    from mars_titan.memory import training_event_cache as cache
    from mars_titan.memory.financial_observations import FinancialObservationSource
    from mars_titan.training.window_indices import phase_sources

    data, original = corpus(tmp_path / "source")
    phases = {k: s.phase for k, s in original.items()}
    monkeypatch.setenv(ENV, str(tmp_path / "cache"))
    runs = phase_sources(data, phases, tmp_path / "index", train_cache=True)
    normal = phase_sources(data, phases, tmp_path / "index", train_cache=False)
    assert (
        type(runs["train"]) is PackedPhaseSource
        and type(runs["validation"]) is FinancialObservationSource
    )
    assert type(normal["train"]) is FinancialObservationSource
    assert runs["train"].identity == normal["train"].identity
    cache.reset_usage()
    assert list(runs["train"].batched_events(block_rows=4))
    assert list(runs["train"].batched_events(start_cursor=1, block_rows=4))
    report = cache.receipt()
    assert report["builds"] == 1 and report["packed_passes"] == 2 and report["verified_rows"] > 0
    assert report["index_sha256"] == runs["train"].identity


def test_invalid_archive_never_masks_a_changed_source(tmp_path, monkeypatch):
    from mars_titan.memory import event_archive

    data, origins = corpus(tmp_path / "source")
    source = PackedPhaseSource(data, origins["train"].path, tmp_path / "cache")
    ref = fingerprint(source.batched_events(block_rows=4), source.specification())
    (tmp_path / "cache" / "events.arrow").unlink()
    with monkeypatch.context() as no_room:

        def quota(*args, **kwargs):
            import errno

            raise OSError(errno.ENOSPC, "Disco lleno")

        no_room.setattr(event_archive, "write", quota)
        assert fingerprint(source.batched_events(block_rows=4), source.specification()) == ref

    def changed():
        raise ValueError("El origen ha cambiado")

    monkeypatch.setattr(source, "_confirm", changed)
    with pytest.raises(ValueError, match="origen ha cambiado"):
        list(source.batched_events(block_rows=4))


@pytest.mark.parametrize("stored", [STORED_BLOCK, 7])
def test_one_copy_serves_every_block_size_with_the_original_blocks(tmp_path, monkeypatch, stored):
    """El ajuste y el lector leen la misma copia con bloques distintos, sin reconstruirla.

    Con un bloque guardado de 7 filas, los bloques de 2, 3 y 4 no lo dividen, y el de 2 parte
    los instantes de tres filas: la copia los reproduce igual que el lector original.
    """
    from mars_titan.memory import training_event_cache as cache

    monkeypatch.setattr(cache, "STORED_BLOCK", stored)
    _, sources = corpus(tmp_path / "src")
    plain = sources["train"]
    spec = plain.specification()
    sizes = (1, 2, 3, 4, 7, STORED_BLOCK)
    reference = {size: fingerprint(plain.batched_events(block_rows=size), spec) for size in sizes}
    largest = max(
        len(block["sample_ids"])
        for event in plain.batched_events(block_rows=STORED_BLOCK)
        for block in event.inputs
    )
    assert 2 < largest <= 7
    root = tmp_path / "cache"
    packed = PackedPhaseSource(plain.dataset, plain.path, root)
    cache.reset_usage()
    for size in sizes:
        assert fingerprint(packed.batched_events(block_rows=size), spec) == reference[size]
    report = cache.receipt()
    assert report["builds"] == 1 and report["packed_passes"] == len(sizes)
    assert "block_fallbacks" not in report
    stamp = json.loads((root / "events.json").read_text())
    assert stamp["block"] == stored and stamp["split_events"] == 0
    assert stamp["verified"] == reference[stored]


def test_a_copy_with_split_instants_serves_only_the_blocks_that_divide_it(tmp_path, monkeypatch):
    """Si un instante ocupa varias filas guardadas, otro bloque lee el original sin reconstruir."""
    from mars_titan.memory import event_archive
    from mars_titan.memory import training_event_cache as cache

    _, sources = corpus(tmp_path / "src")
    plain = sources["train"]
    spec = plain.specification()
    monkeypatch.setattr(cache, "STORED_BLOCK", 2)
    root = tmp_path / "cache"
    packed = PackedPhaseSource(plain.dataset, plain.path, root)
    cache.reset_usage()
    for size in (2, 1, 3):
        expected = fingerprint(plain.batched_events(block_rows=size), spec)
        assert fingerprint(packed.batched_events(block_rows=size), spec) == expected
    report = cache.receipt()
    assert report["builds"] == 1 and report["packed_passes"] == 2
    assert report["block_fallbacks"] == 1
    stamp = json.loads((root / "events.json").read_text())
    assert stamp["block"] == 2 and stamp["split_events"] > 0
    assert event_archive.compatible(stamp, 1) and not event_archive.compatible(stamp, 3)
    with pytest.raises(ValueError, match="bloque"):
        list(event_archive.replay(plain, spec, root / "events.arrow", stamp, cursor=0, block=3))


def test_queue_provides_private_storage_and_respects_explicit_settings(tmp_path, monkeypatch):
    from mars_titan.training import comparison_campaign as c

    layout = c.Layout(tmp_path / "state", tmp_path / "views")
    decl, _ = c.load_declaration()
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv("MARS_TITAN_CUDA_HOST_FINITE", raising=False)
    env = c.job_environment(layout, decl)
    assert (
        env[ENV] == str(layout.state / "packed-inputs")
        and env["MARS_TITAN_CUDA_HOST_FINITE"] == "1"
    )
    monkeypatch.setenv(ENV, "")
    monkeypatch.setenv("MARS_TITAN_CUDA_HOST_FINITE", "0")
    env = c.job_environment(layout, decl)
    assert env[ENV] == "" and env["MARS_TITAN_CUDA_HOST_FINITE"] == "0"


@pytest.mark.parametrize(("model", "expected"), [("titans_mac", True), ("mars_titan_m2", False)])
def test_job_registers_only_the_qualified_kernel_scope(tmp_path, monkeypatch, model, expected):
    from mars_titan.models.titans.host_finite import enabled
    from mars_titan.training import comparison_campaign as c

    decl, _ = c.load_declaration()
    layout = c.Layout(tmp_path / "state", tmp_path / "views")
    job = next(j for j in c.plan(decl, "references") if j.model == model)
    monkeypatch.setenv(ENV, str(tmp_path / "cache"))
    monkeypatch.setenv("MARS_TITAN_CUDA_HOST_FINITE", "1")

    def execute(*args, **kwargs):
        assert enabled() is expected
        return {"status": "paused"}

    monkeypatch.setattr(c, "execute_job", execute)
    assert c.run_job(decl, layout, job) == c.PAUSED
    attempt = c.read_record(layout, job.key)["attempts"][-1]
    assert ("finite_flags" in attempt) is expected and not enabled()
    assert set(attempt["training_input_archive"]["implementation"]) == {
        "mars_titan.memory.event_archive",
        "mars_titan.memory.training_event_cache",
    }
