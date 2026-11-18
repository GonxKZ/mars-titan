"""La caché opcional alimenta al lector real sin adelantar etiquetas ni alterar el cursor."""

import json

import pytest

from benchmarks.comparison_input_archive import fingerprint
from mars_titan.memory.training_event_cache import PackedTrainingSource, source_implementation
from tests.training.test_financial_run import corpus

ENV = "MARS_TITAN_TRAIN_ARCHIVE"


def test_training_uses_verified_copy_on_repeat_and_resume(tmp_path, monkeypatch):
    _, sources = corpus(tmp_path / "src")
    src = sources["train"]
    spec = src.specification()
    ref = {i: fingerprint(src.batched_events(start_cursor=i, block_rows=4), spec) for i in (0, 1)}
    root = tmp_path / "cache"
    monkeypatch.setenv(ENV, str(root))
    src = PackedTrainingSource(src.dataset, src.path, root)
    assert fingerprint(src.batched_events(block_rows=4), spec) == ref[0]
    manifest = json.loads((root / "small.json").read_text())
    assert manifest["verified"] == ref[0] and manifest["index"] == src.identity
    assert fingerprint(src.batched_events(start_cursor=1, block_rows=4), spec) == ref[1]
    assert source_implementation(src)
    assert source_implementation(src).keys() == {
        "mars_titan.memory.event_archive",
        "mars_titan.memory.training_event_cache",
    }
    assert not source_implementation(sources["validation"])


def test_damaged_optional_copy_is_rebuilt_without_touching_the_source(tmp_path, monkeypatch):
    _, sources = corpus(tmp_path / "src")
    source = sources["train"]
    spec = source.specification()
    root = tmp_path / "cache"
    monkeypatch.setenv(ENV, str(root))
    source = PackedTrainingSource(source.dataset, source.path, root)
    original = fingerprint(source.batched_events(block_rows=4), spec)
    path = root / "small.arrow"
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
    source = PackedTrainingSource(source.dataset, source.path, root)
    with monkeypatch.context() as constrained:
        constrained.setattr(cache, "_budget", lambda folder: 0)
        assert fingerprint(source.batched_events(block_rows=4), spec) == original
    with (root / "small.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert fingerprint(source.batched_events(block_rows=4), spec) == original
    assert not (root / "small.arrow").exists()
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
        type(runs["train"]) is PackedTrainingSource
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
    source = PackedTrainingSource(data, origins["train"].path, tmp_path / "cache")
    ref = fingerprint(source.batched_events(block_rows=4), source.specification())
    (tmp_path / "cache" / "small.arrow").unlink()
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


def test_queue_provides_private_storage_and_respects_explicit_settings(tmp_path, monkeypatch):
    from mars_titan.training import comparison_campaign as c

    layout = c.Layout(tmp_path / "state", tmp_path / "views")
    decl, _ = c.load_declaration()
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv("MARS_TITAN_CUDA_HOST_FINITE", raising=False)
    env = c._environment(layout, decl)
    assert (
        env[ENV] == str(layout.state / "packed-inputs")
        and env["MARS_TITAN_CUDA_HOST_FINITE"] == "1"
    )
    monkeypatch.setenv(ENV, "")
    monkeypatch.setenv("MARS_TITAN_CUDA_HOST_FINITE", "0")
    env = c._environment(layout, decl)
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
