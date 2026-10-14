"""Índices de observaciones y escalas M3 compartidos por vista (`training.window_indices`)."""

import threading
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.memory import financial_observations
from mars_titan.training import candidate_walk_forward as candidate
from mars_titan.training import mars_titan_walk_forward as readouts
from mars_titan.training import masked_campaign as engine
from mars_titan.training import titans_walk_forward as wf
from mars_titan.training import window_indices
from mars_titan.training.corpus_inputs import DIGEST_CACHE_ENV, CorpusDataset
from mars_titan.training.window_indices import SHARED_INDICES_ENV
from tests.training.test_titans_walk_forward import views


@pytest.fixture(scope="module")
def view(tmp_path_factory):
    path, protocol = views(tmp_path_factory.mktemp("window-indices"))
    _, _, _, fold = wf._protocol(str(protocol), "fold-000", 42)
    return SimpleNamespace(path=path, phases=wf.window_phases(fold, 1))


@pytest.fixture
def dataset(view):
    return CorpusDataset(view.path, input_policy=HISTORICAL_MASKED)


@pytest.fixture
def shared(tmp_path, monkeypatch):
    root = tmp_path / "shared-indices"
    monkeypatch.setenv(SHARED_INDICES_ENV, str(root))
    return root


@pytest.fixture
def builds(monkeypatch):
    """Índices construidos de verdad: llamadas a `prepare_observation_index` sin manifiesto."""
    built, prepare = [], financial_observations.prepare_observation_index

    def counted(dataset, output, **options):
        if not (output / "manifest.json").exists():
            built.append(output.name)
        return prepare(dataset, output, **options)

    monkeypatch.setattr(window_indices, "prepare_observation_index", counted)
    return built


def test_without_the_shared_root_each_job_keeps_its_indices_in_its_folder(dataset, tmp_path):
    assert window_indices.shared_root() is None
    assert window_indices.index_root(dataset, tmp_path / "job") == tmp_path / "job" / "indices"
    assert window_indices.shared_file(dataset, "m3.json") is None
    assert window_indices.release_view_indices(dataset.identity) == 0


def test_the_shared_root_groups_the_indices_by_view(dataset, shared, tmp_path):
    folder = shared / dataset.identity[:16]
    assert window_indices.index_root(dataset, tmp_path / "job") == folder
    assert window_indices.shared_file(dataset, "m3.json") == folder / "m3.json"


@pytest.mark.parametrize("value", ["relative/indices", "link"])
def test_the_shared_root_must_be_an_absolute_regular_path(dataset, tmp_path, monkeypatch, value):
    if value == "link":
        (tmp_path / "target").mkdir()
        (tmp_path / "link").symlink_to(tmp_path / "target")
        value = str(tmp_path / "link")
    monkeypatch.setenv(SHARED_INDICES_ENV, value)
    with pytest.raises(ValueError, match="ruta absoluta y regular"):
        window_indices.index_root(dataset, tmp_path / "job")


def test_jobs_of_a_view_build_each_phase_index_once_and_reuse_it(view, dataset, shared, builds):
    root = window_indices.index_root(dataset, None)
    first = window_indices.phase_sources(dataset, view.phases, root)
    second = window_indices.phase_sources(dataset, view.phases, root)
    assert sorted(builds) == sorted(source.path.parent.name for source in first.values())
    assert len(builds) == len(view.phases) == 4
    for name, source in first.items():
        assert source.path.parent.name.startswith(f"{name}-")
        assert second[name].identity == source.identity
        assert second[name].path == source.path


def test_the_shared_index_has_the_same_identity_as_a_job_index(view, dataset, shared, tmp_path):
    own = window_indices.phase_sources(dataset, view.phases, tmp_path / "job" / "indices")
    common = window_indices.phase_sources(dataset, view.phases, shared / "view")
    for name in view.phases:
        assert common[name].identity == own[name].identity
        assert common[name].metadata == own[name].metadata


def test_simultaneous_jobs_wait_for_the_first_build_and_reuse_it(view, dataset, shared, builds):
    root, results, errors = window_indices.index_root(dataset, None), [], []
    start = threading.Barrier(4)

    def job():
        try:
            start.wait()
            results.append(window_indices.phase_sources(dataset, view.phases, root))
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=job) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(builds) == len(view.phases)
    assert {tuple(s.identity for s in r.values()) for r in results} == {
        tuple(s.identity for s in results[0].values())
    }


def test_an_index_left_half_built_is_completed_with_the_same_identity(
    view, dataset, shared, monkeypatch
):
    phases = {"train": view.phases["train"]}
    root = window_indices.index_root(dataset, None)
    write = financial_observations.atomic_parquet_batches

    def interrupted(*_, **__):
        raise RuntimeError("corte a mitad del índice")

    monkeypatch.setattr(financial_observations, "atomic_parquet_batches", interrupted)
    with pytest.raises(RuntimeError, match="corte"):
        window_indices.phase_sources(dataset, phases, root)
    (folder,) = [path for path in root.iterdir() if path.is_dir()]
    assert (folder / "identity.json").exists() and not (folder / "manifest.json").exists()
    monkeypatch.setattr(financial_observations, "atomic_parquet_batches", write)
    resumed = window_indices.phase_sources(dataset, phases, root)["train"]
    fresh = window_indices.phase_sources(dataset, phases, root.parent / "fresh")["train"]
    assert resumed.identity == fresh.identity


def test_a_code_change_before_resuming_builds_new_indices_beside_the_old_ones(
    view, dataset, shared, monkeypatch
):
    root = window_indices.index_root(dataset, None)
    old = window_indices.phase_sources(dataset, view.phases, root)
    identity = financial_observations._identity

    def changed(*args):
        return identity(*args) | dict(code=dict(financial_observations="0" * 64))

    # Otro código de los índices al reanudar: carpetas nuevas, sin chocar con las anteriores.
    monkeypatch.setattr(financial_observations, "_identity", changed)
    new = window_indices.phase_sources(dataset, view.phases, root)
    for name, source in new.items():
        assert source.path.parent != old[name].path.parent
        assert source.metadata["identity"]["code"] == dict(financial_observations="0" * 64)
        assert old[name].path.exists()
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    assert window_indices.release_view_indices(dataset.identity) == size


def test_releasing_a_view_removes_its_shared_indices_only(view, dataset, shared):
    root = window_indices.index_root(dataset, None)
    window_indices.phase_sources(dataset, view.phases, root)
    other = shared / "0123456789abcdef"
    other.mkdir()
    (other / "kept").write_bytes(b"x")
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    assert window_indices.release_view_indices(dataset.identity) == size > 0
    assert not root.exists() and (other / "kept").exists()
    assert window_indices.release_view_indices(dataset.identity) == 0


def test_a_cut_while_releasing_never_leaves_a_half_deleted_index(
    view, dataset, shared, monkeypatch
):
    root = window_indices.index_root(dataset, None)
    window_indices.phase_sources(dataset, view.phases, root)
    remove = window_indices.shutil.rmtree

    def cut(*_, **__):
        raise RuntimeError("corte durante el borrado")

    monkeypatch.setattr(window_indices.shutil, "rmtree", cut)
    with pytest.raises(RuntimeError, match="corte"):
        window_indices.release_view_indices(dataset.identity)
    # La vista ya no tiene índices activos: un trabajo los reconstruiría enteros.
    assert not root.exists()
    (retired,) = shared.glob(".released-*")
    monkeypatch.setattr(window_indices.shutil, "rmtree", remove)
    rebuilt = window_indices.phase_sources(dataset, view.phases, root)
    assert all(source.path.exists() for source in rebuilt.values())
    assert window_indices.release_view_indices("f" * 64) > 0
    assert not retired.exists() and root.exists()
    # Otro corte y la misma vista reconstruida: la retirada pendiente no bloquea la siguiente.
    monkeypatch.setattr(window_indices.shutil, "rmtree", cut)
    with pytest.raises(RuntimeError, match="corte"):
        window_indices.release_view_indices(dataset.identity)
    monkeypatch.setattr(window_indices.shutil, "rmtree", remove)
    window_indices.phase_sources(dataset, view.phases, root)
    size = sum(path.stat().st_size for path in shared.rglob("*") if path.is_file())
    assert window_indices.release_view_indices(dataset.identity) == size
    assert not root.exists() and not list(shared.glob(".released-*"))


def test_the_candidate_names_its_indices_by_view_and_phase(view, dataset, shared, builds):
    root = window_indices.index_root(dataset, None)
    titans = window_indices.phase_sources(dataset, view.phases, root)
    sources = candidate.window_sources(dataset, root, ("train", "evaluation"), 1)
    assert builds == [titans[name].path.parent.name for name in view.phases]
    assert {name: s.path for name, s in sources.items()} == {
        name: titans[name].path for name in ("train", "evaluation")
    }


def test_m3_scalers_of_a_window_are_estimated_once_and_read_with_the_same_bits(
    view, dataset, shared, monkeypatch
):
    root = window_indices.index_root(dataset, None)
    train = window_indices.phase_sources(dataset, {"train": view.phases["train"]}, root)["train"]
    recipe = SimpleNamespace(block_rows=2)
    expected = readouts.fit_write_scalers(train, block_rows=2)
    calls, fit = [], readouts.fit_write_scalers

    def counted(*args, **options):
        calls.append(options)
        return fit(*args, **options)

    monkeypatch.setattr(readouts, "fit_write_scalers", counted)
    first = readouts.window_scalers(train, recipe)
    second = readouts.window_scalers(train, recipe)
    assert first == second == expected
    assert asdict(second) == asdict(expected) and len(calls) == 1
    path = readouts._scalers_file(train, 2)
    assert path.parent == root and path.exists()
    assert readouts._scalers_file(train, 4) != path
    # Con otro código de las escalas se estiman de nuevo en otro archivo.
    code = dict(readouts._scaler_code(), **{"mars_titan.memory.write_scores": "0" * 64})
    with monkeypatch.context() as patch:
        patch.setattr(readouts, "_scaler_code", lambda: code)
        assert readouts.window_scalers(train, recipe) == expected and len(calls) == 2
        assert readouts._scalers_file(train, 2) != path and path.exists()
    # Unas escalas de otro tramo con el mismo nombre se rechazan antes de usarlas.
    atomic_json(path, asdict(expected) | dict(source_sha256=sha256(view.path)))
    with pytest.raises(ValueError, match="no proceden del tramo"):
        readouts.window_scalers(train, recipe)


def test_a_campaign_owns_the_shared_root_only_if_no_outer_process_declared_it(
    tmp_path, monkeypatch
):
    for name in (SHARED_INDICES_ENV, DIGEST_CACHE_ENV):
        monkeypatch.delenv(name, raising=False)
    output = tmp_path.resolve()
    assert engine.shared_files(tmp_path) == {
        DIGEST_CACHE_ENV: str(output / engine.DIGESTS),
        SHARED_INDICES_ENV: str(output / engine.SHARED_INDICES),
    }
    # Dentro del recorrido por ventanas la raíz es del recorrido y la campaña no la toca.
    monkeypatch.setenv(SHARED_INDICES_ENV, str(tmp_path / "rolling"))
    assert set(engine.shared_files(tmp_path)) == {DIGEST_CACHE_ENV}


def test_the_campaign_releases_a_view_when_its_last_job_ends_and_the_root_is_its_own(
    view, dataset, shared
):
    root = window_indices.index_root(dataset, None)
    window_indices.phase_sources(dataset, view.phases, root)
    first, second = (dict(id=name, scope="US", window="fold-000") for name in ("a", "b"))
    state = SimpleNamespace(
        owns_indices=False,
        receipts=dict(a={}, b={}),
        window_jobs={("US", "fold-000"): ["a", "b"]},
        views=dict(US=dict(windows={"fold-000": dict(sha256=dataset.identity)})),
    )
    engine._Campaign.release_indices(state, second)
    assert root.exists()
    state.owns_indices, state.receipts = True, dict(a={})
    engine._Campaign.release_indices(state, first)
    assert root.exists()
    state.receipts["b"] = {}
    engine._Campaign.release_indices(state, second)
    assert not root.exists()
