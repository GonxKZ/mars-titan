"""Ventanas de recepción con fixtures nativos, sin ajuste de parámetros."""

import copy
from pathlib import Path

import pytest
import torch
from test_financial_mechanisms import engines
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native
from test_native_mechanism_collection import session

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def window(run, generations=(1, 2), *, through=2, before=None, **limits):
    method = getattr(run, "mechanism_collections_at", None)
    assert method is not None, "Falta la lectura conjunta del prefijo nativo"
    return method(
        generations,
        through_generation=through,
        before=moment(128) if before is None else before,
        **limits,
    )


def first_two(run, native, batches):
    issued = run.step(batches[126], [])
    labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in issued.predictions[:2]]
    second = run.step(batches[127], labels)
    return issued, second


@pytest.mark.parametrize("device", DEVICES)
def test_window_matches_individual_views_and_preserves_late_results(
    native, four_flow_source, tmp_path, device
):
    data = engines(four_flow_source, device)
    output = tmp_path / "window"
    with session(native, output, data) as run:
        first, second = first_two(run, native, four_flow_source["batches"])
        originals = [
            run.mechanism_collection_at(g, through_generation=2, before=moment(128)) for g in (1, 2)
        ]
        rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(device).clone() if device != "cpu" else None
        head = run.snapshot()
        result = window(run)
        assert run.snapshot() == head and torch.equal(torch.random.get_rng_state(), rng)
        if cuda_rng is not None:
            assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
        assert [c["generation"] for c in result["collections"]] == [1, 2]
        for actual, original in zip(result["collections"], originals, strict=True):
            assert {k: actual[k] for k in ("collection", "supervision", "censored")} == {
                k: original[k] for k in ("collection", "supervision", "censored")
            }
            assert actual["resolutions"] == original["as_of"]["resolutions"]
            assert actual["packet_reference"] == original["as_of"]["packet_reference"]
        assert [r["generation"] for r in result["as_of"]["native_records"]] == [1, 2]
        assert [
            t["value"] if t else None for t in result["collections"][0]["supervision"]["outcomes"]
        ] == [
            0.125,
            0.125,
            None,
            None,
        ]
        assert result["collections"][1]["supervision"]["outcomes"] == [None] * 4
        late = [native.Feedback(p.id, 0, moment(127), 99.0) for p in first.predictions[2:]]
        late += [native.Feedback(p.id, 0, moment(128), -0.25) for p in second.predictions[:2]]
        run.step([], late, kind="settlement", cutoff=moment(129))
        assert window(run) == result
        full = window(run, through=3, before=moment(130))
        assert [t["value"] for t in full["collections"][0]["supervision"]["outcomes"]] == [
            0.125,
            0.125,
            99.0,
            99.0,
        ]
        assert [
            t["value"] if t else None for t in full["collections"][1]["supervision"]["outcomes"]
        ] == [
            -0.25,
            -0.25,
            None,
            None,
        ]
        assert full["as_of"]["window_sha256"] != result["as_of"]["window_sha256"]
        run.step([], [], kind="settlement", cutoff=moment(131), close_phase=True)
        closed = window(run, through=4, before=moment(132))
        assert closed["collections"][0]["censored"] == []
        assert {c["row_index"] for c in closed["collections"][1]["censored"]} == {2, 3}
        assert {c["closed_at"] for c in closed["collections"][1]["censored"]} == {moment(131)}
        assert closed["collections"][1]["supervision"]["outcomes"][2:] == [None, None]
    with session(native, output, data, resume=True) as run:
        assert window(run) == result


def test_window_reads_each_native_record_once_and_charges_unique_bytes(
    native, four_flow_source, tmp_path, monkeypatch
):
    data = engines(four_flow_source, "cpu")
    with session(native, tmp_path / "once", data) as run:
        first_two(run, native, four_flow_source["batches"])
        method = native.read_blob
        read = []

        def counted(path, maximum):
            if Path(path).name.startswith("record-"):
                read.append(Path(path).name)
            return method(path, maximum)

        monkeypatch.setattr(native, "read_blob", counted)
        result = window(run)
        assert read == ["record-1.json", "record-2.json"]
        nbytes = sum(c["packet_reference"]["bytes"] for c in result["collections"])
        nbytes += sum((run.output / f"record-{g}.json").stat().st_size for g in (1, 2))
        assert result["as_of"]["source_bytes"] == nbytes
        assert window(run, max_bytes=nbytes) == result
        with pytest.raises(
            (ValueError, RuntimeError), match="bytes|size|limit|tamaño|límite|budget"
        ):
            window(run, max_bytes=nbytes - 1)
        assert [c["supervision"]["outcomes"] for c in result["collections"]] != [[], []]


@pytest.mark.parametrize("device", DEVICES)
def test_sparse_window_never_loads_an_unselected_packet(native, four_flow_source, tmp_path, device):
    data = engines(four_flow_source, device)
    with session(native, tmp_path / "sparse", data) as run:
        first, _ = first_two(run, native, four_flow_source["batches"])
        late = [native.Feedback(p.id, 0, moment(127), -0.5) for p in first.predictions[2:]]
        run.step(four_flow_source["batches"][128], late)
        bundle = run._bundle(run.snapshot()["state"])
        middle = bundle["mechanism_collections"]["entries"][1]["packet"]
        path = run.artifacts.directory / middle["name"]
        saved = path.read_bytes()
        try:
            path.write_bytes(b"Paquete no seleccionado")
            result = window(run, (1, 3), through=3, before=moment(129))
        finally:
            path.write_bytes(saved)
        assert [c["generation"] for c in result["collections"]] == [1, 3]
        assert [r["generation"] for r in result["as_of"]["native_records"]] == [1, 2, 3]
        assert [t["value"] for t in result["collections"][0]["supervision"]["outcomes"]] == [
            0.125,
            0.125,
            -0.5,
            -0.5,
        ]
        assert result["collections"][1]["supervision"]["outcomes"] == [None] * 4


@pytest.mark.parametrize("enabled", [False, True])
def test_unconfirmed_or_disabled_window_is_rejected(native, four_flow_source, tmp_path, enabled):
    data = engines(four_flow_source, "cpu")
    with session(native, tmp_path / "empty", data, enabled=enabled) as run:
        with pytest.raises(ValueError, match="recogida|confirmada"):
            window(run, (1,), through=1)


def test_settlement_generation_is_not_a_collection(native, four_flow_source, tmp_path):
    data = engines(four_flow_source, "cpu")
    with session(native, tmp_path / "settlement", data) as run:
        run.step(four_flow_source["batches"][126], [])
        run.step([], [], kind="settlement", cutoff=moment(127))
        with pytest.raises(ValueError, match="decisión"):
            window(run)


@pytest.mark.parametrize(
    "generations", [(), [], (True,), (0,), (2, 1), (1, 1), (3,), tuple(range(1, 258))]
)
def test_invalid_collection_selection_is_rejected_without_mutating_state(
    native, four_flow_source, tmp_path, generations
):
    data = engines(four_flow_source, "cpu")
    with session(native, tmp_path / "invalid", data) as run:
        first_two(run, native, four_flow_source["batches"])
        before = run.snapshot()
        with pytest.raises(ValueError):
            window(run, generations)
        assert run.snapshot() == before


@pytest.mark.parametrize(
    "options",
    [
        {"through": 1},
        {"through": True},
        {"max_records": 1},
        {"max_bytes": 1},
        {"before": moment(127)},
    ],
)
def test_window_rejects_partial_or_future_prefix(native, four_flow_source, tmp_path, options):
    data = engines(four_flow_source, "cpu")
    with session(native, tmp_path / "bounds", data) as run:
        first_two(run, native, four_flow_source["batches"])
        before = run.snapshot()
        with pytest.raises(ValueError):
            window(run, **options)
        assert run.snapshot() == before


def test_window_ignores_receipts_and_unselected_future_records(native, four_flow_source, tmp_path):
    data = engines(four_flow_source, "cpu")
    with session(native, tmp_path / "future", data) as run:
        first_two(run, native, four_flow_source["batches"])
        expected = window(run)
        run.step([], [], kind="settlement", cutoff=moment(131), close_phase=True)
        bundle = run._bundle(run.snapshot()["state"])
        refs = [e["receipt"] for e in bundle["mechanism_collections"]["entries"]]
        paths = [run.artifacts.directory / r["name"] for r in refs]
        paths.append(run.output / "record-3.json")
        saved = [p.read_bytes() for p in paths]
        try:
            for path in paths:
                path.write_bytes(b"Contenido posterior no consultable")
            assert window(run) == expected
        finally:
            for path, value in zip(paths, saved, strict=True):
                path.write_bytes(value)


def test_window_checks_one_common_head_after_reconstruction(
    native, four_flow_source, tmp_path, monkeypatch
):
    import mars_titan.memory.mechanism_collection_asof as api

    data = engines(four_flow_source, "cpu")
    with session(native, tmp_path / "concurrent", data) as run:
        first_two(run, native, four_flow_source["batches"])
        original = getattr(api, "collections_at", None)
        assert original is not None, "Falta el lector conjunto de colecciones"

        def advance(*args, **kwargs):
            result = original(*args, **kwargs)
            run.step([], [], kind="settlement", cutoff=moment(128))
            return result

        monkeypatch.setattr(api, "collections_at", advance)
        with pytest.raises(ValueError, match="cambió durante"):
            window(run)
        assert run.snapshot()["cursor"] == 3


def test_window_results_do_not_alias_each_other(native, four_flow_source, tmp_path):
    data = engines(four_flow_source, "cpu")
    with session(native, tmp_path / "independent", data) as run:
        first_two(run, native, four_flow_source["batches"])
        result = window(run)
        second = copy.deepcopy(result["collections"][1])
        result["collections"][0]["supervision"]["outcomes"][0]["value"] = 12345.0
        result["collections"][0]["resolutions"].clear()
        assert result["collections"][1] == second
        assert window(run)["collections"][0]["supervision"]["outcomes"][0]["value"] == 0.125
