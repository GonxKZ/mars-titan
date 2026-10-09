"""Banco 128×256 de la GRU con episodios y etiquetas manuales, sin ajustar parámetros."""

import pytest
import torch
from test_native_episode_backend import native as native
from test_native_episode_backend import record

from mars_titan.memory.candidate_bank import (
    KEY_WIDTH,
    VALUE_WIDTH,
    CandidateBankConfig,
    CandidateEpisodeBank,
    CandidateEpisodes,
)
from mars_titan.memory.session_artifacts import SessionArtifacts

CODEC = "c" * 64
REPRESENTATION = "candidate-fixed-v1:manual"


def bank(native, *, capacity=4, seed=73, dtype=torch.float64, codec=CODEC, **config):
    return CandidateEpisodeBank(
        native,
        CandidateBankConfig(capacity=capacity, seed=seed, **config),
        codec_id=codec,
        representation_id=REPRESENTATION,
        dtype=dtype,
        world="manual",
        partition="validation",
        fold="0",
    )


def episodes(ids, dtype=torch.float64, *, zero=()):
    ids = list(ids)
    count = len(ids)
    keys = torch.zeros((count, KEY_WIDTH), dtype=dtype)
    for row, identifier in enumerate(ids):
        if identifier not in zero:
            keys[row, identifier % KEY_WIDTH] = 1.0
    offsets = torch.arange(VALUE_WIDTH, dtype=dtype) / 1024
    return CandidateEpisodes(
        torch.tensor(ids, dtype=torch.int64),
        keys,
        torch.tensor(ids, dtype=dtype).reshape(count, 1) + offsets,
        torch.tensor([[2 * i, 2 * i, 2 * i + 1] for i in ids], dtype=torch.int64).reshape(count, 3),
        torch.tensor([0.125 * i for i in ids], dtype=torch.float64),
    )


def admit(state, ids, **options):
    return state.propose(episodes(ids, state.dtype, **options), confirmed_at=2 * max(ids) + 1)


def same_snapshot(left, right):
    assert set(left) == set(right)
    for name, value in left.items():
        if isinstance(value, torch.Tensor):
            assert value.dtype == right[name].dtype and torch.equal(value, right[name]), name
        else:
            assert value == right[name], name


def replace(value, **changes):
    return CandidateEpisodes(**{**value.__dict__, **changes})


@pytest.mark.parametrize("capacity", [1, 4, 9])
def test_reservoir_reproduces_native_v2_slots_for_the_same_seed_and_ids(native, capacity):
    gru = bank(native, capacity=capacity)
    scope = native.MemoryScope()
    scope.world, scope.partition, scope.fold = "another-arm", "validation", "0"
    scope.representation = "fixed64"
    reference = native.EpisodicMemory(scope, 73, capacity, 2)
    last = 0
    for size in (1, 3, 7, 2, 11, 5, 13):
        ids = list(range(last + 1, last + size + 1))
        last = ids[-1]
        gru = admit(gru, ids)
        for identifier in ids:
            reference.write(record(native, identifier), 2 * ids[-1] + 1)
        assert [r.id for r in gru.records()] == [r.id for r in reference.retained_records()]
        assert gru.seen == reference.seen and gru.size == reference.size


def test_splitting_admissions_keeps_slots_tensors_and_rng(native):
    whole = admit(bank(native, capacity=6), range(1, 41))
    split = bank(native, capacity=6)
    for start in range(1, 41, 7):
        split = admit(split, range(start, min(start + 7, 41)))
    same_snapshot(whole.snapshot(), split.snapshot())


def test_rejected_proposals_leave_the_bank_and_its_draws_unchanged(native):
    state = admit(bank(native, capacity=3), range(1, 6))
    before = state.snapshot()
    valid = episodes([6, 7])
    nan_values = valid.values.clone()
    nan_values[0, 0] = float("nan")
    inf_keys = valid.keys.clone()
    inf_keys[1, 3] = float("inf")
    subunit = valid.keys.clone() * 0.5
    late = valid.times.clone()
    late[1, 2] = 100
    future_input = valid.times.clone()
    future_input[0, 1] = future_input[0, 0] + 1
    immature = valid.times.clone()
    immature[0, 2] = immature[0, 0]
    invalid = [
        replace(valid, ids=torch.tensor([7, 6])),
        replace(valid, ids=torch.tensor([5, 6])),
        replace(valid, values=nan_values),
        replace(valid, keys=inf_keys),
        replace(valid, keys=subunit),
        replace(valid, labels=torch.tensor([float("nan"), 0.0], dtype=torch.float64)),
        replace(valid, times=late),
        replace(valid, times=future_input),
        replace(valid, times=immature),
        replace(valid, keys=valid.keys.float()),
        replace(valid, values=valid.values[:, :64].contiguous()),
        replace(valid, labels=valid.labels.float()),
        replace(valid, keys=valid.keys.clone().requires_grad_()),
        episodes([]),
    ]
    for candidate in invalid:
        with pytest.raises(ValueError):
            state.propose(candidate, confirmed_at=15)
        same_snapshot(state.snapshot(), before)
    with pytest.raises(ValueError, match="retrocede"):
        state.propose(valid, confirmed_at=state.confirmed_at - 1)
    with pytest.raises(ValueError):
        state.propose(valid, confirmed_at=15.0)
    later = admit(bank(native, capacity=3), range(1, 6)).propose(episodes([6]), confirmed_at=1000)
    with pytest.raises(ValueError, match="retrocede"):
        later.propose(episodes([7]), confirmed_at=20)
    tight = bank(native, capacity=3, max_working_bytes=1)
    with pytest.raises(ValueError, match="presupuesto"):
        tight.propose(valid, confirmed_at=15)
    assert tight.snapshot()["reservoir_rng"] == bank(native).snapshot()["reservoir_rng"]
    same_snapshot(state.snapshot(), before)
    restored = state.restore(before)
    same_snapshot(
        state.propose(valid, confirmed_at=15).snapshot(),
        restored.propose(valid, confirmed_at=15).snapshot(),
    )


def test_snapshot_and_proposals_own_their_storage(native):
    incoming = episodes([1, 2, 3])
    state = bank(native).propose(incoming, confirmed_at=7)
    expected = state.snapshot()
    incoming.keys.zero_()
    incoming.labels.zero_()
    exported = state.snapshot()
    exported["values"].zero_()
    exported["ids"].zero_()
    same_snapshot(state.snapshot(), expected)
    view = state.read_view()
    view["keys"].zero_()
    same_snapshot(state.snapshot(), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_precision_is_preserved_and_labels_convert_only_in_the_read_view(native, dtype):
    state = admit(bank(native, dtype=dtype), [1, 2])
    view = state.read_view()
    assert view["keys"].dtype == view["values"].dtype == view["returns"].dtype == dtype
    assert view["labels"].dtype == torch.float64
    assert torch.equal(view["returns"], view["labels"].to(dtype))
    other = torch.float64 if dtype == torch.float32 else torch.float32
    with pytest.raises(ValueError):
        state.propose(episodes([3], other), confirmed_at=7)
    with pytest.raises(ValueError, match="contrato"):
        bank(native, dtype=other).restore(state.snapshot())


def test_fp32_reading_rejects_a_label_outside_its_range(native):
    huge = episodes([1], torch.float32)
    huge = replace(huge, labels=torch.tensor([1e300], dtype=torch.float64))
    state = bank(native, dtype=torch.float32).propose(huge, confirmed_at=3)
    with pytest.raises(ValueError, match="precisión"):
        state.read_view()


def test_full_capacity_view_feeds_eight_neighbors_and_exact_zero_keys(native):
    if not hasattr(native, "CandidateConfig"):
        pytest.skip("El enlace no incluye el candidato GRU")
    config = native.CandidateConfig()
    config.dimensions = [5, 3, 4, 3, 3]
    config.normalization_id = "manual-bank"
    model = native.Candidate(config, "float64", "cpu")
    state = admit(bank(native, capacity=12), range(1, 31), zero={3, 30})
    view = state.read_view()
    assert state.size == 12 and state.seen == 30
    assert view["ids"].tolist() == sorted(r.id for r in state.records())
    memory = model.snapshot(
        view["keys"], view["values"], view["returns"], view["ids"], model.representation_id()
    )
    inputs = native.CandidateInputs(
        torch.zeros((2, 64, 5), dtype=torch.float64),
        *(torch.zeros((2, width), dtype=torch.float64) for width in (3, 4, 3, 3)),
        torch.ones((2, 5), dtype=torch.bool),
    )
    with torch.no_grad():
        result = model.forward(inputs, memory, 2)
    assert tuple(result.read.ids.shape) == (2, 8)
    assert set(result.read.ids.flatten().tolist()) <= set(view["ids"].tolist())
    assert torch.isfinite(result.quantiles).all()


def test_artifact_roundtrip_restores_the_next_admission_exactly(native, tmp_path):
    state = admit(bank(native, capacity=5), range(1, 18))
    artifacts = SessionArtifacts(native, tmp_path / "artifacts")
    reference = artifacts.stage(state.snapshot(), identity="a" * 64, kind="bank")
    restored = bank(native, capacity=5).restore(
        artifacts.read(reference, identity="a" * 64, kind="bank")
    )
    same_snapshot(restored.snapshot(), state.snapshot())
    same_snapshot(admit(restored, range(18, 30)).snapshot(), admit(state, range(18, 30)).snapshot())


def _mutated(snapshot, **changes):
    result = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in snapshot.items()
    }
    result.update(changes)
    return result


def test_restore_rejects_inconsistent_snapshots_without_changing_the_bank(native):
    full = admit(bank(native, capacity=3), range(1, 9))
    partial = admit(bank(native, capacity=3), [1, 2])
    other_seed = admit(bank(native, capacity=3, seed=74), [1, 2]).snapshot()["reservoir_rng"]
    duplicated = full.snapshot()["ids"].clone()
    duplicated[1] = duplicated[0]
    late = full.snapshot()["times"].clone()
    late[0, 2] = 10**6
    unordered = partial.snapshot()
    fields = ("ids", "keys", "values", "times", "labels")
    invalid = [
        _mutated(full.snapshot(), **{name: full.snapshot()[name][:2].clone() for name in fields}),
        _mutated(full.snapshot(), ids=duplicated),
        _mutated(full.snapshot(), times=late),
        _mutated(full.snapshot(), last_id=1),
        _mutated(full.snapshot(), seen=2),
        _mutated(full.snapshot(), reservoir_rng=full.snapshot()["reservoir_rng"] + " "),
        _mutated(partial.snapshot(), reservoir_rng=other_seed),
        _mutated(
            unordered,
            **{
                name: unordered[name].flip(0).contiguous()
                for name in ("ids", "keys", "values", "times", "labels")
            },
        ),
        {**full.snapshot(), "extra": 1},
        bank(native, capacity=3, codec="d" * 64).snapshot(),
    ]
    target = bank(native, capacity=3)
    before = target.snapshot()
    for payload in invalid:
        with pytest.raises(ValueError):
            target.restore(payload)
        same_snapshot(target.snapshot(), before)


def test_configuration_and_identity_bounds(native):
    for options in (
        dict(capacity=0),
        dict(capacity=8193),
        dict(capacity=True),
        dict(seed=-1),
        dict(seed=2**64),
        dict(max_working_bytes=256 * 1024**2 + 1),
    ):
        with pytest.raises(ValueError):
            CandidateBankConfig(**options)
    with pytest.raises(ValueError):
        bank(native, codec="x")
    with pytest.raises(ValueError):
        bank(native, dtype=torch.float16)
    assert bank(native).fingerprint() != bank(native, seed=74).fingerprint()
    assert bank(native).fingerprint() != bank(native, dtype=torch.float32).fingerprint()
    assert bank(native).identity()["scope_in_rng"] is False


def test_maximum_capacity_accepts_a_full_fp64_batch_within_its_budget(native):
    state = bank(native, capacity=8192)
    assert state.estimated_bytes(8192) <= state.config.max_working_bytes
    full = state.propose(episodes(range(1, 8193)), confirmed_at=2 * 8192 + 1)
    assert full.size == 8192 and full.seen == 8192
    assert full.read_view()["ids"].tolist() == list(range(1, 8193))
