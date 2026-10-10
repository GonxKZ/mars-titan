"""M3 sobre los tres índices de M2: mismo presupuesto y otra puntuación del selectivo."""

import copy

import pytest
from test_native_episode_backend import native as native
from test_native_episode_backend import record
from test_write_scores import scalers

from mars_titan.memory import write_scores as ws
from mars_titan.memory.write_policy import CompositeScoreConfig, MatureErrorBank, MatureErrorConfig
from mars_titan.memory.write_scores import DecisionFeatures


def bank(native, policy="m3", **options):
    options = {"capacity": 4, "seed": 73} | options
    config = (
        CompositeScoreConfig(scalers(), **options)
        if policy == "m3"
        else MatureErrorConfig(**options)
    )
    return MatureErrorBank(
        native, config, codec_id="a" * 64, world="fixture", partition="validation", fold="0"
    )


def feature(identifier, anomaly=1.5, age=None, news=False):
    return DecisionFeatures(float(anomaly), None if age is None else float(age), news)


def offer(value, native, identifiers, errors, features=None, at=None):
    incoming = [record(native, i) for i in identifiers]
    options = dict(errors=dict(zip(identifiers, errors, strict=True)))
    if features is not None:
        options["features"] = dict(zip(identifiers, features, strict=True))
    confirmed = at if at is not None else incoming[-1].maturity_at
    return value.propose(incoming, confirmed_at=confirmed, **options)


def test_m3_shares_capacity_quotas_offers_and_reservoir_with_m2(native):
    """Las mismas ofertas a los mismos índices: solo puede cambiar el selectivo."""
    m2, m3 = bank(native, "m2", capacity=8), bank(native, capacity=8)
    assert m2.config.quotas == m3.config.quotas == dict(reservoir=4, selective=2, recent=2)
    for start in (1, 5, 9, 13):
        identifiers = list(range(start, start + 4))
        errors = [0.01 * i * (-1) ** i for i in identifiers]
        m2 = offer(m2, native, identifiers, errors)
        m3 = offer(m3, native, identifiers, errors, [feature(i, i % 3) for i in identifiers])
        left, right = m2.receipt, m3.receipt
        for name in ("index_offers", "score_evaluations", "after_seen", "before_seen"):
            assert left[name] == right[name]
        assert m2.index_ids()["reservoir"] == m3.index_ids()["reservoir"]
        assert m2.index_ids()["recent"] == m3.index_ids()["recent"]
        assert left["physical_slots"] == right["physical_slots"] <= 8
    assert m3.receipt["index_offers"] == 3 * 4 and m3.seen == m2.seen == 16
    assert m2.identity()["recipe"] == "episodic_m2_three_index_v1"
    assert m3.identity()["recipe"] == "episodic_m3_three_index_v1"
    assert m3.identity()["scalers_sha256"] == scalers().fingerprint()
    assert "write_score" not in m2.identity() and m2.write_features == {}


def test_error_only_weights_select_exactly_what_m2_selects(native):
    """x/(x+m) es estrictamente creciente: con w=(1,0,0) el selectivo coincide con M2."""
    m2 = bank(native, "m2", capacity=12)
    only = MatureErrorBank(
        native,
        CompositeScoreConfig(scalers(), capacity=12, seed=73, weights=(1.0, 0.0, 0.0)),
        codec_id="a" * 64,
        world="fixture",
        partition="validation",
        fold="0",
    )
    errors = [0.5, -0.5, 1e-9, -3.0, 0.02, 0.0, 2e-9, 0.75]
    for start in (1, 9, 17):
        identifiers = list(range(start, start + 8))
        values = [errors[(i * 5) % 8] * (1 + i / 100) for i in identifiers]
        m2 = offer(m2, native, identifiers, values)
        only = offer(
            only, native, identifiers, values, [feature(i, 9 - i % 9, i, True) for i in identifiers]
        )
        assert only.index_ids() == m2.index_ids()


def test_selective_index_keeps_the_highest_composite_scores_with_canonical_ties(native):
    value = bank(native, capacity=4)
    # Mismo error: decide la anomalía. Empate exacto entre 3 y 4: gana el ID menor.
    value = offer(
        value,
        native,
        [1, 2, 3, 4],
        [0.02, 0.02, 0.02, 0.02],
        [feature(1, 0.1), feature(2, 0.5), feature(3, 9.0), feature(4, 9.0)],
    )
    rows = {row[0]: row for row in value.receipt["components"]}
    assert rows[3][4] == rows[4][4] > rows[2][4] > rows[1][4]
    assert value.index_ids()["selective"] == (3,)
    # Una publicación contable fresca supera a una anomalía igual con relevancia desconocida.
    value = offer(
        value,
        native,
        [5, 6],
        [0.02, 0.02],
        [feature(5, 9.0, age=0.0, news=True), feature(6, 9.0, age=4000.0)],
    )
    receipt = value.receipt
    assert value.index_ids()["selective"] == (5,)
    assert receipt["selective_new_ids"] == [5]
    assert receipt["selective_rejected_ids"] == [6]
    assert receipt["selective_evicted_ids"] == [3]
    rows = {row[0]: row for row in receipt["components"]}
    assert rows[5][3] == 1.0 and rows[5][5] == 3
    assert rows[6][5] == 1 and rows[6][3] < 0.01
    assert value.write_features[5] == [0.02, 9.0, 0.0, True]


def test_missing_relevance_is_neutral_and_not_zero_in_the_score(native):
    value = offer(
        bank(native, capacity=4),
        native,
        [1, 2],
        [0.02, 0.02],
        [feature(1, 1.5, age=4000.0), feature(2, 1.5)],
    )
    rows = {row[0]: row for row in value.receipt["components"]}
    # Sin fuente conocida la relevancia es la mediana de entrenamiento, no cero.
    assert rows[2][3] == ws.NEUTRAL and rows[2][5] == 0
    assert value.index_ids()["selective"] == (2,)


def test_snapshot_recovery_and_continuation_are_exact(native):
    value = offer(
        bank(native, capacity=8),
        native,
        list(range(1, 9)),
        [0.01 * i for i in range(1, 9)],
        [feature(i, i, age=float(i), news=i % 2 == 0) for i in range(1, 9)],
    )
    snapshot = value.snapshot()
    assert set(snapshot) == {"identity", "indices", "scores", "receipt", "features"}
    restored = bank(native, capacity=8).restore(snapshot)
    assert restored.snapshot() == snapshot
    following = [feature(9, 20.0), feature(10, 0.0, news=True)]
    left = offer(restored, native, [9, 10], [-0.3, 0.0], following)
    right = offer(value, native, [9, 10], [-0.3, 0.0], following)
    assert left.snapshot() == right.snapshot()


@pytest.mark.parametrize(
    "change",
    [
        "missing_features",
        "extra_feature",
        "changed_error",
        "promoted_outsider",
        "score",
        "component",
        "rejected",
        "evicted",
        "scalers",
    ],
)
def test_corrupt_m3_snapshots_do_not_recover(native, change):
    value = offer(
        bank(native, capacity=8),
        native,
        list(range(1, 9)),
        [0.01 * i for i in range(1, 9)],
        [feature(i, 9 - i) for i in range(1, 9)],
    )
    value = offer(value, native, [9, 10], [0.5, 0.0], [feature(9, 0.0), feature(10, 0.0)])
    snapshot = copy.deepcopy(value.snapshot())
    target = bank(native, capacity=8)
    receipt = snapshot["receipt"]
    if change == "missing_features":
        snapshot.pop("features")
    elif change == "extra_feature":
        snapshot["features"][99] = [0.1, 0.1, None, False]
    elif change == "changed_error":
        selective = receipt["index_ids"]["selective"][0]
        snapshot["features"][selective][0] *= 2
    elif change == "promoted_outsider":
        selective = receipt["index_ids"]["selective"]
        outsider = next(i for i in receipt["retained_ids"] if i not in selective)
        snapshot["features"][outsider][:2] = [1e6, 1e6]
    elif change == "score":
        key = next(iter(snapshot["scores"]))
        snapshot["scores"][key] += 1e-12
    elif change == "component":
        receipt["components"][0][2] = 0.999
    elif change == "rejected":
        receipt["selective_rejected_ids"] = []
    elif change == "evicted":
        receipt["selective_evicted_ids"] = [receipt["before_ids"][0]]
    else:
        target = MatureErrorBank(
            native,
            CompositeScoreConfig(scalers(error_median=0.03), capacity=8, seed=73),
            codec_id="a" * 64,
            world="fixture",
            partition="validation",
            fold="0",
        )
    with pytest.raises((ValueError, RuntimeError)):
        target.restore(snapshot)


@pytest.mark.parametrize("policy", ["m2", "m3"])
def test_features_are_required_by_m3_and_refused_by_m2(native, policy):
    value = bank(native, policy)
    before = value.snapshot()
    wrong = None if policy == "m3" else {1: feature(1)}
    with pytest.raises(ValueError, match="rasgos"):
        value.propose([record(native, 1)], errors={1: 0.5}, confirmed_at=3, features=wrong)
    if policy == "m3":
        with pytest.raises(ValueError, match="rasgos"):
            value.propose(
                [record(native, 1)], errors={1: 0.5}, confirmed_at=3, features={2: feature(2)}
            )
        with pytest.raises(ValueError):
            value.propose([record(native, 1)], errors={1: 0.5}, confirmed_at=3, features={1: 1.0})
    assert value.snapshot() == before


@pytest.mark.parametrize(
    "options",
    [
        dict(weights=(0.5, 0.5, 0.5)),
        dict(weights=(1.0, 0.0)),
        dict(weights=[1 / 3, 1 / 3, 1 / 3]),
        dict(weights=(-0.5, 1.0, 0.5)),
        dict(capacity=3),
        dict(scalers=None),
    ],
)
def test_invalid_m3_configurations_are_rejected(options):
    values = dict(scalers=scalers(), capacity=8, seed=73) | options
    with pytest.raises(ValueError):
        CompositeScoreConfig(**values)
