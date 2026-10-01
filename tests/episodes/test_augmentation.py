"""Presupuesto emparejado y caché del padre ligada a las entradas efectivas."""

import copy

import numpy as np
import pytest

from mars_titan.episodes.augmentation import augmentation_windows, paired_world, training_visits
from mars_titan.episodes.parents import ParentCache
from mars_titan.episodes.worlds import WorldConfig, generate_world


def test_each_epoch_preserves_all_base_rows_and_matches_additional_updates():
    world = generate_world(WorldConfig(assets=4, sessions=80, context=8))
    windows = augmentation_windows(world, seed=42, decisions=8, warmup=4)
    added = sum(sum(world.index[i][1] for i in range(w.decision_start, w.stop)) for w in windows)
    assert added >= sum(row[1] for row in world.index) // 4
    assert added < sum(row[1] for row in world.index) // 4 + world.max_assets
    visits = training_visits(world, windows, epoch=1, seed=42)
    base = [v.cohort for v in visits if v.arm == "real"]
    assert sorted(base) == list(range(len(world)))
    assert visits == training_visits(world, windows, epoch=1, seed=42)


def test_parent_cache_recomputes_when_inputs_change_but_not_for_label_only_changes(tmp_path):
    world = generate_world(WorldConfig(assets=3, sessions=40, context=8))
    calls = []

    def predict(inputs):
        assert set(inputs) == set(world.shapes)
        calls.append(1)
        return inputs["news"][:, 0].astype(np.float64)

    with ParentCache(tmp_path / "cache.sqlite", "a" * 64, "analytic-test", predict) as cache:
        raw = world(0)
        first = cache.predict(raw)
        labels = copy.deepcopy(raw)
        labels["target"] += 1
        np.testing.assert_array_equal(cache.predict(labels), first)
        assert len(calls) == 1
        changed = copy.deepcopy(raw)
        changed["inputs"]["news"] += 0.5
        assert not np.array_equal(cache.predict(changed), first)
        assert len(calls) == 2
        with pytest.raises(ValueError):
            cache.predict({**raw, "asset_ids": ["duplicate"] * 3})


def test_paired_worlds_match_varying_complete_cohort_sizes():
    world = generate_world(
        WorldConfig(
            assets=4, sessions=80, context=8, active_assets=tuple(3 + i % 2 for i in range(80))
        )
    )
    for spec in augmentation_windows(world, seed=42, decisions=8, warmup=4):
        synthetic, paired = paired_world(world, spec, seed=43)
        expected = [world.index[i][1] for i in range(spec.decision_start, spec.stop)]
        observed = [synthetic.index[i][1] for i in range(paired.decision_start, paired.stop)]
        assert observed == expected


def test_parent_cache_rejects_future_availability(tmp_path):
    world = generate_world(WorldConfig(assets=3, sessions=40, context=8))
    with ParentCache(
        tmp_path / "cache.sqlite", "a" * 64, "analytic-test", lambda x: np.zeros(len(x["news"]))
    ) as cache:
        raw = world(0)
        raw["available_at"] += 1
        with pytest.raises(ValueError):
            cache.predict(raw)
