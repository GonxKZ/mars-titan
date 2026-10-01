"""Presupuesto de filas, causalidad y cursor de la comparación emparejada."""

from dataclasses import replace

import numpy as np
import pytest

from mars_titan.episodes.augmentation import augmentation_windows, paired_world
from mars_titan.episodes.parents import ParentCache
from mars_titan.episodes.windows import EpisodeView
from mars_titan.episodes.worlds import WorldConfig, generate_world
from mars_titan.posttraining.inputs import PairedInputs, fit_normalization


def sources():
    train = generate_world(
        WorldConfig(
            assets=4, sessions=32, context=4, active_assets=tuple(3 + i % 2 for i in range(32))
        )
    )
    validation = generate_world(
        WorldConfig(assets=4, sessions=12, context=4, partition="validation")
    )
    windows = augmentation_windows(train, seed=7, decisions=3, warmup=2)
    extras = {}
    for i, window in enumerate(windows):
        world, paired = paired_world(train, window, seed=100 + i)
        extras[i] = EpisodeView(world, paired)
    return train, validation, windows, extras


def dataset(tmp_path):
    train, validation, windows, extras = sources()
    cache = ParentCache(
        tmp_path / "parent.sqlite",
        "a" * 64,
        "analytic-test",
        lambda x: x["news"][:, 0].astype(np.float64),
    )
    return PairedInputs(
        train,
        validation,
        cache,
        windows=windows,
        synthetic=lambda i: extras[i],
        synthetic_identity={str(i): view.source.manifest_sha256 for i, view in extras.items()},
    )


def collect(data, condition, *, cursor=None):
    return list(
        data.batches(
            partition="train", condition=condition, batch_size=2, epoch=0, seed=7, cursor=cursor
        )
    )


def test_augmented_epochs_preserve_every_real_row_and_match_exact_updates(tmp_path):
    data = dataset(tmp_path)
    base = collect(data, "real")
    resampled = collect(data, "real_resampled")
    synthetic = collect(data, "real_synthetic")
    real_ids = [key for batch in base for key in batch["sample_ids"]]
    assert len(set(real_ids)) == 98
    for batches in (resampled, synthetic):
        assert [
            key for b in batches if b["origin"] == "real" for key in b["sample_ids"]
        ] == real_ids
        assert all(len(b["target"]) <= 2 for b in batches)
    assert len(resampled) == len(synthetic) == 72
    assert sum(len(b["target"]) for b in resampled) == 126
    assert sum(len(b["target"]) for b in synthetic) == 126
    assert data.budget("real_resampled", 2) == {
        "real_rows": 98,
        "extra_rows": 28,
        "rows": 126,
        "updates": 72,
    }
    data.parent.close()


@pytest.mark.parametrize("condition", ["real", "real_resampled", "real_synthetic"])
def test_cursor_replays_only_unconfirmed_rows(condition, tmp_path):
    data = dataset(tmp_path)
    whole = collect(data, condition)
    tail = collect(data, condition, cursor=whole[5]["confirmed_cursor"])
    assert len(tail) == len(whole) - 6
    for a, b in zip(tail, whole[6:], strict=True):
        assert a["sample_ids"] == b["sample_ids"]
        np.testing.assert_array_equal(a["features"], b["features"])
    assert collect(data, condition, cursor=whole[-1]["confirmed_cursor"]) == []
    bad = dict(whole[5]["confirmed_cursor"], consumed=0)
    with pytest.raises(ValueError, match="cursor"):
        collect(data, condition, cursor=bad)
    data.parent.close()


def test_normalization_uses_only_original_train_and_validation_forbids_augmentation(tmp_path):
    data = dataset(tmp_path)
    values = np.concatenate([b["features"] for b in collect(data, "real")])
    fit = fit_normalization(data, batch_size=2)
    np.testing.assert_allclose(fit["mean"], values.mean(axis=0, dtype=np.float64), atol=1e-14)
    assert fit["samples"] == 98
    assert fit["fit_partition"] == "train"
    with pytest.raises(ValueError, match="validación"):
        list(data.batches(partition="validation", condition="real_synthetic", batch_size=2))
    with pytest.raises(ValueError):
        list(data.batches(partition="test", condition="real", batch_size=2))
    data.parent.close()


def test_rejects_future_labels_and_unmatched_synthetic_shapes(tmp_path):
    data = dataset(tmp_path)
    original = data.train

    class FutureSource:
        def __getattr__(self, name):
            return getattr(original, name)

        def __len__(self):
            return len(original)

        def __call__(self, i):
            raw = original(i)
            raw["target_available_at"][:] = 1_672_531_200_000_000
            return raw

    data.train = FutureSource()
    with pytest.raises(ValueError, match="partición"):
        collect(data, "real")
    data.train = original
    view = data.synthetic(0)
    data.synthetic = lambda _i: EpisodeView(
        view.source, replace(view.window, stop=view.window.stop - 1)
    )
    with pytest.raises(ValueError, match="emparejado"):
        collect(data, "real_synthetic")
    data.parent.close()


def test_augmented_condition_requires_windows_and_the_confirmed_synthetic_identity(tmp_path):
    data = dataset(tmp_path)
    windows = data.windows
    data.windows = ()
    with pytest.raises(ValueError, match="aumento"):
        collect(data, "real_synthetic")
    data.windows = windows
    data.identity["synthetic"]["0"] = "f" * 64
    with pytest.raises(ValueError, match="identidad"):
        collect(data, "real_synthetic")
    data.parent.close()
