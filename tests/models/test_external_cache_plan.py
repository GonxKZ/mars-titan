"""Presupuestos de RAM y disco de XGBoost externo, comprobados sin CUDA ni ajuste."""

import os
from types import SimpleNamespace

import pytest

from mars_titan.models.baselines import external_boosting as external

GIB = 1024**3
# Orden de magnitud de la edición histórica: unos 15,4 millones de filas de ajuste
# y 1.714 características más cinco bits de presencia.
ROWS, FEATURES = 15_400_000, 1719


def plan(**changes):
    values = dict(
        rows=ROWS,
        features=FEATURES,
        max_bin=128,
        on_host=False,
        max_host_cache_bytes=2 * GIB,
        max_disk_cache_bytes=40 * GIB,
        available_ram=20 * GIB,
        free_disk=50 * GIB,
    )
    return external.external_cache_plan(**(values | changes))


def test_historical_host_cache_does_not_fit_and_fails_before_reading():
    with pytest.raises(ValueError, match="host"):
        plan(on_host=True, max_disk_cache_bytes=None, max_host_cache_bytes=24 * GIB)
    estimate = ROWS * (FEATURES * 4 + 16)
    assert 105e9 < estimate < 107e9


def test_disk_plan_reports_dense_estimate_and_global_bound():
    result = plan()
    assert result["cache_location"] == "disk"
    assert result["host_cache_bytes_estimate"] == 0
    # 128 bins locales necesitan 7 bits por valor: las entradas no tienen ausentes.
    assert result["disk_cache_bytes_estimate"] == -(-ROWS * FEATURES * 7 // 8)
    # 1719 * 128 bins globales más el ausente necesitan 18 bits.
    assert result["disk_cache_bytes_global_bins_bound"] == ROWS * FEATURES * 18 // 8
    assert 23e9 < result["disk_cache_bytes_estimate"] < 24e9
    assert 59e9 < result["disk_cache_bytes_global_bins_bound"] < 60e9
    assert result["disk_cache_estimate_basis"].endswith("measured_xgboost_3_3")


@pytest.mark.parametrize(
    "max_bin,bits", [(2, 1), (63, 6), (64, 6), (65, 7), (128, 7), (129, 8), (256, 8), (512, 9)]
)
def test_dense_bits_cover_every_local_bin_without_a_missing_symbol(max_bin, bits):
    result = plan(rows=8, features=1, max_bin=max_bin)
    assert result["disk_cache_bytes_estimate"] == bits


@pytest.mark.parametrize(
    "changes",
    [
        dict(max_disk_cache_bytes=21 * GIB),
        dict(free_disk=21 * GIB),
        dict(other_disk_bytes=30 * GIB),
        dict(on_host=True, max_disk_cache_bytes=None, rows=4000, available_ram=1024),
        dict(on_host=True, max_disk_cache_bytes=None, rows=4000, max_host_cache_bytes=1024),
    ],
)
def test_each_budget_and_real_availability_fails_early(changes):
    with pytest.raises(ValueError):
        plan(**changes)


def test_limits_are_inclusive_and_include_other_disk_artifacts():
    dense = -(-ROWS * FEATURES * 7 // 8)
    assert plan(max_disk_cache_bytes=dense, free_disk=dense)["free_disk_bytes"] == dense
    assert plan(free_disk=dense + 10, other_disk_bytes=10)["other_disk_bytes"] == 10
    with pytest.raises(ValueError, match="disco libre"):
        plan(free_disk=dense + 10, other_disk_bytes=11)
    host = 4000 * (FEATURES * 4 + 16)
    small = dict(on_host=True, max_disk_cache_bytes=None, rows=4000)
    assert plan(**small, available_ram=host)["host_cache_bytes_estimate"] == host
    with pytest.raises(ValueError, match="RAM"):
        plan(**small, available_ram=host - 1)


@pytest.mark.parametrize(
    "changes",
    [
        dict(on_host=True),
        dict(max_disk_cache_bytes=0),
        dict(max_disk_cache_bytes=float(GIB)),
        dict(max_disk_cache_bytes=external.MAX_DISK_CACHE_BYTES + 1),
        dict(rows=0),
        dict(features=True),
        dict(max_bin=1),
        dict(max_host_cache_bytes=25 * GIB),
        dict(free_disk=-1),
    ],
)
def test_invalid_budgets_are_rejected(changes):
    with pytest.raises(ValueError):
        plan(**changes)


def test_fit_rejects_a_disk_budget_for_the_host_cache_before_loading_cuda(tmp_path, monkeypatch):
    def unexpected():
        raise AssertionError("No debe cargar CUDA")

    monkeypatch.setattr(external, "_libraries", unexpected)
    for on_host, budget in ((True, GIB), (False, 0)):
        with pytest.raises(ValueError, match="disco"):
            external.fit_external_boosting(
                lambda: iter(()),
                tmp_path / "cache",
                expected_rows=1,
                on_host=on_host,
                max_disk_cache_bytes=budget,
            )
    assert not (tmp_path / "cache").exists()


def test_directory_bytes_counts_regular_files_without_following_links(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "pages.0").write_bytes(b"x" * 10)
    (tmp_path / "nested" / "pages.1").write_bytes(b"y" * 7)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.write_bytes(b"z" * 1000)
    os.symlink(outside, tmp_path / "link")
    assert external.directory_bytes(tmp_path) == 17


def test_free_disk_uses_the_nearest_existing_parent(tmp_path, monkeypatch):
    calls = []

    def usage(path):
        calls.append(path)
        return SimpleNamespace(free=123)

    monkeypatch.setattr(external.shutil, "disk_usage", usage)
    assert external.free_disk_bytes(tmp_path / "a" / "b") == 123
    assert calls == [tmp_path]
    assert external.available_ram_bytes() > 0
