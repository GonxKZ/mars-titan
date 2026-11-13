"""Cohortes de la campaña de comparación con manifiestos de prueba, sin abrir datos (#836)."""

import copy
import json

import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training import comparison_cohorts as cohorts
from mars_titan.training import screening_cohorts
from mars_titan.training.cohort_contract import cohort_identity
from tests.training.test_screening_cohorts import reconcile, source


@pytest.mark.parametrize("origin", [2018, 2019, 2022, 2023])
def test_cohort_uses_only_the_fit_history_of_its_origin_and_keeps_every_selected_row(origin):
    original = source(origin)
    view, selection = cohorts.project_cohort(original, size=64)
    symbols = [asset["symbol"] for asset in view["assets"]]
    assert len(symbols) == len(set(symbols)) == 64 and symbols == sorted(symbols)
    # A000 no tiene etiquetas de ajuste y nunca es elegible.
    assert "A000" not in symbols and selection["eligible_count"] == 300
    assert selection["information_before"] == f"{origin - 1}-04-01"
    assert selection["evaluation"] == [f"{origin}-01-01", f"{origin + 1}-01-01"]
    assert selection["rule"] == cohorts.RULE and selection["salt"] == cohorts.COHORT_SALT
    assert selection["targets_used"] is False and selection["future_coverage_used"] is False
    expected = [asset for asset in original["assets"] if asset["symbol"] in set(symbols)]
    assert view["assets"] == expected
    assert [item["symbol"] for item in view["coverage"]] == symbols
    assert view["samples"] == sum(asset["samples"] for asset in expected)
    assert view["counts"] == selection["counts"]
    assert view["counts"]["train"] == sum(asset["counts"]["train"] for asset in expected)
    assert view["temporal_view"] == original["temporal_view"] and view["roots"] == original["roots"]
    assert view["cohort_complete"] is False and view["scope"] == "development_snapshot"
    assert view["final_test_opened"] is False and view["candidate_count"] == 64
    assert cohort_identity(view, input_policy=HISTORICAL_MASKED) == "original_audited"
    assert original == source(origin)


def test_order_future_counts_and_added_fields_cannot_choose_or_drop_an_asset():
    original = source()
    expected, selection = cohorts.project_cohort(original, size=128)
    changed = copy.deepcopy(original)
    changed["assets"].reverse()
    changed["coverage"].reverse()
    for asset in changed["assets"]:
        asset["counts"].update(validation=0, calibration=0, evaluation=0)
        asset["future_target"] = -1000
        if asset["counts"]["train"]:
            asset["counts"]["train"] = 1
    actual, repeated = cohorts.project_cohort(reconcile(changed), size=128)
    assert repeated["assets_sha256"] == selection["assets_sha256"]
    assert [a["symbol"] for a in actual["assets"]] == [a["symbol"] for a in expected["assets"]]
    assert actual["counts"]["evaluation"] == 0 and len(actual["assets"]) == 128


def test_cohorts_are_nested_by_size_and_independent_of_the_screening_cohorts():
    original = source()
    small, _ = cohorts.project_cohort(original, size=32)
    large, _ = cohorts.project_cohort(original, size=256)
    names = lambda view: {asset["symbol"] for asset in view["assets"]}  # noqa: E731
    assert names(small) < names(large)
    screening, _ = screening_cohorts.project_origin_cohort(original, origin=2019, size=32)
    # Otra sal ordena los activos de otra forma: las dos cohortes no coinciden.
    assert names(small) != names(screening)
    everything, selection = cohorts.project_cohort(original, size=300)
    assert len(everything["assets"]) == selection["eligible_count"] == 300


@pytest.mark.parametrize("size", [0, 301, cohorts.MAX_SIZE + 1, True, 64.0, "64"])
def test_size_must_fit_the_eligible_assets(size):
    with pytest.raises(ValueError, match="activos"):
        cohorts.project_cohort(source(), size=size)


@pytest.mark.parametrize(
    "change",
    [
        dict(scope="development_snapshot"),
        dict(cohort_complete=False),
        dict(failed_assets=1),
        dict(markets=["US", "CN"]),
        dict(final_test_opened=True),
        dict(data_subset=dict(name="liquid")),
        dict(kind="other"),
        dict(roots=dict(prepared="relative", samples="/a", labels="/b")),
    ],
)
def test_only_a_complete_us_view_without_other_filters_is_a_source(change):
    original = source()
    original.update(change)
    with pytest.raises(ValueError):
        cohorts.project_cohort(original, size=8)


def test_counts_and_symbols_of_the_source_are_checked():
    original = source()
    original["counts"]["train"] += 1
    with pytest.raises(ValueError, match="no concilian"):
        cohorts.project_cohort(original, size=8)
    original = source()
    original["assets"][3]["symbol"] = original["coverage"][3]["symbol"] = "../x"
    with pytest.raises(ValueError, match="identidad US"):
        cohorts.project_cohort(original, size=8)
    original = source()
    original["assets"][3]["counts"]["train"] = -1
    with pytest.raises(ValueError, match="enteros no negativos"):
        cohorts.project_cohort(original, size=8)


def test_written_cohort_is_reused_and_a_changed_source_or_size_is_refused(tmp_path):
    origin = tmp_path / "views/fold-014/manifest.json"
    atomic_json(origin, source())
    manifest, receipt = cohorts.write_cohort(origin, tmp_path / "cohort", size=16)
    view = json.loads(manifest.read_text())
    assert len(view["assets"]) == 16 and receipt["manifest_sha256"] == sha256(manifest)
    assert receipt["kind"] == cohorts.KIND and receipt["source"]["sha256"] == sha256(origin)
    assert receipt["row_files_opened"] is False and receipt["optimizer_steps"] == 0
    stamp = manifest.stat().st_mtime_ns
    again, repeated = cohorts.write_cohort(origin, tmp_path / "cohort", size=16)
    assert again == manifest and repeated == receipt and manifest.stat().st_mtime_ns == stamp
    with pytest.raises(ValueError, match="no corresponde"):
        cohorts.write_cohort(origin, tmp_path / "cohort", size=32)
    changed = source()
    changed["assets"][5]["samples"] += 1
    changed["coverage"][5]["samples"] += 1
    changed["samples"] += 1
    atomic_json(origin, changed)
    with pytest.raises(ValueError, match="no corresponde"):
        cohorts.write_cohort(origin, tmp_path / "cohort", size=16)
    # Un manifiesto de cohorte alterado tampoco se reutiliza.
    atomic_json(origin, source())
    view["candidate_count"] = 17
    atomic_json(manifest, view)
    with pytest.raises(ValueError, match="no corresponde"):
        cohorts.write_cohort(origin, tmp_path / "cohort", size=16)
