"""Cohortes de la campaña de comparación con manifiestos de prueba, sin abrir datos (#836)."""

import copy
import hashlib
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


def joint_source(origin=2019):
    """Vista conjunta ficticia: los activos de US de la de prueba y otros tantos de CN."""
    document = source(origin)
    view = document.pop("temporal_view")
    chinese = []
    for asset in document["assets"][:40]:
        other = dict(asset, market="CN", symbol=f"{600000 + len(chinese)}.SH")
        chinese.append(other)
    document["assets"] = sorted(
        document["assets"] + chinese, key=lambda asset: (asset["market"], asset["symbol"])
    )
    document["coverage"] = [
        dict(market=a["market"], symbol=a["symbol"], state="encoded", samples=a["samples"])
        for a in document["assets"]
    ]
    document["markets"] = ["CN", "US"]
    document["temporal_views"] = {
        market: dict(view, protocol=dict(view["protocol"], market=market))
        for market in ("US", "CN")
    }
    document["candidate_count"] = len(document["assets"])
    document["samples"] = sum(asset["samples"] for asset in document["assets"])
    return reconcile(document)


def write_source(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1))
    return path


@pytest.mark.parametrize(
    "build, markets, size", [(source, ["US"], 301), (joint_source, ["CN", "US"], 341)]
)
def test_a_full_view_cohort_is_the_source_byte_for_byte_with_its_receipt(
    tmp_path, build, markets, size
):
    path = write_source(tmp_path / "views/fold/manifest.json", build(2019))
    manifest, receipt = cohorts.write_full_view(path, tmp_path / "state/cohorts/2019", size=size)
    assert manifest.read_bytes() == path.read_bytes()
    selection = receipt["selection"]
    assert receipt["kind"] == cohorts.FULL_VIEW_KIND and receipt["manifest_sha256"] == sha256(path)
    assert receipt["source"] == dict(path=str(path.resolve()), sha256=sha256(path))
    assert selection["rule"] == cohorts.FULL_VIEW_RULE and selection["markets"] == markets
    assert selection["size"] == size and selection["evaluation"] == ["2019-01-01", "2020-01-01"]
    assert selection["targets_used"] is False and selection["future_coverage_used"] is False
    assert selection["counts"] == build(2019)["counts"]
    # La huella de los activos es la de sus pares (mercado, símbolo) ordenados.
    pairs = sorted([asset["market"], asset["symbol"]] for asset in build(2019)["assets"])
    expected = hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()
    assert selection["assets_sha256"] == expected
    assert json.loads((tmp_path / "state/cohorts/2019/cohort.json").read_text()) == receipt
    # Repetirla la reutiliza; con otra vista o con otro tamaño se rechaza.
    again = cohorts.write_full_view(path, tmp_path / "state/cohorts/2019", size=size)
    assert again == (manifest, receipt)
    with pytest.raises(ValueError, match="espera"):
        cohorts.write_full_view(path, tmp_path / "state/cohorts/otra", size=size - 1)
    assert not (tmp_path / "state/cohorts/otra").exists()
    changed = build(2019)
    changed["assets"][0]["samples"] += 1
    changed["coverage"][0]["samples"] += 1
    changed["samples"] += 1
    other = write_source(tmp_path / "views/otra/manifest.json", changed)
    with pytest.raises(ValueError, match="no es esta vista completa"):
        cohorts.write_full_view(other, tmp_path / "state/cohorts/2019", size=size)
    assert manifest.read_bytes() == path.read_bytes()


def test_the_joint_view_keeps_both_markets_one_window_and_every_asset():
    document = joint_source(2022)
    markets, fold, identities = cohorts.full_view(document)
    assert markets == ["CN", "US"] and fold["evaluation"] == ["2022-01-01", "2023-01-01"]
    assert len(identities) == 341 and identities == sorted(identities)
    assert sum(market == "CN" for market, _ in identities) == 40


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d.update(scope="development_snapshot"), "vista histórica completa"),
        (lambda d: d.update(cohort_complete=False), "vista histórica completa"),
        (lambda d: d.update(failed_assets=1), "vista histórica completa"),
        (lambda d: d.update(final_test_opened=True), "vista histórica completa|calendarios"),
        (lambda d: d.update(data_subset="x"), "vista histórica completa"),
        (lambda d: d.update(markets=["US"]), "calendarios|mercados"),
        (lambda d: d["assets"].append(dict(d["assets"][0])), "repite un activo|concilian"),
        (
            lambda d: (
                d["assets"][0].update(symbol="../x"),
                d["coverage"][0].update(symbol="../x"),
            ),
            "identidad válida",
        ),
        (lambda d: d["assets"][0]["counts"].update(train=-1), "enteros no negativos"),
        (lambda d: d["counts"].update(train=d["counts"]["train"] + 1), "no concilian"),
        (lambda d: d["roots"].update(labels="relativa"), "rutas absolutas"),
        (
            lambda d: d["temporal_views"]["CN"].update(
                fold=dict(d["temporal_views"]["CN"]["fold"], id="fold-000")
            ),
            "ventana|cortes",
        ),
    ],
)
def test_a_full_view_refuses_filters_other_markets_and_inconsistent_counts(change, message):
    document = joint_source()
    change(document)
    with pytest.raises(ValueError, match=message):
        cohorts.full_view(document)


def test_a_joint_view_cannot_lose_a_whole_market():
    document = joint_source()
    document["assets"] = [asset for asset in document["assets"] if asset["market"] == "US"]
    document["coverage"] = [item for item in document["coverage"] if item["market"] == "US"]
    document["candidate_count"] = len(document["assets"])
    document["samples"] = sum(asset["samples"] for asset in document["assets"])
    with pytest.raises(ValueError, match="calendarios|mercado"):
        cohorts.full_view(reconcile(document))


TECHNICAL = dict(technical="long_horizon_v1")


def test_a_cohort_with_an_input_extension_declares_it_and_is_another_view(tmp_path):
    origin = tmp_path / "views/fold-014/manifest.json"
    atomic_json(origin, source())
    plain, _ = cohorts.write_cohort(origin, tmp_path / "sin", size=16)
    manifest, receipt = cohorts.write_cohort(
        origin, tmp_path / "con", size=16, extensions=TECHNICAL
    )
    view, base = json.loads(manifest.read_text()), json.loads(plain.read_text())
    # Los mismos activos y contratos; solo se añade la declaración, y la huella cambia.
    assert view == dict(base, input_extensions=TECHNICAL) and sha256(manifest) != sha256(plain)
    assert receipt["input_extensions"] == TECHNICAL
    assert receipt["manifest_sha256"] == sha256(manifest)
    again = cohorts.write_cohort(origin, tmp_path / "con", size=16, extensions=TECHNICAL)
    assert again == (manifest, receipt)
    # Una cohorte guardada con extensión no vale para la campaña sin ella, ni al revés.
    with pytest.raises(ValueError, match="no corresponde"):
        cohorts.write_cohort(origin, tmp_path / "con", size=16)
    with pytest.raises(ValueError, match="no corresponde"):
        cohorts.write_cohort(origin, tmp_path / "sin", size=16, extensions=TECHNICAL)


def test_a_full_view_with_an_input_extension_is_the_view_plus_its_declaration(tmp_path):
    path = write_source(tmp_path / "views/fold/manifest.json", joint_source(2019))
    folder = tmp_path / "state/cohorts/2019"
    manifest, receipt = cohorts.write_full_view(path, folder, size=341, extensions=TECHNICAL)
    assert json.loads(manifest.read_text()) == dict(joint_source(2019), input_extensions=TECHNICAL)
    assert receipt["manifest_sha256"] == sha256(manifest) != sha256(path)
    assert receipt["input_extensions"] == TECHNICAL and receipt["source"]["sha256"] == sha256(path)
    assert receipt["selection"]["markets"] == ["CN", "US"]
    assert cohorts.write_full_view(path, folder, size=341, extensions=TECHNICAL) == (
        manifest,
        receipt,
    )
    with pytest.raises(ValueError, match="no es esta vista completa"):
        cohorts.write_full_view(path, folder, size=341)
    view = json.loads(manifest.read_text())
    view["candidate_count"] += 1
    atomic_json(manifest, view)
    with pytest.raises(ValueError, match="no es esta vista completa"):
        cohorts.write_full_view(path, folder, size=341, extensions=TECHNICAL)


@pytest.mark.parametrize("extensions", [{}, dict(technical="otra"), dict(noticias="x"), "long"])
def test_only_the_declared_input_extension_is_accepted(tmp_path, extensions):
    origin = tmp_path / "views/fold-014/manifest.json"
    atomic_json(origin, source())
    with pytest.raises(ValueError, match="extensión de entradas"):
        cohorts.write_cohort(origin, tmp_path / "cohort", size=16, extensions=extensions)
    assert not (tmp_path / "cohort").exists()
