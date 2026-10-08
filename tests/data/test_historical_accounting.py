"""Contabilidad histórica por moneda, sin objetivos ni modelos."""

import importlib
import json
import math
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from mars_titan.data.china_preparation import derive_chinese_preparation
from mars_titan.data.cohort_preparation import prepare_cohort_asset
from mars_titan.data.corpus_encoding import encode_corpus
from mars_titan.data.input_policy import HISTORICAL_MASKED, policy_identity
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training.cohort_contract import representation_identity
from tests.data.test_chinese_fact_history import edition as earlier_edition
from tests.data.test_chinese_samples import inputs as chinese_fixture
from tests.data.test_cohort_samples import Encoders
from tests.data.test_historical_input_masks import prices_only

POLICY = "historical_disjoint_accounting_v1"


def api():
    try:
        return importlib.import_module("mars_titan.data.historical_accounting")
    except ModuleNotFoundError:
        pytest.fail("Falta el enriquecimiento contable histórico")


@pytest.fixture
def history(tmp_path):
    cn_dir = tmp_path / "cn"
    cn_dir.mkdir()
    cn = chinese_fixture.__wrapped__(cn_dir)
    parent = json.loads(cn["parent_preparation"].read_text())
    prepared = Path(parent["prepared_root"])
    cn_manifest = prepared / "CN/000001.SZ/manifest.json"
    reviewed = tmp_path / "reviewed"
    strict = derive_chinese_preparation(
        cn_manifest, cn["facts_edition"], reviewed / "prepared/CN/000001.SZ", clock=cn["clock"]
    )
    origin = json.loads(cn_manifest.read_text())
    origin.update(policy_identity(HISTORICAL_MASKED), schema_version=4, missing_sources=[])
    origin["policy"].update(policy_identity(HISTORICAL_MASKED))
    atomic_json(cn_manifest, origin)
    other = prepared / "CN/000002.SZ"
    shutil.copytree(cn_manifest.parent, other)
    atomic_json(other / "manifest.json", dict(origin, symbol="000002.SZ"))

    us_dir = tmp_path / "us"
    us_dir.mkdir()
    raw, asset, us_clock = prices_only(us_dir)
    fact_path = raw / "facts.json"
    facts = {
        "Assets": {
            "units": {
                "USD": [dict(val=100.0, end="2022-12-31", filed="2023-01-03", accn="a")],
                "CAD": [dict(val=900.0, end="2022-12-31", filed=str(us_clock.days[65]), accn="b")],
            }
        },
        "Liabilities": {
            "units": {
                "USD": [dict(val=20.0, end="2022-12-31", filed="2023-01-03", accn="a")],
                "CAD": [dict(val=450.0, end="2022-12-31", filed=str(us_clock.days[65]), accn="b")],
            }
        },
        "CashAndCashEquivalentsAtCarryingValue": {
            "units": {
                "USD": [dict(val=0.0, end="2022-12-31", filed="2023-01-03", accn="a")],
                "CAD": [dict(val=50.0, end="2022-12-31", accn="unknown")],
            }
        },
    }
    fact_path.write_text(json.dumps({"filings": [{"facts": {"us-gaap": facts}}]}))
    asset["paths"]["fundamentals"] = [fact_path.name]
    asset["hashes"][fact_path.name] = sha256(fact_path)
    prepare_cohort_asset(
        raw,
        prepared,
        asset,
        us_clock,
        cohort="original_audited",
        reviews={},
        input_policy=HISTORICAL_MASKED,
    )
    assets = [
        dict(
            market=market,
            symbol=symbol,
            state="prepared",
            manifest_sha256=sha256(prepared / market / symbol / "manifest.json"),
        )
        for market, symbol in (("US", "A"), ("CN", "000001.SZ"), ("CN", "000002.SZ"))
    ]
    assets.append(
        dict(market="CN", symbol="600000.SS", state="missing_required_prices", missing=["prices"])
    )
    manifest = tmp_path / "historical.json"
    atomic_json(
        manifest,
        dict(
            **policy_identity(HISTORICAL_MASKED),
            schema_version=2,
            kind="prepared_cohort",
            status="completed",
            cohort_id="original_audited",
            candidate_count=len(assets),
            assets=assets,
            prepared_root=str(prepared),
            failed_assets=0,
            configuration=dict(markets=["US", "CN"]),
            training_ready=False,
        ),
    )

    lineage = [dict(symbol="000001.SZ", edition=str(reviewed))]
    config = dict(
        policy="copied_reviewed_chinese_editions_v1",
        lineage=lineage,
        sources={
            str((reviewed / "prepared/CN/000001.SZ/manifest.json").resolve()): sha256(
                reviewed / "prepared/CN/000001.SZ/manifest.json"
            )
        },
    )
    atomic_json(reviewed / "configuration.json", config)
    encoded = dict(
        schema_version=2,
        kind="materialized_corpus",
        cohort_id="original_audited",
        candidate_count=1,
        assets=[dict(market="CN", symbol="000001.SZ")],
        final_test_opened=False,
    )
    atomic_json(reviewed / "encoded/manifest.json", encoded)
    atomic_json(reviewed / "supervised/manifest.json", dict(encoded, kind="corpus_supervision"))
    atomic_json(
        reviewed / "report.json",
        dict(
            schema_version=1,
            kind="chinese_corpus_union",
            status="completed",
            training_ready=False,
            final_test_opened=False,
            scope="development_snapshot",
            cohort_complete=False,
            candidate_count=1,
            configuration_sha256=sha256(reviewed / "configuration.json"),
            lineage=lineage,
            artifacts={
                name: sha256(reviewed / name)
                for name in ("encoded/manifest.json", "supervised/manifest.json")
            },
        ),
    )
    return dict(
        manifest=manifest,
        prepared=prepared,
        reviewed=reviewed,
        cn=cn,
        strict_cn=strict,
        clocks={"US": us_clock, "CN": cn["clock"]},
    )


def encode(history, preparation, output, **kwargs):
    return encode_corpus(
        preparation,
        output,
        macros={},
        macro_indicators=["a", "b"],
        encoders=Encoders(),
        clocks=history["clocks"],
        input_policy=HISTORICAL_MASKED,
        accounting_policy=POLICY,
        **kwargs,
    )


def test_common_context_keeps_usd_cad_and_observed_zero_without_converting(history, tmp_path):
    output = tmp_path / "encoded"
    result = encode(history, history["manifest"], output)
    assert result["failed_assets"] == 0
    rows = pq.read_table(output / "samples/US/A/samples.parquet").to_pylist()
    assert len(rows) == 5
    assert all(
        len(row["fundamentals"]) == 78 and len(row["fundamental_missing_reasons"]) == 26
        for row in rows
    )
    first, last = rows[0]["fundamentals"], rows[-1]["fundamentals"]
    assert first[0] == pytest.approx(math.log1p(100))
    assert first[26] == 1.0
    assert first[5] == 0.0 and first[31] == 1.0
    assert first[15] == 0.0 and first[41] == 0.0
    assert last[15] == pytest.approx(math.log1p(900)) and last[41] == 1.0
    assert last[10] == pytest.approx(math.log1p(0.2))
    assert last[23:26] == [0.0, 0.0, 0.0]
    cad_available = history["clocks"]["US"].date_available(str(history["clocks"]["US"].days[65]))
    assert last[67] == pytest.approx(
        math.log1p((rows[-1]["prediction_at"] - cad_available).total_seconds() / 86400)
    )
    assert first[67] == 0.0
    assert rows[-1]["fundamental_missing_reasons"][20] == "unknown_publication"
    before = (output / "samples/US/A/samples.parquet").read_bytes()
    repeated = encode(history, history["manifest"], output)
    assert repeated["reused_assets"] == 3
    assert (output / "samples/US/A/samples.parquet").read_bytes() == before


def test_cn_enrichment_preserves_all_price_windows_and_common_representation(history, tmp_path):
    before = {
        p: (sha256(p), p.stat().st_mtime_ns) for p in history["prepared"].rglob("*") if p.is_file()
    }
    enriched = tmp_path / "enriched"
    result = api().prepare_historical_accounting(
        history["manifest"], history["reviewed"], enriched, clocks=history["clocks"]
    )
    assert result["status"] == "completed" and result["candidate_count"] == 4
    assert result["assets"][-1]["state"] == "missing_required_prices"
    output = tmp_path / "encoded"
    report = encode(history, enriched / "manifest.json", output)
    assert report["failed_assets"] == 0 and report["training_ready"] is False
    identities = []
    for market, symbol in (("US", "A"), ("CN", "000001.SZ"), ("CN", "000002.SZ")):
        folder = output / "samples" / market / symbol
        receipt = json.loads((folder / "manifest.json").read_text())
        identities.append(representation_identity(receipt, input_policy=HISTORICAL_MASKED))
        rows = pq.read_table(folder / "samples.parquet").to_pylist()
        price_rows = pq.read_table(
            history["prepared"] / market / symbol / "prices.parquet"
        ).to_pylist()
        assert [row["session"] for row in rows] == [row["session"] for row in price_rows[63:]]
        assert all(row["presence"][-1] is False and len(row["fundamentals"]) == 78 for row in rows)
        if symbol == "000001.SZ":
            earlier = [r for r in rows if r["session"] < "2023-03-10"]
            later = [r for r in rows if r["session"] >= "2023-03-10"]
            assert earlier and later
            assert all(not r["presence"][3] for r in earlier)
            assert all(r["fundamentals"][49] == 1.0 for r in later)
            assert all(
                r["fundamentals"][23] == pytest.approx(math.log1p(100_000_000)) for r in later
            )
        elif symbol == "000002.SZ":
            assert all(not r["presence"][3] and not any(r["fundamentals"]) for r in rows)
    assert identities[0] == identities[1] == identities[2]
    assert before == {p: (sha256(p), p.stat().st_mtime_ns) for p in before}


@pytest.mark.parametrize("status", ["running", "paused", "completed_with_errors"])
def test_active_parent_is_rejected_before_copying(history, tmp_path, status):
    parent = json.loads(history["manifest"].read_text())
    parent["status"] = status
    atomic_json(history["manifest"], parent)
    output = tmp_path / "enriched"
    with pytest.raises(ValueError, match="completa|preparación"):
        api().prepare_historical_accounting(
            history["manifest"], history["reviewed"], output, clocks=history["clocks"]
        )
    assert not output.exists()


def test_historical_cn_import_is_explicit_and_preserves_cas_provenance(history, tmp_path):
    source = history["prepared"] / "CN/000001.SZ/manifest.json"
    with pytest.raises(ValueError):
        derive_chinese_preparation(
            source,
            history["cn"]["facts_edition"],
            tmp_path / "strict",
            clock=history["clocks"]["CN"],
        )
    result = derive_chinese_preparation(
        source,
        history["cn"]["facts_edition"],
        tmp_path / "historical",
        clock=history["clocks"]["CN"],
        input_policy=HISTORICAL_MASKED,
    )
    assert result["schema_version"] == 4 and result["input_policy"] == HISTORICAL_MASKED
    facts = pq.read_table(tmp_path / "historical/fundamentals.parquet").to_pylist()
    assert all(row["unit"] == "CNY" and row["accounting_standard"] == "CAS" for row in facts)
    assert (tmp_path / "historical/fundamentals.parquet").read_bytes() == (
        history["cn"]["facts_edition"] / "fundamentals.parquet"
    ).read_bytes()


def test_recovery_keeps_all_candidates_and_rejects_changed_sources(history, tmp_path):
    output = tmp_path / "enriched"
    paused = api().prepare_historical_accounting(
        history["manifest"],
        history["reviewed"],
        output,
        clocks=history["clocks"],
        stop_after_assets=1,
    )
    assert paused["status"] == "paused"
    first_asset = output / "prepared/US/A"
    before = {p: (sha256(p), p.stat().st_mtime_ns) for p in first_asset.rglob("*") if p.is_file()}
    complete = api().prepare_historical_accounting(
        history["manifest"], history["reviewed"], output, clocks=history["clocks"]
    )
    assert complete["status"] == "completed" and len(complete["assets"]) == 4
    assert before == {p: (sha256(p), p.stat().st_mtime_ns) for p in before}
    changed = history["cn"]["facts_edition"] / "fundamentals.parquet"
    changed.write_bytes(changed.read_bytes() + b"alterado")
    with pytest.raises(ValueError, match="huella|fuente|partición"):
        api().prepare_historical_accounting(
            history["manifest"], history["reviewed"], output, clocks=history["clocks"]
        )


def refresh_reviewed_parent(history, edit):
    root = history["reviewed"]
    path = root / "prepared/CN/000001.SZ/manifest.json"
    parent = json.loads(path.read_text())
    edit(parent)
    atomic_json(path, parent)
    config = json.loads((root / "configuration.json").read_text())
    config["sources"][str(path.resolve())] = sha256(path)
    atomic_json(root / "configuration.json", config)
    report = json.loads((root / "report.json").read_text())
    report["configuration_sha256"] = sha256(root / "configuration.json")
    atomic_json(root / "report.json", report)


def test_earlier_cn_publication_and_later_revision_keep_their_own_dates(history, tmp_path):
    source = history["cn"]["facts_edition"]
    earlier = tmp_path / "earlier"
    earlier_edition(source, earlier, history["clocks"]["CN"])
    receipt = json.loads((earlier / "report.json").read_text())
    reference = dict(
        path=str(earlier / "report.json"),
        sha256=sha256(earlier / "report.json"),
        identity=receipt["identity"],
        artifact_sha256=receipt["sha256"],
    )
    refresh_reviewed_parent(history, lambda r: r["parents"].update(additional_facts=[reference]))
    output = tmp_path / "enriched"
    api().prepare_historical_accounting(
        history["manifest"], history["reviewed"], output, clocks=history["clocks"]
    )
    encode(history, output / "manifest.json", tmp_path / "encoded")
    rows = pq.read_table(tmp_path / "encoded/samples/CN/000001.SZ/samples.parquet").to_pylist()
    earlier_rows = [row for row in rows if row["session"] < "2023-03-10"]
    later_rows = [row for row in rows if row["session"] >= "2023-03-10"]
    assert earlier_rows and later_rows
    assert all(
        row["fundamentals"][23] == pytest.approx(math.log1p(90_000_000)) for row in earlier_rows
    )
    assert all(
        row["input_availability"]["fundamentals"]
        == history["clocks"]["CN"].date_available("2022-03-10")
        for row in earlier_rows
    )
    assert all(
        row["fundamentals"][23] == pytest.approx(math.log1p(100_000_000)) for row in later_rows
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("unit", "USD"),
        ("accounting_standard", "IFRS"),
        ("value_exact", "999"),
        ("source_file", "foreign.json"),
        ("filed", "2024-01-01"),
        ("concept", "us-gaap:Assets:USD"),
    ],
)
def test_cn_semantic_errors_fail_even_if_outer_hashes_are_updated(history, tmp_path, field, value):
    import pyarrow as pa

    directory = history["cn"]["facts_edition"]
    path = directory / "fundamentals.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0][field] = value
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    report = json.loads((directory / "report.json").read_text())
    report["sha256"] = sha256(path)
    atomic_json(directory / "report.json", report)

    def update(parent):
        parent["parents"]["facts"].update(
            sha256=sha256(directory / "report.json"), artifact_sha256=sha256(path)
        )

    refresh_reviewed_parent(history, update)
    with pytest.raises(ValueError):
        api().prepare_historical_accounting(
            history["manifest"],
            history["reviewed"],
            tmp_path / "enriched",
            clocks=history["clocks"],
        )
    assert not (tmp_path / "enriched/manifest.json").exists()


def test_common_accounting_rejects_complete_decision_filters(history, tmp_path):
    with pytest.raises(ValueError, match="filtra|completitud"):
        encode(history, history["manifest"], tmp_path / "filtered", admitted_decisions={})
    with pytest.raises(ValueError, match="históric"):
        encode_corpus(
            history["manifest"],
            tmp_path / "strict",
            macros={},
            encoders=Encoders(),
            accounting_policy=POLICY,
        )
    assert not (tmp_path / "filtered").exists() and not (tmp_path / "strict").exists()


def test_projection_order_cannot_be_changed_in_a_confirmed_receipt(history, tmp_path):
    output = tmp_path / "encoded"
    encode(history, history["manifest"], output)
    path = output / "samples/US/A/manifest.json"
    report = json.loads(path.read_text())
    report["accounting_projection"]["source_concepts"].reverse()
    atomic_json(path, report)
    result = encode(history, history["manifest"], output)
    assert result["failed_assets"] == 1
    assert result["coverage"][0]["state"] == "failed"


def test_enriched_preparation_cannot_silently_use_the_previous_accounting_catalog(
    history, tmp_path
):
    enriched = tmp_path / "enriched"
    api().prepare_historical_accounting(
        history["manifest"], history["reviewed"], enriched, clocks=history["clocks"]
    )
    with pytest.raises(ValueError, match="contable|catálogo"):
        encode_corpus(
            enriched / "manifest.json",
            tmp_path / "old-context",
            macros={},
            macro_indicators=["a", "b"],
            encoders=Encoders(),
            clocks=history["clocks"],
            input_policy=HISTORICAL_MASKED,
        )
    assert not (tmp_path / "old-context").exists()
