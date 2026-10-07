"""Inventario local de cierres contables sin conceder admisión temporal."""

import importlib
import json
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import atomic_json, sha256


def module():
    try:
        return importlib.import_module("mars_titan.data.china_inventory")
    except ModuleNotFoundError:
        pytest.fail("Falta el inventario reproducible de balances CN")


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "dataset"
    prepared = tmp_path / "cohort" / "prepared"
    assets = []
    for symbol, payload in (
        (
            "000001.SZ",
            '[{"ts_code":"000001.SZ","end_date":"20241231","reserved_only":99},'
            '{"ts_code":"000001.SZ","end_date":"20221231",'
            '"total_assets":1000000000000.0000000001,"update_flag":"0"},'
            '{"ts_code":"000001.SZ","end_date":"20221231",'
            '"total_assets":1000000000000.0000000002,"update_flag":"1"},'
            '{"ts_code":"000001.SZ","end_date":"20211231",'
            '"total_assets":10,"update_flag":"0"},'
            '{"ts_code":"000001.SZ","end_date":"20211231",'
            '"total_assets":10.000e0,"update_flag":"1"}]',
        ),
        (
            "600000.SS",
            '[{"ts_code":"600000.SH","end_date":"2022-12-31","total_assets":12,'
            '"currency":"CNY","unit_multiplier":1000,"accounting_standard":"CAS",'
            '"ann_date":"20230331"},'
            '{"ts_code":"600000.SH","end_date":"20211231","ann_date":"20240101",'
            '"late_only":123}]',
        ),
    ):
        balance = source / "table" / symbol / "balance_sheet.jsonl"
        balance.parent.mkdir(parents=True)
        balance.write_text(payload)
        child = prepared / "CN" / symbol / "manifest.json"
        atomic_json(
            child,
            dict(
                schema_version=3,
                market="CN",
                symbol=symbol,
                cohort_id="original_audited",
                policy=dict(cutoff="2023-12-31"),
                sources={balance.relative_to(source).as_posix(): sha256(balance)},
                counts=dict(prices=3, news=2, fundamentals=0),
                chart_policy="regenerate_from_past_prices",
                training_ready=False,
            ),
        )
        assets.append(
            dict(market="CN", symbol=symbol, state="prepared", manifest_sha256=sha256(child))
        )
    assets.append(
        dict(
            market="CN",
            symbol="000012.SZ",
            state="missing_modalities",
            missing=["prices", "fundamentals", "charts"],
        )
    )
    manifest = tmp_path / "cohort" / "manifest.json"
    atomic_json(
        manifest,
        dict(
            schema_version=1,
            kind="prepared_cohort",
            status="completed",
            cohort_id="original_audited",
            configuration=dict(markets=["CN"], cutoff="2023-12-31", source_root=str(source)),
            prepared_root=str(prepared),
            candidate_count=3,
            failed_assets=0,
            assets=assets,
            training_ready=False,
        ),
    )
    return dict(manifest=manifest, output=tmp_path / "inventory")


def edit_json(path, update):
    content = json.loads(path.read_text())
    update(content)
    atomic_json(path, content)


def child_path(inputs, symbol="000001.SZ"):
    return inputs["manifest"].parent / "prepared" / "CN" / symbol / "manifest.json"


def refresh_child(inputs, symbol="000001.SZ"):
    edit_json(
        inputs["manifest"],
        lambda m: next(a for a in m["assets"] if a["symbol"] == symbol).update(
            manifest_sha256=sha256(child_path(inputs, symbol))
        ),
    )


def balance_path(inputs):
    return (
        inputs["manifest"].parent.parent / "dataset" / "table" / "000001.SZ" / "balance_sheet.jsonl"
    )


def replace_balance(inputs, text):
    balance_path(inputs).write_text(text)
    edit_json(
        child_path(inputs),
        lambda c: c.update(
            sources={"table/000001.SZ/balance_sheet.jsonl": sha256(balance_path(inputs))}
        ),
    )
    refresh_child(inputs)


def read_results(inputs):
    inventory = json.loads((inputs["output"] / "inventory.json").read_text())
    queue = pq.read_table(inputs["output"] / "reconciliation-queue.parquet").to_pylist()
    return inventory, {(row["symbol"], row["period_end"]): row for row in queue}


def test_inventory_keeps_all_candidates_original_locators_and_decimal_duplicates(inputs):
    report = module().inventory_chinese_balances(**inputs)
    inventory, queue = read_results(inputs)
    assert report["candidate_count"] == 3
    assert report["counts"] == {"inventoried": 2, "not_prepared": 1}
    assert report["period_jobs"] == 4
    assert report["configuration"]["periods"] == ["2021-12-31", "2022-12-31"]
    assert inventory["candidates"][2]["exclusion_reasons"] == ["prices", "fundamentals", "charts"]
    old = queue["000001.SZ", "2021-12-31"]
    assert old["original_record_ordinals"] == [4, 5]
    assert old["records"] == old["inspected_records"] == 2
    assert old["exact_duplicate_records"] == 0
    assert old["duplicate_records_without_update_flag"] == 1
    assert (
        old["record_fingerprints"][0]["without_update_flag_sha256"]
        == old["record_fingerprints"][1]["without_update_flag_sha256"]
    )
    recent = queue["000001.SZ", "2022-12-31"]
    assert recent["original_record_ordinals"] == [2, 3]
    assert recent["duplicate_records_without_update_flag"] == 0
    assert recent["differing_numeric_fields"] == ["total_assets"]
    assert (
        recent["record_fingerprints"][0]["without_update_flag_sha256"]
        != recent["record_fingerprints"][1]["without_update_flag_sha256"]
    )
    assert recent["source_locator"] == "array_record_1_based"
    assert recent["balance_sha256"] == sha256(balance_path(inputs))
    for name, artifact in report["artifacts"].items():
        assert sha256(inputs["output"] / name) == artifact["sha256"]
    assert inventory["training_ready"] is report["training_ready"] is False
    assert report["final_test_opened"] is False
    assert "1000000000000.0000000001" not in (inputs["output"] / "inventory.json").read_text()


def test_declared_context_and_ss_sh_are_preserved_without_inference(inputs):
    module().inventory_chinese_balances(**inputs)
    _, queue = read_results(inputs)
    row = queue["600000.SS", "2022-12-31"]
    assert row["source_symbols"] == ["600000.SH"]
    assert json.loads(row["declared_metadata_json"])["units"] == {
        "currency": ["CNY"],
        "unit_multiplier": ["1000"],
    }
    unverified = queue["000001.SZ", "2021-12-31"]
    assert json.loads(unverified["declared_metadata_json"])["units"] == {}
    assert "currency_and_scale_not_declared" in unverified["pending_reasons"]
    assert all(r["temporal_admission_granted"] is False for r in queue.values())


def test_only_requested_closes_before_declared_2024_publication_are_materialized(inputs):
    module().inventory_chinese_balances(**inputs)
    inventory, queue = read_results(inputs)
    excluded = queue["600000.SS", "2021-12-31"]
    assert excluded["records"] == 1 and excluded["inspected_records"] == 0
    assert excluded["record_fingerprints"] == []
    assert "declared_publication_from_2024_not_materialized" in excluded["pending_reasons"]
    assert all(
        "reserved_only" not in row["numeric_fields"] and "late_only" not in row["numeric_fields"]
        for row in queue.values()
    )
    assert inventory["original_context_recovered"] is False


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.update(cohort_id="externally_verified"),
        lambda m: m.update(candidate_count=True),
        lambda m: m.update(candidate_count=2),
        lambda m: m["configuration"].update(markets=["US"]),
        lambda m: m["configuration"].update(cutoff="2024-01-01"),
        lambda m: m["assets"][0].update(market="US"),
        lambda m: m["assets"][0].update(symbol="../../outside"),
        lambda m: m["assets"].append(m["assets"][0]) or m.update(candidate_count=4),
        lambda m: m.update(failed_assets=1),
    ],
)
def test_invalid_census_is_rejected_without_an_edition(inputs, change):
    edit_json(inputs["manifest"], change)
    with pytest.raises(ValueError):
        module().inventory_chinese_balances(**inputs)
    assert not inputs["output"].exists()


def test_duplicate_census_keys_are_rejected(inputs):
    payload = (
        inputs["manifest"]
        .read_text()
        .replace('"candidate_count": 3', '"candidate_count": 3, "candidate_count": 3')
    )
    inputs["manifest"].write_text(payload)
    with pytest.raises(ValueError, match="duplicada"):
        module().inventory_chinese_balances(**inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize(
    "payload",
    [
        '[{"end_date":"20211231","end_date":"20221231"}]',
        '[{"end_date":"20221231","total_assets":1,"total_assets":2}]',
        '[{"end_date":"20221231"},',
        '{"item":{"end_date":"20221231"}}',
        "[1]",
        '[{"end_date":"20221231","nested":{"value":1}}]',
        '[{"ts_code":"000002.SZ","end_date":"20221231","total_assets":1}]',
    ],
)
def test_invalid_original_is_an_asset_error_and_does_not_hide_other_candidates(inputs, payload):
    replace_balance(inputs, payload)
    report = module().inventory_chinese_balances(**inputs)
    inventory, queue = read_results(inputs)
    assert report["status"] == "completed_with_errors"
    assert report["counts"] == {"source_requires_review": 1, "inventoried": 1, "not_prepared": 1}
    assert inventory["candidates"][0]["exclusion_reasons"]
    assert {symbol for symbol, _ in queue} == {"600000.SS"}


@pytest.mark.parametrize("which", ["child", "balance"])
def test_wrong_source_hash_is_not_treated_as_an_inventoried_asset(inputs, which):
    path = child_path(inputs) if which == "child" else balance_path(inputs)
    path.write_bytes(path.read_bytes() + b" ")
    report = module().inventory_chinese_balances(**inputs)
    assert report["counts"]["source_requires_review"] == 1
    assert report["period_jobs"] == 2


@pytest.mark.parametrize("relative", ["../balance_sheet.jsonl", "/tmp/balance_sheet.jsonl"])
def test_balance_locator_cannot_escape_the_declared_source_root(inputs, relative):
    edit_json(child_path(inputs), lambda c: c.update(sources={relative: "a" * 64}))
    refresh_child(inputs)
    result = module().inventory_chinese_balances(**inputs)
    assert result["counts"]["source_requires_review"] == 1


def test_intermediate_symlink_is_not_followed(inputs):
    balance = balance_path(inputs)
    original = balance.parent.rename(balance.parent.with_name("elsewhere"))
    balance.parent.symlink_to(original, target_is_directory=True)
    result = module().inventory_chinese_balances(**inputs)
    assert result["counts"]["source_requires_review"] == 1


@pytest.mark.parametrize("which", ["census", "child", "balance"])
def test_source_changed_during_staging_prevents_publication(inputs, monkeypatch, which):
    api = module()
    original = api.atomic_parquet

    def write_and_change(path, table):
        original(path, table)
        source = {
            "census": inputs["manifest"],
            "child": child_path(inputs),
            "balance": balance_path(inputs),
        }[which]
        source.write_bytes(source.read_bytes() + b" ")

    monkeypatch.setattr(api, "atomic_parquet", write_and_change)
    with pytest.raises(ValueError, match="cambi"):
        api.inventory_chinese_balances(**inputs)
    assert not inputs["output"].exists()


def test_interrupted_write_publishes_nothing_and_a_new_attempt_can_succeed(inputs, monkeypatch):
    api = module()
    original = api.atomic_parquet

    def fail(path, table):
        original(path, table)
        raise OSError("Corte simulado")

    monkeypatch.setattr(api, "atomic_parquet", fail)
    with pytest.raises(OSError, match="Corte"):
        api.inventory_chinese_balances(**inputs)
    assert not inputs["output"].exists()
    monkeypatch.setattr(api, "atomic_parquet", original)
    assert api.inventory_chinese_balances(**inputs)["period_jobs"] == 4
    with pytest.raises(FileExistsError):
        api.inventory_chinese_balances(**inputs)


@pytest.mark.parametrize("where", ["source", "prepared"])
def test_output_must_be_separate_from_inputs(inputs, where):
    metadata = json.loads(inputs["manifest"].read_text())
    root = Path(
        metadata["configuration"]["source_root"] if where == "source" else metadata["prepared_root"]
    )
    inputs["output"] = root / "new_inventory"
    with pytest.raises(ValueError):
        module().inventory_chinese_balances(**inputs)
    assert not inputs["output"].exists()


def test_cli_uses_the_same_fixed_period_contract(inputs, capsys):
    module().main(["--manifest", str(inputs["manifest"]), "--output", str(inputs["output"])])
    result = json.loads(capsys.readouterr().out)
    assert result["candidate_count"] == 3
    assert result["period_jobs"] == 4


def test_explicit_quarters_are_selected_and_declared_before_reading(inputs):
    replace_balance(
        inputs,
        '[{"ts_code":"000001.SZ","end_date":"20220331","total_assets":20}, '
        '{"ts_code":"000001.SZ","end_date":"20221231","total_assets":30}]',
    )
    result = module().inventory_chinese_balances(**inputs, periods=["2022-03-31"])
    inventory, queue = read_results(inputs)
    assert result["configuration"]["periods"] == ["2022-03-31"]
    assert inventory["configuration"]["periods"] == ["2022-03-31"]
    assert set(queue) == {("000001.SZ", "2022-03-31"), ("600000.SS", "2022-03-31")}
    assert queue["000001.SZ", "2022-03-31"]["original_record_ordinals"] == [1]
    assert queue["600000.SS", "2022-03-31"]["records"] == 0
    assert "no_records_at_requested_close" in queue["600000.SS", "2022-03-31"]["pending_reasons"]


@pytest.mark.parametrize(
    "periods",
    [
        [],
        ["2024-01-01"],
        ["1989-12-31"],
        ["2022-02-30"],
        ["20221231"],
        ["2022-12-31", "2021-12-31"],
        ["2021-12-31", "2021-12-31"],
        [True],
        "2022-12-31",
        [f"{year}-12-31" for year in range(1800, 1929)],
    ],
)
def test_invalid_period_selection_never_opens_a_source(inputs, monkeypatch, periods):
    api = module()

    def forbidden(*args, **kwargs):
        pytest.fail("La selección inválida llegó a leer fuentes")

    monkeypatch.setattr(api, "read_manifest", forbidden)
    with pytest.raises(ValueError):
        api.inventory_chinese_balances(**inputs, periods=periods)
    assert not inputs["output"].exists()


def test_cli_accepts_explicit_periods(inputs, capsys):
    module().main(
        [
            "--manifest",
            str(inputs["manifest"]),
            "--output",
            str(inputs["output"]),
            "--period",
            "2021-12-31",
            "--period",
            "2023-03-31",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["configuration"]["periods"] == ["2021-12-31", "2023-03-31"]


def test_too_many_candidates_are_rejected_before_opening_children(inputs):
    edit_json(
        inputs["manifest"],
        lambda m: m.update(
            candidate_count=1025,
            assets=[
                dict(
                    market="CN", symbol=f"{index:06}.SZ", state="prepared", manifest_sha256="a" * 64
                )
                for index in range(1025)
            ],
        ),
    )
    with pytest.raises(ValueError):
        module().inventory_chinese_balances(**inputs)
    assert not inputs["output"].exists()


def test_record_limit_reports_asset_failure_even_when_requested_closes_are_absent(inputs):
    replace_balance(inputs, "[" + ",".join(["{}"] * 100001) + "]")
    result = module().inventory_chinese_balances(**inputs)
    inventory, _ = read_results(inputs)
    assert result["counts"]["source_requires_review"] == 1
    assert "presupuesto de registros" in inventory["candidates"][0]["detail"]


def test_large_balance_is_rejected_before_hashing_or_parsing(inputs, monkeypatch):
    balance = balance_path(inputs)
    with balance.open("r+b") as stream:
        stream.truncate(64 * 1024**2 + 1)
    api = module()
    original = api.sha256

    def bounded_hash(path):
        if Path(path) == balance:
            pytest.fail("Se intentó leer el balance que excede el límite")
        return original(path)

    monkeypatch.setattr(api, "sha256", bounded_hash)
    result = api.inventory_chinese_balances(**inputs)
    assert result["counts"]["source_requires_review"] == 1


def test_total_selected_records_have_a_global_budget(inputs, monkeypatch):
    api = module()
    monkeypatch.setattr(api, "_MAX_RECORDS", 5)
    with pytest.raises(ValueError, match="total"):
        api.inventory_chinese_balances(**inputs)
    assert not inputs["output"].exists()


def test_failed_preparation_keeps_its_reason_without_reading_nonexistent_sources(inputs):
    def failed(meta):
        meta["assets"][0].update(
            state="failed", error_type="ValueError", detail="Faltan originales"
        )
        meta.update(failed_assets=1, status="completed_with_errors")

    edit_json(inputs["manifest"], failed)
    result = module().inventory_chinese_balances(**inputs)
    inventory, _ = read_results(inputs)
    assert result["counts"] == {"not_prepared": 2, "inventoried": 1}
    assert inventory["candidates"][0]["preparation_error"] == {
        "error_type": "ValueError",
        "detail": "Faltan originales",
    }


def test_ss_and_sh_aliases_cannot_duplicate_the_same_census_candidate(inputs):
    def duplicate(meta):
        meta["assets"].append(
            dict(market="CN", symbol="600000.SH", state="prepared", manifest_sha256="a" * 64)
        )
        meta["candidate_count"] += 1

    edit_json(inputs["manifest"], duplicate)
    with pytest.raises(ValueError, match="duplicados"):
        module().inventory_chinese_balances(**inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize("state", [[], {}])
def test_malformed_state_has_an_explicit_validation_error(inputs, state):
    edit_json(inputs["manifest"], lambda m: m["assets"][0].update(state=state))
    with pytest.raises(ValueError):
        module().inventory_chinese_balances(**inputs)


def test_equal_decimal_lexemes_are_exact_duplicates_when_metadata_also_matches(inputs):
    replace_balance(
        inputs,
        '[{"end_date":"20211231","total_assets":10}, '
        '{"total_assets":1.000e1,"end_date":"20211231"}]',
    )
    module().inventory_chinese_balances(**inputs)
    _, queue = read_results(inputs)
    period = queue["000001.SZ", "2021-12-31"]
    assert period["exact_duplicate_records"] == 1
    assert period["record_fingerprints"][0]["sha256"] == period["record_fingerprints"][1]["sha256"]


@pytest.mark.parametrize("name", ["inventory.json", "reconciliation-queue.parquet", "report.json"])
def test_altered_staged_artifacts_are_not_published(inputs, monkeypatch, name):
    api = module()
    original = api.atomic_json

    def write_and_corrupt(path, content):
        original(path, content)
        if path.name == "report.json":
            target = path.parent / name
            if name == "report.json":
                damaged = json.loads(target.read_text())
                damaged["training_ready"] = 0
                target.write_text(json.dumps(damaged))
            else:
                target.write_bytes(target.read_bytes() + b"alterado")

    monkeypatch.setattr(api, "atomic_json", write_and_corrupt)
    with pytest.raises(ValueError):
        api.inventory_chinese_balances(**inputs)
    assert not inputs["output"].exists()


def test_code_identity_is_captured_before_reading_sources(inputs, monkeypatch):
    api = module()
    read, digest = api.read_manifest, api.sha256
    changed = False

    def read_then_change(*args, **kwargs):
        nonlocal changed
        result = read(*args, **kwargs)
        changed = True
        return result

    def changed_code(path):
        return "a" * 64 if changed and Path(path).name == "china_inventory.py" else digest(path)

    monkeypatch.setattr(api, "read_manifest", read_then_change)
    monkeypatch.setattr(api, "sha256", changed_code)
    with pytest.raises(ValueError, match="código cambió"):
        api.inventory_chinese_balances(**inputs)
    assert not inputs["output"].exists()


def test_malformed_close_cannot_be_normalized_into_a_requested_period(inputs):
    replace_balance(
        inputs,
        '[{"ts_code":"000001.SZ","end_date":"2021--12--31","total_assets":10}]',
    )
    result = module().inventory_chinese_balances(**inputs)
    inventory, queue = read_results(inputs)
    assert result["counts"]["source_requires_review"] == 1
    assert inventory["candidates"][0]["status"] == "source_requires_review"
    assert {symbol for symbol, _ in queue} == {"600000.SS"}


@pytest.mark.parametrize(
    ("publication", "reason"),
    [
        (" 2024-01-01", "declared_publication_from_2024_not_materialized"),
        ("20240101 ", "declared_publication_from_2024_not_materialized"),
        ("unknown", "declared_publication_uninterpretable_not_materialized"),
        (1704067200000, "declared_publication_uninterpretable_not_materialized"),
        (False, "declared_publication_uninterpretable_not_materialized"),
        ("2023-02-30", "declared_publication_uninterpretable_not_materialized"),
        ("2023/03/31", "declared_publication_uninterpretable_not_materialized"),
    ],
)
def test_future_or_uninterpretable_publication_is_never_materialized(inputs, publication, reason):
    replace_balance(
        inputs,
        json.dumps(
            [
                dict(
                    ts_code="000001.SZ",
                    end_date="20211231",
                    ann_date=publication,
                    hidden_amount=123,
                )
            ]
        ),
    )
    module().inventory_chinese_balances(**inputs)
    _, queue = read_results(inputs)
    period = queue["000001.SZ", "2021-12-31"]
    assert period["original_record_ordinals"] == [1]
    assert period["records"] == 1 and period["inspected_records"] == 0
    assert period["record_fingerprints"] == []
    assert period["numeric_fields"] == []
    assert reason in period["pending_reasons"]


@pytest.mark.parametrize("publication", [" 2023-03-31 ", "20230331", 20230331, "", None])
def test_publication_selection_accepts_calendar_dates_without_claiming_admission(
    inputs, publication
):
    replace_balance(
        inputs,
        json.dumps(
            [
                dict(
                    ts_code="000001.SZ",
                    end_date="2021-12-31",
                    ann_date=publication,
                    total_assets=10,
                )
            ]
        ),
    )
    module().inventory_chinese_balances(**inputs)
    _, queue = read_results(inputs)
    period = queue["000001.SZ", "2021-12-31"]
    assert period["inspected_records"] == 1
    assert period["numeric_fields"] == ["total_assets"]
    assert period["temporal_admission_granted"] is False


@pytest.mark.parametrize("damaged", [False, True])
def test_inventory_links_contiguous_exhaustive_queue_ranges_without_period_details(inputs, damaged):
    # Una exclusión entre activos no debe desplazar ni solapar sus localizadores.
    edit_json(inputs["manifest"], lambda m: m["assets"].insert(1, m["assets"].pop()))
    if damaged:
        replace_balance(inputs, "[")
    result = module().inventory_chinese_balances(**inputs)
    inventory = json.loads((inputs["output"] / "inventory.json").read_text())
    table = pq.read_table(inputs["output"] / "reconciliation-queue.parquet")
    assert inventory["queue"] == {
        "path": "reconciliation-queue.parquet",
        "rows": result["period_jobs"],
        "indexing": "zero_based_half_open",
    }
    cursor = 0
    for candidate in inventory["candidates"]:
        assert (
            not {"periods", "numeric_fields", "record_fingerprints", "declared_metadata"}
            & candidate.keys()
        )
        bounds = candidate["queue_rows"]
        assert bounds["start"] == cursor
        start, stop = bounds["start"], bounds["stop"]
        rows = table.slice(start, stop - start).to_pylist()
        if candidate["status"] == "inventoried":
            assert [row["period_end"] for row in rows] == ["2021-12-31", "2022-12-31"]
            assert {row["symbol"] for row in rows} == {candidate["symbol"]}
            assert candidate["pending_reasons"] == sorted(
                {reason for row in rows for reason in row["pending_reasons"]}
            )
        else:
            assert start == stop
        cursor = stop
    assert cursor == len(table) == result["period_jobs"]


def test_distinct_values_use_exact_decimal_equality_and_only_hash_complete_records(monkeypatch):
    api = module()
    fingerprint, seen = api._fingerprint, []

    def observe(record):
        seen.append(tuple(record))
        return fingerprint(record)

    monkeypatch.setattr(api, "_fingerprint", observe)
    rows = [
        (
            1,
            dict(
                end_date="20211231",
                equivalent=10,
                zero=0,
                precise=Decimal("1000000000000.0000000001"),
            ),
        ),
        (
            2,
            dict(
                end_date="20211231",
                equivalent=Decimal("1e1"),
                zero=Decimal("-0.0"),
                precise=Decimal("1000000000000.0000000002"),
            ),
        ),
        (
            3,
            dict(
                end_date="20211231",
                equivalent=Decimal("10.000"),
                zero=Decimal("0E9"),
                precise=Decimal("1000000000000.0000000001"),
            ),
        ),
    ]
    result = api._period_summary(
        "2021-12-31", {i: ("2021-12-31", ()) for i in (1, 2, 3)}, rows, "000001.SZ"
    )
    assert result["differing_numeric_fields"] == ["precise"]
    assert result["exact_duplicate_records"] == 1
    assert result["record_fingerprints"][0]["sha256"] == result["record_fingerprints"][2]["sha256"]
    assert len(seen) == 6
    assert all("end_date" in fields for fields in seen)
