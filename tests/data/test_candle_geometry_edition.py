"""Derivación por lotes desde una edición confirmada, sin cargar codificadores."""

import json
from datetime import timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data import candle_geometry_edition as edition
from mars_titan.data.candle_geometry import MODEL, geometry_chart
from mars_titan.data.corpus_encoding import encode_corpus
from mars_titan.data.storage import atomic_json, sha256
from tests.data.test_candle_geometry_integration import prepared


def source_edition(tmp_path, *, company_factors=False):
    preparation, options = prepared(tmp_path)
    options["company_factors"] = company_factors
    source = tmp_path / "original"
    encode_corpus(preparation, source, **options)
    # Los valores distintos de cero detectan una recodificación accidental del texto.
    path = source / "samples/US/A/samples.parquet"
    table = pq.read_table(path)
    news = pa.array([[0.25] * 384] * len(table), type=table.schema.field("news").type)
    table = table.set_column(table.schema.get_field_index("news"), "news", news)
    table = table.append_column("provenance_note", pa.array(["source-only"] * len(table)))
    table = table.replace_schema_metadata({b"origin": b"fixture-provenance"})
    replace_samples(source, table)
    return source, options["clocks"]


def replace_samples(root, table):
    folder = root / "samples/US/A"
    pq.write_table(table, folder / "samples.parquet")
    receipt = json.loads((folder / "manifest.json").read_text())
    receipt["samples_sha256"] = sha256(folder / "samples.parquet")
    atomic_json(folder / "manifest.json", receipt)


def test_derive_preserves_every_other_column_and_reuses_no_models(tmp_path, monkeypatch):
    source, clocks = source_edition(tmp_path)
    from mars_titan.data import corpus_encoding

    def no_models(*args, **kwargs):
        raise AssertionError("La derivación no carga codificadores")

    monkeypatch.setattr(corpus_encoding, "FrozenEncoders", no_models)
    before = {p: sha256(p) for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "geometry"
    result = edition.derive_geometry_edition(source, output, batch_rows=2, clocks=clocks)
    assert result["cohort_complete"] and not result["training_ready"]
    assert not result["final_test_opened"] and result["samples"] == 5
    assert result["configuration"]["encoders"]["image_model"] == MODEL
    assert result["chart_derivation"]["comparison"]["same_inputs"]
    assert result["chart_derivation"]["timing_seconds"]["geometry"] > 0
    assert result["chart_derivation"]["timing_seconds"]["read"] > 0
    assert result["chart_derivation"]["timing_seconds"]["write"] > 0
    old, new = [pq.read_table(p / "samples/US/A/samples.parquet") for p in (source, output)]
    columns = [name for name in old.column_names if name not in {"charts", "chart_hash"}]
    assert old.select(columns).equals(new.select(columns), check_metadata=True)
    raw = pq.read_table(tmp_path / "prepared/US/A/prices.parquet").to_pydict()
    ohlc = np.column_stack([raw[name] for name in ("open", "high", "low", "close")])
    for row in new.to_pylist():
        present = np.ones(64, dtype=bool)
        final = row["price_end_index"] + 1
        present[10 - (final - 63)] = False
        digest, vector = geometry_chart(ohlc, end_index=row["price_end_index"], present=present)
        assert row["chart_hash"] == digest
        np.testing.assert_array_equal(row["charts"], vector)
    assert all(sha256(path) == digest for path, digest in before.items())
    with pytest.raises(FileExistsError):
        edition.derive_geometry_edition(source, output, clocks=clocks)


@pytest.mark.parametrize("field", ["price_end_index", "session", "prediction_at", "order"])
def test_rejects_a_sample_bound_to_a_different_close(tmp_path, field):
    source, clocks = source_edition(tmp_path)
    table = pq.read_table(source / "samples/US/A/samples.parquet")
    rows = table.to_pylist()
    if field == "order":
        rows.reverse()
    elif field == "session":
        rows[0][field] = rows[1][field]
    elif field == "prediction_at":
        rows[0][field] += timedelta(minutes=1)
    else:
        rows[0][field] += 1
    replace_samples(source, pa.Table.from_pylist(rows, schema=table.schema))
    output = tmp_path / "geometry"
    with pytest.raises(ValueError, match="índice|calendario|sesión|orden"):
        edition.derive_geometry_edition(source, output, clocks=clocks)
    assert not output.exists()


def test_rejects_inconsistent_calendar_identity(tmp_path):
    source, clocks = source_edition(tmp_path)
    clocks["US"].decisions[2] += timedelta(seconds=1)
    with pytest.raises(ValueError, match="calendario"):
        edition.derive_geometry_edition(source, tmp_path / "geometry", clocks=clocks)


@pytest.mark.parametrize("changed", ["samples", "prices", "manifest"])
def test_source_changed_during_geometry_is_never_published(tmp_path, monkeypatch, changed):
    source, clocks = source_edition(tmp_path)
    paths = dict(
        samples=source / "samples/US/A/samples.parquet",
        prices=tmp_path / "prepared/US/A/prices.parquet",
        manifest=source / "manifest.json",
    )
    calculate = edition.geometry_chart
    called = False

    def mutate(*args, **kwargs):
        nonlocal called
        if not called:
            called = True
            with paths[changed].open("ab") as stream:
                stream.write(b" ")
        return calculate(*args, **kwargs)

    monkeypatch.setattr(edition, "geometry_chart", mutate)
    output = tmp_path / "geometry"
    with pytest.raises(ValueError, match="cambi|huella"):
        edition.derive_geometry_edition(source, output, batch_rows=2, clocks=clocks)
    assert not output.exists()


def test_interruption_leaves_no_published_edition_and_can_resume(tmp_path, monkeypatch):
    source, clocks = source_edition(tmp_path)
    output = tmp_path / "geometry"
    compare = edition.compare_representations

    def interrupt(*args, **kwargs):
        raise InterruptedError("Corte tras confirmar el activo")

    monkeypatch.setattr(edition, "compare_representations", interrupt)
    with pytest.raises(InterruptedError):
        edition.derive_geometry_edition(source, output, clocks=clocks)
    assert not output.exists()

    def already_calculated(*args, **kwargs):
        raise AssertionError("No debe recalcular un activo confirmado")

    monkeypatch.setattr(edition, "geometry_chart", already_calculated)
    monkeypatch.setattr(edition, "compare_representations", compare)
    result = edition.derive_geometry_edition(source, output, clocks=clocks)
    assert result["chart_derivation"]["reused_assets"] == 1
    assert result["samples"] == 5


def test_interrupted_parquet_is_not_confirmed(tmp_path, monkeypatch):
    source, clocks = source_edition(tmp_path)
    output = tmp_path / "geometry"
    original = edition.atomic_parquet_batches

    def fail(path, tables):
        def interrupted():
            yield next(iter(tables))
            raise InterruptedError("Corte durante un lote")

        return original(path, interrupted())

    monkeypatch.setattr(edition, "atomic_parquet_batches", fail)
    with pytest.raises(InterruptedError):
        edition.derive_geometry_edition(source, output, batch_rows=2, clocks=clocks)
    assert not output.exists()
    monkeypatch.setattr(edition, "atomic_parquet_batches", original)
    result = edition.derive_geometry_edition(source, output, batch_rows=2, clocks=clocks)
    assert result["chart_derivation"]["reused_assets"] == 0


def test_limits_reject_oversized_decoded_batches(tmp_path, monkeypatch):
    source, clocks = source_edition(tmp_path)
    monkeypatch.setattr(edition, "MAX_BYTES", 100)
    with pytest.raises(ValueError, match="presupuesto"):
        edition.derive_geometry_edition(source, tmp_path / "geometry", clocks=clocks)


@pytest.mark.parametrize("batch", [0, True, 513])
def test_batch_limit_is_validated_before_reading(batch, tmp_path):
    with pytest.raises(ValueError, match="lote"):
        edition.derive_geometry_edition(tmp_path / "missing", tmp_path / "new", batch_rows=batch)


def test_derived_window_does_not_depend_on_later_prepared_prices(tmp_path):
    source, clocks = source_edition(tmp_path)
    configuration = json.loads((source / "configuration.json").read_text())
    timing = dict(read=0.0, price_preparation=0.0, geometry=0.0)
    prices = edition._prices(
        tmp_path / "prepared/US/A/prices.parquet",
        clocks["US"],
        configuration["price_window"],
        timing,
    )
    path = source / "samples/US/A/samples.parquet"
    first = pq.read_table(path, columns=["price_end_index"])[0][0].as_py()

    def materialize():
        return pa.concat_tables(
            list(
                edition._replace_batches(
                    path, prices, clocks["US"], 0.0, 2, timing, dict(largest_input_batch_bytes=0)
                )
            )
        )

    before = materialize()
    prices[1][first + 1 :] *= 7.0
    after = materialize()
    assert before.slice(0, 1).equals(after.slice(0, 1), check_metadata=True)
    assert before["chart_hash"].slice(1).to_pylist() != after["chart_hash"].slice(1).to_pylist()


def test_company_factors_are_copied_byte_for_byte(tmp_path):
    source, clocks = source_edition(tmp_path, company_factors=True)
    output = tmp_path / "geometry"
    edition.derive_geometry_edition(source, output, clocks=clocks)
    relative = "samples/US/A/company-factors.parquet"
    assert sha256(source / relative) == sha256(output / relative)


@pytest.mark.parametrize("changed", ["samples", "receipt", "identity"])
def test_recovery_rejects_tampered_confirmed_output(tmp_path, monkeypatch, changed):
    source, clocks = source_edition(tmp_path)
    output = tmp_path / "geometry"
    compare = edition.compare_representations

    def interrupt(*args, **kwargs):
        raise InterruptedError

    monkeypatch.setattr(edition, "compare_representations", interrupt)
    with pytest.raises(InterruptedError):
        edition.derive_geometry_edition(source, output, clocks=clocks)
    stage = tmp_path / ".geometry.geometry-staging"
    folder = stage / "samples/US/A"
    if changed == "samples":
        with (folder / "samples.parquet").open("ab") as stream:
            stream.write(b" ")
    else:
        path = folder / "manifest.json" if changed == "receipt" else stage / "derivation.json"
        value = json.loads(path.read_text())
        if changed == "receipt":
            value["encoders"]["image_model"] = "other"
        else:
            value["representation"] = "other"
        atomic_json(path, value)
    monkeypatch.setattr(edition, "compare_representations", compare)
    with pytest.raises(ValueError, match="recuperación"):
        edition.derive_geometry_edition(source, output, clocks=clocks)
    assert not output.exists()


def test_empty_encoded_asset_and_missing_asset_keep_coverage(tmp_path):
    source, clocks = source_edition(tmp_path)
    folder = source / "samples/US/A"
    table = pq.read_table(folder / "samples.parquet").slice(0, 0)
    replace_samples(source, table)
    receipt = json.loads((folder / "manifest.json").read_text())
    receipt["samples"] = 0
    atomic_json(folder / "manifest.json", receipt)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["samples"] = manifest["coverage"][0]["samples"] = 0
    manifest["assets"] = []
    manifest["candidate_count"] += 1
    manifest["coverage"].append(dict(market="US", symbol="EMPTY", state="missing_required_prices"))
    atomic_json(source / "manifest.json", manifest)
    output = tmp_path / "geometry"
    result = edition.derive_geometry_edition(source, output, clocks=clocks)
    assert result["coverage"][-1] == manifest["coverage"][-1]
    assert result["candidate_count"] == 2 and result["samples"] == 0
    assert pq.read_table(output / "samples/US/A/samples.parquet").schema == table.schema


def test_cli_uses_no_caller_supplied_encoder(tmp_path, capsys):
    source, _ = source_edition(tmp_path)
    assert edition.main(["--source", str(source), "--output", str(tmp_path / "geometry")]) == 0
    assert json.loads(capsys.readouterr().out)["samples"] == 5


def test_source_symlink_and_output_inside_source_are_rejected(tmp_path):
    source, clocks = source_edition(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="enlace"):
        edition.derive_geometry_edition(link, tmp_path / "geometry", clocks=clocks)
    with pytest.raises(ValueError, match="fuera"):
        edition.derive_geometry_edition(source, source / "geometry", clocks=clocks)


@pytest.mark.parametrize("field", ["available_at", "session", "missing"])
def test_prepared_prices_reject_future_and_missing_availability(tmp_path, field):
    source, clocks = source_edition(tmp_path)
    path = tmp_path / "prepared/US/A/prices.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    if field == "session":
        rows[-1][field] = "2024-01-03"
    else:
        rows[-1]["available_at"] = None if field == "missing" else clocks["US"].decisions[-1]
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    config = json.loads((source / "configuration.json").read_text())
    with pytest.raises(ValueError, match="futuro|reserva|ausentes"):
        edition._prices(
            path, clocks["US"], config["price_window"], dict(read=0.0, price_preparation=0.0)
        )


def test_invalid_chart_width_is_rejected_before_derivation(tmp_path):
    source, clocks = source_edition(tmp_path)
    path = source / "samples/US/A/samples.parquet"
    table = pq.read_table(path)
    table = table.set_column(
        table.schema.get_field_index("charts"), "charts", pa.array([[0.0] * 4] * len(table))
    )
    replace_samples(source, table)
    with pytest.raises(ValueError, match="512"):
        edition.derive_geometry_edition(source, tmp_path / "geometry", clocks=clocks)


def test_comparison_also_keeps_schema_metadata(tmp_path):
    source, clocks = source_edition(tmp_path)
    output = tmp_path / "geometry"
    edition.derive_geometry_edition(source, output, clocks=clocks)
    table = pq.read_table(output / "samples/US/A/samples.parquet")
    replace_samples(output, table.replace_schema_metadata({b"origin": b"changed"}))
    with pytest.raises(ValueError, match="columnas"):
        edition.compare_representations(source, output)


def test_changed_source_bytes_restored_before_verification_are_rejected(tmp_path, monkeypatch):
    source, clocks = source_edition(tmp_path)
    path = tmp_path / "prepared/US/A/prices.parquet"
    original = path.read_bytes()
    calculate = edition.geometry_chart

    def restore(*args, **kwargs):
        path.write_bytes(original)
        return calculate(*args, **kwargs)

    monkeypatch.setattr(edition, "geometry_chart", restore)
    with pytest.raises(ValueError, match="cambió"):
        edition.derive_geometry_edition(source, tmp_path / "geometry", clocks=clocks)


def test_unconfirmed_source_hash_is_rejected_before_reading_rows(tmp_path, monkeypatch):
    source, clocks = source_edition(tmp_path)
    with (source / "samples/US/A/samples.parquet").open("ab") as stream:
        stream.write(b" ")

    def cannot_calculate(*args, **kwargs):
        raise AssertionError("No se puede calcular sobre un origen sin verificar")

    monkeypatch.setattr(edition, "geometry_chart", cannot_calculate)
    with pytest.raises(ValueError, match="huella"):
        edition.derive_geometry_edition(source, tmp_path / "geometry", clocks=clocks)


def test_asset_receipt_must_be_bound_to_its_configuration(tmp_path):
    source, clocks = source_edition(tmp_path)
    path = source / "samples/US/A/configuration.json"
    config = json.loads(path.read_text())
    config["calendar"] = "0" * 64
    atomic_json(path, config)
    with pytest.raises(ValueError, match="preparación|calendario"):
        edition.derive_geometry_edition(source, tmp_path / "geometry", clocks=clocks)


def test_only_complete_confirmed_corpus_is_accepted(tmp_path):
    source, clocks = source_edition(tmp_path)
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["cohort_complete"] = False
    atomic_json(path, manifest)
    with pytest.raises(ValueError, match="completa"):
        edition.derive_geometry_edition(source, tmp_path / "geometry", clocks=clocks)
