"""Sustitución activo a activo de la edición anterior, solo tras verificar la nueva."""

import json
from functools import partial

import numpy as np
import pyarrow.parquet as pq
import pytest

from mars_titan.data import corpus_encoding
from mars_titan.data.corpus_encoding import encode_corpus
from mars_titan.data.edition_comparison import compare_asset, compare_editions
from mars_titan.data.edition_substitution import SubstitutionError, substitute_asset
from mars_titan.data.storage import sha256
from tests.data.test_vector_carry import Encoders, TF32Encoders, first_edition

GAP = 2.0**-20


class Reencoded(Encoders):
    """Identidad en FP32 estricto con un desplazamiento fijo de los gráficos frente a TF32."""

    def images(self, pngs):
        return super().images(pngs) + np.float32(GAP)


def editions(tmp_path):
    manifest, previous, kwargs = first_edition(tmp_path, TF32Encoders)
    return manifest, previous, kwargs, tmp_path / "current", tmp_path / "records"


def samples(root):
    return root / "samples/US/A/samples.parquet"


def test_each_verified_asset_replaces_the_previous_samples_after_its_record(tmp_path):
    manifest, previous, kwargs, current, records = editions(tmp_path)
    before = json.loads((previous / "samples/US/A/manifest.json").read_text())
    replaced = sha256(samples(previous))
    result = encode_corpus(
        manifest,
        current,
        encoders=Reencoded(),
        on_confirmed=partial(substitute_asset, previous, current, records),
        **kwargs,
    )
    assert result["cohort_complete"] is True
    assert not samples(previous).exists()
    # Se conservan el manifiesto, la configuración y los factores anteriores como constancia.
    assert json.loads((previous / "samples/US/A/manifest.json").read_text()) == before
    assert (previous / "configuration.json").exists()
    assert (previous / "samples/US/A/company-factors.parquet").exists()
    record = json.loads((records / "US/A.json").read_text())
    assert record["previous"]["samples_sha256"] == replaced == before["samples_sha256"]
    assert record["current"]["samples_sha256"] == sha256(samples(current))
    comparison = record["comparison"]
    assert comparison["other_differences"] == [] and comparison["dropped_sessions"] == []
    charts = comparison["reencoded_charts"]
    assert charts["compared"] == comparison["common"] == result["samples"] > 0
    assert charts["identical"] == 0
    assert charts["max_abs"] == pytest.approx(GAP, rel=1e-3)
    # Los vectores de prueba son constantes, así que ambas medidas relativas coinciden.
    assert charts["max_rel"] == pytest.approx(charts["max_norm_rel"], rel=1e-4)
    assert charts["max_rel"] > 0
    # Repetir el recorrido no vuelve a verificar ni a borrar nada.
    again = encode_corpus(
        manifest,
        current,
        encoders=Reencoded(),
        on_confirmed=partial(substitute_asset, previous, current, records),
        **kwargs,
    )
    assert again["reused_assets"] == 1
    assert json.loads((records / "US/A.json").read_text()) == record


def test_the_summary_aggregates_the_differences_of_reencoded_charts(tmp_path):
    manifest, previous, kwargs, current, _ = editions(tmp_path)
    encode_corpus(manifest, current, encoders=Reencoded(), **kwargs)
    summary = compare_editions(previous, current, tmp_path / "comparison", workers=1)
    charts = summary["reencoded_charts"]
    assert charts["compared"] == summary["common_sessions"] > 0 and charts["identical"] == 0
    assert charts["max_abs"] == pytest.approx(GAP, rel=1e-3)
    assert summary["other_differences"] == 0


def test_the_same_encoder_must_reproduce_each_vector_of_the_same_chart(tmp_path):
    manifest, previous, kwargs, current, _ = editions(tmp_path)
    encode_corpus(manifest, current, encoders=TF32Encoders(), **kwargs)
    same = compare_asset(previous, current, "US", "A")
    assert same["reencoded_charts"] is None and same["other_differences"] == []
    summary = compare_editions(previous, current, tmp_path / "comparison", workers=1)
    assert summary["reencoded_charts"] is None
    rows = pq.read_table(samples(current)).to_pylist()
    rows[0]["charts"] = [value + GAP for value in rows[0]["charts"]]
    table = pq.read_table(samples(current))
    pq.write_table(type(table).from_pylist(rows, schema=table.schema), samples(current))
    changed = compare_asset(previous, current, "US", "A")
    assert changed["other_differences"] == [[rows[0]["session"], "charts_with_same_png"]]


def rewrite(root, change):
    table = pq.read_table(samples(root))
    rows = change(table.to_pylist())
    pq.write_table(type(table).from_pylist(rows, schema=table.schema), samples(root))
    receipt = root / "samples/US/A/manifest.json"
    data = json.loads(receipt.read_text())
    receipt.write_text(json.dumps({**data, "samples_sha256": sha256(samples(root))}))


def forge(root):
    """El Parquet sigue siendo legible, pero el recibo declara otra huella."""
    receipt = root / "samples/US/A/manifest.json"
    receipt.write_text(json.dumps({**json.loads(receipt.read_text()), "samples_sha256": "0" * 64}))


@pytest.mark.parametrize(
    "damage", ["previous_hash", "current_hash", "undeclared_change", "dropped_session"]
)
def test_a_failed_verification_stops_without_deleting_anything(tmp_path, damage):
    manifest, previous, kwargs, current, records = editions(tmp_path)
    if damage == "previous_hash":
        forge(previous)
    elif damage == "undeclared_change":

        def change(rows):
            rows[-1]["macro"] = [value + 1 for value in rows[-1]["macro"]]
            return rows

        rewrite(previous, change)
    kept = samples(previous).read_bytes()
    encode_corpus(manifest, current, encoders=Reencoded(), **kwargs)
    if damage == "current_hash":
        forge(current)
    elif damage == "dropped_session":
        rewrite(current, lambda rows: rows[1:])
    with pytest.raises(SubstitutionError, match="US/A"):
        substitute_asset(previous, current, records, "US", "A")
    assert samples(previous).read_bytes() == kept
    assert not (records / "US/A.json").exists()


def test_the_previous_samples_survive_a_record_that_cannot_be_written(tmp_path, monkeypatch):
    from mars_titan.data import edition_substitution

    manifest, previous, kwargs, current, records = editions(tmp_path)
    encode_corpus(manifest, current, encoders=Reencoded(), **kwargs)
    kept = samples(previous).read_bytes()

    def full_disk(path, value):
        raise OSError("Sin espacio")

    monkeypatch.setattr(edition_substitution, "atomic_json", full_disk)
    with pytest.raises(OSError):
        substitute_asset(previous, current, records, "US", "A")
    assert samples(previous).read_bytes() == kept


def test_an_asset_left_pending_by_the_collection_replaces_nothing(tmp_path):
    from mars_titan.data.vector_carry import CollectingEncoders

    manifest, previous, kwargs, current, records = editions(tmp_path)
    kept = samples(previous).read_bytes()
    result = encode_corpus(
        manifest,
        current,
        encoders=CollectingEncoders(Reencoded.spec),
        on_confirmed=partial(substitute_asset, previous, current, records),
        **kwargs,
    )
    assert result["failed_assets"] == 1 and "GPU" in result["coverage"][0]["detail"]
    assert samples(previous).read_bytes() == kept and not records.exists()


def test_the_encoding_stops_at_the_first_failed_substitution(tmp_path):
    manifest, previous, kwargs, current, records = editions(tmp_path)
    samples(previous).write_bytes(samples(previous).read_bytes() + b"x")
    kept = samples(previous).read_bytes()
    with pytest.raises(SubstitutionError):
        encode_corpus(
            manifest,
            current,
            encoders=Reencoded(),
            on_confirmed=partial(substitute_asset, previous, current, records),
            **kwargs,
        )
    assert samples(previous).read_bytes() == kept and not records.exists()


def test_no_asset_starts_while_the_free_disk_is_below_the_reserve(tmp_path, monkeypatch):
    manifest, previous, kwargs, current, records = editions(tmp_path)
    kept = samples(previous).read_bytes()
    reserve = 15 * 1024**3
    run = partial(
        encode_corpus,
        manifest,
        current,
        encoders=Reencoded(),
        on_confirmed=partial(substitute_asset, previous, current, records),
        min_free_disk_bytes=reserve,
        **kwargs,
    )
    monkeypatch.setattr(corpus_encoding, "_free_disk_bytes", lambda path: reserve - 1)
    with pytest.raises(OSError, match="reserva"):
        run()
    # El disco baja de la reserva después de arrancar y antes del activo.
    free = iter([reserve, reserve - 1])
    monkeypatch.setattr(corpus_encoding, "_free_disk_bytes", lambda path: next(free))
    result = run()
    assert result["stop_reason"] == "disk_reserve" and result["samples"] == 0
    assert samples(previous).read_bytes() == kept and not samples(current).exists()


def test_records_and_editions_never_live_inside_the_edition_they_replace(tmp_path):
    manifest, previous, kwargs, current, records = editions(tmp_path)
    encode_corpus(manifest, current, encoders=Reencoded(), **kwargs)
    kept = samples(previous).read_bytes()
    for arguments in ((previous, current, previous / "records"), (previous, previous, records)):
        with pytest.raises(SubstitutionError, match="fuera"):
            substitute_asset(*arguments, "US", "A")
    assert samples(previous).read_bytes() == kept and not records.exists()
    # Un activo que la edición anterior no tenía no sustituye nada.
    assert substitute_asset(previous, current, records, "US", "Z") is None
