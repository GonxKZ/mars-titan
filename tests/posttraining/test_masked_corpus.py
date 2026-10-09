"""Corpus ordenado de la edición con máscaras, con la ruta estricta separada."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED, STRICT_INPUTS, policy_identity
from mars_titan.environments import corpus_source
from mars_titan.environments.corpus_source import ParquetCohortSource, prepare_causal_corpus
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.posttraining.masked_fixture import masked_ordered, masked_view, sources
from tests.training.test_temporal_corpus import inputs as inputs
from tests.training.test_temporal_corpus import prepare


def reader_rows(view, partition):
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    rows = {}
    for batch in dataset.batches(partition=partition, batch_size=2, epoch=0, seed=0):
        for index, key in enumerate(batch["sample_ids"]):
            rows[key] = (
                batch["target"][index],
                batch["presence"][index].copy(),
                {name: values[index].copy() for name, values in batch["inputs"].items()},
            )
    return rows


def test_masked_ordered_corpus_keeps_presence_bits_of_every_row(tmp_path):
    view, ordered, report = masked_ordered(tmp_path)
    assert report["status"] == "completed"
    assert {key: report[key] for key in ("input_policy", "mask_contract")} == policy_identity(
        HISTORICAL_MASKED
    )
    assert report["identity"]["input_policy"] == HISTORICAL_MASKED
    assert "data/input_policy.py" in report["identity"]["code"]
    assert set(report["partitions"]) == {"train", "validation"}
    train, validation = sources(ordered)
    with train, validation:
        assert train.masked and train.input_policy == HISTORICAL_MASKED
        column = train.file.schema_arrow.field("presence").type
        assert pa.types.is_list(column) and pa.types.is_boolean(column.value_type)
        for source in (train, validation):
            expected = reader_rows(view, source.partition)
            seen = set()
            for position in range(len(source)):
                raw = source(position)
                assert raw["presence"].dtype == np.bool_ and not raw["presence"].flags.writeable
                for index, asset in enumerate(raw["asset_ids"]):
                    key = f"{asset}/{raw['prediction_at']}"
                    target, presence, values = expected[key]
                    assert raw["target"][index] == target
                    np.testing.assert_array_equal(raw["presence"][index], presence)
                    for name, value in values.items():
                        np.testing.assert_array_equal(raw["inputs"][name][index], value)
                    seen.add(key)
            assert seen == set(expected)
        # El fixture contiene macro observada y ausente en el ajuste.
        bits = np.concatenate([train(i)["presence"] for i in range(len(train))])
        assert bits[:, 4].any() and not bits[:, 4].all()


def test_every_reader_declares_the_policy_of_the_ordered_corpus(tmp_path, inputs):
    view = masked_view(tmp_path / "masked")
    with pytest.raises(ValueError, match="política"):
        prepare_causal_corpus(view, tmp_path / "strict-output", batch_size=2)
    with pytest.raises(ValueError, match="política"):
        prepare_causal_corpus(view, tmp_path / "other", input_policy="historical_masked_v0")
    ordered = tmp_path / "masked-ordered"
    prepare_causal_corpus(view, ordered, batch_size=2, input_policy=HISTORICAL_MASKED)
    with pytest.raises(ValueError, match="política"):
        ParquetCohortSource(ordered / "manifest.json", partition="train")
    prepare(inputs, tmp_path / "views")
    strict = tmp_path / "strict-ordered"
    report = prepare_causal_corpus(tmp_path / "views/fold-000/manifest.json", strict, batch_size=2)
    assert "input_policy" not in report and "input_policy" not in report["identity"]
    assert "data/input_policy.py" not in report["identity"]["code"]
    assert "presence" not in pq.read_schema(strict / report["partitions"]["train"]["path"]).names
    with pytest.raises(ValueError, match="política"):
        ParquetCohortSource(
            strict / "manifest.json", partition="train", input_policy=HISTORICAL_MASKED
        )
    with ParquetCohortSource(strict / "manifest.json", partition="train") as source:
        assert not source.masked and "presence" not in source(0)


def test_a_manifest_that_hides_its_policy_is_still_bound_to_its_masked_source(tmp_path):
    _, ordered, report = masked_ordered(tmp_path)
    for key in ("input_policy", "mask_contract"):
        report.pop(key)
        report["identity"].pop(key)
    ordered.write_text(json.dumps(report))
    # La supervisión de origen conserva la política aunque el recibo ordenado la oculte.
    with pytest.raises(ValueError, match="política"):
        ParquetCohortSource(ordered, partition="train", input_policy=STRICT_INPUTS)


def table(presence, **values):
    count = len(presence)
    columns = {
        name: pa.FixedSizeListArray.from_arrays(
            pa.array(np.asarray(values.get(name, np.zeros((count, 2)))).reshape(-1), pa.float32()),
            2,
        )
        for name in ("news", "charts", "fundamentals", "macro")
    }
    columns["presence"] = pa.FixedSizeListArray.from_arrays(
        pa.array(np.asarray(presence).reshape(-1), pa.bool_()), 5
    )
    return pa.table(columns)


def arrays(source):
    return {
        name: source[name].combine_chunks().flatten().to_numpy().reshape(len(source), -1)
        for name in ("news", "charts", "fundamentals", "macro")
    } | {"prices": np.ones((len(source), 3), dtype=np.float32)}


@pytest.mark.parametrize(
    ("presence", "values", "message"),
    [
        ([[True, False, True, False, False]], dict(news=[[0.5, 0.0]]), "cero"),
        ([[True, True, True, False, False]], dict(macro=[[1.0, 0.0]]), "cero"),
        ([[True, True, False, True, True]], {}, "obligatorios"),
        ([[False, True, True, True, True]], {}, "obligatorios"),
    ],
)
def test_presence_reader_rejects_inconsistent_absences(presence, values, message):
    source = table(presence, **values)
    with pytest.raises(ValueError, match=message):
        corpus_source._presence(source, arrays(source))


def test_presence_reader_accepts_an_observed_zero_and_rejects_short_rows():
    source = table([[True, True, True, False, True]])
    bits = corpus_source._presence(source, arrays(source))
    np.testing.assert_array_equal(bits, [[True, True, True, False, True]])
    short = source.set_column(
        source.schema.get_field_index("presence"),
        "presence",
        pa.array([[True, True, True, False]], pa.list_(pa.bool_())),
    )
    with pytest.raises(ValueError, match="cinco"):
        corpus_source._presence(short, arrays(short))


def test_a_declared_policy_is_checked_even_without_a_temporal_source(tmp_path):
    from tests.posttraining.test_session_capacity import legacy_ordered

    manifest = legacy_ordered(tmp_path, 2)
    meta = json.loads(manifest.read_text())
    meta |= policy_identity(HISTORICAL_MASKED)
    meta["identity"] |= policy_identity(HISTORICAL_MASKED)
    manifest.write_text(json.dumps(meta))
    # Sin vista temporal, solo la declaración del recibo separa ambas ediciones.
    with pytest.raises(ValueError, match="política"):
        ParquetCohortSource(manifest, partition="train", input_policy=STRICT_INPUTS)
    with pytest.raises(ValueError, match="columnas"):
        ParquetCohortSource(manifest, partition="train", input_policy=HISTORICAL_MASKED)
