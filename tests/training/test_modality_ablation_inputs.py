"""Ablación de modalidades en el lector: la misma entrada que una ausencia real.

Los corpus son técnicos, con patrones de presencia fijados por activo y fila. No hay
modelos ni pasos de optimizador.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pyarrow as pa
import pytest

from mars_titan.data import modality_ablation as ablation
from mars_titan.data.input_policy import HISTORICAL_MASKED, MODALITIES, STRICT_INPUTS
from mars_titan.memory.financial_observations import (
    FinancialObservationSource,
    prepare_observation_index,
)
from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.training import corpus_inputs
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.training.historical_temporal_fixture import historical_temporal_fixture

START, END = 946_684_800_000_000, 1_704_067_200_000_000
US_2023 = 1_672_531_200_000_000


def pattern(market, symbol, row):
    """Noticias y fundamentales presentes en filas alternas y distintas por activo."""
    number = int(symbol[1:])
    return (number + row) % 2 == 0, (number + row) % 3 != 0, row % 4 != 1


def without(variant):
    """El mismo patrón con las modalidades de la variante ausentes de verdad."""
    masked = ablation.VARIANTS[variant]

    def presence(market, symbol, row):
        news, fundamentals, macro = pattern(market, symbol, row)
        return news and "news" not in masked, fundamentals and "fundamentals" not in masked, macro

    return presence


@pytest.fixture(scope="module")
def corpora(tmp_path_factory):
    root = tmp_path_factory.mktemp("ablation-inputs")
    original = historical_temporal_fixture(root / "original", assets=3, presence=pattern)
    absent = {
        variant: historical_temporal_fixture(root / variant, assets=3, presence=without(variant))
        for variant in ablation.VARIANTS
    }
    return original.parent, {variant: value.parent for variant, value in absent.items()}


def supervised(dataset, partition):
    """Lotes completos de un tramo, con los campos que recibe un modelo."""
    return [
        dict(
            inputs={name: value.copy() for name, value in batch["inputs"].items()},
            presence=batch["presence"].copy(),
            target=batch["target"].copy(),
            prediction_at=batch["prediction_at"].copy(),
            input_available_at=batch["input_available_at"].copy(),
            sample_ids=list(batch["sample_ids"]),
            market=list(batch["market"]),
        )
        for batch in dataset.batches(partition=partition, batch_size=4, epoch=0, seed=7)
    ]


def observed(dataset):
    """Observaciones sin etiquetas, la lectura de los recorridos con memoria."""
    return [
        dict(
            inputs={name: value.copy() for name, value in batch["inputs"].items()},
            presence=batch["presence"].copy(),
            prediction_at=batch["prediction_at"].copy(),
            input_available_at=batch["input_available_at"].copy(),
            sample_ids=list(batch["sample_ids"]),
        )
        for batch in dataset.observation_batches(start=START, end=END, batch_size=5)
    ]


def assert_same(left, right):
    assert len(left) == len(right) and left
    for first, second in zip(left, right, strict=True):
        assert set(first) == set(second)
        for key, value in first.items():
            if key == "inputs":
                assert set(value) == set(second[key])
                for name, array in value.items():
                    assert array.dtype == second[key][name].dtype
                    np.testing.assert_array_equal(array, second[key][name])
            elif isinstance(value, np.ndarray):
                assert value.dtype == second[key].dtype
                np.testing.assert_array_equal(value, second[key])
            else:
                assert value == second[key]


def reader(path, variant=None):
    return CorpusDataset(path, input_policy=HISTORICAL_MASKED, modality_ablation=variant)


@pytest.mark.parametrize("variant", list(ablation.VARIANTS))
def test_ablated_reading_is_exactly_a_real_absence(corpora, variant):
    original, absent = corpora
    masked, real = reader(original, variant), reader(absent[variant])
    for partition in ("train", "validation"):
        assert_same(supervised(masked, partition), supervised(real, partition))
    assert_same(observed(masked), observed(real))


def events(path, folder, variant=None, *, blocks=None):
    """Observaciones de un tramo con calentamiento, como las leen los modelos con memoria."""
    dataset = reader(path, variant)
    phase = FinancialPhase("validation", START, US_2023, END, END)
    source = FinancialObservationSource(
        dataset, prepare_observation_index(dataset, folder, phase=phase)
    )
    stream = source.events() if blocks is None else source.batched_events(block_rows=blocks)
    return list(stream)


@pytest.mark.parametrize("blocks", [None, 2])
@pytest.mark.parametrize("variant", list(ablation.VARIANTS))
def test_memory_sources_warm_up_and_predict_with_a_real_absence(corpora, tmp_path, variant, blocks):
    """Titans-MAC, MARS-TITAN, CM-v1 y la GRU candidata leen su tramo con este índice."""
    original, absent = corpora
    masked = events(original, tmp_path / "masked", variant, blocks=blocks)
    real = events(absent[variant], tmp_path / "real", blocks=blocks)
    assert [(e.at, e.close_phase, len(e.labels)) for e in masked] == [
        (e.at, e.close_phase, len(e.labels)) for e in real
    ]
    assert_same(
        [batch for event in masked for batch in event.inputs],
        [batch for event in real for batch in event.inputs],
    )
    # El calentamiento anterior al tramo también llega sin la modalidad.
    warmup = [batch for event in masked if event.at < US_2023 for batch in event.inputs]
    assert warmup
    plain = events(original, tmp_path / "plain", blocks=blocks)
    indices = [MODALITIES.index(name) for name in ablation.VARIANTS[variant]]
    before = [batch for event in plain if event.at < US_2023 for batch in event.inputs]
    assert any(batch["presence"][:, indices].any() for batch in before)


@pytest.mark.parametrize("variant", list(ablation.VARIANTS))
def test_rows_without_the_masked_modalities_keep_every_input(corpora, variant):
    original, _ = corpora
    plain, masked = (
        supervised(reader(original), "train"),
        supervised(reader(original, variant), "train"),
    )
    indices = [MODALITIES.index(name) for name in ablation.VARIANTS[variant]]
    touched = untouched = 0
    for before, after in zip(plain, masked, strict=True):
        assert before["sample_ids"] == after["sample_ids"]
        present = before["presence"][:, indices].any(axis=1)
        touched += int(present.sum())
        untouched += int((~present).sum())
        for key in ("presence", "input_available_at", "target"):
            np.testing.assert_array_equal(before[key][~present], after[key][~present])
        for name, values in before["inputs"].items():
            np.testing.assert_array_equal(values[~present], after["inputs"][name][~present])
            if name in ablation.VARIANTS[variant]:
                assert not after["inputs"][name].any()
            else:
                np.testing.assert_array_equal(values, after["inputs"][name])
        assert not after["presence"][:, indices].any()
        others = [i for i in range(len(MODALITIES)) if i not in indices]
        np.testing.assert_array_equal(before["presence"][:, others], after["presence"][:, others])
    assert touched and untouched


def test_normal_reading_keeps_its_inputs_and_identity(corpora):
    original, _ = corpora
    default = CorpusDataset(original, input_policy=HISTORICAL_MASKED)
    explicit = reader(original, None)
    assert default.modality_ablation is explicit.modality_ablation is None
    assert default.identity == explicit.identity
    for partition in ("train", "validation"):
        assert_same(supervised(default, partition), supervised(explicit, partition))
    masked = reader(original, "mask_news")
    # La ablación no cambia la identidad de la vista: el recibo la declara aparte.
    assert masked.identity == default.identity and masked.modality_ablation == "mask_news"
    # Con noticias presentes en el fixture, la lectura ablacionada sí cambia.
    news = [batch["inputs"]["news"] for batch in supervised(default, "train")]
    assert any(values.any() for values in news)


def test_ablation_is_rejected_outside_the_masked_edition_or_with_an_unknown_variant(corpora):
    original, _ = corpora
    with pytest.raises(ValueError, match="no está declarada"):
        reader(original, "mask_macro")
    with pytest.raises(ValueError, match="no está declarada"):
        reader(original, "mask_prices")
    with pytest.raises(ValueError, match="edición con máscaras"):
        CorpusDataset(original, input_policy=STRICT_INPUTS, modality_ablation="mask_news")


def test_the_reader_validates_the_original_rows_before_masking(corpora, monkeypatch):
    """Una fila original incoherente se rechaza aunque la ablación fuera a ocultarla."""
    original, _ = corpora
    real = corpus_inputs._presence

    def broken(table, vectors, representation):
        if table["news_count"].to_numpy().any():
            raise ValueError("fila original incoherente")
        return real(table, vectors, representation)

    monkeypatch.setattr(corpus_inputs, "_presence", broken)
    with pytest.raises(ValueError, match="fila original incoherente"):
        supervised(reader(original, "mask_news"), "train")


STAMP = pa.timestamp("us", tz="UTC")
MOMENT = datetime(2010, 3, 1, 21, tzinfo=UTC)


def samples(*, fixed=True, reasons=True):
    """Tres filas: todo presente, sin noticias y sin fundamentales, con fechas distintas."""
    lists = (
        (lambda kind, width: pa.list_(kind, width)) if fixed else (lambda kind, _: pa.list_(kind))
    )
    presence = [
        [True, True, True, True, True],
        [True, False, True, True, True],
        [True, True, True, False, True],
    ]
    available = []
    for row in presence:
        stamps = {
            "prices": MOMENT - timedelta(hours=5),
            "news": MOMENT - timedelta(minutes=10),
            "charts": MOMENT - timedelta(hours=5),
            "fundamentals": MOMENT - timedelta(days=30),
            "macro": MOMENT - timedelta(days=2),
        }
        available.append(
            {name: stamps[name] if row[i] else None for i, name in enumerate(MODALITIES)}
        )
    columns = dict(
        prediction_at=pa.array([MOMENT] * 3, STAMP),
        price_end_index=pa.array([10, 11, 12], pa.int64()),
        news=pa.array([[0.5, -0.5], [0.0, 0.0], [0.25, 0.75]], lists(pa.float32(), 2)),
        charts=pa.array([[1.0, 2.0]] * 3, lists(pa.float32(), 2)),
        fundamentals=pa.array(
            [[2.0, 1.0, 3.5], [4.0, 1.0, 1.5], [0.0, 0.0, 0.0]], lists(pa.float32(), 3)
        ),
        macro=pa.array([[1.0, 1.0, 0.0]] * 3, lists(pa.float32(), 3)),
        presence=pa.array(presence, pa.list_(pa.bool_(), 5)),
        news_count=pa.array([3, 0, 1], pa.int64()),
        input_availability=pa.array(available, pa.struct([(name, STAMP) for name in MODALITIES])),
    )
    if reasons:
        columns["missing_reasons"] = pa.array(
            [
                {name: None for name in MODALITIES},
                {**{name: None for name in MODALITIES}, "news": "no_admissible_value"},
                {**{name: None for name in MODALITIES}, "fundamentals": "source_missing"},
            ],
            pa.struct([(name, pa.string()) for name in MODALITIES]),
        )
    return pa.table(columns)


def field(table, name):
    return table["input_availability"].combine_chunks().field(name).to_pylist()


@pytest.mark.parametrize("fixed", [True, False])
@pytest.mark.parametrize("variant", list(ablation.VARIANTS))
def test_table_ablation_writes_the_mask_contract_of_a_real_absence(variant, fixed):
    table = samples(fixed=fixed)
    result = ablation.ablate_samples(table, variant)
    masked = ablation.VARIANTS[variant]
    assert result.schema == table.schema and result.num_rows == 3
    presence = np.array(result["presence"].to_pylist())
    before = np.array(table["presence"].to_pylist())
    for index, name in enumerate(MODALITIES):
        if name in masked:
            assert not presence[:, index].any()
            assert all(value is None for value in field(result, name))
        else:
            np.testing.assert_array_equal(presence[:, index], before[:, index])
            assert field(result, name) == field(table, name)
    for name in ("news", "charts", "fundamentals", "macro"):
        values = np.array(result[name].to_pylist())
        if name in masked:
            assert values.dtype == np.float64 and not values.any()
        else:
            np.testing.assert_array_equal(values, np.array(table[name].to_pylist()))
    counts = result["news_count"].to_pylist()
    assert counts == ([0, 0, 0] if "news" in masked else [3, 0, 1])
    reasons = result["missing_reasons"].to_pylist()
    for row, (record, original) in enumerate(
        zip(reasons, table["missing_reasons"].to_pylist(), strict=True)
    ):
        for name in MODALITIES:
            if name in masked and before[row, MODALITIES.index(name)]:
                assert record[name] == ablation.REASON
            else:
                # Una ausencia real conserva su causa.
                assert record[name] == original[name]
    # La tabla resultante cumple las comprobaciones del lector.
    vectors = corpus_inputs._vectors(result, historical=True)
    representation = dict(fundamental_concepts=["a"], macro_indicators=["m"])
    bits = corpus_inputs._presence(result, vectors, representation)
    bounds, valid = corpus_inputs._availability(result, presence=bits)
    assert valid.all()
    expected = {
        "mask_news": [
            MOMENT - timedelta(hours=5),
            MOMENT - timedelta(hours=5),
            MOMENT - timedelta(hours=5),
        ],
        "mask_fundamentals": [
            MOMENT - timedelta(minutes=10),
            MOMENT - timedelta(hours=5),
            MOMENT - timedelta(minutes=10),
        ],
        "mask_news_and_fundamentals": [MOMENT - timedelta(hours=5)] * 3,
    }[variant]
    stamps = [int(moment.timestamp() * 1_000_000) for moment in expected]
    np.testing.assert_array_equal(bounds, stamps)


def test_table_ablation_rejects_columns_it_cannot_make_absent():
    table = samples().append_column("news_available_at", pa.array([MOMENT] * 3, STAMP))
    with pytest.raises(ValueError, match="no sabe ausentar"):
        ablation.ablate_samples(table, "mask_news")
    with pytest.raises(ValueError, match="necesita presencia"):
        ablation.ablate_samples(samples().drop_columns(["news_count"]), "mask_news")
    with pytest.raises(ValueError, match="no está declarada"):
        ablation.ablate_samples(samples(), "mask_charts")
    empty = samples().slice(0, 0)
    assert ablation.ablate_samples(empty, "mask_news").equals(empty)
    plain = samples(fixed=False)
    ragged = plain.set_column(
        plain.schema.get_field_index("news"),
        "news",
        pa.array([[0.5, -0.5], [0.0], [0.25, 0.75]], pa.list_(pa.float32())),
    )
    with pytest.raises(ValueError, match="anchura común"):
        ablation.ablate_samples(ragged, "mask_news")


def test_ablation_identity_declares_the_mask_contract():
    identity = ablation.ablation_identity("mask_news_and_fundamentals")
    assert identity["modalities"] == ["news", "fundamentals"]
    assert identity["presence"] is False and identity["available_at"] is None
    assert identity["fill"] == "missing_fill" and identity["missing_reason"] == "modality_ablation"
    assert identity["mask_contract"]["missing_fill"] == 0.0
