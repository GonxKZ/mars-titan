"""Perturbaciones declaradas de las entradas en el lector, sin modelos ni optimizadores.

Las ráfagas de ausencia se comparan con un corpus técnico donde esas filas carecen de verdad
de la modalidad. El ruido se comprueba sobre tablas escritas en la prueba.
"""

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data import input_perturbations as perturbations
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import sha256
from mars_titan.data.temporal import MarketClock
from mars_titan.training import carried_predictions
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.training.historical_temporal_fixture import DEFAULT_DAYS, historical_temporal_fixture
from tests.training.test_modality_ablation_inputs import (
    MOMENT,
    STAMP,
    assert_same,
    events,
    observed,
    pattern,
    reader,
    samples,
    supervised,
)

BURST = dict(kind=perturbations.BURST, modalities=("news", "fundamentals"), length_days=21)
NOISE = dict(kind=perturbations.NOISE, modalities=("news", "charts", "fundamentals", "macro"))
# Cada lectura rechaza una sesión repetida con su propia guarda. La supervisada lo hace porque
# la fila repetida no tiene etiqueta ni exclusión propias.
DUPLICATE = "etiqueta o una exclusión|orden único por activo|orden temporal único"
CLOCK = MarketClock("US", "1999-01-01", "2024-01-05")
MOMENTS = [
    int(CLOCK.decisions[CLOCK.days.index(date.fromisoformat(day))].timestamp() * 1_000_000)
    for day in DEFAULT_DAYS
]


def bursts(market, symbol, share):
    return perturbations.burst_rows(f"{market}/{symbol}", MOMENTS, 21, share)


def without_bursts(share):
    """El patrón del fixture con noticias y fundamentales ausentes de verdad en las ráfagas."""

    def presence(market, symbol, row):
        news, fundamentals, macro = pattern(market, symbol, row)
        inside = bursts(market, symbol, share)[row]
        return news and not inside, fundamentals and not inside, macro

    return presence


@pytest.fixture(scope="module")
def corpora(tmp_path_factory):
    root = tmp_path_factory.mktemp("perturbation-inputs")
    original = historical_temporal_fixture(root / "original", assets=3, presence=pattern)
    absent = historical_temporal_fixture(root / "bursts", assets=3, presence=without_bursts(0.5))
    return original.parent, absent.parent


@pytest.fixture
def declared(monkeypatch):
    """Variantes de prueba con intensidades que marcan filas en un corpus de 13 sesiones."""
    variants = dict(
        burst_test=dict(BURST, share=0.5),
        burst_zero=dict(BURST, share=0.0),
        noise_test=dict(NOISE, scale=0.5),
        noise_zero=dict(NOISE, scale=0.0),
    )
    for name, spec in variants.items():
        monkeypatch.setitem(perturbations.PERTURBATIONS, name, spec)
    return variants


def test_a_burst_reads_exactly_as_a_real_absence_in_those_sessions(corpora, declared):
    original, absent = corpora
    # La ráfaga del fixture deja sin noticias o fundamentales filas que los tenían.
    marked = [bursts("US", f"A{number}", 0.5) for number in range(3)]
    assert 0 < sum(map(np.sum, marked)) < 3 * len(MOMENTS)
    perturbed, real = reader(original, "burst_test"), reader(absent)
    for partition in ("train", "validation"):
        assert_same(supervised(perturbed, partition), supervised(real, partition))
    assert_same(observed(perturbed), observed(real))


@pytest.mark.parametrize("blocks", [None, 2])
def test_memory_sources_read_the_burst_as_a_real_absence(corpora, declared, tmp_path, blocks):
    """Titans-MAC, MARS-TITAN, CM-v1 y la GRU candidata leen su tramo con este índice."""
    original, absent = corpora
    perturbed = events(original, tmp_path / "burst", "burst_test", blocks=blocks)
    real = events(absent, tmp_path / "real", blocks=blocks)
    assert [(e.at, e.close_phase, len(e.labels)) for e in perturbed] == [
        (e.at, e.close_phase, len(e.labels)) for e in real
    ]
    assert_same(
        [batch for event in perturbed for batch in event.inputs],
        [batch for event in real for batch in event.inputs],
    )


def test_zero_intensity_reads_the_original_inputs(corpora, declared):
    original, _ = corpora
    for variant in ("burst_zero", "noise_zero"):
        for partition in ("train", "validation"):
            assert_same(
                supervised(reader(original, variant), partition),
                supervised(reader(original), partition),
            )


def test_noise_changes_only_observed_values_and_is_the_same_on_every_read(corpora, declared):
    original, _ = corpora
    clean = supervised(reader(original), "train")
    first = supervised(reader(original, "noise_test"), "train")
    assert_same(first, supervised(reader(original, "noise_test"), "train"))
    for before, after in zip(clean, first, strict=True):
        np.testing.assert_array_equal(before["presence"], after["presence"])
        np.testing.assert_array_equal(before["input_available_at"], after["input_available_at"])
        for name in ("news", "charts"):
            present = before["inputs"][name] != 0
            assert (after["inputs"][name][~present] == 0).all()
            assert (after["inputs"][name][present] != before["inputs"][name][present]).all()


def test_burst_rows_depend_only_on_the_asset_block_and_nest_by_share():
    moments = np.array(MOMENTS, dtype=np.int64)
    low = perturbations.burst_rows("US/A0", moments, 21, 0.1)
    high = perturbations.burst_rows("US/A0", moments, 21, 0.25)
    assert not (low & ~high).any()
    # El resultado de cada fila no depende de las demás filas que se lean con ella.
    for row in range(len(moments)):
        assert perturbations.burst_rows("US/A0", moments[row : row + 1], 21, 0.25)[0] == high[row]
    # Dos fechas del mismo bloque de 21 días comparten resultado.
    day = 86_400_000_000
    start = 946_684_800_000_000 + 21 * day * 100
    same = perturbations.burst_rows("US/A0", [start, start + 20 * day], 21, 0.5)
    assert same[0] == same[1]
    with pytest.raises(ValueError, match="anterior a 2000"):
        perturbations.burst_rows("US/A0", [start - 21 * day * 101], 21, 0.5)
    # Cada activo tiene sus propias ráfagas.
    blocks = [start + 21 * day * block for block in range(64)]
    first = perturbations.burst_rows("US/A0", blocks, 21, 0.5)
    assert not np.array_equal(first, perturbations.burst_rows("US/A1", blocks, 21, 0.5))


def test_a_burst_writes_the_absence_contract_only_in_its_rows():
    table = samples()
    rows = np.array([True, False, True])
    result = perturbations.absent_rows(table, ("news", "fundamentals"), rows, "input_perturbation")
    assert result.schema == table.schema
    presence = np.array(result["presence"].to_pylist())
    before = np.array(table["presence"].to_pylist())
    np.testing.assert_array_equal(presence[1], before[1])
    assert not presence[[0, 2]][:, [1, 3]].any()
    assert result["news_count"].to_pylist() == [0, 0, 0]
    news = values(result, "news")
    assert (news[[0, 2]] == 0).all()
    np.testing.assert_array_equal(news[1], values(table, "news")[1])
    reasons = result["missing_reasons"].to_pylist()
    # Solo cambian las causas de lo que estaba presente en las filas de la ráfaga.
    assert [row["news"] for row in reasons] == [
        "input_perturbation",
        "no_admissible_value",
        "input_perturbation",
    ]
    assert [row["fundamentals"] for row in reasons] == [
        "input_perturbation",
        None,
        "source_missing",
    ]
    availability = result["input_availability"].to_pylist()
    assert availability[1] == table["input_availability"].to_pylist()[1]
    assert availability[0]["news"] is None and availability[2]["fundamentals"] is None


def noisy(table, asset="US/A0", scale=0.5):
    return perturbations.perturb_samples(table, dict(NOISE, scale=scale), asset)


def values(table, name):
    return np.array(table[name].to_pylist(), dtype=np.float32)


@pytest.mark.parametrize("fixed", [True, False])
def test_noise_keeps_masks_ages_absent_blocks_and_zeros(fixed):
    table = samples(fixed=fixed)
    result = noisy(table)
    assert result.schema == table.schema
    for name in ("presence", "news_count", "input_availability", "missing_reasons"):
        assert result[name].to_pylist() == table[name].to_pylist()
    news, before = values(result, "news"), values(table, "news")
    # La fila sin noticias sigue a cero y las presentes cambian.
    assert (news[1] == 0).all() and (news[[0, 2]] != before[[0, 2]]).all()
    fundamentals, original = values(result, "fundamentals"), values(table, "fundamentals")
    # Valor, máscara y edad: solo cambia el valor de un concepto observado.
    assert (fundamentals[:2, 0] != original[:2, 0]).all()
    np.testing.assert_array_equal(fundamentals[:, 1:], original[:, 1:])
    np.testing.assert_array_equal(fundamentals[2], original[2])


def test_noise_leaves_the_unobserved_concepts_of_a_present_block_bit_for_bit():
    # El contrato exige cero en un concepto no observado, y un factor negativo lo convertiría
    # en -0.0. Con escala 1 y 64 fechas hay factores negativos en las dos posiciones.
    rows = 64
    moments = [MOMENT + timedelta(days=day) for day in range(rows)]
    table = pa.table(
        dict(
            prediction_at=pa.array(moments, STAMP),
            fundamentals=pa.array(
                [[2.0, 0.0, 1.0, 0.0, 3.0, 0.0]] * rows, pa.list_(pa.float32(), 6)
            ),
            presence=pa.array([[True] * 5] * rows, pa.list_(pa.bool_(), 5)),
        )
    )
    spec = dict(kind=perturbations.NOISE, modalities=("fundamentals",), scale=1.0)
    key = perturbations._key(perturbations.NOISE, "US/A0", "fundamentals")
    stamps = table["prediction_at"].cast(pa.int64()).to_numpy()
    assert (perturbations._normal(key, stamps, 2) < -1).any(axis=0).all()
    result = values(perturbations.perturb_samples(table, spec, "US/A0"), "fundamentals")
    original = values(table, "fundamentals")
    assert (result[:, 0] != original[:, 0]).all()
    np.testing.assert_array_equal(result[:, 1:].view(np.uint32), original[:, 1:].view(np.uint32))


def test_noise_is_the_declared_multiplicative_factor_and_does_not_depend_on_grouping():
    table = samples()
    result = noisy(table, scale=0.5)
    moments = table["prediction_at"].cast(pa.int64()).to_numpy()
    key = perturbations._key(perturbations.NOISE, "US/A0", "charts")
    factor = 1.0 + 0.5 * perturbations._normal(key, moments, 2)
    expected = (values(table, "charts").astype(np.float64) * factor).astype(np.float32)
    np.testing.assert_array_equal(values(result, "charts"), expected)
    # Fila a fila da lo mismo que la tabla entera.
    for row in range(table.num_rows):
        part = noisy(table.slice(row, 1))
        for name in ("news", "charts", "fundamentals", "macro"):
            np.testing.assert_array_equal(values(part, name)[0], values(result, name)[row])
    assert noisy(table, scale=0.0).equals(table)


def test_macro_noise_is_shared_by_the_assets_of_a_market_and_the_rest_is_not():
    table = samples()
    first, second = noisy(table, "US/A0"), noisy(table, "US/A1")
    np.testing.assert_array_equal(values(first, "macro"), values(second, "macro"))
    assert not np.array_equal(values(first, "news"), values(second, "news"))
    assert not np.array_equal(values(first, "macro"), values(noisy(table, "CN/A0"), "macro"))


@pytest.mark.parametrize(
    "spec",
    [
        dict(BURST, share=1.5),
        dict(BURST, share=1),
        dict(BURST, share=0.1, length_days=0),
        dict(BURST, share=0.1, modalities=("charts",)),
        dict(NOISE, scale=-0.1),
        dict(NOISE, scale=0.1, modalities=("prices",)),
        dict(NOISE, scale=0.1, modalities=()),
        dict(NOISE, scale=0.1, extra=True),
        dict(kind="shuffle", modalities=("news",), scale=0.1),
    ],
)
def test_invalid_perturbations_are_rejected(spec):
    with pytest.raises(ValueError):
        perturbations.check_spec(spec)


def test_every_declared_perturbation_is_valid_and_identified():
    for name, spec in perturbations.PERTURBATIONS.items():
        perturbations.check_spec(spec)
        identity = perturbations.variant_identity(name)
        assert identity["variant"] == name and identity["seed"] == perturbations.SEED
        assert identity["modalities"] == list(spec["modalities"])
        assert identity["random"] == perturbations.RANDOM
        record = carried_predictions.ablation_record(name)
        assert record == dict(modality_ablation=identity)
    assert perturbations.variant_identity("mask_news")["missing_reason"] == "modality_ablation"
    assert not set(perturbations.PERTURBATIONS) & {"mask_news", "mask_fundamentals"}
    with pytest.raises(ValueError, match="no está declarada"):
        perturbations.variant_identity("noise_99pct")


def test_unknown_variants_are_rejected_by_the_reader(corpora):
    original, _ = corpora
    with pytest.raises(ValueError, match="no está declarada"):
        reader(original, "noise_99pct")


# Guardas: lo que el lector rechaza en lugar de entregarlo al modelo.


def change_samples(manifest, transform):
    content = json.loads(Path(manifest).read_text())
    entry = content["assets"][0]
    path = Path(content["roots"]["samples"]) / entry["market"] / entry["symbol"] / "samples.parquet"
    table = pq.read_table(path)
    pq.write_table(transform(table), path, row_group_size=3)
    entry["samples_sha256"] = sha256(path)
    Path(manifest).write_text(json.dumps(content))


@pytest.fixture
def manifest(tmp_path):
    return historical_temporal_fixture(tmp_path / "corpus", assets=1, presence=pattern).parent


def reads(manifest, folder):
    """Lecturas supervisadas, observaciones y eventos con calentamiento de los modelos."""
    return dict(
        train=lambda: supervised(CorpusDataset(manifest, input_policy=HISTORICAL_MASKED), "train"),
        observed=lambda: observed(CorpusDataset(manifest, input_policy=HISTORICAL_MASKED)),
        events=lambda: events(manifest, folder),
    )


def test_a_duplicated_session_is_rejected_by_every_read(manifest, tmp_path):
    def duplicate(table):
        return pa.concat_tables([table.slice(0, 3), table.slice(2, 1), table.slice(3)])

    change_samples(manifest, duplicate)
    for read in reads(manifest, tmp_path / "index").values():
        with pytest.raises(ValueError, match=DUPLICATE):
            read()


def test_availability_after_the_decision_is_rejected_by_every_read_that_uses_the_row(
    manifest, tmp_path
):
    # La fila 2 (15 de enero de 2009) tiene etiqueta aceptada de entrenamiento.
    def late(table):
        rows = table.to_pylist()
        rows[2]["input_availability"]["charts"] = rows[2]["prediction_at"] + timedelta(days=1)
        return pa.Table.from_pylist(rows, schema=table.schema)

    change_samples(manifest, late)
    for read in reads(manifest, tmp_path / "index").values():
        with pytest.raises(ValueError, match="ausente o futura"):
            read()
