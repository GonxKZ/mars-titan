"""Lectura de ventanas v3.1: paridad bit a bit sin huecos y relleno anulado con huecos."""

import numpy as np
import pytest

from mars_titan.data.price_windows import (
    absent_positions,
    calendar_digest,
    price_window_contract,
    window_rows,
)
from mars_titan.data.temporal import MarketClock
from mars_titan.training.corpus_inputs import (
    CorpusDataset,
    _calendar_price_contexts,
    _price_contexts,
)

ABSENT = ["2019-04-29", "2019-04-30"]


@pytest.fixture(scope="module")
def clock():
    return MarketClock("CN", "2019-01-01", "2019-12-31")


def ohlcv(rows, seed=11):
    generator = np.random.default_rng(seed)
    low = np.exp(generator.normal(2, 0.2, rows))
    high = low * generator.uniform(1.0, 1.05, rows)
    opening, close = (low + (high - low) * generator.uniform(0, 1, rows) for _ in range(2))
    volume = generator.integers(0, 10_000, rows).astype(np.float64)
    volume[::17] = 0
    return np.column_stack([opening, high, low, close, volume])


def test_full_windows_are_bit_identical_to_the_previous_reader_plus_a_presence_bit():
    prices = ohlcv(300)
    ends = np.array([63, 64, 150, 299])
    rows = ends[:, None] - np.arange(63, -1, -1)
    result = _calendar_price_contexts(prices, rows)
    legacy = _price_contexts(prices, ends, 64)
    assert result.shape == (4, 64, 6) and result.dtype == np.float32
    assert result[..., :5].view(np.uint32).tolist() == legacy.view(np.uint32).tolist()
    assert (result[..., 5] == 1).all()


@pytest.mark.parametrize("gaps", [[61, 62], [0], [0, 1], [30]])
def test_gapped_windows_anchor_on_the_first_observed_close_and_leave_exact_zeros(gaps):
    prices = ohlcv(300)
    present = np.ones(64, dtype=bool)
    present[gaps] = False
    end = 200
    rows = np.full(64, -1)
    rows[present] = np.arange(end - present.sum() + 1, end + 1)
    result = _calendar_price_contexts(prices, rows[None])[0]
    window = prices[rows[present]]
    assert result[:, 5].tolist() == present.astype(np.float32).tolist()
    assert result[~present].view(np.uint32).tolist() == np.zeros((len(gaps), 6), np.uint32).tolist()
    expected = np.log(window[:, :4] / window[0, 3]).astype(np.float32)
    assert result[present, :4].tolist() == expected.tolist()
    assert result[present][0, 3] == 0
    volume = np.log1p(window[:, 4] / window[:, 4].mean()).astype(np.float32)
    assert result[present, 4].tolist() == volume.tolist()


def test_invalid_observed_values_are_rejected_in_gapped_windows():
    prices = ohlcv(100)
    prices[90, 3] = -1
    rows = np.full(64, -1)
    rows[2:] = np.arange(38, 100)
    with pytest.raises(ValueError, match="OHLCV"):
        _calendar_price_contexts(prices, rows[None])


def dataset(clock, absent=ABSENT):
    calendar = dict(
        start=clock.days[0].isoformat(),
        end=clock.days[-1].isoformat(),
        decisions_sha256=calendar_digest(clock),
    )
    reader = object.__new__(CorpusDataset)
    reader.price_window = price_window_contract({"CN": calendar}, {"CN": list(absent)})
    reader.context, reader._calendars = 64, {}
    return reader


def test_reader_places_rows_on_the_declared_calendar(clock):
    absent = absent_positions(clock, ABSENT)
    positions = np.setdiff1d(np.arange(len(clock.days)), absent)
    available = np.array(
        [round(clock.decisions[p].timestamp() * 1_000_000) for p in positions], dtype=np.int64
    )
    prices = ohlcv(len(positions))
    reader = dataset(clock)
    ends = np.array([70, 80, 120, 200])
    result = reader.price_windows({"market": "CN"}, prices, available, ends)
    expected = _calendar_price_contexts(prices, window_rows(positions, ends, 64, absent))
    assert result.view(np.uint32).tolist() == expected.view(np.uint32).tolist()
    assert (result[..., 5] == 0).sum() == sum(
        int(np.isin(np.arange(positions[e] - 63, positions[e] + 1), absent).sum()) for e in ends
    )
    # Un precio fuera del calendario declarado no se coloca por aproximación.
    moved = available.copy()
    moved[5] += 1
    with pytest.raises(ValueError, match="calendario declarado"):
        reader.price_windows({"market": "CN"}, prices, moved, ends)
    # Sin el hueco declarado, la misma serie ya no forma ventanas admisibles.
    with pytest.raises(ValueError, match="solo para este activo"):
        dataset(clock, absent=[]).price_windows({"market": "CN"}, prices, available, ends)
    with pytest.raises(ValueError, match="mercado"):
        reader.price_windows({"market": "US"}, prices, available, ends)


def test_reader_without_contract_keeps_the_previous_windows(clock):
    prices = ohlcv(120)
    reader = object.__new__(CorpusDataset)
    reader.price_window, reader.context = None, 64
    ends = np.array([63, 119])
    result = reader.price_windows({"market": "CN"}, prices, None, ends)
    assert np.array_equal(result, _price_contexts(prices, ends, 64))


def test_a_different_calendar_is_rejected(clock):
    reader = dataset(clock)
    reader.price_window["calendars"]["CN"]["decisions_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="calendario reconstruido"):
        reader.price_windows({"market": "CN"}, ohlcv(10), np.zeros(10, np.int64), np.array([5]))


def fixture_windows(root, market_absent=None):
    """Ventanas de precios por clave de muestra, leídas del corpus y de su primera vista."""
    from mars_titan.data.input_policy import HISTORICAL_MASKED
    from tests.training.historical_temporal_fixture import historical_temporal_fixture
    from tests.training.test_historical_temporal import prepare

    fixture = historical_temporal_fixture(root / "source", market_absent=market_absent)
    prepare(fixture, root / "views")
    result = {}
    for name, path in (("corpus", fixture.parent), ("view", root / "views/fold-000/manifest.json")):
        reader = CorpusDataset(path, input_policy=HISTORICAL_MASKED)
        for partition in ("train", "validation"):
            for batch in reader.batches(partition=partition, batch_size=3, epoch=0, seed=0):
                for index, key in enumerate(batch["sample_ids"]):
                    result.setdefault(name, {})[key] = batch["inputs"]["prices"][index].copy()
    return result, reader, fixture.clocks["US"]


def test_declared_market_absence_reaches_reader_views_and_the_ordered_corpus(tmp_path):
    from datetime import date

    from mars_titan.data.input_policy import HISTORICAL_MASKED
    from mars_titan.environments.corpus_source import ParquetCohortSource, prepare_causal_corpus

    gap = "2022-12-01"
    reference, _, _ = fixture_windows(tmp_path / "reference")
    gapped, reader, clock = fixture_windows(tmp_path / "gapped", {"US": [gap]})
    assert reader.price_channels == 6
    assert reader.price_window["market_absent_sessions"] == {"US": [gap]}
    hole = clock.days.index(date.fromisoformat(gap))
    position = {round(t.timestamp() * 1_000_000): i for i, t in enumerate(clock.decisions)}
    crossed = 0
    for name in ("corpus", "view"):
        assert reference[name].keys() == gapped[name].keys() and reference[name]
        for key, legacy in reference[name].items():
            window = gapped[name][key]
            assert legacy.shape == (64, 5) and window.shape == (64, 6)
            end = position[int(key.rsplit("/", 1)[1])]
            if end - 63 <= hole <= end:
                crossed += 1
                # El hueco ocupa su posición del calendario y no se rellena con otro precio.
                absent = np.flatnonzero(window[:, 5] == 0)
                assert absent.tolist() == [hole - end + 63]
                assert window[absent].view(np.uint32).tolist() == [[0] * 6]
                assert window[0, 3] == 0 or hole == end - 63
                assert not np.array_equal(window[:, :5], legacy)
            else:
                # Sin huecos, la ventana es la misma bit a bit con el bit de presencia a uno.
                assert (window[:, 5] == 1).all()
                assert window[:, :5].view(np.uint32).tolist() == legacy.view(np.uint32).tolist()
    assert crossed >= 2
    view = tmp_path / "gapped/views/fold-000/manifest.json"
    report = prepare_causal_corpus(
        view, tmp_path / "ordered", batch_size=2, input_policy=HISTORICAL_MASKED
    )
    assert report["shapes"]["prices"] == [64, 6]
    with ParquetCohortSource(
        tmp_path / "ordered/manifest.json", partition="train", input_policy=HISTORICAL_MASKED
    ) as source:
        windows = np.concatenate([source(i)["inputs"]["prices"] for i in range(len(source))])
    assert windows.shape[1:] == (64, 6) and set(np.unique(windows[..., 5])) <= {0.0, 1.0}


def test_information_view_declares_the_presence_bit_as_a_dependency_of_every_price_channel():
    from mars_titan.training.information_inputs import _price_nodes

    legacy, gate = _price_nodes(64, 5)
    assert gate == [] and [node["name"] for node in legacy] == [
        f"prices/{name}" for name in ("open", "high", "low", "close", "volume")
    ]
    assert legacy[0]["value"] == list(range(0, 320, 5))
    assert legacy[0]["dependencies"] == ["prices/close"] and legacy[3]["dependencies"] == []
    nodes, gate = _price_nodes(64, 6)
    assert gate == ["prices/present"]
    columns = sorted(index for node in nodes for index in node["value"])
    assert columns == list(range(384))
    by_name = {node["name"]: node for node in nodes}
    assert by_name["prices/present"]["value"] == list(range(5, 384, 6))
    assert by_name["prices/present"]["dependencies"] == []
    for name in ("open", "high", "low", "close", "volume"):
        assert "prices/present" in by_name[f"prices/{name}"]["dependencies"]
    with pytest.raises(ValueError, match="canales"):
        _price_nodes(64, 4)


def test_temporal_views_admit_a_short_history_window_completed_by_a_market_gap(tmp_path):
    from datetime import date

    import pyarrow.parquet as pq

    from mars_titan.data.input_policy import HISTORICAL_MASKED
    from tests.training.historical_temporal_fixture import (
        DEFAULT_DAYS,
        historical_temporal_fixture,
    )
    from tests.training.test_historical_temporal import prepare

    clock = MarketClock("US", "1999-01-01", "2024-01-05")
    decision = clock.days.index(date.fromisoformat("2022-11-15"))
    fixture = historical_temporal_fixture(
        tmp_path / "source",
        days={"US": list(DEFAULT_DAYS[3:])},
        market_absent={"US": [clock.days[decision - 30].isoformat()]},
        price_start=clock.days[decision - 63].isoformat(),
    )
    asset = fixture.metadata["assets"][0]
    samples = pq.read_table(
        f"{fixture.metadata['roots']['samples']}/US/{asset['symbol']}/samples.parquet"
    )
    # La primera ventana reúne 63 filas y el hueco de mercado, así que termina en la fila 62.
    assert samples.column("price_end_index").to_pylist()[0] == 62
    prepare(fixture, tmp_path / "views")
    reader = CorpusDataset(
        tmp_path / "views/fold-000/manifest.json", input_policy=HISTORICAL_MASKED
    )
    absent = [
        (batch["inputs"]["prices"][..., 5] == 0).sum(axis=1)
        for partition in ("train", "validation")
        for batch in reader.batches(partition=partition, batch_size=3, epoch=0, seed=0)
    ]
    assert np.concatenate(absent).max() == 1


def test_financial_observations_carry_the_same_gapped_windows_as_the_reader(tmp_path):
    from mars_titan.data.input_policy import HISTORICAL_MASKED
    from mars_titan.memory import financial_observations as api
    from mars_titan.memory.financial_session import FinancialPhase

    windows, _, _ = fixture_windows(tmp_path, {"US": ["2022-12-01"]})
    dataset = CorpusDataset(
        tmp_path / "source/parent/manifest.json", input_policy=HISTORICAL_MASKED
    )
    start, close = 946_684_800_000_000, 1_672_531_200_000_000
    phase = FinancialPhase("train", start, start, close, close)
    manifest = api.prepare_observation_index(dataset, tmp_path / "index", phase=phase)
    seen, gapped = 0, 0
    for event in api.FinancialObservationSource(dataset, manifest).events():
        for batch in event.inputs:
            assert batch["inputs"]["prices"].shape[1:] == (64, 6)
            for index, key in enumerate(batch["sample_ids"]):
                expected = windows["corpus"].get(key)
                if expected is not None:
                    seen += 1
                    observed = batch["inputs"]["prices"][index]
                    gapped += int((observed[:, 5] == 0).any())
                    assert observed.view(np.uint32).tolist() == expected.view(np.uint32).tolist()
    assert seen > 0 and gapped > 0
