"""Claves de la memoria por vecinos desde la vista: definición del laboratorio y contratos."""

import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data import technical_features as tf
from mars_titan.evaluation import posttraining_pilots as pilots
from mars_titan.memory import view_keys as vk

DAY_US = 86_400_000_000
FIRST = np.datetime64("2010-01-04")
CLOSE = {"US": 21, "CN": 7}
SPLIT = (("train", 300), ("validation", 40), ("calibration", 30), ("evaluation", 50))


def _sessions(count):
    days = np.arange(FIRST, FIRST + np.timedelta64(count * 2, "D"))
    weekday = (days.astype("datetime64[D]").view("int64") - 4) % 7
    return days[weekday < 5][:count]


def _prices(rng, days, *, start=100.0, flat=()):
    rows = len(days)
    close = start * np.exp(np.cumsum(rng.normal(0, 0.02, rows)))
    opened = close * np.exp(rng.normal(0, 0.01, rows))
    high = np.maximum(opened, close) * (1 + rng.uniform(0, 0.02, rows))
    low = np.minimum(opened, close) * (1 - rng.uniform(0, 0.02, rows))
    volume = rng.uniform(1e5, 1e6, rows)
    for i in flat:  # sesiones sin negociación
        opened[i] = close[i] = high[i] = low[i]
        volume[i] = 0.0
    return dict(open=opened, high=high, low=low, close=close, volume=volume)


def _write_prices(path, days, columns, hour):
    path.parent.mkdir(parents=True, exist_ok=True)
    moments = days.astype("datetime64[us]").astype(np.int64) + hour * 3_600_000_000
    table = pa.table(
        {
            **{k: pa.array(v, pa.float64()) for k, v in columns.items()},
            "available_at": pa.array(moments, pa.timestamp("us", tz="UTC")),
        }
    )
    pq.write_table(table, path)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_view(root, *, markets=("US",), assets=4, seed=0, edit=None, partitions=None):
    """Una vista pequeña: precios preparados, etiquetas por activo, factor de mercado y manifiesto.

    `edit(market, symbol, days, columns)` puede cambiar los precios de un activo antes de
    escribirlos y `partitions` sustituye la partición de cada día.
    """
    rng = np.random.default_rng(seed)
    days = _sessions(sum(n for _, n in SPLIT) + 30)
    names = [p for p, n in SPLIT for _ in range(n)] + [None] * 30
    if partitions is not None:
        names = partitions(list(names))
    prepared, labels = root / "prepared", root / "labels"
    manifest = dict(roots=dict(prepared=str(prepared), labels=str(labels)), assets=[])
    manifest["market_factors"] = {}
    for market in markets:
        hour = CLOSE[market]
        factor = prepared / market / "FACTOR" / "prices.parquet"
        _write_prices(factor, days, _prices(rng, days), hour)
        manifest["market_factors"][market] = dict(
            prices_path=str(factor), prices_sha256=_digest(factor)
        )
        for a in range(assets):
            symbol = f"S{a}"
            # El activo 1 empieza tarde: le falta historia al principio.
            own = days[40:] if a == 1 else days
            columns = _prices(rng, own, flat=(5, 6, 7) if a == 2 else ())
            if edit is not None:
                edit(market, symbol, own, columns)
            path = prepared / market / symbol / "prices.parquet"
            _write_prices(path, own, columns, hour)
            moments = days.astype("datetime64[us]").astype(np.int64) + (hour * 60 + 5) * 60_000_000
            label = labels / market / symbol / "labels.parquet"
            label.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.table(
                    dict(
                        prediction_at=pa.array(moments, pa.timestamp("us", tz="UTC")),
                        target=pa.array(rng.normal(0, 0.01, len(days)), pa.float64()),
                        partition=pa.array(names, pa.string()),
                    )
                ),
                label,
            )
            manifest["assets"].append(
                dict(
                    market=market,
                    symbol=symbol,
                    prices_sha256=_digest(path),
                    labels_sha256=_digest(label),
                )
            )
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def build(tmp_path, name="claves", **options):
    view = write_view(tmp_path / f"vista-{name}", **options)
    output = tmp_path / name
    receipt = vk.build(view, output, log=lambda _: None)
    return output, receipt


def load(output):
    arrays = {name: np.load(output / name) for name in vk.OUTPUTS}
    arrays["names"] = json.loads((output / "names.json").read_text())["assets"]
    return arrays


def test_the_lab_columns_keep_long_horizon_v1_and_add_sixteen_with_missing_history_as_nan():
    rng = np.random.default_rng(3)
    days = _sessions(400)
    columns = _prices(rng, days, flat=(300, 301))
    prices = np.column_stack([columns[c] for c in ("open", "high", "low", "close", "volume")])
    factor = _prices(rng, days)
    market = factor["close"] / factor["open"] - 1
    values = vk.lab_columns(prices, days, days, market)
    assert values.dtype == np.float32 and values.shape == (400, 46)
    np.testing.assert_array_equal(values[:, :30], tf.technical_features(prices, days, days, market))
    extra = values[:, 30:]
    named = dict(zip(vk.LAB_EXTRA + vk.LAB_PERCENT, extra.T, strict=True))
    # Sin las 252 sesiones de la volatilidad anual, las medias escaladas faltan (NaN, no cero).
    assert (
        np.isnan(named["intraday_5"][:252]).all() and np.isfinite(named["intraday_5"][252:]).all()
    )
    assert np.isnan(named["intraday_mean_252_percent"][:251]).all()
    assert np.isnan(named["overnight_mean_252_percent"][:252]).all()
    assert np.isnan(named["zero_volume_63"][:62]).all()
    # Definiciones a mano en la fila 330.
    i = 330
    log_close = np.log(columns["close"])
    returns = np.r_[0.0, np.diff(log_close)]
    year = returns[i - 251 : i + 1].std()
    intraday = np.log(columns["close"] / columns["open"])
    expected = intraday[i - 4 : i + 1].mean() / year * np.sqrt(5)
    assert named["intraday_5"][i] == pytest.approx(expected, rel=1e-5)
    assert named["intraday_mean_252_percent"][i] == pytest.approx(
        intraday[i - 251 : i + 1].mean() * 100, rel=1e-5
    )
    stale = (columns["volume"] <= 0) | (
        (columns["open"] == columns["close"]) & (columns["high"] == columns["low"])
    )
    assert named["zero_volume_63"][i] == pytest.approx(stale[i - 62 : i + 1].mean(), rel=1e-6)
    span = columns["high"][i] - columns["low"][i]
    location = (2 * columns["close"][i] - columns["high"][i] - columns["low"][i]) / span
    assert named["close_location_1"][i] == pytest.approx(location, rel=1e-6)
    alpha, beta, residual, _ = tf.market_model(columns["close"] / columns["open"] - 1, market)
    realized = columns["close"][i] / columns["open"][i] - 1 - alpha[i - 1] - beta[i - 1] * market[i]
    assert named["residual_1"][i] == pytest.approx(realized / residual[i - 1], rel=1e-5)
    # Sin factor de mercado, la etiqueta realizada y el alfa faltan.
    alone = vk.lab_columns(prices, days)
    for name in ("residual_1", "residual_5", "residual_21", "alpha_21", "alpha_63"):
        assert np.isnan(alone[:, 30 + (vk.LAB_EXTRA + vk.LAB_PERCENT).index(name)]).all()


def test_the_scale_uses_only_the_train_rows_and_each_row_only_its_past(tmp_path):
    base, receipt = build(tmp_path, "base")
    last = sum(n for _, n in SPLIT) - 1

    def late(market, symbol, days, columns):
        if symbol == "S0":
            columns["close"][last - 10 :] *= 1.5

    changed, changed_receipt = build(tmp_path, "cambiada", edit=late)
    assert changed_receipt["markets"]["US"]["center"] == receipt["markets"]["US"]["center"]
    assert changed_receipt["markets"]["US"]["spread"] == receipt["markets"]["US"]["spread"]
    a, b = load(base), load(changed)
    start = np.datetime64(_sessions(last + 1)[last - 10]).astype(np.int64)
    earlier = a["day.npy"] < start
    np.testing.assert_array_equal(a["X.npy"][earlier], b["X.npy"][earlier])
    np.testing.assert_array_equal(a["XR.npy"][earlier], b["XR.npy"][earlier])
    assert not np.array_equal(a["X.npy"][~earlier], b["X.npy"][~earlier])


def test_the_scale_is_the_robust_one_of_the_lab_on_the_train_rows(tmp_path):
    raw = np.array(
        [[1.0, 5.0, np.nan], [2.0, 5.0, 1.0], [4.0, 5.0, 2.0], [100.0, 5.0, 3.0]], np.float32
    )
    train = np.array([True, True, True, False])
    center, spread = vk.robust_scale(raw, train)
    np.testing.assert_array_equal(center, np.array([2.0, 5.0, 1.5], np.float32))
    # MAD 1 → 1,4826; columna constante → desviación 0 → escala 1; NaN fuera del ajuste.
    np.testing.assert_allclose(spread, [1.4826, 1.0, 0.5 * 1.4826], rtol=1e-6)
    out = vk.scaled(raw, center, spread)
    assert out.dtype == np.float32
    assert out[3, 0] == np.float32(5.0)  # recortada a +5
    assert out[0, 2] == 0.0  # el valor que falta queda en cero


def test_ranks_are_stable_gaussian_scores_within_each_session(tmp_path):
    output, _ = build(tmp_path)
    data = load(output)
    bounds = data["bounds.npy"]
    assert np.all(np.diff(bounds) > 0)
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        assert len(np.unique(data["day.npy"][start:end])) == 1
        block = data["X.npy"][start:end]
        n = end - start
        for j in (0, 4, 16, 45):
            order = np.argsort(block[:, j], kind="stable")
            ranks = np.empty(n, np.float32)
            ranks[order] = np.arange(1, n + 1, dtype=np.float32)
            level = (ranks - np.float32(0.5)) * (np.float32(1.0) / np.float32(n))
            expected = torch.special.ndtri(torch.from_numpy(level)).clamp(-5, 5).numpy()
            np.testing.assert_array_equal(data["XR.npy"][start:end, j], expected)
    # has_market_model vale lo mismo en todos los activos: el rango sigue su orden en la vista.
    first = slice(bounds[0], bounds[1])
    column = tf.NAMES.index("has_market_model")
    assert np.all(np.diff(data["XR.npy"][first, column]) > 0)


def test_the_rank_level_multiplies_by_the_fp32_inverse_like_the_lab_on_the_gpu():
    # Siete filas: (r − ½)·(1/7) y (r − ½)/7 difieren en FP32 en cuatro de ellas.
    values = np.arange(21, dtype=np.float32).reshape(7, 3)[::-1].copy()
    ranks = vk.session_ranks(values, np.array([0, 7]))
    order = np.arange(7, 0, -1, dtype=np.float32)
    level = (order - np.float32(0.5)) * (np.float32(1.0) / np.float32(7))
    assert np.count_nonzero(level != (order - np.float32(0.5)) / np.float32(7)) == 4
    expected = torch.special.ndtri(torch.from_numpy(level)).clamp(-5, 5).numpy()
    for column in range(3):
        np.testing.assert_array_equal(ranks[:, column], expected)


def test_only_the_rows_after_the_fit_are_written_in_the_lab_format(tmp_path):
    output, receipt = build(tmp_path)
    data = load(output)
    sizes = {p: n for p, n in SPLIT}
    assert set(np.unique(data["session_part.npy"])) == {1, 2, 3}
    assert (
        len(data["bounds.npy"]) - 1
        == sizes["validation"] + sizes["calibration"] + sizes["evaluation"]
    )
    assert data["X.npy"].shape == data["XR.npy"].shape == (len(data["day.npy"]), 46)
    assert np.all(np.diff(data["day.npy"]) >= 0)
    # La memoria lee estas claves con la función que leía la caché del laboratorio.
    rows = np.arange(0, len(data["day.npy"]), 7)
    assets = np.array(data["names"])[data["asset.npy"][rows]]
    inputs, _ = pilots.lab_inputs(output, output / "names.json", assets, data["day.npy"][rows])
    np.testing.assert_array_equal(inputs, np.hstack([data["X.npy"], data["XR.npy"]])[rows])
    for name, digest in receipt["outputs"].items():
        assert _digest(output / name) == digest
    assert receipt["final_test_opened"] is False
    assert receipt["view_sha256"] == _digest(tmp_path / "vista-claves" / "manifest.json")


def test_a_joint_view_gives_each_market_what_its_own_view_gives(tmp_path):
    joint, joint_receipt = build(tmp_path, "conjunta", markets=("US", "CN"))
    alone, alone_receipt = build(tmp_path, "sola", markets=("US", "CN"))
    a = load(joint)
    view = tmp_path / "vista-conjunta" / "manifest.json"
    only = vk.build(view, tmp_path / "solo-us", markets=["US"], log=lambda _: None)
    b = load(tmp_path / "solo-us")
    us = np.char.startswith(np.array(a["names"])[a["asset.npy"]], "US/")
    np.testing.assert_array_equal(a["X.npy"][us], b["X.npy"])
    np.testing.assert_array_equal(a["XR.npy"][us], b["XR.npy"])
    assert only["markets"]["US"] == joint_receipt["markets"]["US"]
    assert joint_receipt["markets"]["CN"]["center"] != joint_receipt["markets"]["US"]["center"]
    # Cada sesión es de un solo mercado.
    for start, end in zip(a["bounds.npy"][:-1], a["bounds.npy"][1:], strict=True):
        assert len(set(us[start:end])) == 1


def test_a_file_that_is_not_the_one_of_the_view_is_rejected(tmp_path):
    view = write_view(tmp_path / "vista")
    path = tmp_path / "vista" / "prepared" / "US" / "S2" / "prices.parquet"
    table = pq.read_table(path)
    pq.write_table(table.slice(0, len(table) - 1), path)
    with pytest.raises(
        ValueError, match="S2/prices.parquet no es el archivo de precios de la vista"
    ):
        vk.build(view, tmp_path / "claves", log=lambda _: None)
    assert not (tmp_path / "claves").exists()


def test_the_final_test_and_unknown_partitions_are_never_read(tmp_path):
    def into_2024(names):
        return names  # el último día de evaluación se mueve a 2024 abajo

    view = write_view(tmp_path / "vista")
    label = tmp_path / "vista" / "labels" / "US" / "S0" / "labels.parquet"
    table = pq.read_table(label)
    moments = table["prediction_at"].cast(pa.int64()).to_numpy().copy()
    last = sum(n for _, n in SPLIT) - 1
    moments[last] = np.datetime64("2024-01-02T21:05").astype("datetime64[us]").astype(np.int64)
    table = table.set_column(0, "prediction_at", pa.array(moments, pa.timestamp("us", tz="UTC")))
    pq.write_table(table, label)
    manifest = json.loads(view.read_text())
    manifest["assets"][0]["labels_sha256"] = _digest(label)
    view.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="test final"):
        vk.build(view, tmp_path / "claves", log=lambda _: None)

    def unknown(names):
        names[-1] = "otra"
        return names

    other = write_view(tmp_path / "otra", partitions=unknown)
    with pytest.raises(ValueError, match="particiones desconocidas"):
        vk.build(other, tmp_path / "claves", log=lambda _: None)


def test_existing_keys_are_never_overwritten(tmp_path):
    output, _ = build(tmp_path)
    before = _digest(output / "X.npy")
    with pytest.raises(ValueError, match="ya existe"):
        vk.build(tmp_path / "vista-claves" / "manifest.json", output, log=lambda _: None)
    assert _digest(output / "X.npy") == before


def test_the_comparison_counts_every_bit_that_differs_from_the_lab(tmp_path):
    output, _ = build(tmp_path)
    copy = tmp_path / "laboratorio"
    copy.mkdir()
    for name in (*vk.OUTPUTS, "names.json"):
        (copy / name).write_bytes((output / name).read_bytes())
    same = vk.compare(output, copy, copy / "names.json")
    assert same["matched"] == same["rows"] == same["lab_rows"]
    assert same["X.npy"]["different_bits"] == same["XR.npy"]["different_bits"] == 0
    ranks = np.load(copy / "XR.npy")
    ranks[3, 7] = np.nextafter(ranks[3, 7], np.float32(np.inf))
    np.save(copy / "XR.npy", ranks)
    moved = vk.compare(output, copy, copy / "names.json")
    assert moved["XR.npy"]["different_bits"] == 1 and moved["XR.npy"]["max_ulp"] == 1
    assert moved["XR.npy"]["columns_with_differences"] == [vk.COLUMNS[7]]
