"""Artefacto de señales de panel precalculadas (`data.panel_signals`): formato y rechazos."""

import hashlib
import json

import numpy as np
import pytest

from mars_titan.data import technical_features
from mars_titan.data.storage import atomic_json, sha256

EXPECTED_NAMES = (
    "bv_21",
    "jump_var_ratio_21",
    "vol_of_vol_63",
    "realized_skew_signed_21",
    "peer_resid_mom_5",
    "peer_resid_mom_21",
    "peer_resid_mom_63",
    "liq_cond_rev_5",
    "signed_abn_turn_21",
    "kaufman_er_21",
    "amihud_asym_up_down_21",
    "rank_vel_5_21",
    "rank_accel_5_21_63",
    "peer_rev_5_x_idio_vol_surge",
    "limit_hit_streak_5",
    "post_halt_age_decay",
)
PANEL_NAME = "price_volume_16_v1"
SESSIONS = ["2022-12-28", "2022-12-29", "2022-12-30", "2023-01-03", "2023-12-29"]
SYMBOLS = ["AAA", "BBB", "CCC"]


def _encoded_values(sessions=SESSIONS, symbols=SYMBOLS, width=16):
    """Array float32 acotado a ±8 donde cada celda identifica (sesión, símbolo, señal)."""
    t, n = len(sessions), len(symbols)
    grid = np.zeros((t, n, width), dtype=np.float32)
    for ti in range(t):
        for ni in range(n):
            for ki in range(width):
                grid[ti, ni, ki] = np.float32(-7.0 + 1.5 * ti + 0.3 * ni + 0.01 * ki)
    return grid


def _raw_artifact(
    root,
    *,
    name=PANEL_NAME,
    market="US",
    array=None,
    manifest_updates=None,
    manifest_removals=(),
    fortran=False,
):
    """Escribir un artefacto crudo (incluso inválido) con huellas para probar el lector."""
    folder = root / name / market
    folder.mkdir(parents=True, exist_ok=True)
    if array is None:
        array = _encoded_values()
    if fortran:
        array = np.asfortranarray(array)
    panel_path = folder / "panel.npy"
    np.save(panel_path, array, allow_pickle=False)
    manifest = {
        "schema_version": 1,
        "kind": "panel_signals_v1",
        "name": name,
        "market": market,
        "names": list(EXPECTED_NAMES),
        "sessions": list(SESSIONS),
        "symbols": list(SYMBOLS),
        "shape": list(array.shape),
        "dtype": "float32",
        "panel_sha256": sha256(panel_path),
        "final_test_opened": False,
        "source": {"tool": "test_fixture"},
    }
    if manifest_updates:
        manifest.update(manifest_updates)
    for key in manifest_removals:
        manifest.pop(key, None)
    manifest_path = folder / "manifest.json"
    atomic_json(manifest_path, manifest)
    return manifest_path, sha256(manifest_path)


def test_catalog_declares_price_volume_16_v1_in_exact_order_and_limit():
    from mars_titan.data import panel_signals

    assert panel_signals.ENV == "MARS_TITAN_PANEL_SIGNALS"
    assert panel_signals.KIND == "panel_signals_v1"
    assert panel_signals.LIMIT == technical_features.LIMIT == 8.0
    assert panel_signals.PANELS == {PANEL_NAME: EXPECTED_NAMES}
    assert len(panel_signals.PANELS[PANEL_NAME]) == 16


@pytest.mark.parametrize("market", ["US", "CN"])
def test_write_panel_and_reader_roundtrip_uses_memmap_and_exact_days(tmp_path, market):
    from mars_titan.data import panel_signals

    values = _encoded_values()
    values[0, 0, 0] = np.float32(8.0)
    values[0, 0, 1] = np.float32(-8.0)
    manifest_path, digest = panel_signals.write_panel(
        tmp_path,
        name=PANEL_NAME,
        market=market,
        sessions=SESSIONS,
        symbols=SYMBOLS,
        values=values,
        source={"producer": "external_tool", "version": 3},
    )
    assert manifest_path == tmp_path / PANEL_NAME / market / "manifest.json"
    assert digest == sha256(manifest_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["kind"] == "panel_signals_v1"
    assert document["name"] == PANEL_NAME
    assert document["market"] == market
    assert document["names"] == list(EXPECTED_NAMES)
    assert document["sessions"] == SESSIONS
    assert document["symbols"] == SYMBOLS
    assert document["shape"] == [len(SESSIONS), len(SYMBOLS), 16]
    assert document["dtype"] == "float32"
    assert document["panel_sha256"] == sha256(manifest_path.with_name("panel.npy"))
    assert document["final_test_opened"] is False
    assert document["source"] == {"producer": "external_tool", "version": 3}

    reader = panel_signals.PanelSignals(
        tmp_path, name=PANEL_NAME, market=market, manifest_sha256=digest
    )
    assert isinstance(reader._data, np.memmap)
    assert reader._data.dtype == np.float32
    assert reader._data.flags.c_contiguous
    assert reader.width == 16

    # Día exacto, huecos antes/entre/después y orden arbitrario: nunca día anterior ni posterior.
    queried = np.array(
        ["2022-12-27", "2022-12-28", "2022-12-31", "2023-01-03", "2023-06-01", "2024-01-02"],
        dtype="datetime64[D]",
    )
    got = reader.rows("AAA", queried)
    assert got.dtype == np.float32 and got.shape == (len(queried), 16)
    np.testing.assert_array_equal(got[0], np.zeros(16, dtype=np.float32))
    np.testing.assert_array_equal(got[1], values[0, 0])
    np.testing.assert_array_equal(got[2], np.zeros(16, dtype=np.float32))
    np.testing.assert_array_equal(got[3], values[3, 0])
    np.testing.assert_array_equal(got[4], np.zeros(16, dtype=np.float32))
    np.testing.assert_array_equal(got[5], np.zeros(16, dtype=np.float32))

    # Símbolo ausente y vector vacío devuelven ceros con la forma correcta.
    missing = reader.rows("ZZZ", queried)
    assert missing.dtype == np.float32 and missing.shape == (len(queried), 16)
    np.testing.assert_array_equal(missing, np.zeros((len(queried), 16), dtype=np.float32))
    empty = reader.rows("AAA", np.array([], dtype="datetime64[D]"))
    assert empty.dtype == np.float32 and empty.shape == (0, 16)


def test_panel_sha256_is_computed_once_per_process_and_rehashed_on_change(tmp_path, monkeypatch):
    from mars_titan.data import panel_signals

    _, digest = panel_signals.write_panel(
        tmp_path,
        name=PANEL_NAME,
        market="US",
        sessions=SESSIONS,
        symbols=SYMBOLS,
        values=_encoded_values(),
    )
    calls = []
    real_sha256 = panel_signals.sha256

    def counted(path):
        calls.append(path.name)
        return real_sha256(path)

    monkeypatch.setattr(panel_signals, "sha256", counted)
    first = panel_signals.PanelSignals(
        tmp_path, name=PANEL_NAME, market="US", manifest_sha256=digest
    )
    second = panel_signals.PanelSignals(
        tmp_path, name=PANEL_NAME, market="US", manifest_sha256=digest
    )
    assert calls == ["panel.npy"]
    first.verify()
    second.verify()
    assert calls == ["panel.npy"]

    # Alterar panel.npy cambia la firma de stat: se vuelve a calcular el SHA-256 y se rechaza.
    panel_path = tmp_path / PANEL_NAME / "US" / "panel.npy"
    raw = bytearray(panel_path.read_bytes())
    raw[-1] ^= 0x01
    panel_path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="panel.npy|sha256|huella|cambió"):
        panel_signals.PanelSignals(tmp_path, name=PANEL_NAME, market="US", manifest_sha256=digest)
    assert calls == ["panel.npy", "panel.npy"]
    with pytest.raises(ValueError, match="cambió|huella|panel"):
        first.verify()


def test_panel_root_fails_closed_when_env_missing_or_invalid(tmp_path, monkeypatch):
    from mars_titan.data import panel_signals

    monkeypatch.delenv(panel_signals.ENV, raising=False)
    with pytest.raises(ValueError, match=panel_signals.ENV):
        panel_signals.panel_root()

    monkeypatch.setenv(panel_signals.ENV, "")
    with pytest.raises(ValueError, match=panel_signals.ENV):
        panel_signals.panel_root()

    monkeypatch.setenv(panel_signals.ENV, "relative/panels")
    with pytest.raises(ValueError, match=panel_signals.ENV):
        panel_signals.panel_root()

    monkeypatch.setenv(panel_signals.ENV, str(tmp_path / "no-existe"))
    with pytest.raises(ValueError, match=panel_signals.ENV):
        panel_signals.panel_root()

    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    monkeypatch.setenv(panel_signals.ENV, str(link))
    with pytest.raises(ValueError, match=panel_signals.ENV):
        panel_signals.panel_root()

    monkeypatch.setenv(panel_signals.ENV, str(real))
    assert panel_signals.panel_root() == real


def test_rejects_unknown_panel_name(tmp_path):
    from mars_titan.data import panel_signals

    with pytest.raises(ValueError, match="catálogo|panel"):
        panel_signals.write_panel(
            tmp_path,
            name="unknown_panel_v1",
            market="US",
            sessions=SESSIONS,
            symbols=SYMBOLS,
            values=_encoded_values(),
        )
    _, digest = _raw_artifact(tmp_path, name="unknown_panel_v1")
    with pytest.raises(ValueError, match="catálogo|panel"):
        panel_signals.PanelSignals(
            tmp_path, name="unknown_panel_v1", market="US", manifest_sha256=digest
        )


def test_rejects_signal_names_in_wrong_order_or_different(tmp_path):
    from mars_titan.data import panel_signals

    swapped = list(EXPECTED_NAMES)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError, match="nombres|orden"):
        panel_signals.write_panel(
            tmp_path,
            name=PANEL_NAME,
            market="US",
            sessions=SESSIONS,
            symbols=SYMBOLS,
            values=_encoded_values(),
            names=swapped,
        )
    _, digest = _raw_artifact(tmp_path, manifest_updates={"names": swapped})
    with pytest.raises(ValueError, match="nombres|orden"):
        panel_signals.PanelSignals(tmp_path, name=PANEL_NAME, market="US", manifest_sha256=digest)


def test_rejects_mismatched_manifest_sha256_and_panel_sha256(tmp_path):
    from mars_titan.data import panel_signals

    _, digest = panel_signals.write_panel(
        tmp_path,
        name=PANEL_NAME,
        market="US",
        sessions=SESSIONS,
        symbols=SYMBOLS,
        values=_encoded_values(),
    )
    wrong = ("0" if digest[0] != "0" else "1") + digest[1:]
    with pytest.raises(ValueError, match="manifest|sha256|huella"):
        panel_signals.PanelSignals(tmp_path, name=PANEL_NAME, market="US", manifest_sha256=wrong)

    # Manifiesto con panel_sha256 que no coincide con los bytes de panel.npy.
    _, bad_panel_digest = _raw_artifact(
        tmp_path / "bad_data",
        manifest_updates={"panel_sha256": hashlib.sha256(b"other").hexdigest()},
    )
    with pytest.raises(ValueError, match="panel.npy|sha256|huella"):
        panel_signals.PanelSignals(
            tmp_path / "bad_data", name=PANEL_NAME, market="US", manifest_sha256=bad_panel_digest
        )


def test_rejects_wrong_shape_dtype_and_fortran_order(tmp_path):
    from mars_titan.data import panel_signals

    # Forma distinta en panel.npy respecto a los ejes declarados.
    wrong_shape = np.zeros((len(SESSIONS), len(SYMBOLS), 15), dtype=np.float32)
    with pytest.raises(ValueError, match="forma|shape"):
        panel_signals.write_panel(
            tmp_path / "w1",
            name=PANEL_NAME,
            market="US",
            sessions=SESSIONS,
            symbols=SYMBOLS,
            values=wrong_shape,
        )
    _, d1 = _raw_artifact(
        tmp_path / "r1",
        array=wrong_shape,
        manifest_updates={"shape": [len(SESSIONS), len(SYMBOLS), 16]},
    )
    with pytest.raises(ValueError, match="forma|shape"):
        panel_signals.PanelSignals(
            tmp_path / "r1", name=PANEL_NAME, market="US", manifest_sha256=d1
        )

    _, d_manifest_shape = _raw_artifact(
        tmp_path / "r2",
        manifest_updates={"shape": [len(SESSIONS) + 1, len(SYMBOLS), 16]},
    )
    with pytest.raises(ValueError, match="forma|shape"):
        panel_signals.PanelSignals(
            tmp_path / "r2", name=PANEL_NAME, market="US", manifest_sha256=d_manifest_shape
        )

    # Tipo distinto (float64) tanto al escribir como al leer.
    as_f64 = _encoded_values().astype(np.float64)
    with pytest.raises(ValueError, match="float32|tipo|dtype"):
        panel_signals.write_panel(
            tmp_path / "w2",
            name=PANEL_NAME,
            market="US",
            sessions=SESSIONS,
            symbols=SYMBOLS,
            values=as_f64,
        )
    _, d_f64 = _raw_artifact(tmp_path / "r3", array=as_f64)
    with pytest.raises(ValueError, match="float32|tipo|dtype"):
        panel_signals.PanelSignals(
            tmp_path / "r3", name=PANEL_NAME, market="US", manifest_sha256=d_f64
        )

    _, d_manifest_dtype = _raw_artifact(tmp_path / "r4", manifest_updates={"dtype": "float64"})
    with pytest.raises(ValueError, match="float32|tipo|dtype"):
        panel_signals.PanelSignals(
            tmp_path / "r4", name=PANEL_NAME, market="US", manifest_sha256=d_manifest_dtype
        )

    # Orden Fortran rechazado.
    _, d_fortran = _raw_artifact(tmp_path / "r5", fortran=True)
    with pytest.raises(ValueError, match="orden C|contiguo"):
        panel_signals.PanelSignals(
            tmp_path / "r5", name=PANEL_NAME, market="US", manifest_sha256=d_fortran
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, 8.001, -8.001])
def test_rejects_non_finite_and_out_of_bounds_values_on_write_and_read(tmp_path, bad_value):
    from mars_titan.data import panel_signals

    bad = _encoded_values()
    bad[1, 1, 3] = np.float32(bad_value)
    with pytest.raises(ValueError, match="finito|cota|±8"):
        panel_signals.write_panel(
            tmp_path / "w",
            name=PANEL_NAME,
            market="US",
            sessions=SESSIONS,
            symbols=SYMBOLS,
            values=bad,
        )
    _, digest = _raw_artifact(tmp_path / "r", array=bad)
    reader = panel_signals.PanelSignals(
        tmp_path / "r", name=PANEL_NAME, market="US", manifest_sha256=digest
    )
    # Leer un día sin el valor malo funciona; leer el día con el valor inválido falla en cerrado.
    ok = reader.rows("BBB", np.array(["2022-12-28"], dtype="datetime64[D]"))
    assert ok.shape == (1, 16)
    with pytest.raises(ValueError, match="finito|cota|±8"):
        reader.rows("BBB", np.array(["2022-12-29"], dtype="datetime64[D]"))


def test_rejects_wrong_market_and_sealed_final_test_fields(tmp_path):
    from mars_titan.data import panel_signals

    with pytest.raises(ValueError, match="mercado"):
        panel_signals.write_panel(
            tmp_path / "w",
            name=PANEL_NAME,
            market="JP",
            sessions=SESSIONS,
            symbols=SYMBOLS,
            values=_encoded_values(),
        )
    # Artefacto guardado en carpeta US pero cuyo manifiesto declara CN.
    _, d_wrong_mkt = _raw_artifact(tmp_path / "r1", market="US", manifest_updates={"market": "CN"})
    with pytest.raises(ValueError, match="mercado"):
        panel_signals.PanelSignals(
            tmp_path / "r1", name=PANEL_NAME, market="US", manifest_sha256=d_wrong_mkt
        )
    # final_test_opened = True rechazado.
    _, d_open = _raw_artifact(tmp_path / "r2", manifest_updates={"final_test_opened": True})
    with pytest.raises(ValueError, match="final_test_opened|test final"):
        panel_signals.PanelSignals(
            tmp_path / "r2", name=PANEL_NAME, market="US", manifest_sha256=d_open
        )
    # Clase (kind) o schema_version incorrectos.
    for change in (
        {"kind": "panel_signals_v2"},
        {"schema_version": 2},
        {"schema_version": True},
        {"name": "other"},
        {"source": "not_a_dict"},
    ):
        _, digest = _raw_artifact(tmp_path / f"r_{next(iter(change))}", manifest_updates=change)
        with pytest.raises(ValueError):
            panel_signals.PanelSignals(
                tmp_path / f"r_{next(iter(change))}",
                name=PANEL_NAME,
                market="US",
                manifest_sha256=digest,
            )


@pytest.mark.parametrize(
    "bad_symbols",
    [
        ["BBB", "AAA", "CCC"],
        ["AAA", "AAA", "BBB"],
        [],
        [""],
        ["AAA", 123],
    ],
)
def test_rejects_unsorted_duplicate_or_invalid_symbols(tmp_path, bad_symbols):
    from mars_titan.data import panel_signals

    arr = np.zeros((len(SESSIONS), max(len(bad_symbols), 1), 16), dtype=np.float32)
    with pytest.raises(ValueError, match="símbolos"):
        panel_signals.write_panel(
            tmp_path / "w",
            name=PANEL_NAME,
            market="US",
            sessions=SESSIONS,
            symbols=bad_symbols,
            values=arr[:, : len(bad_symbols), :],
        )
    _, digest = _raw_artifact(
        tmp_path / "r",
        array=arr[:, : len(bad_symbols), :] if bad_symbols else arr,
        manifest_updates={
            "symbols": bad_symbols,
            "shape": [len(SESSIONS), len(bad_symbols), 16],
        },
    )
    with pytest.raises(ValueError, match="símbolos"):
        panel_signals.PanelSignals(
            tmp_path / "r", name=PANEL_NAME, market="US", manifest_sha256=digest
        )


@pytest.mark.parametrize(
    "bad_sessions, match",
    [
        (["2022-12-30", "2022-12-29"], "crecientes|sesiones"),
        (["2022-12-29", "2022-12-29"], "crecientes|sesiones"),
        (["2023-12-29", "2024-01-01"], "2024|sellado"),
        (["2023-12-29", "2024-06-03"], "2024|sellado"),
        (["2022-1-3"], "ISO|sesiones"),
        (["2022-02-30"], "ISO|sesiones"),
        ([], "sesiones"),
    ],
)
def test_rejects_unsorted_invalid_or_2024_sessions(tmp_path, bad_sessions, match):
    from mars_titan.data import panel_signals

    t = len(bad_sessions)
    arr = np.zeros((max(t, 1), len(SYMBOLS), 16), dtype=np.float32)
    with pytest.raises(ValueError, match=match):
        panel_signals.write_panel(
            tmp_path / "w",
            name=PANEL_NAME,
            market="US",
            sessions=bad_sessions,
            symbols=SYMBOLS,
            values=arr[:t],
        )
    _, digest = _raw_artifact(
        tmp_path / "r",
        array=arr[:t] if t else arr,
        manifest_updates={"sessions": bad_sessions, "shape": [t, len(SYMBOLS), 16]},
    )
    with pytest.raises(ValueError, match=match):
        panel_signals.PanelSignals(
            tmp_path / "r", name=PANEL_NAME, market="US", manifest_sha256=digest
        )


def test_rejects_symlinks_and_missing_files_and_bad_days_in_rows(tmp_path):
    from mars_titan.data import panel_signals

    _, digest = panel_signals.write_panel(
        tmp_path / "real",
        name=PANEL_NAME,
        market="US",
        sessions=SESSIONS,
        symbols=SYMBOLS,
        values=_encoded_values(),
    )
    # Falta panel.npy
    missing_dir = tmp_path / "missing" / PANEL_NAME / "US"
    missing_dir.mkdir(parents=True)
    (missing_dir / "manifest.json").write_bytes(
        (tmp_path / "real" / PANEL_NAME / "US" / "manifest.json").read_bytes()
    )
    with pytest.raises(ValueError, match="panel.npy|artefacto"):
        panel_signals.PanelSignals(
            tmp_path / "missing", name=PANEL_NAME, market="US", manifest_sha256=digest
        )

    # Enlace simbólico en panel.npy rechazado.
    sym_dir = tmp_path / "sym" / PANEL_NAME / "US"
    sym_dir.mkdir(parents=True)
    (sym_dir / "manifest.json").write_bytes(
        (tmp_path / "real" / PANEL_NAME / "US" / "manifest.json").read_bytes()
    )
    (sym_dir / "panel.npy").symlink_to(tmp_path / "real" / PANEL_NAME / "US" / "panel.npy")
    with pytest.raises(ValueError, match="regular|enlace|artefacto"):
        panel_signals.PanelSignals(
            tmp_path / "sym", name=PANEL_NAME, market="US", manifest_sha256=digest
        )

    reader = panel_signals.PanelSignals(
        tmp_path / "real", name=PANEL_NAME, market="US", manifest_sha256=digest
    )
    with pytest.raises(ValueError, match="sesiones|días"):
        reader.rows("AAA", np.array([["2022-12-28"]], dtype="datetime64[D]"))
    with pytest.raises(ValueError, match="sesiones|días|NaT"):
        reader.rows("AAA", np.array(["NaT"], dtype="datetime64[D]"))


def test_checked_panel_spec_validates_structure_digests_and_markets():
    from mars_titan.data import panel_signals

    sha = "a" * 64
    valid = {"name": PANEL_NAME, "sha256": {"US": sha}}
    assert panel_signals.checked_panel_spec(valid) == valid
    assert panel_signals.checked_panel_spec(valid, markets=["US"]) == valid
    joint = {"name": PANEL_NAME, "sha256": {"US": sha, "CN": "b" * 64}}
    assert panel_signals.checked_panel_spec(joint, markets=["CN", "US"]) == joint

    for bad in (
        None,
        PANEL_NAME,
        {},
        {"name": PANEL_NAME},
        {"name": PANEL_NAME, "sha256": {}},
        {"name": PANEL_NAME, "sha256": {"US": sha}, "extra": 1},
        {"name": "otro_panel", "sha256": {"US": sha}},
        {"name": PANEL_NAME, "sha256": {"JP": sha}},
        {"name": PANEL_NAME, "sha256": {"US": "A" * 64}},
        {"name": PANEL_NAME, "sha256": {"US": "abc"}},
        {"name": PANEL_NAME, "sha256": {"US": 123}},
    ):
        with pytest.raises(ValueError):
            panel_signals.checked_panel_spec(bad)

    with pytest.raises(ValueError, match="mercado"):
        panel_signals.checked_panel_spec(valid, markets=["CN"])
    with pytest.raises(ValueError, match="mercado"):
        panel_signals.checked_panel_spec(valid, markets=["US", "CN"])
    with pytest.raises(ValueError, match="mercado"):
        panel_signals.checked_panel_spec(joint, markets=["US"])
