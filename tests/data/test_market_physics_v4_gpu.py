"""Pruebas TDD para física de mercado v4 vectorizada en PyTorch y ZeroCopySessionStore."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from mars_titan.data import market_physics_v4 as mp4
from mars_titan.data import market_physics_v4_gpu as mp4_gpu


def _synthetic_ohlcv_panel(
    rng: np.random.Generator,
    t_len: int = 260,
    n_assets: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    drifts = rng.uniform(-0.0002, 0.0008, size=(1, n_assets))
    vols = rng.uniform(0.008, 0.028, size=(1, n_assets))
    rets = drifts + vols * rng.standard_normal((t_len, n_assets))
    base_px = rng.uniform(15.0, 220.0, size=(1, n_assets))
    close = base_px * np.exp(np.cumsum(rets, axis=0))
    open_ = close * np.exp(0.004 * rng.standard_normal((t_len, n_assets)))
    high = np.maximum(open_, close) * np.exp(np.abs(0.007 * rng.standard_normal((t_len, n_assets))))
    low = np.minimum(open_, close) * np.exp(-np.abs(0.007 * rng.standard_normal((t_len, n_assets))))
    vol_base = rng.uniform(10.0, 15.0, size=(1, n_assets))
    volume = np.exp(vol_base + 0.6 * rng.standard_normal((t_len, n_assets)))
    return open_, high, low, close, volume


def _make_multimodal_arrays(
    rng: np.random.Generator,
    t_len: int = 26,
    n_assets: int = 8,
) -> dict[str, np.ndarray]:
    n_phys = len(mp4.ASSET_V4_NAMES)
    asset_ids = np.broadcast_to(np.arange(n_assets, dtype=np.int64)[None, :], (t_len, n_assets))
    return dict(
        asset_ids=asset_ids.copy(),
        prices=rng.standard_normal((t_len, n_assets, 16)).astype(np.float32),
        fundamentals=rng.standard_normal((t_len, n_assets, 6)).astype(np.float32),
        macro=rng.standard_normal((t_len, n_assets, 4)).astype(np.float32),
        news=rng.standard_normal((t_len, n_assets, 8)).astype(np.float32),
        charts=rng.standard_normal((t_len, n_assets, 8)).astype(np.float32),
        modality_mask=(rng.random((t_len, n_assets, 5)) > 0.2).astype(np.float32),
        physics_v4=rng.standard_normal((t_len, n_assets, n_phys)).astype(np.float32),
        high_snr_context=rng.standard_normal((t_len, n_assets, 8)).astype(np.float32),
        exposures=rng.standard_normal((t_len, n_assets, 4)).astype(np.float32),
        rmt_loadings=rng.standard_normal((t_len, n_assets, 5)).astype(np.float32),
        target_returns=rng.standard_normal((t_len, n_assets, 3)).astype(np.float32),
    )


def test_batched_asset_microstructure_shape_finite_bounds_and_edge_cases():
    """Devuelve [T, N, 24] float32 finito en [-8, 8] y tolera 1D, T=0 y volumen nulo."""
    rng = np.random.default_rng(11)
    o, h, lo, c, v = _synthetic_ohlcv_panel(rng, t_len=180, n_assets=6)
    # Activo con volumen cero y barra plana + un NaN aislado
    v[:, 0] = 0.0
    o[10, 1] = h[10, 1] = lo[10, 1] = c[10, 1] = 42.0
    c[25, 2] = np.nan

    out = mp4_gpu.batched_asset_microstructure_v4_torch(
        torch.from_numpy(o),
        torch.from_numpy(h),
        torch.from_numpy(lo),
        torch.from_numpy(c),
        torch.from_numpy(v),
    )
    assert out.shape == (180, 6, len(mp4.ASSET_V4_NAMES))
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert float(out.min()) >= -mp4.LIMIT
    assert float(out.max()) <= mp4.LIMIT

    # Soporte 1D (un solo activo): paridad estricta en FP32 e identidad bit-a-bit en FP64
    out_1d = mp4_gpu.batched_asset_microstructure_v4_torch(
        torch.from_numpy(o[:, 3]),
        torch.from_numpy(h[:, 3]),
        torch.from_numpy(lo[:, 3]),
        torch.from_numpy(c[:, 3]),
        torch.from_numpy(v[:, 3]),
    )
    assert out_1d.shape == (180, len(mp4.ASSET_V4_NAMES))
    torch.testing.assert_close(out_1d, out[:, 3, :], atol=1e-5, rtol=1e-5)
    out_f64 = mp4_gpu.batched_asset_microstructure_v4_torch(
        torch.from_numpy(o),
        torch.from_numpy(h),
        torch.from_numpy(lo),
        torch.from_numpy(c),
        torch.from_numpy(v),
        compute_dtype=torch.float64,
    )
    out_1d_f64 = mp4_gpu.batched_asset_microstructure_v4_torch(
        torch.from_numpy(o[:, 3]),
        torch.from_numpy(h[:, 3]),
        torch.from_numpy(lo[:, 3]),
        torch.from_numpy(c[:, 3]),
        torch.from_numpy(v[:, 3]),
        compute_dtype=torch.float64,
    )
    assert torch.equal(out_1d_f64, out_f64[:, 3, :])

    # T = 0 vacío
    empty = torch.empty((0, 4), dtype=torch.float32)
    out_empty = mp4_gpu.batched_asset_microstructure_v4_torch(empty, empty, empty, empty, empty)
    assert out_empty.shape == (0, 4, len(mp4.ASSET_V4_NAMES))
    assert out_empty.dtype == torch.float32


def test_batched_asset_microstructure_matches_numpy_reference_across_assets():
    """Paridad matemática exacta con asset_microstructure_v4 de NumPy en todos los 24 rasgos."""
    rng = np.random.default_rng(42)
    t_len, n_assets = 280, 10
    o, h, lo, c, v = _synthetic_ohlcv_panel(rng, t_len=t_len, n_assets=n_assets)
    # Incluir caso extremo de volumen cero en un activo
    v[:, -1] = 0.0

    batched = mp4_gpu.batched_asset_microstructure_v4_torch(
        torch.from_numpy(o),
        torch.from_numpy(h),
        torch.from_numpy(lo),
        torch.from_numpy(c),
        torch.from_numpy(v),
    )
    ref = np.stack(
        [
            mp4.asset_microstructure_v4(o[:, j], h[:, j], lo[:, j], c[:, j], v[:, j])
            for j in range(n_assets)
        ],
        axis=1,
    )
    assert batched.shape == ref.shape
    # Post-warmup (t >= 63): paridad exacta en los 24 rasgos (sin sesgo espurio de ret[0]=0)
    np.testing.assert_allclose(batched.numpy()[63:], ref[63:], atol=1e-5, rtol=1e-4)
    # En warmup (t < 63): paridad exacta en los 19 rasgos que no dependen de ret_cc[0]/ret_co[0]
    day0_sensitive = {
        mp4.ASSET_V4_NAMES.index(n)
        for n in (
            "vol_yang_zhang_21",
            "vol_yang_zhang_63",
            "price_vol_corr_21",
            "realized_skew_21",
            "realized_kurt_21",
        )
    }
    unaffected = [i for i in range(len(mp4.ASSET_V4_NAMES)) if i not in day0_sensitive]
    np.testing.assert_allclose(
        batched.numpy()[:63, :, unaffected], ref[:63, :, unaffected], atol=1e-5, rtol=1e-4
    )


def test_batched_asset_microstructure_strict_bit_for_bit_causality():
    """Mutar t > t0 nunca altera ni un solo bit en filas <= t0."""
    rng = np.random.default_rng(7)
    o, h, lo, c, v = _synthetic_ohlcv_panel(rng, t_len=300, n_assets=8)
    t0 = 175

    f1 = mp4_gpu.batched_asset_microstructure_v4_torch(
        torch.from_numpy(o),
        torch.from_numpy(h),
        torch.from_numpy(lo),
        torch.from_numpy(c),
        torch.from_numpy(v),
    )
    o2, h2, lo2, c2, v2 = o.copy(), h.copy(), lo.copy(), c.copy(), v.copy()
    c2[t0 + 1 :] *= 3.5
    h2[t0 + 1 :] *= 4.0
    lo2[t0 + 1 :] *= 0.2
    v2[t0 + 1 :] *= 10.0
    o2[t0 + 2] *= 1e4
    f2 = mp4_gpu.batched_asset_microstructure_v4_torch(
        torch.from_numpy(o2),
        torch.from_numpy(h2),
        torch.from_numpy(lo2),
        torch.from_numpy(c2),
        torch.from_numpy(v2),
    )
    assert torch.equal(f1[: t0 + 1], f2[: t0 + 1])
    assert not torch.equal(f1[t0 + 1 :], f2[t0 + 1 :])


def test_batched_rmt_marchenko_pastur_recovers_signal_modes_and_matches_numpy():
    """RMT batched en PyTorch recupera >= 3 modos reales y empata con rmt_clean_correlation."""
    rng = np.random.default_rng(13)
    t_len, n_assets = 120, 36
    factors = rng.standard_normal((t_len, 3))
    loadings = rng.standard_normal((n_assets, 3))
    returns = factors @ loadings.T + 0.45 * rng.standard_normal((t_len, n_assets))

    res_gpu = mp4_gpu.batched_rmt_clean_correlation_torch(
        torch.from_numpy(returns), top_k=5, return_clean_corr=True
    )
    res_np = mp4.rmt_clean_correlation(returns, top_k=5)

    assert int(res_gpu["n_signal_modes"].item()) == res_np["n_signal_modes"]
    assert int(res_gpu["n_signal_modes"].item()) >= 3
    np.testing.assert_allclose(
        float(res_gpu["absorption_ratio"].item()), res_np["absorption_ratio"], atol=1e-5
    )
    np.testing.assert_allclose(
        float(res_gpu["spectral_entropy"].item()), res_np["spectral_entropy"], atol=1e-5
    )
    clean_corr = res_gpu["clean_corr"].numpy()
    np.testing.assert_allclose(clean_corr, clean_corr.T, atol=1e-5)
    np.testing.assert_allclose(clean_corr, res_np["clean_corr"], atol=1e-4)
    assert float(np.linalg.eigvalsh(clean_corr).min()) > 0.0
    np.testing.assert_allclose(res_gpu["loadings"].numpy(), res_np["loadings"], atol=1e-4)


def test_batched_panel_physics_parity_and_strict_causality():
    """Paridad numérica con panel_market_physics_v4 y causalidad bit-a-bit estricta."""
    rng = np.random.default_rng(31)
    t_len, n_assets = 96, 28
    returns = 0.015 * rng.standard_normal((t_len, n_assets))
    liq = rng.normal(15.0, 1.5, size=(t_len, n_assets))
    vol = np.abs(0.015 + 0.005 * rng.standard_normal((t_len, n_assets)))
    macro = rng.standard_normal((t_len, 4))

    phys_t = mp4_gpu.batched_panel_physics_v4_torch(
        torch.from_numpy(returns),
        torch.from_numpy(liq),
        torch.from_numpy(vol),
        macro=torch.from_numpy(macro),
        rmt_window=40,
        top_k_rmt=5,
        maturity_lag=1,
    )
    phys_np = mp4.panel_market_physics_v4(
        returns, liq, vol, macro=macro, rmt_window=40, top_k_rmt=5, maturity_lag=1
    )

    assert phys_t["rmt_absorption"].shape == (t_len,)
    assert phys_t["rmt_entropy"].shape == (t_len,)
    assert phys_t["rmt_n_signal_modes"].shape == (t_len,)
    assert phys_t["rmt_loadings"].shape == (t_len, n_assets, 5)
    assert phys_t["wasserstein"].shape == (t_len,)
    assert phys_t["cross_dispersion"].shape == (t_len,)
    assert phys_t["lead_lag"].shape == (t_len, n_assets)
    assert phys_t["macro_surprises"].shape == (t_len, 4)
    assert phys_t["macro_betas"].shape == (t_len, n_assets, 4)
    assert phys_t["high_snr_context"].shape == (t_len, n_assets, 8)
    assert phys_t["high_snr_context"].dtype == torch.float32
    for key, val in phys_t.items():
        assert torch.isfinite(val).all(), f"Valor no finito en {key}"

    np.testing.assert_allclose(
        phys_t["rmt_absorption"].numpy(), phys_np["rmt_absorption"], atol=1e-5
    )
    np.testing.assert_allclose(phys_t["rmt_entropy"].numpy(), phys_np["rmt_entropy"], atol=1e-5)
    np.testing.assert_allclose(phys_t["wasserstein"].numpy(), phys_np["wasserstein"], atol=1e-5)
    np.testing.assert_allclose(
        phys_t["cross_dispersion"].numpy(), phys_np["cross_dispersion"], atol=1e-5
    )
    np.testing.assert_allclose(phys_t["lead_lag"].numpy(), phys_np["lead_lag"], atol=1e-5)
    np.testing.assert_allclose(
        phys_t["macro_surprises"].numpy(), phys_np["macro_surprises"], atol=1e-5
    )
    np.testing.assert_allclose(phys_t["macro_betas"].numpy(), phys_np["macro_betas"], atol=1e-4)
    np.testing.assert_allclose(
        phys_t["high_snr_context"].numpy(), phys_np["high_snr_context"], atol=1e-4
    )

    # Causalidad estricta en t = 70 (maturity_lag=1 -> mutar returns[70:] no altera t <= 70)
    ret_mut = returns.copy()
    ret_mut[70:] += 0.75
    phys_ret_mut = mp4_gpu.batched_panel_physics_v4_torch(
        torch.from_numpy(ret_mut),
        torch.from_numpy(liq),
        torch.from_numpy(vol),
        macro=torch.from_numpy(macro),
        rmt_window=40,
        top_k_rmt=5,
        maturity_lag=1,
    )
    for key in (
        "rmt_absorption",
        "rmt_entropy",
        "rmt_n_signal_modes",
        "rmt_loadings",
        "wasserstein",
        "cross_dispersion",
        "lead_lag",
        "macro_betas",
        "high_snr_context",
    ):
        assert torch.equal(phys_t[key][:71], phys_ret_mut[key][:71]), f"Fuga causal en {key}"

    # Mutar macro[70:] no altera t <= 69
    mac_mut = macro.copy()
    mac_mut[70:] = -25.0
    phys_mac_mut = mp4_gpu.batched_panel_physics_v4_torch(
        torch.from_numpy(returns),
        torch.from_numpy(liq),
        torch.from_numpy(vol),
        macro=torch.from_numpy(mac_mut),
        rmt_window=40,
        top_k_rmt=5,
        maturity_lag=1,
    )
    assert torch.equal(phys_t["macro_surprises"][:70], phys_mac_mut["macro_surprises"][:70])
    assert torch.equal(phys_t["high_snr_context"][:70], phys_mac_mut["high_snr_context"][:70])

    # Sin macro (macro=None)
    phys_nomac = mp4_gpu.batched_panel_physics_v4_torch(
        torch.from_numpy(returns),
        torch.from_numpy(liq),
        torch.from_numpy(vol),
        macro=None,
        rmt_window=40,
        top_k_rmt=5,
        maturity_lag=1,
    )
    assert phys_nomac["macro_surprises"].shape == (t_len, 0)
    assert phys_nomac["macro_betas"].shape == (t_len, n_assets, 0)
    assert torch.isfinite(phys_nomac["high_snr_context"]).all()


def test_batched_panel_physics_rmt_modes_wasserstein_shock_and_lead_lag_recovery():
    """Verifica detección física real: modos RMT, pico Wasserstein y señal Lead-Lag."""
    # 1) Recuperación de 3 modos de señal RMT en panel
    rng_rmt = np.random.default_rng(17)
    t_rmt, n_rmt = 90, 32
    f3 = rng_rmt.standard_normal((t_rmt, 3))
    l3 = rng_rmt.standard_normal((n_rmt, 3))
    r_rmt = 0.01 * (f3 @ l3.T + 0.35 * rng_rmt.standard_normal((t_rmt, n_rmt)))
    liq_rmt = rng_rmt.normal(14.0, 1.0, size=(t_rmt, n_rmt))
    vol_rmt = np.full((t_rmt, n_rmt), 0.015)
    phys_rmt = mp4_gpu.batched_panel_physics_v4_torch(
        torch.from_numpy(r_rmt),
        torch.from_numpy(liq_rmt),
        torch.from_numpy(vol_rmt),
        rmt_window=60,
        top_k_rmt=5,
        maturity_lag=1,
    )
    assert int(phys_rmt["rmt_n_signal_modes"][-1].item()) >= 3
    assert float(phys_rmt["rmt_absorption"][-1].item()) > 0.50

    # 2) Pico de distancia de Wasserstein ante quiebre de régimen
    rng_w = np.random.default_rng(21)
    t_w, n_w = 120, 44
    panel_w = 0.01 * rng_w.standard_normal((t_w, n_w))
    panel_w[90:] = -0.02 + 0.05 * rng_w.standard_normal((30, n_w))
    phys_w = mp4_gpu.batched_panel_physics_v4_torch(
        torch.from_numpy(panel_w),
        torch.ones((t_w, n_w)),
        torch.full((t_w, n_w), 0.02),
        rmt_window=40,
        top_k_rmt=4,
        maturity_lag=1,
    )
    wass = phys_w["wasserstein"].numpy()
    calm_mean = float(np.mean(wass[50:89]))
    shock_mean = float(np.mean(wass[93:105]))
    assert shock_mean > calm_mean * 3.0

    # 3) Recuperación causal de Lead-Lag (líderes líquidos t-1 -> seguidores t)
    rng_ll = np.random.default_rng(99)
    t_ll, n_ll = 140, 30
    liq_vec = np.linspace(10.0, 20.0, n_ll)
    liq_ll = np.broadcast_to(liq_vec[None, :], (t_ll, n_ll)).copy()
    r_ll = 0.01 * rng_ll.standard_normal((t_ll, n_ll))
    leader_factor = 0.015 * rng_ll.standard_normal(t_ll)
    r_ll[:, 24:] += leader_factor[:, None]
    leader_mean = r_ll[:, 24:].mean(axis=1)
    for t in range(1, t_ll):
        r_ll[t, :24] += 0.8 * leader_mean[t - 1]
    phys_ll = mp4_gpu.batched_panel_physics_v4_torch(
        torch.from_numpy(r_ll),
        torch.from_numpy(liq_ll),
        torch.full((t_ll, n_ll), 0.015),
        rmt_window=40,
        top_k_rmt=4,
        maturity_lag=1,
    )
    spill = phys_ll["lead_lag"].numpy()
    corr = float(np.corrcoef(spill[60:, :24].ravel(), r_ll[60:, :24].ravel())[0, 1])
    assert corr > 0.50


def test_batched_multimodal_fusion_v4_parity_and_causality():
    """Fusión multimodal vectorizada O(log T) empata con NumPy y es bit-a-bit causal."""
    rng = np.random.default_rng(123)
    t_len, n_assets = 48, 14
    news_emb = rng.standard_normal((t_len, n_assets, 8)).astype(np.float32)
    news_present = rng.random((t_len, n_assets)) > 0.55
    chart_emb = rng.standard_normal((t_len, n_assets, 8)).astype(np.float32)
    chart_present = rng.random((t_len, n_assets)) > 0.25
    fund_vals = rng.standard_normal((t_len, n_assets, 6)).astype(np.float32)
    fund_present = rng.random((t_len, n_assets)) > 0.4
    macro_surp = rng.standard_normal((t_len, 4)).astype(np.float64)
    macro_betas = rng.standard_normal((t_len, n_assets, 4)).astype(np.float64)

    mm_gpu = mp4_gpu.batched_multimodal_fusion_v4_torch(
        news_emb=torch.from_numpy(news_emb),
        news_present=torch.from_numpy(news_present),
        chart_emb=torch.from_numpy(chart_emb),
        chart_present=torch.from_numpy(chart_present),
        fund_vals=torch.from_numpy(fund_vals),
        fund_present=torch.from_numpy(fund_present),
        macro_surprises=torch.from_numpy(macro_surp),
        macro_betas=torch.from_numpy(macro_betas),
    )
    mm_np = mp4.multimodal_fusion_features_v4(
        news_emb=news_emb,
        news_present=news_present,
        chart_emb=chart_emb,
        chart_present=chart_present,
        fund_vals=fund_vals,
        fund_present=fund_present,
        macro_surprises=macro_surp,
        macro_betas=macro_betas,
    )
    assert mm_gpu["features"].shape == mm_np["features"].shape
    assert mm_gpu["modality_presence"].shape == mm_np["modality_presence"].shape
    np.testing.assert_allclose(mm_gpu["features"].numpy(), mm_np["features"], atol=1e-5)
    np.testing.assert_allclose(
        mm_gpu["modality_presence"].numpy(), mm_np["modality_presence"], atol=1e-6
    )

    news2 = news_emb.copy()
    news2[30:] = -50.0
    macro2 = macro_surp.copy()
    macro2[30:] = 25.0
    mm_gpu2 = mp4_gpu.batched_multimodal_fusion_v4_torch(
        news_emb=torch.from_numpy(news2),
        news_present=torch.from_numpy(news_present),
        chart_emb=torch.from_numpy(chart_emb),
        chart_present=torch.from_numpy(chart_present),
        fund_vals=torch.from_numpy(fund_vals),
        fund_present=torch.from_numpy(fund_present),
        macro_surprises=torch.from_numpy(macro2),
        macro_betas=torch.from_numpy(macro_betas),
    )
    assert torch.equal(mm_gpu["features"][:30], mm_gpu2["features"][:30])


def test_zero_copy_session_store_mmap_roundtrip_and_zero_copy_tensor_slices(tmp_path: Path):
    """Guarda/carga vía mmap_mode='r', empata exacto y crea tensores zero-copy sin avisos."""
    rng = np.random.default_rng(77)
    t_len, n_assets = 26, 8
    arrays = _make_multimodal_arrays(rng, t_len=t_len, n_assets=n_assets)
    years = np.arange(2000, 2000 + t_len, dtype=np.int64)

    store_dir = tmp_path / "packed_24y"
    mp4_gpu.ZeroCopySessionStore.save(store_dir, session_years=years, arrays=arrays)
    store = mp4_gpu.ZeroCopySessionStore.load(store_dir, mmap_mode="r")

    assert len(store) == t_len
    assert set(store.keys()) == set(mp4_gpu.STORE_ARRAY_KEYS)
    np.testing.assert_array_equal(store.session_years, years)

    # Todos los arrays respaldados por np.memmap solo lectura (page-cache compartido entre workers)
    for key in mp4_gpu.STORE_ARRAY_KEYS:
        raw = store.raw_array(key)
        assert isinstance(raw, np.memmap), f"{key} debe cargarse vía mmap_mode='r'"
        assert not raw.flags.writeable
        np.testing.assert_array_equal(raw, arrays[key])

    # Indexación O(1) y slice de rango zero-copy (mismo puntero en memoria, sin UserWarning)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batch_t = store[5]
        slice_dict = store.get_slice(3, 9)
    assert len(caught) == 0

    for key in mp4_gpu.STORE_ARRAY_KEYS:
        raw = store.raw_array(key)
        assert batch_t[key].data_ptr() == raw[5].ctypes.data
        np.testing.assert_array_equal(batch_t[key].numpy(), arrays[key][5])
        assert slice_dict[key].data_ptr() == raw[3:9].ctypes.data
        np.testing.assert_array_equal(slice_dict[key].numpy(), arrays[key][3:9])

    # Índice negativo
    last_batch = store[-1]
    np.testing.assert_array_equal(last_batch["prices"].numpy(), arrays["prices"][-1])

    report = store.memory_footprint_report(workers=6)
    assert report["is_mmap_backed"] is True
    assert report["shared_ram_duplication_factor"] == 1
    assert report["total_bytes"] > 0


def test_zero_copy_session_store_chronological_splits_and_2024_seal(tmp_path: Path):
    """Partición 2000-2022 train, 2023 val y sello estricto 2024+ con purga y embargo."""
    rng = np.random.default_rng(101)
    # 2 sesiones por año de 2000 a 2025 -> 52 sesiones totales
    years = np.repeat(np.arange(2000, 2026, dtype=np.int64), 2)
    arrays = _make_multimodal_arrays(rng, t_len=len(years), n_assets=6)

    store_dir = tmp_path / "chronological_store"
    mp4_gpu.ZeroCopySessionStore.save(store_dir, session_years=years, arrays=arrays)
    store = mp4_gpu.ZeroCopySessionStore.load(store_dir, mmap_mode="r")

    # 1) Por defecto: 2000-2022 train (46), 2023 val (2), 2024+ sellado (4 retenidas)
    splits = store.chronological_splits(
        train_start_year=2000,
        train_end_year=2022,
        val_year=2023,
        sealed_test_start_year=2024,
        active_horizons=(1,),
        purge_sessions=2,
        embargo_sessions=1,
        allow_zero_purge=True,
    )
    assert len(splits["train"]) == 44  # 46 - 2 purgadas
    assert len(splits["val"]) == 1  # 2 - 1 embargada
    assert len(splits["sealed_test"]) == 0
    assert splits["n_sealed_withheld"] == 4
    assert splits["sealed_test_accessible"] is False
    assert int(years[splits["train_indices"]].max()) <= 2022
    assert set(years[splits["val_indices"]].tolist()) == {2023}

    # Las vistas de split siguen siendo zero-copy respecto al memmap subyacente
    first_val = splits["val"][0]
    raw_idx = int(splits["val_indices"][0])
    assert first_val["prices"].data_ptr() == store.raw_array("prices")[raw_idx].ctypes.data

    # 2) slice_years rehúsa acceder a >= 2024 sin autorización explícita
    with pytest.raises(ValueError, match="2024"):
        store.slice_years(2020, 2024)

    train_view = store.slice_years(2000, 2022)
    val_view = store.slice_years(2023, 2023)
    assert len(train_view) == 46
    assert len(val_view) == 2

    # Solo desbloquea 2024+ con allow_sealed_test=True
    unlocked = store.chronological_splits(
        allow_sealed_test=True, purge_sessions=0, embargo_sessions=0, allow_zero_purge=True
    )
    assert len(unlocked["sealed_test"]) == 4
    assert unlocked["n_sealed_withheld"] == 0
    assert unlocked["sealed_test_accessible"] is True

    # 3) Rechazar años no cronológicos al guardar
    bad_years = years.copy()
    bad_years[5] = 1999
    with pytest.raises(ValueError, match="cronológ"):
        mp4_gpu.ZeroCopySessionStore.save(
            tmp_path / "bad_order", session_years=bad_years, arrays=arrays
        )

    # 4) Slice por índice devuelve vista zero-copy
    sub_view = store[2:5]
    assert len(sub_view) == 3
    assert sub_view[0]["prices"].data_ptr() == store.raw_array("prices")[2].ctypes.data


def test_zero_copy_session_store_ragged_from_sessions_and_h2d_prefetch(tmp_path: Path):
    """Empaquetado ragged (N_t variable por sesión) y flujo iter_pinned_h2d en CPU."""
    rng = np.random.default_rng(55)
    counts = [10, 14, 8, 12]
    years = [2020, 2021, 2022, 2023]
    sessions = []
    for n_i in counts:
        sessions.append(
            dict(
                asset_ids=torch.arange(100, 100 + n_i, dtype=torch.int64),
                prices=torch.from_numpy(rng.standard_normal((n_i, 12)).astype(np.float32)),
                fundamentals=torch.from_numpy(rng.standard_normal((n_i, 4)).astype(np.float32)),
                macro=torch.from_numpy(rng.standard_normal((n_i, 3)).astype(np.float32)),
                news=torch.from_numpy(rng.standard_normal((n_i, 6)).astype(np.float32)),
                charts=torch.from_numpy(rng.standard_normal((n_i, 6)).astype(np.float32)),
                modality_mask=torch.ones((n_i, 5), dtype=torch.float32),
                physics_v4=torch.from_numpy(rng.standard_normal((n_i, 24)).astype(np.float32)),
                high_snr_context=torch.from_numpy(rng.standard_normal((n_i, 8)).astype(np.float32)),
                exposures=torch.from_numpy(rng.standard_normal((n_i, 4)).astype(np.float32)),
                rmt_loadings=torch.from_numpy(rng.standard_normal((n_i, 5)).astype(np.float32)),
                target_returns=torch.from_numpy(rng.standard_normal((n_i, 3)).astype(np.float32)),
            )
        )

    store = mp4_gpu.ZeroCopySessionStore.from_sessions(tmp_path / "ragged_store", sessions, years)
    assert store.layout == "ragged"
    assert len(store) == 4
    for idx, orig in enumerate(sessions):
        loaded = store[idx]
        for key in mp4_gpu.STORE_ARRAY_KEYS:
            assert loaded[key].shape == orig[key].shape
            torch.testing.assert_close(loaded[key], orig[key], atol=0.0, rtol=0.0)
        # Verificar zero-copy real dentro del memmap contiguo
        off = int(store.session_offsets[idx])
        assert loaded["prices"].data_ptr() == store.raw_array("prices")[off].ctypes.data
        assert (
            loaded["asset_ids"].data_ptr()
            == store.raw_array("asset_ids")[off : off + 1].ctypes.data
        )

    # iter_pinned_h2d sobre CPU devuelve exactamente las sesiones solicitadas
    splits = store.chronological_splits(
        train_start_year=2020,
        train_end_year=2022,
        val_year=2023,
        purge_sessions=0,
        embargo_sessions=0,
        allow_zero_purge=True,
    )
    yielded = list(splits["train"].iter_pinned_h2d(device=torch.device("cpu")))
    assert len(yielded) == 3
    for i, batch in enumerate(yielded):
        torch.testing.assert_close(batch["prices"], sessions[i]["prices"], atol=0.0, rtol=0.0)


def test_batched_panel_physics_handles_nans_maturity_lag_2_and_rejects_lookahead():
    """Tolera NaNs en retornos/liquidez/vol/macro, respeta maturity_lag=2 y bloquea lag < 0."""
    rng = np.random.default_rng(2026)
    t_len, n_assets = 85, 24
    returns = 0.015 * rng.standard_normal((t_len, n_assets))
    liq = rng.normal(14.0, 1.2, size=(t_len, n_assets))
    vol = np.abs(0.02 + 0.005 * rng.standard_normal((t_len, n_assets)))
    macro = rng.standard_normal((t_len, 3))

    # Inyectar NaNs dispersos en todas las entradas
    returns[rng.random(returns.shape) < 0.06] = np.nan
    liq[rng.random(liq.shape) < 0.05] = np.nan
    vol[rng.random(vol.shape) < 0.05] = np.nan
    macro[rng.random(macro.shape) < 0.08] = np.nan

    phys_t = mp4_gpu.batched_panel_physics_v4_torch(
        torch.from_numpy(returns),
        torch.from_numpy(liq),
        torch.from_numpy(vol),
        macro=torch.from_numpy(macro),
        rmt_window=36,
        top_k_rmt=4,
        maturity_lag=2,
    )
    phys_np = mp4.panel_market_physics_v4(
        returns, liq, vol, macro=macro, rmt_window=36, top_k_rmt=4, maturity_lag=2
    )
    for key, val in phys_t.items():
        assert torch.isfinite(val).all(), f"No finito con NaNs en {key}"
    np.testing.assert_allclose(phys_t["wasserstein"].numpy(), phys_np["wasserstein"], atol=1e-5)
    np.testing.assert_allclose(phys_t["lead_lag"].numpy(), phys_np["lead_lag"], atol=1e-5)
    np.testing.assert_allclose(
        phys_t["macro_surprises"].numpy(), phys_np["macro_surprises"], atol=1e-5
    )
    np.testing.assert_allclose(
        phys_t["high_snr_context"].numpy(), phys_np["high_snr_context"], atol=1e-4
    )

    with pytest.raises(ValueError, match="maturity_lag"):
        mp4_gpu.batched_panel_physics_v4_torch(
            torch.from_numpy(returns),
            torch.from_numpy(liq),
            torch.from_numpy(vol),
            maturity_lag=-1,
        )


def test_gpu_physics_hot_paths_have_zero_host_sync_traps():
    """Audita vía AST que ninguna ruta GPU física contenga `.item()`, `.cpu()` ni `nonzero`."""
    import ast
    import inspect

    source = inspect.getsource(mp4_gpu)
    tree = ast.parse(source)
    hot_fns = {
        "batched_asset_microstructure_v4_torch",
        "batched_rmt_clean_correlation_torch",
        "batched_causal_wasserstein_shift_torch",
        "batched_causal_lead_lag_spillover_torch",
        "batched_macro_surprises_and_betas_torch",
        "batched_panel_physics_v4_torch",
        "batched_multimodal_fusion_v4_torch",
        "gaussian_rank_features",
    }
    forbidden_attrs = {"item", "cpu", "nonzero"}
    found_fns: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in hot_fns:
            found_fns.add(node.name)
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute) and child.attr in forbidden_attrs:
                    raise AssertionError(
                        f"Trampa de sincronización host `{child.attr}` detectada en {node.name}"
                    )
    assert found_fns == hot_fns


def test_zero_copy_session_store_cuda_double_buffer_dma_sync_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verifica sincronización por evento de doble búfer DMA y record_stream en ruta CUDA."""
    rng = np.random.default_rng(314)
    t_len, n_assets = 4, 5
    arrays = dict(
        prices=rng.standard_normal((t_len, n_assets, 4)).astype(np.float32),
        target_returns=rng.standard_normal((t_len, n_assets, 1)).astype(np.float32),
    )
    years = np.array([2020, 2021, 2022, 2023], dtype=np.int64)
    mp4_gpu.ZeroCopySessionStore.save(tmp_path / "cuda_proto", session_years=years, arrays=arrays)
    store = mp4_gpu.ZeroCopySessionStore.load(tmp_path / "cuda_proto", mmap_mode="r")

    trace: list[tuple[object, ...]] = []
    recorded_streams: list[object] = []
    # Rastrea si cada búfer fijado tiene una transferencia DMA aún en vuelo
    in_flight_bufs: set[int] = set()
    event_to_bufs: dict[int, list[int]] = {}
    pending_bufs_for_event: list[int] = []

    class _FakeStream:
        def __init__(self, device=None, name: str = "prefetch"):
            self.device = device
            self.name = name

        def wait_stream(self, other):
            trace.append(("wait_stream", self.name, other.name))

    class _FakeStreamCtx:
        def __init__(self, s):
            self._s = s

        def __enter__(self):
            trace.append(("enter_stream", self._s.name))
            return self._s

        def __exit__(self, exc_type, exc, tb):
            trace.append(("exit_stream", self._s.name))
            return False

    class _FakeEvent:
        def __init__(self):
            self._id = id(self)
            self.recorded = False

        def record(self, stream=None):
            self.recorded = True
            event_to_bufs[self._id] = list(pending_bufs_for_event)
            pending_bufs_for_event.clear()
            trace.append(("event_record", self._id, getattr(stream, "name", None)))

        def synchronize(self):
            assert self.recorded, "synchronize() llamado sobre evento no registrado"
            for bid in event_to_bufs.get(self._id, ()):
                in_flight_bufs.discard(bid)
            trace.append(("event_sync", self._id))

    class _FakeGpuTensor:
        def __init__(self, data: torch.Tensor):
            self.data = data

        def record_stream(self, stream):
            recorded_streams.append(stream.name)
            trace.append(("record_stream", stream.name))

    class _FakePinnedBuf:
        def __init__(self, shape, dtype):
            self._id = id(self)
            self.shape = tuple(shape)
            self.dtype = dtype
            self._data = torch.empty(shape, dtype=dtype)

        def copy_(self, src, non_blocking=False):
            if self._id in in_flight_bufs:
                raise RuntimeError("DMA race: sobrescritura de staging buffer aún en vuelo")
            self._data.copy_(src)
            trace.append(("copy_", self._id))
            return self

        def to(self, device, non_blocking=False):
            assert non_blocking is True
            in_flight_bufs.add(self._id)
            pending_bufs_for_event.append(self._id)
            trace.append(("h2d_queue", self._id, str(device)))
            return _FakeGpuTensor(self._data.clone())

    cur_stream = _FakeStream(name="default")
    orig_empty = torch.empty

    def _fake_empty(*args, **kwargs):
        if kwargs.get("pin_memory"):
            shape = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args
            return _FakePinnedBuf(shape, kwargs.get("dtype", torch.float32))
        return orig_empty(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: _FakeStream(device=device))
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: _FakeStreamCtx(s))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: cur_stream)
    monkeypatch.setattr(torch, "empty", _fake_empty)

    yielded = list(store.iter_pinned_h2d(device="cuda:0", use_pinned=True, slab_sessions=1))
    assert len(yielded) == t_len
    for i, batch in enumerate(yielded):
        np.testing.assert_array_equal(batch["prices"].data.numpy(), arrays["prices"][i])
        np.testing.assert_array_equal(
            batch["target_returns"].data.numpy(), arrays["target_returns"][i]
        )

    # 4 sesiones × len(store.keys()) (prices, target_returns, asset_ids) => 12 record_stream
    assert recorded_streams == ["default"] * (t_len * len(store.keys()))
    # 4 eventos registrados (uno por sesión) y al menos 2 sincronizaciones al reusar slots 0 y 1
    assert sum(1 for item in trace if item[0] == "event_record") == t_len
    assert sum(1 for item in trace if item[0] == "event_sync") == t_len - 2


def test_batched_microstructure_exact_moments_halt_masking_and_split_detector():
    """Paridad exacta de pv_corr_21/skew_21/kurt_21 en GPU, máscara volume<=0 y detector splits."""
    from scipy import stats

    rng = np.random.default_rng(808)
    o, h, lo, c, v = _synthetic_ohlcv_panel(rng, t_len=120, n_assets=6)
    v[40, 0] = 0.0
    c[40, 0] = c[39, 0] * 1.05
    h[40, 0] = max(o[40, 0], c[40, 0])
    lo[40, 0] = min(o[40, 0], c[40, 0])
    v[:, 1] = 0.0  # activo totalmente suspendido

    out = mp4_gpu.batched_asset_microstructure_v4_torch(
        torch.from_numpy(o),
        torch.from_numpy(h),
        torch.from_numpy(lo),
        torch.from_numpy(c),
        torch.from_numpy(v),
    )
    idx = {name: i for i, name in enumerate(mp4.ASSET_V4_NAMES)}
    # pv_corr_21 estrictamente acotado en [-1, 1]
    pv = out[:, :, idx["price_vol_corr_21"]]
    assert float(pv.min()) >= -1.0 - 1e-6
    assert float(pv.max()) <= 1.0 + 1e-6

    # Verificar exactitud en t=95 para activo 2 (sin halts) frente a ventana directa [75..95]
    t_chk = 95
    log_c = np.log(c[:, 2])
    ret_cc = np.zeros_like(log_c)
    ret_cc[1:] = log_c[1:] - log_c[:-1]
    log_v = np.log1p(v[:, 2])
    r_win = ret_cc[t_chk - 20 : t_chk + 1]
    v_win = log_v[t_chk - 20 : t_chk + 1]
    assert float(out[t_chk, 2, idx["price_vol_corr_21"]]) == pytest.approx(
        float(np.corrcoef(r_win, v_win)[0, 1]), abs=1e-5
    )
    assert float(out[t_chk, 2, idx["realized_skew_21"]]) == pytest.approx(
        float(stats.skew(r_win, bias=True)), abs=1e-5
    )
    assert float(out[t_chk, 2, idx["realized_kurt_21"]]) == pytest.approx(
        float(stats.kurtosis(r_win, fisher=True, bias=True)), abs=1e-5
    )

    # Activo con halt aislado en t=40 no pega Amihud a 8.0; activo suspendido queda en 0.0
    assert float(out[40, 0, idx["amihud_up_21"]]) < 4.0
    assert float(out[:, 1, idx["amihud_up_21"]].abs().max()) == 0.0
    assert float(out[:, 1, idx["amihud_down_21"]].abs().max()) == 0.0

    # Paridad total con NumPy post-warmup (t >= 63) y en rasgos sin sesgo Day-0
    ref = np.stack(
        [
            mp4.asset_microstructure_v4(o[:, j], h[:, j], lo[:, j], c[:, j], v[:, j])
            for j in range(6)
        ],
        axis=1,
    )
    np.testing.assert_allclose(out.numpy()[63:], ref[63:], atol=1e-5, rtol=1e-4)

    # Detector de splits sin ajustar
    o_sp, c_sp = o.copy(), c.copy()
    o_sp[60:, 3] *= 0.5
    c_sp[60:, 3] *= 0.5
    assert mp4_gpu.detect_unadjusted_split_jumps(torch.from_numpy(o), torch.from_numpy(c))[
        "n_jumps"
    ] == 0
    assert (
        mp4_gpu.detect_unadjusted_split_jumps(torch.from_numpy(o_sp), torch.from_numpy(c_sp))[
            "n_jumps"
        ]
        >= 1
    )


def test_batched_rmt_pairwise_overlap_and_lead_lag_maturity_lag_strict():
    """RMT batched usa solape cnt_ij-1 y Lead-Lag usa leader_ret[t - maturity_lag]."""
    rng = np.random.default_rng(515)
    t_len, n_assets = 63, 12
    r = 0.015 * rng.standard_normal((t_len, n_assets))
    base = 0.02 * rng.standard_normal(30)
    r[:33, 0] = np.nan
    r[:33, 1] = np.nan
    r[33:, 0] = base
    r[33:, 1] = base + 1e-4 * rng.standard_normal(30)
    r[:-2, 2] = np.nan
    r[2:, 3] = np.nan

    res_t = mp4_gpu.batched_rmt_clean_correlation_torch(
        torch.from_numpy(r), top_k=3, return_clean_corr=True, min_overlap=5
    )
    res_np = mp4.rmt_clean_correlation(r, top_k=3, min_overlap=5)
    assert float(res_t["raw_corr"][0, 1]) > 0.99
    assert float(res_t["raw_corr"][2, 3]) == 0.0
    np.testing.assert_allclose(res_t["clean_corr"].numpy(), res_np["clean_corr"], atol=1e-4)

    # Lead-Lag con maturity_lag=2: mutar t >= 70 no altera ni un bit en t <= 71 ([:72])
    r2 = 0.015 * rng.standard_normal((90, 20))
    liq2 = rng.normal(14.0, 1.0, size=(90, 20))
    s1 = mp4_gpu.batched_causal_lead_lag_spillover_torch(
        torch.from_numpy(r2), torch.from_numpy(liq2), window=30, maturity_lag=2
    )
    r2_m, liq2_m = r2.copy(), liq2.copy()
    r2_m[70:] *= -40.0
    liq2_m[70:] += 100.0
    s2 = mp4_gpu.batched_causal_lead_lag_spillover_torch(
        torch.from_numpy(r2_m), torch.from_numpy(liq2_m), window=30, maturity_lag=2
    )
    assert torch.equal(s1[:72], s2[:72])
    np.testing.assert_allclose(
        s1.numpy(),
        mp4.causal_lead_lag_spillover(r2, liq2, window=30, maturity_lag=2),
        atol=1e-5,
    )
    with pytest.raises(ValueError, match="maturity_lag"):
        mp4_gpu.batched_panel_physics_v4_torch(
            torch.from_numpy(r2), torch.from_numpy(liq2), torch.ones((90, 20)), maturity_lag=0
        )


def test_zero_copy_store_asset_ids_safe_purge_embargo_and_gaussian_ranks(tmp_path: Path):
    """Store persiste asset_ids int64, exige purge>=max_h y embargo>=max(5,ceil(0.2*h)), y ranks."""
    from mars_titan.models.titans.session_ranks import gaussian_ranks

    rng = np.random.default_rng(909)
    # 100 sesiones de 2022 + 30 sesiones de 2023 + 10 sesiones de 2024 (sellado)
    years = np.concatenate(
        [
            np.full(100, 2022, dtype=np.int64),
            np.full(30, 2023, dtype=np.int64),
            np.full(10, 2024, dtype=np.int64),
        ]
    )
    n_assets = 8
    arrays = _make_multimodal_arrays(rng, t_len=len(years), n_assets=n_assets)
    asset_ids = np.broadcast_to(
        np.array([101, 105, 203, 307, 409, 512, 618, 725], dtype=np.int64),
        (len(years), n_assets),
    ).copy()
    arrays["asset_ids"] = asset_ids

    store_dir = tmp_path / "p0_upgraded_store"
    mp4_gpu.ZeroCopySessionStore.save(store_dir, session_years=years, arrays=arrays)
    store = mp4_gpu.ZeroCopySessionStore.load(store_dir, mmap_mode="r")

    # asset_ids es memmap int64 zero-copy en cada sesión
    assert isinstance(store.raw_array("asset_ids"), np.memmap)
    b0 = store[0]
    assert b0["asset_ids"].dtype == torch.int64
    assert b0["asset_ids"].data_ptr() == store.raw_array("asset_ids")[0].ctypes.data
    np.testing.assert_array_equal(b0["asset_ids"].numpy(), asset_ids[0])

    # Rechazar asset_ids duplicados dentro de una misma sesión
    bad_ids = asset_ids.copy()
    bad_ids[3, 1] = bad_ids[3, 0]
    bad_arrays = dict(arrays, asset_ids=bad_ids)
    with pytest.raises(ValueError, match="duplicad"):
        mp4_gpu.ZeroCopySessionStore.save(
            tmp_path / "dup_ids", session_years=years, arrays=bad_arrays
        )

    # Por defecto (active_horizons=(1,5,10,21,42,63)): purge=63, embargo=13!
    safe_splits = store.chronological_splits(
        train_start_year=2022, train_end_year=2022, val_year=2023
    )
    assert safe_splits["purge_sessions"] == 63
    assert safe_splits["embargo_sessions"] == 13
    assert len(safe_splits["train"]) == 100 - 63
    assert len(safe_splits["val"]) == 30 - 13
    assert len(safe_splits["sealed_test"]) == 0
    assert safe_splits["n_sealed_withheld"] == 10

    # Rechazar purge_sessions=0 o purge_sessions=5 sin allow_zero_purge=True
    with pytest.raises(ValueError, match="purge_sessions"):
        store.chronological_splits(
            train_start_year=2022, train_end_year=2022, val_year=2023, purge_sessions=5
        )
    with pytest.raises(ValueError, match="embargo_sessions"):
        store.chronological_splits(
            train_start_year=2022, train_end_year=2022, val_year=2023, embargo_sessions=2
        )

    # Cross-sectional Gaussian ranks: paridad exacta con session_ranks.gaussian_ranks
    x_2d = torch.from_numpy(arrays["physics_v4"][5].copy())
    x_2d[0, 0] = x_2d[1, 0]  # empate
    x_2d[2, 1] = float("nan")
    g_ours = mp4_gpu.gaussian_rank_features(x_2d)
    present = torch.isfinite(x_2d)
    g_ref = gaussian_ranks(torch.where(present, x_2d, torch.zeros_like(x_2d)), present)
    torch.testing.assert_close(g_ours, g_ref.clamp(-mp4.LIMIT, mp4.LIMIT), atol=1e-5, rtol=1e-5)

    # Integración en ZeroCopySessionStore.get_session(..., gaussian_rank_keys=...)
    ranked_batch = store.get_session(5, gaussian_rank_keys=("physics_v4",))
    torch.testing.assert_close(
        ranked_batch["physics_v4"],
        mp4_gpu.gaussian_rank_features(torch.from_numpy(arrays["physics_v4"][5])),
        atol=1e-5,
        rtol=1e-5,
    )
    # asset_ids y modality_mask no se alteran y mantienen zero-copy
    assert ranked_batch["asset_ids"].data_ptr() == store.raw_array("asset_ids")[5].ctypes.data
    assert (
        ranked_batch["modality_mask"].data_ptr() == store.raw_array("modality_mask")[5].ctypes.data
    )


def test_rmt_dual_gram_fast_path_exact_parity_fp32_and_fp64():
    """Dual W×W Gram (W < N) empata con NumPy en FP64/FP32, con inactivos y NaNs, CPU y CUDA."""
    rng = np.random.default_rng(2027)
    w_len, n_assets = 32, 128
    factors = rng.standard_normal((w_len, 3))
    load_true = rng.standard_normal((n_assets, 3))
    ret_complete = (factors @ load_true.T + 0.35 * rng.standard_normal((w_len, n_assets))).astype(
        np.float64
    )

    # 1) Ventana completa W < N: FP64 exacto y FP32 (por defecto) frente a referencia NumPy
    res_np = mp4.rmt_clean_correlation(ret_complete, top_k=5)
    res_f64 = mp4_gpu.batched_rmt_clean_correlation_torch(
        torch.from_numpy(ret_complete),
        top_k=5,
        return_clean_corr=True,
        compute_dtype=torch.float64,
    )
    res_f32 = mp4_gpu.batched_rmt_clean_correlation_torch(
        torch.from_numpy(ret_complete),
        top_k=5,
        return_clean_corr=True,
        compute_dtype=torch.float32,
    )
    for res, tol_s, tol_m in ((res_f64, 1e-6, 1e-5), (res_f32, 1e-4, 2e-4)):
        assert int(res["n_signal_modes"].item()) == res_np["n_signal_modes"]
        np.testing.assert_allclose(
            float(res["absorption_ratio"].item()), res_np["absorption_ratio"], atol=tol_s
        )
        np.testing.assert_allclose(
            float(res["spectral_entropy"].item()), res_np["spectral_entropy"], atol=tol_s
        )
        np.testing.assert_allclose(res["loadings"].numpy(), res_np["loadings"], atol=tol_m)
        np.testing.assert_allclose(res["clean_corr"].numpy(), res_np["clean_corr"], atol=tol_m)
        np.testing.assert_allclose(res["raw_corr"].numpy(), res_np["raw_corr"], atol=tol_m)

    # 2) Ventana W < N con activos todo-NaN / constantes (inactivos) sigue tomando vía dual exacta
    ret_inact = ret_complete.copy()
    ret_inact[:, :10] = np.nan
    ret_inact[:, 10:15] = 0.01
    res_inact_np = mp4.rmt_clean_correlation(ret_inact, top_k=5)
    res_inact_f64 = mp4_gpu.batched_rmt_clean_correlation_torch(
        torch.from_numpy(ret_inact),
        top_k=5,
        return_clean_corr=True,
        compute_dtype=torch.float64,
    )
    assert int(res_inact_f64["n_signal_modes"].item()) == res_inact_np["n_signal_modes"]
    np.testing.assert_allclose(
        float(res_inact_f64["absorption_ratio"].item()),
        res_inact_np["absorption_ratio"],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        float(res_inact_f64["spectral_entropy"].item()),
        res_inact_np["spectral_entropy"],
        atol=1e-6,
    )
    # Los 3 modos de señal no degenerados empatan exactos; modos 3..4 pertenecen al subespacio
    # degenerado lambda=1.0 de los 15 activos inactivos (cualquier base ortonormal es equivalente)
    load_inact = res_inact_f64["loadings"].numpy()
    np.testing.assert_allclose(load_inact[:, :3], res_inact_np["loadings"][:, :3], atol=1e-5)
    np.testing.assert_allclose(load_inact[15:, 3:], 0.0, atol=1e-6)
    np.testing.assert_allclose(load_inact[:15, 3:].T @ load_inact[:15, 3:], np.eye(2), atol=1e-6)
    np.testing.assert_allclose(
        res_inact_f64["clean_corr"].numpy(), res_inact_np["clean_corr"], atol=1e-5
    )

    # 3) Lote mixto [B=3, W=32, N=128]: completa + inactivos + NaNs parciales (fallback)
    ret_partial = ret_complete.copy()
    ret_partial[:8, 20:25] = np.nan
    batch_r = np.stack([ret_complete, ret_inact, ret_partial], axis=0)
    res_batch = mp4_gpu.batched_rmt_clean_correlation_torch(
        torch.from_numpy(batch_r), top_k=4, return_clean_corr=True, compute_dtype=torch.float64
    )
    for b_i, r_single in enumerate((ret_complete, ret_inact, ret_partial)):
        ref_b = mp4.rmt_clean_correlation(r_single, top_k=4)
        assert int(res_batch["n_signal_modes"][b_i].item()) == ref_b["n_signal_modes"]
        np.testing.assert_allclose(
            float(res_batch["absorption_ratio"][b_i].item()),
            ref_b["absorption_ratio"],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            res_batch["clean_corr"][b_i].numpy(), ref_b["clean_corr"], atol=1e-4
        )

    # 4) Panel completo W < N (rmt_window=24, N=80) + verificación real en CUDA si disponible
    t_p, n_p = 45, 80
    r_pan = (
        rng.standard_normal((t_p, 3)) @ rng.standard_normal((3, n_p))
        + 0.4 * rng.standard_normal((t_p, n_p))
    ) * 0.01
    liq_p = rng.normal(14.0, 1.0, size=(t_p, n_p))
    vol_p = np.full((t_p, n_p), 0.015)
    phys_cpu = mp4_gpu.batched_panel_physics_v4_torch(
        torch.from_numpy(r_pan),
        torch.from_numpy(liq_p),
        torch.from_numpy(vol_p),
        rmt_window=24,
        top_k_rmt=4,
        maturity_lag=1,
    )
    phys_f64 = mp4_gpu.batched_panel_physics_v4_torch(
        torch.from_numpy(r_pan),
        torch.from_numpy(liq_p),
        torch.from_numpy(vol_p),
        rmt_window=24,
        top_k_rmt=4,
        maturity_lag=1,
        compute_dtype=torch.float64,
    )
    np.testing.assert_allclose(
        phys_cpu["rmt_absorption"].numpy(), phys_f64["rmt_absorption"].numpy(), atol=1e-4
    )
    np.testing.assert_allclose(
        phys_cpu["rmt_entropy"].numpy(), phys_f64["rmt_entropy"].numpy(), atol=1e-4
    )
    np.testing.assert_allclose(
        phys_cpu["wasserstein"].numpy(), phys_f64["wasserstein"].numpy(), atol=1e-5
    )
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        res_cuda = mp4_gpu.batched_rmt_clean_correlation_torch(
            torch.from_numpy(ret_complete).to(dev), top_k=5, return_clean_corr=True
        )
        np.testing.assert_allclose(
            res_cuda["clean_corr"].cpu().numpy(), res_np["clean_corr"], atol=2e-4
        )
        np.testing.assert_allclose(
            res_cuda["loadings"].cpu().numpy(), res_np["loadings"], atol=2e-4
        )


def test_causal_affine_scan_broadcast_and_multimodal_fusion_low_vram():
    """_causal_affine_scan acepta decay [T,N,1] broadcastable y fusión chunked usa <1 GB VRAM."""
    rng = np.random.default_rng(707)
    t_len, n_assets, dim = 64, 16, 32
    a_scalar = torch.from_numpy(rng.uniform(0.5, 0.99, size=(t_len, n_assets, 1)))
    b_vec = torch.from_numpy(rng.standard_normal((t_len, n_assets, dim)))
    out_bcast = mp4_gpu._causal_affine_scan(a_scalar, b_vec, init=0.0)
    out_full = mp4_gpu._causal_affine_scan(a_scalar.expand_as(b_vec), b_vec, init=0.0)
    torch.testing.assert_close(out_bcast, out_full, atol=1e-12, rtol=1e-12)

    # Verificar paridad FP32 vs FP64 en batched_multimodal_fusion_v4_torch
    news_emb = rng.standard_normal((t_len, n_assets, dim)).astype(np.float32)
    news_pres = rng.random((t_len, n_assets)) > 0.5
    chart_emb = rng.standard_normal((t_len, n_assets, dim)).astype(np.float32)
    chart_pres = rng.random((t_len, n_assets)) > 0.3
    fund_vals = rng.standard_normal((t_len, n_assets, 8)).astype(np.float32)
    fund_pres = rng.random((t_len, n_assets)) > 0.4
    ms = rng.standard_normal((t_len, 4)).astype(np.float32)
    mb = rng.standard_normal((t_len, n_assets, 4)).astype(np.float32)
    mm_f32 = mp4_gpu.batched_multimodal_fusion_v4_torch(
        news_emb=news_emb,
        news_present=news_pres,
        chart_emb=chart_emb,
        chart_present=chart_pres,
        fund_vals=fund_vals,
        fund_present=fund_pres,
        macro_surprises=ms,
        macro_betas=mb,
        compute_dtype=torch.float32,
    )
    mm_f64 = mp4_gpu.batched_multimodal_fusion_v4_torch(
        news_emb=news_emb,
        news_present=news_pres,
        chart_emb=chart_emb,
        chart_present=chart_pres,
        fund_vals=fund_vals,
        fund_present=fund_pres,
        macro_surprises=ms,
        macro_betas=mb,
        compute_dtype=torch.float64,
    )
    torch.testing.assert_close(mm_f32["features"], mm_f64["features"], atol=1e-5, rtol=1e-5)

    # En CUDA real: T=500, N=2600, D=768 (entradas en CPU/BF16, workspace GPU < 1.0 GB)
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        t_big, n_big, d_big = 500, 2600, 768
        # Usar vistas broadcast/compactas en CPU para no gastar 8 GB de RAM host en el test
        ne_big = torch.randn((t_big, 1, d_big), dtype=torch.bfloat16).expand(t_big, n_big, d_big)
        ce_big = torch.randn((t_big, 1, d_big), dtype=torch.bfloat16).expand(t_big, n_big, d_big)
        fv_big = torch.randn((t_big, n_big, 8), dtype=torch.float32, device=dev)
        np_big = torch.rand((t_big, n_big), device=dev) > 0.7
        cp_big = torch.rand((t_big, n_big), device=dev) > 0.5
        fp_big = torch.rand((t_big, n_big), device=dev) > 0.6
        ms_big = torch.randn((t_big, 4), dtype=torch.float32, device=dev)
        mb_big = torch.randn((t_big, n_big, 4), dtype=torch.float32, device=dev)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(dev)
        base_mem = torch.cuda.memory_allocated(dev)
        out_big = mp4_gpu.batched_multimodal_fusion_v4_torch(
            news_emb=ne_big,
            news_present=np_big,
            chart_emb=ce_big,
            chart_present=cp_big,
            fund_vals=fv_big,
            fund_present=fp_big,
            macro_surprises=ms_big,
            macro_betas=mb_big,
            device=dev,
        )
        torch.cuda.synchronize()
        peak_delta_gb = (torch.cuda.max_memory_allocated(dev) - base_mem) / (1024.0**3)
        assert out_big["features"].shape == (t_big, n_big, len(mp4.MULTIMODAL_V4_NAMES))
        assert out_big["features"].device.type == "cuda"
        assert torch.isfinite(out_big["features"]).all()
        assert peak_delta_gb < 1.0, f"Pico de VRAM workspace excedió 1 GB: {peak_delta_gb:.3f} GB"


def test_zero_copy_store_high_water_ragged_pinned_slab_and_gpu_resident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """High-water pinned buffers (cero realloc en ragged), bulk slab staging y gpu_resident."""
    rng = np.random.default_rng(606)
    counts = [10, 24, 12, 18, 8, 22, 16, 14]
    years = [2020] * len(counts)
    sessions = [
        dict(
            asset_ids=torch.arange(1000, 1000 + n_i, dtype=torch.int64),
            prices=torch.from_numpy(rng.standard_normal((n_i, 12)).astype(np.float32)),
            target_returns=torch.from_numpy(rng.standard_normal((n_i, 2)).astype(np.float32)),
        )
        for n_i in counts
    ]
    store = mp4_gpu.ZeroCopySessionStore.from_sessions(tmp_path / "ragged_hw", sessions, years)

    # 1) Mock CUDA per-session (slab_sessions=1): contar cuántas veces se llama pin_memory=True
    alloc_shapes: list[tuple[int, ...]] = []

    class _MockStream:
        def __init__(self, device=None):
            self.device = device

        def wait_stream(self, other):
            pass

    class _MockCtx:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _MockEvent:
        def record(self, stream=None):
            pass

        def synchronize(self):
            pass

    class _MockGpuTensor:
        def __init__(self, t: torch.Tensor):
            self.t = t

        def record_stream(self, stream):
            pass

    class _MockPinnedTensor:
        def __init__(self, shape, dtype):
            self.shape = tuple(shape)
            self.dtype = dtype
            self._buf = torch.empty(shape, dtype=dtype)

        def __getitem__(self, item):
            return _MockPinnedView(self._buf[item])

        def copy_(self, src, non_blocking=False):
            self._buf.copy_(src)
            return self

        def to(self, device, non_blocking=False):
            return _MockGpuTensor(self._buf.clone())

    class _MockPinnedView:
        def __init__(self, view: torch.Tensor):
            self._view = view

        def copy_(self, src, non_blocking=False):
            self._view.copy_(src)
            return self

        def to(self, device, non_blocking=False):
            return _MockGpuTensor(self._view.clone())

    orig_empty = torch.empty

    def _counting_empty(*args, **kwargs):
        if kwargs.get("pin_memory"):
            raw_sh = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args
            shape = tuple(raw_sh)
            alloc_shapes.append(shape)
            return _MockPinnedTensor(shape, kwargs.get("dtype", torch.float32))
        return orig_empty(*args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(torch.cuda, "Stream", _MockStream)
        mp.setattr(torch.cuda, "Event", _MockEvent)
        mp.setattr(torch.cuda, "stream", lambda s: _MockCtx())
        mp.setattr(torch.cuda, "current_stream", lambda device=None: _MockStream())
        mp.setattr(torch, "empty", _counting_empty)

        out_ragged = list(
            store.iter_pinned_h2d(device="cuda:0", use_pinned=True, slab_sessions=1)
        )
        assert len(out_ragged) == len(sessions)
        for i, b in enumerate(out_ragged):
            torch.testing.assert_close(b["prices"].t, sessions[i]["prices"], atol=0.0, rtol=0.0)
            torch.testing.assert_close(
                b["asset_ids"].t, sessions[i]["asset_ids"], atol=0.0, rtol=0.0
            )
        # High-water max_N=24: exactamente 2 slots × 3 claves = 6 allocs (jamás 24!)
        assert len(alloc_shapes) == 2 * len(store.keys())
        assert all(sh[0] == max(counts) for sh in alloc_shapes)

    # 2) Verificación CUDA real: bulk slab staging (dense & ragged) + gpu_resident + early break
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        # Ragged slab staging
        slab_out = list(
            store.iter_pinned_h2d(device=dev, use_pinned=True, slab_sessions=3)
        )
        assert len(slab_out) == len(sessions)
        for i, b in enumerate(slab_out):
            assert b["prices"].device.type == "cuda"
            torch.testing.assert_close(b["prices"].cpu(), sessions[i]["prices"], atol=0.0, rtol=0.0)
            torch.testing.assert_close(
                b["asset_ids"].cpu(), sessions[i]["asset_ids"], atol=0.0, rtol=0.0
            )

        # Ragged gaussian_rank_keys sobre slab en GPU coincide con CPU
        ranked_gpu = list(
            store.iter_pinned_h2d(
                device=dev, use_pinned=True, slab_sessions=3, gaussian_rank_keys=("prices",)
            )
        )
        for i, b in enumerate(ranked_gpu):
            ref_r = mp4_gpu.gaussian_rank_features(sessions[i]["prices"])
            torch.testing.assert_close(b["prices"].cpu(), ref_r, atol=1e-5, rtol=1e-5)

        # Early break limpia el hilo de prefetch sin race ni error en la siguiente iteración
        it = store.iter_pinned_h2d(device=dev, use_pinned=True, slab_sessions=2)
        first = next(it)
        torch.testing.assert_close(first["prices"].cpu(), sessions[0]["prices"], atol=0.0, rtol=0.0)
        it.close()

        # Modo gpu_resident=True: zero-copy GPU slices en llamadas repetidas + clear_gpu_cache()
        res_out1 = list(store.iter_pinned_h2d(device=dev, gpu_resident=True))
        res_out2 = list(store.iter_pinned_h2d(device=dev, gpu_resident=True))
        assert len(res_out1) == len(sessions)
        for i in range(len(sessions)):
            assert res_out1[i]["prices"].data_ptr() == res_out2[i]["prices"].data_ptr()
            torch.testing.assert_close(
                res_out2[i]["prices"].cpu(), sessions[i]["prices"], atol=0.0, rtol=0.0
            )
        store.clear_gpu_cache()
        store.clear_pinned_cache()


def test_market_physics_zero_scalar_h2d_syncs_day0_mask_and_centered_var_hygiene():
    """Audita cero torch.tensor() H2D ocultos, máscara Day-0 ret[0] y varianza centrada T=6000."""
    import ast
    import inspect

    source = inspect.getsource(mp4_gpu)
    tree = ast.parse(source)
    hot_fns = {
        "_apply_ffd_2d",
        "_rolling_var_2d",
        "batched_asset_microstructure_v4_torch",
        "_rmt_batch_core",
        "batched_rmt_clean_correlation_torch",
        "batched_causal_wasserstein_shift_torch",
        "batched_causal_lead_lag_spillover_torch",
        "batched_macro_surprises_and_betas_torch",
        "batched_panel_physics_v4_torch",
        "batched_multimodal_fusion_v4_torch",
        "gaussian_rank_features",
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in hot_fns:
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "tensor"
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == "torch"
                ):
                    raise AssertionError(
                        f"Copia H2D/sync oculta `torch.tensor(...)` detectada en {node.name}"
                    )

    # 1) Day-0 masking: retornos constantes +2% desde t=1 -> sin sesgo espurio por ret[0]=0.0
    t_d0, n_d0 = 40, 2
    log_c = np.log(100.0) + 0.02 * np.arange(t_d0, dtype=np.float64)[:, None] * np.ones((1, n_d0))
    c_d0 = np.exp(log_c)
    o_d0 = c_d0.copy()
    h_d0 = c_d0 * 1.005
    l_d0 = c_d0 * 0.995
    v_d0 = np.full((t_d0, n_d0), 1e5, dtype=np.float64)
    out_d0 = mp4_gpu.batched_asset_microstructure_v4_torch(o_d0, h_d0, l_d0, c_d0, v_d0)
    idx = {name: i for i, name in enumerate(mp4.ASSET_V4_NAMES)}
    # En t=10 (< 21), ret_cc[1..10] son idénticos (+0.02) -> skew/kurt/pv_corr deben ser EXACTOS 0.0
    assert float(out_d0[10, 0, idx["realized_skew_21"]]) == 0.0
    assert float(out_d0[10, 0, idx["realized_kurt_21"]]) == 0.0
    assert float(out_d0[10, 0, idx["price_vol_corr_21"]]) == 0.0

    # 2) Varianza centrada sin cancelación catastrófica en T=6000 desplazado (+1e6)
    x_const = torch.full((6000, 4), 1e6, dtype=torch.float64)
    v_centered = mp4_gpu._rolling_var_2d(x_const, 63)
    assert float(v_centered.abs().max()) == 0.0

    # 3) Lead-Lag centrado sin cancelación en serie constante T=6000
    r_const = torch.full((6000, 6), 0.02, dtype=torch.float32)
    liq_const = torch.linspace(10.0, 20.0, 6).unsqueeze(0).expand(6000, 6).clone()
    ll_const = mp4_gpu.batched_causal_lead_lag_spillover_torch(r_const, liq_const, window=40)
    assert float(ll_const.abs().max()) == 0.0


def test_wave3a_zero_copy_store_missingness_featurizer_toggle_encoded_triplets_and_4families(
    tmp_path: Path,
):
    """Wave-3A: toggle `apply_missingness_featurizer` en dense/ragged, v3.1 y 4 familias."""
    from mars_titan.data.dataset_enrichment_sota import ENCODED_TO_MODEL_MODALITY_INDEX
    from mars_titan.models.mmarrs import build_family_model

    rng = np.random.default_rng(3101)
    t_len, n_assets = 8, 12
    years = np.full(t_len, 2022, dtype=np.int64)

    # Construir tripletas reales `encoded_v31_78` (fundamentals) y `encoded_v31_420` (macro)
    # con offset log(CNY) ~ 24.0 y ausencias extremas (incl. sesión t=5 con 100% aux ausente + NaN!)
    f_vals = rng.normal(24.0, 1.2, size=(t_len, n_assets, 26)).astype(np.float32)
    f_obs = (rng.random((t_len, n_assets, 26)) > 0.70).astype(np.float32)
    f_lage = (rng.uniform(0.0, 5.5, size=(t_len, n_assets, 26)) * f_obs).astype(np.float32)
    f_vals[f_obs < 0.5] = 0.0
    fundamentals = np.concatenate([f_vals, f_obs, f_lage], axis=-1)

    m_vals = rng.normal(1.5, 0.8, size=(t_len, n_assets, 140)).astype(np.float32)
    m_obs = (rng.random((t_len, n_assets, 140)) > 0.15).astype(np.float32)
    m_lage = (rng.uniform(0.0, 3.0, size=(t_len, n_assets, 140)) * m_obs).astype(np.float32)
    macro = np.concatenate([m_vals, m_obs, m_lage], axis=-1)

    prices = rng.standard_normal((t_len, n_assets, 16)).astype(np.float32)
    news = rng.standard_normal((t_len, n_assets, 8)).astype(np.float32)
    charts = rng.standard_normal((t_len, n_assets, 8)).astype(np.float32)
    # Máscara en orden `encoded-history-v3.1`: (prices, news, charts, fundamentals, macro)
    enc_mask = np.ones((t_len, n_assets, 5), dtype=np.float32)
    enc_mask[:, :, 1] = (rng.random((t_len, n_assets)) > 0.6).astype(np.float32)  # news
    enc_mask[:, :, 2] = (rng.random((t_len, n_assets)) > 0.2).astype(np.float32)  # charts
    enc_mask[:, :, 3] = (f_obs.sum(axis=-1) > 0).astype(np.float32)  # fundamentals
    enc_mask[:, :, 4] = 1.0  # macro

    # Inyectar sesión t=5 con 100% modalidades auxiliares ausentes + NaN/Inf crudos
    enc_mask[5, :, 1:] = 0.0
    fundamentals[5, :, :26] = np.nan
    fundamentals[5, 0, 0] = np.inf
    news[5] = np.nan
    charts[5] = -np.inf
    macro[5, :, :140] = np.nan

    arrays = dict(
        asset_ids=np.broadcast_to(np.arange(n_assets, dtype=np.int64)[None, :], (t_len, n_assets))
        .copy(),
        prices=prices,
        fundamentals=fundamentals,
        macro=macro,
        news=news,
        charts=charts,
        modality_mask=enc_mask,
        physics_v4=rng.standard_normal((t_len, n_assets, 24)).astype(np.float32),
        high_snr_context=rng.standard_normal((t_len, n_assets, 8)).astype(np.float32),
        exposures=rng.standard_normal((t_len, n_assets, 4)).astype(np.float32),
        rmt_loadings=rng.standard_normal((t_len, n_assets, 5)).astype(np.float32),
        target_returns=(0.01 * rng.standard_normal((t_len, n_assets, 6))).astype(np.float32),
    )

    store_dir = tmp_path / "dense_missingness"
    mp4_gpu.ZeroCopySessionStore.save(store_dir, session_years=years, arrays=arrays)

    # 1) Default `apply_missingness_featurizer=False`: 100% bit-exact zero-copy (retrocompatible!)
    raw_store = mp4_gpu.ZeroCopySessionStore.load(store_dir, mmap_mode="r")
    b_raw = raw_store[0]
    assert b_raw["prices"].data_ptr() == raw_store.raw_array("prices")[0].ctypes.data
    assert "modality_freshness" not in b_raw

    # 2) Toggle `apply_missingness_featurizer=True` + `modality_mask_layout="encoded_v31"`
    feat_store = mp4_gpu.ZeroCopySessionStore.load(
        store_dir,
        mmap_mode="r",
        apply_missingness_featurizer=True,
        modality_mask_layout="encoded_v31",
    )
    b0 = feat_store[0]
    b5 = feat_store[5]  # 100% aux ausentes + NaN/Inf!
    assert "modality_freshness" in b0
    assert b0["modality_freshness"].shape == (n_assets, 5)
    # Verificar remapeo exacto `ENCODED_TO_MODEL_MODALITY_INDEX = (0, 3, 4, 1, 2)`
    expected_m0 = torch.from_numpy(enc_mask[0][:, list(ENCODED_TO_MODEL_MODALITY_INDEX)])
    torch.testing.assert_close(b0["modality_mask"], expected_m0, atol=0.0, rtol=0.0)

    # Preservar tripletas `encoded_v31_78` (obs in {0, 1}, log_age >= 0) y SIN saturar log(CNY)~24!
    f_san0, f_obs0, f_lage0 = torch.split(b0["fundamentals"], 26, dim=-1)
    assert torch.isfinite(b0["fundamentals"]).all()
    assert set(torch.unique(f_obs0).tolist()).issubset({0.0, 1.0})
    assert float(f_lage0.min()) >= 0.0
    # Si se hubiera aplicado clamp directo sin robust_cs_z, obs habrían saturado todas a +8.0
    if bool((f_obs0 > 0.5).any()):
        assert float(f_san0[f_obs0 > 0.5].abs().mean()) < 3.5

    m_san0, m_obs0, m_lage0 = torch.split(b0["macro"], 140, dim=-1)
    assert torch.isfinite(m_san0).all()
    assert set(torch.unique(m_obs0).tolist()).issubset({0.0, 1.0})
    assert float(m_lage0.min()) >= 0.0

    # Sesión t=5 (100% aux ausentes + NaN/Inf): 0 NaNs / 0 Infs en todas las claves!
    for k, v in b5.items():
        if v.dtype.is_floating_point:
            assert torch.isfinite(v).all(), f"No finito en sesión 100% ausente para {k}"
    assert float(b5["fundamentals"].abs().max()) == 0.0
    assert float(b5["news"].abs().max()) == 0.0
    assert float(b5["charts"].abs().max()) == 0.0

    # Vista y `iter_pinned_h2d` propagan `apply_missingness_featurizer=True`
    view = feat_store[0:6]
    yielded = list(view.iter_pinned_h2d(device="cpu"))
    assert len(yielded) == 6
    torch.testing.assert_close(yielded[5]["fundamentals"], b5["fundamentals"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(yielded[0]["modality_mask"], expected_m0, atol=0.0, rtol=0.0)

    # 3) Ragged store con `apply_missingness_featurizer=True`
    ragged_sess = []
    for s_i, n_i in enumerate((10, 8, 12)):
        ragged_sess.append({k: v[s_i, :n_i] for k, v in arrays.items()})
    rag_store = mp4_gpu.ZeroCopySessionStore.from_sessions(
        tmp_path / "ragged_missingness",
        ragged_sess,
        [2020, 2021, 2022],
        apply_missingness_featurizer=True,
        modality_mask_layout="encoded_v31",
    )
    rb1 = rag_store[1]
    assert rb1["prices"].shape == (8, 16)
    assert "modality_freshness" in rb1
    assert all(torch.isfinite(v).all() for v in rb1.values() if v.dtype.is_floating_point)

    # 4) Compatibilidad directa con LAS 4 FAMILIAS INDEPENDIENTES en `b5` (100% aux ausentes)
    dims = dict(
        prices_dim=16,
        fundamentals_dim=78,
        macro_dim=420,
        news_dim=8,
        charts_dim=8,
        physics_dim=24,
        context_dim=8,
        exposures_dim=4,
        rmt_dim=5,
        hidden_dim=16,
        allow_short_synthetic=True,
    )
    for var_id in (
        "t15b_full_v2",
        "nb_v2_full",
        "v14c_tied_xasset_cf",
        "cog_v2_10tools_soft_k3",
    ):
        model = build_family_model(var_id, market="US", **dims).eval()
        with torch.no_grad():
            out_m = model.forward_session(b5)
        assert torch.isfinite(out_m["pred_return"]).all(), f"{var_id} no finito en b5"


def test_wave3a_enrich_session_store_sota_dense_ragged_causality_2024_seal_and_zero_syncs(
    tmp_path: Path,
):
    """Wave-3A: `with_sota_enrichment` dense/ragged, causalidad bit-exact, sello 2024 y 0 sync."""
    import ast
    import inspect

    from mars_titan.data.dataset_enrichment_sota import ENRICHED_SOTA_NAMES

    rng = np.random.default_rng(3102)
    t_len, n_assets = 40, 10
    o, h, lo, c, v = _synthetic_ohlcv_panel(rng, t_len=t_len, n_assets=n_assets)
    years = np.full(t_len, 2022, dtype=np.int64)
    arrays = _make_multimodal_arrays(rng, t_len=t_len, n_assets=n_assets)
    # Inyectar NaNs en fundamentales/news para probar limpieza conjunta
    arrays["fundamentals"][10:15, :5] = np.nan
    arrays["news"][20:25, :] = np.nan
    arrays["modality_mask"][:, :, 0] = 1.0

    src_dir = tmp_path / "src_dense"
    mp4_gpu.ZeroCopySessionStore.save(src_dir, session_years=years, arrays=arrays)
    store = mp4_gpu.ZeroCopySessionStore.load(src_dir, mmap_mode="r")

    # 1) Enriquecimiento dense con OHLCV explícito + auditoría integral + paridad bajo BF16 autocast
    ohlcv_dict = dict(open=o, high=h, low=lo, close=c, volume=v)
    enr_store, report = store.with_sota_enrichment(
        tmp_path / "enr_dense_fp32",
        ohlcv=ohlcv_dict,
        append_to_physics_v4=True,
        apply_missingness_featurizer=True,
        run_audit=True,
    )
    assert "enriched_sota" in enr_store.keys()
    n_enr = len(ENRICHED_SOTA_NAMES)
    assert enr_store.raw_array("enriched_sota").shape == (t_len, n_assets, n_enr)
    assert enr_store.raw_array("physics_v4").shape == (t_len, n_assets, 24 + n_enr)
    assert np.isfinite(enr_store.raw_array("enriched_sota")).all()
    assert np.isfinite(enr_store.raw_array("physics_v4")).all()
    assert np.isfinite(enr_store.raw_array("fundamentals")).all()
    assert report["n_enriched_features"] == len(ENRICHED_SOTA_NAMES)
    assert report["strictly_pre_2024"] is True
    assert report["quality_audit"]["causality_check"]["bit_exact_causal"] is True
    assert report["quality_audit"]["missingness_stress"]["all_passed"] is True
    assert isinstance(report["sha256_contract"], str) and len(report["sha256_contract"]) == 64

    # Paridad bajo `torch.autocast("cpu", dtype=torch.bfloat16)` (FP32 maestro en enriquecimiento!)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        enr_bf16, _ = store.with_sota_enrichment(
            tmp_path / "enr_dense_bf16",
            ohlcv=ohlcv_dict,
            append_to_physics_v4=True,
            apply_missingness_featurizer=True,
            run_audit=False,
        )
    np.testing.assert_allclose(
        enr_bf16.raw_array("enriched_sota"),
        enr_store.raw_array("enriched_sota"),
        atol=0.0,
        rtol=0.0,
    )

    # 2) Causalidad estricta bit a bit en `enrich_session_store_sota` (mutar t > t_cut)
    t_cut = 25
    o_mut, h_mut, lo_mut, c_mut, v_mut = o.copy(), h.copy(), lo.copy(), c.copy(), v.copy()
    c_mut[t_cut + 1 :] *= 3.2
    h_mut[t_cut + 1 :] *= 3.8
    lo_mut[t_cut + 1 :] *= 0.4
    v_mut[t_cut + 1 :] *= 9.0
    enr_mut, _ = store.with_sota_enrichment(
        tmp_path / "enr_dense_mut",
        ohlcv=dict(open=o_mut, high=h_mut, low=lo_mut, close=c_mut, volume=v_mut),
        append_to_physics_v4=False,
        apply_missingness_featurizer=False,
        run_audit=False,
    )
    enr_clean_nofeat, _ = store.with_sota_enrichment(
        tmp_path / "enr_dense_clean_nofeat",
        ohlcv=ohlcv_dict,
        append_to_physics_v4=False,
        apply_missingness_featurizer=False,
        run_audit=False,
    )
    np.testing.assert_array_equal(
        enr_clean_nofeat.raw_array("enriched_sota")[: t_cut + 1],
        enr_mut.raw_array("enriched_sota")[: t_cut + 1],
    )

    # 3) Ragged store (IPO/delisting con N_t variable) + derivación OHLCV causal cuando ohlcv=None
    rag_sessions = []
    rag_counts = [8 if t < 15 else (10 if t < 30 else 7) for t in range(t_len)]
    for t_i, n_i in enumerate(rag_counts):
        rag_sessions.append({k: v[t_i, :n_i] for k, v in arrays.items()})
    rag_store = mp4_gpu.ZeroCopySessionStore.from_sessions(
        tmp_path / "src_ragged", rag_sessions, years
    )
    rag_enr, rag_rep = rag_store.with_sota_enrichment(
        tmp_path / "enr_ragged",
        ohlcv=None,
        append_to_physics_v4=True,
        apply_missingness_featurizer=True,
        run_audit=True,
    )
    assert rag_enr.layout == "ragged"
    assert rag_enr.raw_array("enriched_sota").shape == (sum(rag_counts), len(ENRICHED_SOTA_NAMES))
    assert rag_enr.raw_array("physics_v4").shape == (sum(rag_counts), 24 + len(ENRICHED_SOTA_NAMES))
    assert np.isfinite(rag_enr.raw_array("enriched_sota")).all()
    assert rag_rep["ohlcv_source"] == "derived_causal_proxy"

    # 4) Sello estricto `2024+` en `with_sota_enrichment`
    years_2024 = years.copy()
    years_2024[-1] = 2024
    s24_dir = tmp_path / "src_2024"
    mp4_gpu.ZeroCopySessionStore.save(s24_dir, session_years=years_2024, arrays=arrays)
    s24 = mp4_gpu.ZeroCopySessionStore.load(s24_dir, mmap_mode="r")
    with pytest.raises(ValueError, match="2024"):
        s24.with_sota_enrichment(tmp_path / "enr_2024_blocked", ohlcv=ohlcv_dict)

    # 5) Auditoría AST cero host-syncs en helpers hot-path nuevos de `market_physics_v4_gpu.py`
    source = inspect.getsource(mp4_gpu)
    tree = ast.parse(source)
    new_hot = {"_remap_encoded_modality_columns", "_apply_missingness_featurizer_to_session"}
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in new_hot:
            found.add(node.name)
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute) and child.attr in {"item", "cpu", "nonzero"}:
                    raise AssertionError(f"Host-sync `{child.attr}` detectado en {node.name}")
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "tensor"
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == "torch"
                ):
                    raise AssertionError(f"Copia H2D `torch.tensor` detectada en {node.name}")
    assert found == new_hot


def test_wave3a_adversarial_199_permuted_rotating_ohlcv_reload_layout_cn_clip_and_causal_switch(
    tmp_path: Path,
):
    """Regresión `#199` (P0-2, P1-1, P1-2, P1-5): OHLCV permutado, reload layout, CN clip, proxy."""
    from mars_titan.data.dataset_enrichment_sota import (
        ENCODED_TO_MODEL_MODALITY_INDEX,
        ENRICHED_SOTA_NAMES,
    )

    rng = np.random.default_rng(19901)
    t_len, n_assets = 32, 6
    years = np.full(t_len, 2022, dtype=np.int64)
    o, h, lo, c, v = _synthetic_ohlcv_panel(rng, t_len=t_len, n_assets=n_assets)
    arrays = _make_multimodal_arrays(rng, t_len=t_len, n_assets=n_assets)

    # 1) P0-2a: Dense store con `asset_ids` permutados entre sesiones + OHLCV 2D explícito
    canon_dir = tmp_path / "p02_canon"
    mp4_gpu.ZeroCopySessionStore.save(canon_dir, session_years=years, arrays=arrays)
    canon_enr, _ = mp4_gpu.ZeroCopySessionStore.load(canon_dir).with_sota_enrichment(
        tmp_path / "p02_canon_enr",
        ohlcv=dict(open=o, high=h, low=lo, close=c, volume=v),
        append_to_physics_v4=False,
        apply_missingness_featurizer=False,
    )

    perms = np.stack([rng.permutation(n_assets) for _ in range(t_len)], axis=0)
    perm_arrays = {
        k: np.take_along_axis(val, perms[:, :, None] if val.ndim == 3 else perms, 1)
        for k, val in arrays.items()
    }
    o_p = np.take_along_axis(o, perms, 1)
    h_p = np.take_along_axis(h, perms, 1)
    l_p = np.take_along_axis(lo, perms, 1)
    c_p = np.take_along_axis(c, perms, 1)
    v_p = np.take_along_axis(v, perms, 1)

    perm_dir = tmp_path / "p02_perm"
    mp4_gpu.ZeroCopySessionStore.save(perm_dir, session_years=years, arrays=perm_arrays)
    perm_enr, _ = mp4_gpu.ZeroCopySessionStore.load(perm_dir).with_sota_enrichment(
        tmp_path / "p02_perm_enr",
        ohlcv=dict(open=o_p, high=h_p, low=l_p, close=c_p, volume=v_p),
        append_to_physics_v4=False,
        apply_missingness_featurizer=False,
    )
    # Des-permutar salida permutada -> debe empatar exacto (0.0 diff) con la canónica!
    inv_perms = np.argsort(perms, axis=1)
    unperm_out = np.take_along_axis(
        np.asarray(perm_enr.raw_array("enriched_sota")), inv_perms[:, :, None], 1
    )
    np.testing.assert_allclose(
        unperm_out, np.asarray(canon_enr.raw_array("enriched_sota")), atol=1e-6, rtol=1e-6
    )

    # 2) P0-2b: Dense store con `asset_ids` rotativos (N_uniq=8 > N_sess=6) sin IndexError
    rot_arrays = {k: val.copy() for k, val in arrays.items()}
    rot_ids = np.broadcast_to(
        np.arange(n_assets, dtype=np.int64)[None, :], (t_len, n_assets)
    ).copy()
    rot_ids[t_len // 2 :, -2:] = np.array([6, 7], dtype=np.int64)
    rot_arrays["asset_ids"] = rot_ids
    rot_dir = tmp_path / "p02_rot"
    mp4_gpu.ZeroCopySessionStore.save(rot_dir, session_years=years, arrays=rot_arrays)
    rot_enr, _ = mp4_gpu.ZeroCopySessionStore.load(rot_dir).with_sota_enrichment(
        tmp_path / "p02_rot_enr",
        ohlcv=dict(open=o, high=h, low=lo, close=c, volume=v),
        append_to_physics_v4=False,
        apply_missingness_featurizer=False,
    )
    assert rot_enr.raw_array("enriched_sota").shape == (t_len, n_assets, len(ENRICHED_SOTA_NAMES))
    assert np.isfinite(rot_enr.raw_array("enriched_sota")).all()

    # 3) P1-1: `modality_mask_layout="encoded_v31"` persiste en manifest.json tras recarga
    #    y remapea `modality_mask` aun con `apply_missingness_featurizer=False` (prices zero-copy)
    enc_m = np.array(
        [[1.0, 1.0, 0.0, 0.0, 1.0] for _ in range(n_assets)], dtype=np.float32
    )
    sess_enc = [
        dict(
            asset_ids=np.arange(n_assets, dtype=np.int64),
            prices=arrays["prices"][t],
            fundamentals=arrays["fundamentals"][t],
            macro=arrays["macro"][t],
            news=arrays["news"][t],
            charts=arrays["charts"][t],
            modality_mask=enc_m.copy(),
        )
        for t in range(4)
    ]
    enc_dir = tmp_path / "p11_enc_reload"
    mp4_gpu.ZeroCopySessionStore.from_sessions(
        enc_dir, sess_enc, [2020, 2021, 2022, 2023], modality_mask_layout="encoded_v31"
    )
    expected_mod_m = enc_m[:, list(ENCODED_TO_MODEL_MODALITY_INDEX)]
    reloaded_nofeat = mp4_gpu.ZeroCopySessionStore.load(
        enc_dir, mmap_mode="r", apply_missingness_featurizer=False
    )
    assert reloaded_nofeat.modality_mask_layout == "encoded_v31"
    b_nf = reloaded_nofeat[0]
    assert b_nf["prices"].data_ptr() == reloaded_nofeat.raw_array("prices")[0].ctypes.data
    np.testing.assert_array_equal(b_nf["modality_mask"].numpy(), expected_mod_m)

    reloaded_feat = mp4_gpu.ZeroCopySessionStore.load(
        enc_dir, mmap_mode="r", apply_missingness_featurizer=True
    )
    assert reloaded_feat.modality_mask_layout == "encoded_v31"
    np.testing.assert_array_equal(reloaded_feat[0]["modality_mask"].numpy(), expected_mod_m)

    # 4) P1-2: `apply_missingness_featurizer=True` + `append_to_physics_v4=True` NO recorta señales
    #    dispersas CN (99% ceros) en `physics_v4[..., -18:]` vs `enriched_sota`
    n_cn = 100
    o_cn, h_cn, l_cn, c_cn, v_cn = _synthetic_ohlcv_panel(rng, t_len=12, n_assets=n_cn)
    cn_arr = _make_multimodal_arrays(rng, t_len=12, n_assets=n_cn)
    # 98% fundamentales ausentes, solo 2 activos con valores fuertes en `encoded_v31_78`
    f78 = np.zeros((12, n_cn, 78), dtype=np.float32)
    f78[:, 0, :26] = 28.0
    f78[:, 1, :26] = 20.0
    f78[:, :2, 26:52] = 1.0
    f78[:, :2, 52:78] = 0.2
    cn_arr["fundamentals"] = f78
    cn_arr["modality_mask"][:, :, 0] = 1.0
    cn_arr["modality_mask"][:, :, 1] = 0.0
    cn_arr["modality_mask"][:, :2, 1] = 1.0
    cn_dir = tmp_path / "p12_cn_sparse"
    mp4_gpu.ZeroCopySessionStore.save(
        cn_dir, session_years=np.full(12, 2022, dtype=np.int64), arrays=cn_arr
    )
    cn_enr, _ = mp4_gpu.ZeroCopySessionStore.load(cn_dir).with_sota_enrichment(
        tmp_path / "p12_cn_enr",
        ohlcv=dict(open=o_cn, high=h_cn, low=l_cn, close=c_cn, volume=v_cn),
        append_to_physics_v4=True,
        apply_missingness_featurizer=True,
    )
    n_enr = len(ENRICHED_SOTA_NAMES)
    np.testing.assert_array_equal(
        np.asarray(cn_enr.raw_array("physics_v4"))[..., -n_enr:],
        np.asarray(cn_enr.raw_array("enriched_sota")),
    )

    # 5) P1-5: Mutar fila futura a valor negativo en `prices` jamás cambia el modo de `t <= t_cut`
    pos_arrays = {k: val.copy() for k, val in arrays.items()}
    pos_arrays["prices"] = np.abs(pos_arrays["prices"]) + 10.0  # todo positivo
    p15_a = tmp_path / "p15_pos"
    mp4_gpu.ZeroCopySessionStore.save(p15_a, session_years=years, arrays=pos_arrays)
    mut_arrays = {k: val.copy() for k, val in pos_arrays.items()}
    t_cut = 18
    mut_arrays["prices"][t_cut + 1 :, 0, 0] = -5.0  # futuro negativo!
    p15_b = tmp_path / "p15_neg_future"
    mp4_gpu.ZeroCopySessionStore.save(p15_b, session_years=years, arrays=mut_arrays)

    ea, _ = mp4_gpu.ZeroCopySessionStore.load(p15_a).with_sota_enrichment(
        tmp_path / "p15_out_a", ohlcv=None, apply_missingness_featurizer=False
    )
    eb, _ = mp4_gpu.ZeroCopySessionStore.load(p15_b).with_sota_enrichment(
        tmp_path / "p15_out_b", ohlcv=None, apply_missingness_featurizer=False
    )
    np.testing.assert_array_equal(
        np.asarray(ea.raw_array("enriched_sota"))[: t_cut + 1],
        np.asarray(eb.raw_array("enriched_sota"))[: t_cut + 1],
    )

    # P1-5b: Bloquear subdirectorio dentro de `store_dir`
    with pytest.raises(ValueError, match="solo lectura"):
        mp4_gpu.ZeroCopySessionStore.load(p15_a).with_sota_enrichment(
            p15_a / "nested_forbidden", ohlcv=dict(open=o, high=h, low=lo, close=c, volume=v)
        )

    # 6) `#201` P0-3 + cero warnings: store OHLCV-only (sin clave `"prices"`) + `featurizer=True`
    ohlcv_only_dir = tmp_path / "p03_ohlcv_only"
    mp4_gpu.ZeroCopySessionStore.save(
        ohlcv_only_dir,
        session_years=years,
        arrays=dict(
            open=o.astype(np.float32),
            high=h.astype(np.float32),
            low=lo.astype(np.float32),
            close=c.astype(np.float32),
            volume=v.astype(np.float32),
            target_returns=arrays["target_returns"],
        ),
    )
    with warnings.catch_warnings(record=True) as caught_w:
        warnings.simplefilter("always")
        ohlcv_enr, ohlcv_rep = mp4_gpu.ZeroCopySessionStore.load(
            ohlcv_only_dir, mmap_mode="r"
        ).with_sota_enrichment(
            tmp_path / "p03_ohlcv_only_enr",
            ohlcv=None,
            apply_missingness_featurizer=True,
        )
    non_writable_w = [w for w in caught_w if "not writable" in str(w.message)]
    assert len(non_writable_w) == 0
    assert ohlcv_rep["ohlcv_source"] == "store_ohlcv_keys"
    assert np.isfinite(ohlcv_enr.raw_array("enriched_sota")).all()


def test_wave3a_adversarial_212_encoded_v31_single_remap_all_iterator_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Regresión `#212` P0: cero doble remapeo `encoded_v31 -> model` en las 5 rutas."""
    from mars_titan.data.dataset_enrichment_sota import ENCODED_TO_MODEL_MODALITY_INDEX

    rng = np.random.default_rng(21201)
    t_len, n_assets = 4, 6
    years = np.array([2020, 2021, 2022, 2023], dtype=np.int64)
    arrays = _make_multimodal_arrays(rng, t_len=t_len, n_assets=n_assets)
    # Máscara y edades asimétricas: `(0,3,4,1,2)^2 == identidad`, así cualquier doble remapeo falla!
    enc_mask = np.broadcast_to(
        np.array([1.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)[None, None, :],
        (t_len, n_assets, 5),
    ).copy()
    enc_age = np.broadcast_to(
        np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)[None, None, :],
        (t_len, n_assets, 5),
    ).copy()
    arrays["modality_mask"] = enc_mask
    arrays["modality_age"] = enc_age

    expected_mask = enc_mask[0][:, list(ENCODED_TO_MODEL_MODALITY_INDEX)]
    expected_age = enc_age[0][:, list(ENCODED_TO_MODEL_MODALITY_INDEX)]
    expected_age_masked = expected_age * expected_mask

    dense_dir = tmp_path / "enc_dense_212"
    mp4_gpu.ZeroCopySessionStore.save(
        dense_dir, session_years=years, arrays=arrays, modality_mask_layout="encoded_v31"
    )
    st = mp4_gpu.ZeroCopySessionStore.load(
        dense_dir, mmap_mode="r", apply_missingness_featurizer=True
    )

    # 1) `get_session` (featurizer=True y False)
    for af in (False, True):
        b = st.get_session(0, apply_missingness_featurizer=af)
        np.testing.assert_array_equal(b["modality_mask"].numpy(), expected_mask)
        exp_a = expected_age_masked if af else expected_age
        np.testing.assert_array_equal(b["modality_age"].numpy(), exp_a)

    # 2) `_iter_gpu_resident` contiguo Y no contiguo (`[0, 2]` — repro exacto `#212`!)
    for idx_arr in (np.array([0, 1]), np.array([0, 2])):
        for af in (False, True):
            st._gpu_resident_cache.clear()
            batches = list(
                st._iter_gpu_resident(torch.device("cpu"), idx_arr, (), apply_featurizer=af)
            )
            assert len(batches) == len(idx_arr)
            exp_a = expected_age_masked if af else expected_age
            for b in batches:
                np.testing.assert_array_equal(b["modality_mask"].numpy(), expected_mask)
                np.testing.assert_array_equal(b["modality_age"].numpy(), exp_a)

    # 3) `_iter_gpu_resident` ragged contiguo y no contiguo
    rag_sessions = [
        {k: v[t, : n_assets - (t % 2)] for k, v in arrays.items()} for t in range(t_len)
    ]
    rag_dir = tmp_path / "enc_rag_212"
    st_rag = mp4_gpu.ZeroCopySessionStore.from_sessions(
        rag_dir, rag_sessions, years, modality_mask_layout="encoded_v31"
    )
    for idx_arr in (np.array([0, 1]), np.array([0, 2])):
        for af in (False, True):
            st_rag._gpu_resident_cache.clear()
            batches = list(
                st_rag._iter_gpu_resident(torch.device("cpu"), idx_arr, (), apply_featurizer=af)
            )
            assert len(batches) == len(idx_arr)
            exp_a = expected_age_masked if af else expected_age
            for pos, b in enumerate(batches):
                n_i = n_assets - (int(idx_arr[pos]) % 2)
                np.testing.assert_array_equal(b["modality_mask"].numpy(), expected_mask[:n_i])
                np.testing.assert_array_equal(b["modality_age"].numpy(), exp_a[:n_i])

    # 4) & 5) `iter_pinned_h2d` CUDA contigua (`slab_sessions=2`) y por sesión (`slab_sessions=1`,
    #    índices no contiguos `[0, 2]` — segundo repro exacto `#212`!) simulada en CPU
    class _MockStream:
        def __init__(self, device=None):
            self.device = device

        def wait_stream(self, other):
            return None

    class _MockStreamCtx:
        def __init__(self, s):
            self._s = s

        def __enter__(self):
            return self._s

        def __exit__(self, exc_type, exc, tb):
            return False

    class _MockEvent:
        def record(self, stream=None):
            return None

        def synchronize(self):
            return None

    orig_to = torch.Tensor.to

    def _cpu_to(self, *args, **kwargs):
        if args and isinstance(args[0], (str, torch.device)) and str(args[0]).startswith("cuda"):
            args = (torch.device("cpu"), *args[1:])
        if "device" in kwargs and str(kwargs["device"]).startswith("cuda"):
            kwargs = {**kwargs, "device": torch.device("cpu")}
        kwargs.pop("non_blocking", None)
        return orig_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.cuda, "Stream", _MockStream)
    monkeypatch.setattr(torch.cuda, "Event", _MockEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: _MockStreamCtx(s))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: _MockStream(device))
    monkeypatch.setattr(torch.Tensor, "to", _cpu_to)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, s: None)

    for indices, slab_sessions in (([0, 1, 2], 2), ([0, 2], 1)):
        for af in (False, True):
            batches = list(
                st.iter_pinned_h2d(
                    "cuda:0",
                    indices=indices,
                    use_pinned=False,
                    apply_missingness_featurizer=af,
                    slab_sessions=slab_sessions,
                    gpu_resident=False,
                )
            )
            assert len(batches) == len(indices)
            exp_a = expected_age_masked if af else expected_age
            for b in batches:
                np.testing.assert_array_equal(b["modality_mask"].numpy(), expected_mask)
                np.testing.assert_array_equal(b["modality_age"].numpy(), exp_a)


def test_rmt_free_2factor_rotation_features_parity_causality_bf16_and_zero_extra_eigh(
    monkeypatch: pytest.MonkeyPatch,
):
    """#216/#195-5B: `rmt_gap_ratio_1_2` y `rmt_top1_vel_21` gratis (0 eigh extra, causal, BF16)."""
    import ast
    import inspect

    rng = np.random.default_rng(21602)
    t_len, n_assets = 130, 40
    # Regimen 1 (t=0..79): 1-factor crisis (lambda_1 >> lambda_2)
    # Regimen 2 (t=80..129): 2-factor style rotation US 2015 (lambda_1 ~ lambda_2!)
    f1 = rng.standard_normal((t_len, 1))
    f2 = rng.standard_normal((t_len, 1))
    eps = 0.005 * rng.standard_normal((t_len, n_assets))
    w1 = np.ones((1, n_assets))
    w2 = np.concatenate([np.ones((1, n_assets // 2)), -np.ones((1, n_assets - n_assets // 2))], 1)
    rets = np.zeros((t_len, n_assets), dtype=np.float32)
    rets[:80] = (0.025 * f1[:80] @ w1 + 0.002 * f2[:80] @ w2 + eps[:80]).astype(np.float32)
    rets[80:] = (0.015 * f1[80:] @ w1 + 0.014 * f2[80:] @ w2 + eps[80:]).astype(np.float32)
    liq = np.exp(rng.standard_normal((t_len, n_assets))).astype(np.float32)
    vol = np.full((t_len, n_assets), 0.02, dtype=np.float32)

    # 1) Cero eigendecomposiciones extra y contrato `return_extended=False` 100% retrocompatible
    eigh_calls = [0]
    orig_eigh = torch.linalg.eigh

    def _counting_eigh(*args, **kwargs):
        eigh_calls[0] += 1
        return orig_eigh(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "eigh", _counting_eigh)
    base = mp4_gpu.batched_panel_physics_v4_torch(rets, liq, vol, rmt_window=30, top_k_rmt=5)
    calls_base = eigh_calls[0]
    assert calls_base > 0
    assert base["high_snr_context"].shape == (t_len, n_assets, 8)
    assert "extra_rmt_regime_features" not in base

    eigh_calls[0] = 0
    ext = mp4_gpu.batched_panel_physics_v4_torch(
        rets, liq, vol, rmt_window=30, top_k_rmt=5, return_extended=True
    )
    calls_ext = eigh_calls[0]
    assert calls_ext == calls_base, f"Se esperaba 0 eigh extra: {calls_ext} vs {calls_base}"

    # Todas las claves base son bit-a-bit idénticas
    for k in base:
        assert torch.equal(base[k], ext[k])

    assert mp4_gpu.RMT_REGIME_EXTRA_NAMES == ("rmt_gap_ratio_1_2", "rmt_top1_vel_21")
    assert ext["rmt_gap_ratio_1_2"].shape == (t_len,)
    assert ext["rmt_top1_vel_21"].shape == (t_len,)
    assert ext["extra_rmt_regime_features"].shape == (t_len, n_assets, 2)
    for k_ext in (
        "rmt_gap_ratio_1_2",
        "rmt_gap_ratio_1_2_raw",
        "rmt_top1_share",
        "rmt_top1_vel_21",
        "extra_rmt_regime_features",
    ):
        assert ext[k_ext].dtype == torch.float32
        assert torch.isfinite(ext[k_ext]).all()
        assert float(ext[k_ext].abs().max()) <= mp4_gpu.LIMIT

    # 2) Recuperación económica: transición 1-factor -> 2-factor desploma gap_ratio y top1_vel!
    assert float(ext["rmt_gap_ratio_1_2_raw"][70]) > float(ext["rmt_gap_ratio_1_2_raw"][120]) + 1.0
    assert float(ext["rmt_gap_ratio_1_2"][95:125].min()) < -1.0
    assert float(ext["rmt_top1_vel_21"][90:120].min()) < -0.05

    # 3) Paridad exacta dual WxW vs full NxN en `batched_rmt_clean_correlation_torch(..., True)`
    win_dual = torch.from_numpy(rets[40:65])  # W=25 < N=40 -> ruta dual WxW
    r_dual = mp4_gpu.batched_rmt_clean_correlation_torch(
        win_dual,
        top_k=3,
        return_clean_corr=False,
        compute_dtype=torch.float64,
        return_extended=True,
    )
    # Inyectar un único NaN en activo 0 en t=0 para forzar ruta full NxN en otro subconjunto
    win_no_nan_full = win_dual[:, :20]  # W=25 >= N=20 -> ruta full NxN
    r_full_chk = mp4_gpu.batched_rmt_clean_correlation_torch(
        win_no_nan_full, top_k=3, return_clean_corr=False, return_extended=True
    )
    assert "rmt_gap_ratio_1_2" in r_dual and "rmt_top1_share" in r_dual
    assert "rmt_gap_ratio_1_2" in r_full_chk and "rmt_top1_share" in r_full_chk
    # Comparación directa dual vs full sobre la misma ventana W=25, N=40 vía `_rmt_full_nxn_path`
    z_d = (win_dual.double() - win_dual.double().mean(0, keepdim=True)) / (
        win_dual.double().std(0, unbiased=True, keepdim=True)
    )
    ref_full = mp4_gpu._rmt_full_nxn_path(
        z_d.unsqueeze(0),
        torch.ones((1, 25, n_assets), dtype=torch.bool),
        torch.tensor([25.0], dtype=torch.float64),
        top_k=3,
        return_clean_corr=False,
        min_overlap=5,
    )
    torch.testing.assert_close(
        r_dual["rmt_gap_ratio_1_2"], ref_full["rmt_gap_ratio_1_2"][0].float(), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        r_dual["rmt_top1_share"], ref_full["rmt_top1_share"][0].float(), atol=1e-5, rtol=1e-5
    )

    # 4) Causalidad estricta bit-a-bit (mutar t > t_cut -> diff == 0.0 en <= t_cut)
    t_cut = 88
    rets_mut = rets.copy()
    rets_mut[t_cut + 1 :] = -5.0 * rets_mut[t_cut + 1 :]
    ext_mut = mp4_gpu.batched_panel_physics_v4_torch(
        rets_mut, liq, vol, rmt_window=30, top_k_rmt=5, return_extended=True
    )
    for k_ext in ("rmt_gap_ratio_1_2", "rmt_top1_vel_21", "extra_rmt_regime_features"):
        assert torch.equal(ext[k_ext][: t_cut + 1], ext_mut[k_ext][: t_cut + 1])

    # 5) Paridad bajo BF16 autocast + auditoría AST cero syncs en helper nuevo
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        ext_bf16 = mp4_gpu.batched_panel_physics_v4_torch(
            rets, liq, vol, rmt_window=30, top_k_rmt=5, return_extended=True
        )
    torch.testing.assert_close(
        ext_bf16["extra_rmt_regime_features"],
        ext["extra_rmt_regime_features"],
        atol=1e-4,
        rtol=1e-4,
    )

    tree = ast.parse(inspect.getsource(mp4_gpu))
    found_helper = False
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "compute_rmt_regime_rotation_features"
        ):
            found_helper = True
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute) and child.attr in {"item", "cpu", "nonzero"}:
                    raise AssertionError(f"Host-sync `{child.attr}` en {node.name}")
    assert found_helper


def test_enrich_session_store_opt_in_market_macro_indices_dense_ragged_bf16_and_causality(
    tmp_path: Path,
):
    """#231: `include_market_macro_indices` opt-in (dense/ragged, 0 regresiones, causal, BF16)."""
    from mars_titan.data.dataset_enrichment_sota import ENRICHED_SOTA_DIM, ENRICHED_SOTA_NAMES
    from mars_titan.data.market_indices_macro_sota import (
        KEEP_ASSET_INDEX_MACRO_DIM,
        KEEP_ASSET_INDEX_MACRO_NAMES,
        MARKET_REGIME_INDEX_DIM,
        MARKET_REGIME_INDEX_NAMES,
    )

    rng = np.random.default_rng(23101)
    t_len, n_assets = 85, 16
    o, h, lo, c, v = _synthetic_ohlcv_panel(rng, t_len=t_len, n_assets=n_assets)
    years = np.full(t_len, 2022, dtype=np.int64)
    years[-20:] = 2023

    # Dense store con `asset_ids` permutados en sesión impar para probar scatter/gather exacto
    ids = np.broadcast_to(np.arange(n_assets, dtype=np.int64)[None, :], (t_len, n_assets)).copy()
    ids[1::2] = ids[1::2, ::-1]
    base_arrs = _make_multimodal_arrays(rng, t_len=t_len, n_assets=n_assets)
    base_arrs["asset_ids"] = ids
    base_arrs["macro"] = rng.standard_normal((t_len, n_assets, 12)).astype(np.float32)
    base_arrs["modality_mask"] = np.ones((t_len, n_assets, 5), dtype=np.float32)

    o_st, h_st, l_st, c_st, v_st = (a.copy() for a in (o, h, lo, c, v))
    for arr in (o_st, h_st, l_st, c_st, v_st):
        arr[1::2] = arr[1::2, ::-1]
    ohlcv_store = dict(open=o_st, high=h_st, low=l_st, close=c_st, volume=v_st)

    src_dir = tmp_path / "src_dense_231"
    mp4_gpu.ZeroCopySessionStore.save(src_dir, session_years=years, arrays=base_arrs)
    store = mp4_gpu.ZeroCopySessionStore.load(src_dir, mmap_mode="r")

    # 1) Default `include_market_macro_indices=False`: 100% retrocompatible (24 dims, sin regime)
    st_off, rep_off = store.with_sota_enrichment(
        tmp_path / "out_off", ohlcv=ohlcv_store, apply_missingness_featurizer=True
    )
    assert rep_off["n_enriched_features"] == ENRICHED_SOTA_DIM == 24
    assert st_off.raw_array("enriched_sota").shape == (t_len, n_assets, 24)
    assert st_off.market_regime_indices is None
    assert st_off.session_keys() == ()
    assert "market_regime_indices" not in st_off[0]

    # 2) Opt-in `include_market_macro_indices=True`: 24 + 7 KEEP = 31 dims + `[T, 18]` regime
    st_on, rep_on = store.with_sota_enrichment(
        tmp_path / "out_on",
        ohlcv=ohlcv_store,
        include_market_macro_indices=True,
        apply_missingness_featurizer=True,
        run_audit=True,
    )
    expected_dim = ENRICHED_SOTA_DIM + KEEP_ASSET_INDEX_MACRO_DIM
    assert expected_dim == 31
    assert rep_on["include_market_macro_indices"] is True
    assert rep_on["n_enriched_features"] == expected_dim
    assert (
        tuple(rep_on["enriched_sota_names"]) == ENRICHED_SOTA_NAMES + KEEP_ASSET_INDEX_MACRO_NAMES
    )
    assert "stock_rel_to_vw_5" not in rep_on["enriched_sota_names"]
    assert "stock_rel_to_vw_21" not in rep_on["enriched_sota_names"]
    assert rep_on["market_regime_index_dim"] == MARKET_REGIME_INDEX_DIM == 18
    assert tuple(rep_on["market_regime_index_names"]) == MARKET_REGIME_INDEX_NAMES
    assert rep_on["quality_audit"]["causality_check"]["bit_exact_causal"] is True
    assert rep_on["quality_audit"]["causality_check"]["max_abs_diff"] == 0.0
    assert rep_on["market_macro_audit"]["overall_passed"] is True
    assert rep_on["market_macro_audit"]["bf16_autocast_check"]["max_abs_diff"] == 0.0

    # Primeras 24 cols son bit-a-bit idénticas al modo `include_market_macro_indices=False`!
    np.testing.assert_array_equal(
        st_on.raw_array("enriched_sota")[..., :24], st_off.raw_array("enriched_sota")
    )
    assert st_on.raw_array("enriched_sota").shape == (t_len, n_assets, 31)
    assert st_on.raw_array("physics_v4").shape[-1] == base_arrs["physics_v4"].shape[-1] + 31

    # Acceso `[T, 18]` zero-copy en `market_regime_indices`, `raw_array`, `get_session`,
    # `get_slice`, `_iter_gpu_resident` y `iter_pinned_h2d`
    reg_arr = st_on.market_regime_indices
    assert reg_arr is not None and reg_arr.shape == (t_len, 18)
    assert np.isfinite(reg_arr).all()
    assert st_on.session_keys() == ("market_regime_indices",)
    assert st_on[5]["market_regime_indices"].shape == (18,)
    assert (
        st_on[5]["market_regime_indices"].data_ptr()
        == st_on.raw_array("market_regime_indices")[5].ctypes.data
    )
    assert st_on.get_slice(2, 7)["market_regime_indices"].shape == (5, 18)
    gpu_batches = list(st_on._iter_gpu_resident(torch.device("cpu"), np.array([0, 1, 2]), ()))
    assert len(gpu_batches) == 3 and gpu_batches[2]["market_regime_indices"].shape == (18,)
    h2d_batches = list(st_on.iter_pinned_h2d("cpu", indices=[0, 2, 4]))
    assert len(h2d_batches) == 3 and h2d_batches[1]["market_regime_indices"].shape == (18,)

    # 3) Paridad exacta BF16 autocast con `include_market_macro_indices=True`
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        st_bf, _ = store.with_sota_enrichment(
            tmp_path / "out_on_bf16",
            ohlcv=ohlcv_store,
            include_market_macro_indices=True,
            apply_missingness_featurizer=True,
        )
    np.testing.assert_array_equal(
        st_bf.raw_array("enriched_sota"), st_on.raw_array("enriched_sota")
    )
    np.testing.assert_array_equal(
        st_bf.raw_array("market_regime_indices"), st_on.raw_array("market_regime_indices")
    )

    # 4) Layout RAGGED con N_t variable: `[TotalRows, 31]` + session-level `[T, 18]` sin errores!
    rag_sessions = []
    rag_o, rag_h, rag_l, rag_c, rag_v = [], [], [], [], []
    for t_i in range(t_len):
        n_i = n_assets - (t_i % 3)
        rag_sessions.append({k: arr[t_i, :n_i] for k, arr in base_arrs.items()})
        rag_o.append(o_st[t_i, :n_i])
        rag_h.append(h_st[t_i, :n_i])
        rag_l.append(l_st[t_i, :n_i])
        rag_c.append(c_st[t_i, :n_i])
        rag_v.append(v_st[t_i, :n_i])
    st_rag = mp4_gpu.ZeroCopySessionStore.from_sessions(
        tmp_path / "src_rag_231", rag_sessions, years
    )
    ohlcv_rag = dict(
        open=np.concatenate(rag_o),
        high=np.concatenate(rag_h),
        low=np.concatenate(rag_l),
        close=np.concatenate(rag_c),
        volume=np.concatenate(rag_v),
    )
    st_rag_on, rep_rag = st_rag.with_sota_enrichment(
        tmp_path / "out_rag_on",
        ohlcv=ohlcv_rag,
        include_market_macro_indices=True,
        apply_missingness_featurizer=True,
    )
    assert st_rag_on.layout == "ragged"
    assert st_rag_on.raw_array("enriched_sota").shape[-1] == 31
    assert st_rag_on.market_regime_indices is not None
    assert st_rag_on.market_regime_indices.shape == (t_len, 18)
    assert st_rag_on[4]["market_regime_indices"].shape == (18,)
    assert st_rag_on[4]["enriched_sota"].shape == (n_assets - (4 % 3), 31)
    assert rep_rag["n_enriched_features"] == 31
    rag_gpu_b = list(st_rag_on._iter_gpu_resident(torch.device("cpu"), np.array([0, 1, 2]), ()))
    assert rag_gpu_b[1]["market_regime_indices"].shape == (18,)







