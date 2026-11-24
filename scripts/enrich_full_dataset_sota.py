#!/usr/bin/env python3
"""CLI oficial de enriquecimiento SOTA pre-2024 (`US 2000–2023` y `CN 2006–2023`).

Materializa y audita las señales causales ortogonales (`ENRICHED_SOTA_NAMES`) y la
featurización resiliente a ausencias (`MissingnessResilientFeaturizer`) sobre:
  - `--mode lab_cache`: cachés de laboratorio (`x3-us-2013-tabla`, `x4-us-2015`,
    `x3-cn-2013-real`), tanto con `X.npy` completo como esqueletos índice+objetivo
    unidos causalmente a `prepared-v3.1` / `encoded-history-v3.1`.
  - `--mode full_history`: historia completa `prepared-v3.1` + `encoded-history-v3.1` +
    `targets-v3.1` con ingestión streaming por bloques de activos y cálculo FP32 por slabs
    temporales de fundamentales para acotar RAM (`<4 GB` en US 24 años × 4.213 activos).

Invariantes estrictos (FM-8, sello temporal y fail-closed):
  1. Separación absoluta de mercados (`US` vs `CN`, jamás mezclados).
  2. Sello estricto `< 2024-01-01` (epoch day `< 19723`, incluyendo `target_available_at`).
  3. Jamás escribe dentro de directorios fuente de solo lectura.
  4. Fail-closed: si falla finitud, causalidad bit-exact, pantalla de fuga `<0.10` o estrés de
     ausencias, registra el `.json.gz`, NO emite `.npz` y devuelve código de salida `!= 0`.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import math
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from mars_titan.data.dataset_enrichment_sota import (  # noqa: E402
    ENCODED_TO_MODEL_MODALITY_INDEX,
    ENRICHED_SOTA_NAMES,
    LIMIT,
    MissingnessResilientFeaturizer,
    _causal_embedding_drift,
    _compute_avail_surprise_21,
    _compute_pit_fundamental_composites,
    audit_dataset_enrichment_quality,
    audit_laboratorio_cache_readonly,
    enrich_full_dataset_sota_torch,
)
from mars_titan.data.market_indices_macro_sota import (  # noqa: E402
    FACTOR_PORTFOLIO_NAMES,
    KEEP_ASSET_INDEX_MACRO_NAMES,
    MARKET_REGIME_INDEX_DIM,
    MARKET_REGIME_INDEX_NAMES,
    audit_market_indices_macro_value,
    compute_market_indices_and_macro_sota_torch,
    select_keep_asset_index_macro_features,
)
from mars_titan.data.market_physics_v4_gpu import gaussian_rank_features  # noqa: E402

VALID_MARKETS: tuple[str, ...] = ("US", "CN")
VALID_MODES: tuple[str, ...] = ("lab_cache", "full_history")
PRE_2024_EPOCH_DAY_LIMIT: int = 19723  # 2024-01-01
PRE_2024_DATE_LIMIT: str = "2024-01-01"
EPOCH_ORIGIN = datetime.date(1970, 1, 1)

DEFAULT_LAB_CACHES: dict[str, Path] = {
    "US": Path("/home/gpic/.local/state/mars-titan/laboratorio/x3-us-2013-tabla"),
    "CN": Path("/home/gpic/.local/state/mars-titan/laboratorio/x3-cn-2013-real"),
}
DEFAULT_MATERIALIZATION_ROOT = Path(
    "/home/gonzalo/.local/state/mars-titan/history-2000-20261007/materialization"
)


def _epoch_day_to_iso(day: int) -> str:
    return (EPOCH_ORIGIN + datetime.timedelta(days=int(day))).isoformat()


def _to_json_safe(obj: Any) -> Any:
    """Convierte recursivamente `ndarray`, escalares NumPy, tensores y `Path` a JSON puro."""
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _iso_to_epoch_day(date_str: str) -> int:
    return (datetime.date.fromisoformat(date_str[:10]) - EPOCH_ORIGIN).days


def _guard_output_not_inside_sources(
    output_dir: Path | None,
    source_dirs: Sequence[Path | None],
) -> None:
    if output_dir is None:
        return
    out_res = output_dir.resolve()
    for src in source_dirs:
        if src is None:
            continue
        src_res = src.resolve()
        if out_res == src_res or src_res in out_res.parents:
            raise ValueError(
                f"Directorio fuente de solo lectura protegido: output_dir={out_res} "
                f"no puede coincidir ni estar dentro de {src_res}"
            )


def _strip_market_prefix(asset_entry: str, market: str) -> str:
    prefix = f"{market}/"
    if "/" in asset_entry:
        ent_mkt, tkr = asset_entry.split("/", 1)
        if ent_mkt.upper() != market:
            raise ValueError(
                f"FM-8 violado: activo {asset_entry!r} no pertenece a --market {market}"
            )
        return tkr
    return asset_entry[len(prefix) :] if asset_entry.startswith(prefix) else asset_entry


def _audit_lab_cache_general(cache_dir: Path) -> dict[str, Any]:
    """Audita caché de laboratorio completa (`X.npy`) o esqueleto (`x4-us-2015` sin `X.npy`)."""
    day_path = cache_dir / "day.npy"
    bounds_path = cache_dir / "bounds.npy"
    y_path = cache_dir / "y.npy"
    if not day_path.is_file() or not bounds_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(f"Caché de laboratorio incompleta: {cache_dir}")

    days = np.load(day_path, mmap_mode="r")
    min_day = int(np.min(days)) if days.size else 0
    max_day = int(np.max(days)) if days.size else 0
    if max_day >= PRE_2024_EPOCH_DAY_LIMIT:
        raise ValueError(
            f"Partición 2024+ sellada detectada en {cache_dir} "
            f"(max_day={max_day} >= {PRE_2024_EPOCH_DAY_LIMIT}): rechazado"
        )

    has_x = (cache_dir / "X.npy").is_file()
    has_x3 = (cache_dir / "X3.npy").is_file()
    if has_x:
        rep = audit_laboratorio_cache_readonly(cache_dir)
        rep["has_x_npy"] = True
        rep["has_x3_npy"] = has_x3
        rep["min_date"] = _epoch_day_to_iso(min_day)
        rep["max_date"] = _epoch_day_to_iso(max_day)
        return rep

    bounds = np.load(bounds_path, mmap_mode="r")
    sizes = np.diff(bounds)
    y_mm = np.load(y_path, mmap_mode="r")
    n_rows = int(y_mm.shape[0])
    step = max(1, n_rows // 200_000)
    ys = np.asarray(y_mm[::step], dtype=np.float64)
    return dict(
        cache_dir=str(cache_dir),
        strictly_pre_2024=True,
        min_day_epoch=min_day,
        max_day_epoch=max_day,
        min_date=_epoch_day_to_iso(min_day),
        max_date=_epoch_day_to_iso(max_day),
        n_rows=n_rows,
        n_features_x=0,
        has_x_npy=False,
        has_x3_npy=has_x3,
        n_sessions=int(len(bounds) - 1),
        session_size_min=int(sizes.min()) if sizes.size else 0,
        session_size_median=int(np.median(sizes)) if sizes.size else 0,
        session_size_max=int(sizes.max()) if sizes.size else 0,
        y_nan_rate=float(np.mean(~np.isfinite(ys))) if ys.size else 0.0,
        y_std=float(np.nanstd(ys)) if ys.size else 0.0,
    )


def _extract_fixed_list_2d(col: Any, width: int) -> np.ndarray:
    """Convierte una columna Arrow `fixed_size_list<float>[width]` a `ndarray[N, width]` rápido."""
    combined = col.combine_chunks()
    n_rows = len(combined)
    if n_rows == 0:
        return np.empty((0, width), dtype=np.float32)
    flat = combined.values.to_numpy(zero_copy_only=False)
    return np.asarray(flat, dtype=np.float32).reshape(n_rows, width)


def _extract_fixed_bool_2d(col: Any, width: int) -> np.ndarray:
    combined = col.combine_chunks()
    n_rows = len(combined)
    if n_rows == 0:
        return np.empty((0, width), dtype=bool)
    flat = combined.values.to_numpy(zero_copy_only=False)
    return np.asarray(flat, dtype=bool).reshape(n_rows, width)


def _timestamp_is_sealed_2024(ts_val: Any) -> bool:
    if ts_val is None:
        return False
    if isinstance(ts_val, datetime.datetime):
        return ts_val.date().isoformat() >= PRE_2024_DATE_LIMIT
    if isinstance(ts_val, datetime.date):
        return ts_val.isoformat() >= PRE_2024_DATE_LIMIT
    return str(ts_val)[:10] >= PRE_2024_DATE_LIMIT


def _ingest_tickers_streaming(
    *,
    market: str,
    tickers: Sequence[str],
    session_dates: Sequence[str],
    materialization_root: Path,
    asset_chunk_size: int,
    device: torch.device,
    include_targets: bool = True,
    fund_slab_mb: float = 512.0,
    scratch_dir: Path | None = None,
) -> dict[str, Any]:
    """Ingiere OHLCV y modalidades PIT por bloques reales de activos en FP32 con RAM acotada."""
    t_len = len(session_dates)
    n_assets = len(tickers)
    date_to_idx = {d: i for i, d in enumerate(session_dates)}

    open_p = np.full((t_len, n_assets), np.nan, dtype=np.float32)
    high_p = np.full((t_len, n_assets), np.nan, dtype=np.float32)
    low_p = np.full((t_len, n_assets), np.nan, dtype=np.float32)
    close_p = np.full((t_len, n_assets), np.nan, dtype=np.float32)
    vol_p = np.zeros((t_len, n_assets), dtype=np.float32)
    target_p = np.full((t_len, n_assets), np.nan, dtype=np.float32)

    news_sig_p = np.zeros((t_len, n_assets), dtype=np.float32)
    chart_drift_p = np.zeros((t_len, n_assets), dtype=np.float32)
    news_mask_p = np.zeros((t_len, n_assets), dtype=bool)
    charts_mask_p = np.zeros((t_len, n_assets), dtype=bool)
    fund_mask_p = np.zeros((t_len, n_assets), dtype=bool)
    macro_mask_p = np.zeros((t_len, n_assets), dtype=bool)

    # Fundamentales SIEMPRE en FP32 exacto (cero cuantización FP16!): memmap temporal si excede RAM
    fund_budget_bytes = max(1, int(float(fund_slab_mb) * 1024 * 1024))
    fund_bytes_f32 = t_len * n_assets * 78 * 4
    use_fund_memmap = scratch_dir is not None and fund_bytes_f32 > fund_budget_bytes
    if use_fund_memmap:
        assert scratch_dir is not None
        fund_mm_path = scratch_dir / "fundamentals_f32.mmap"
        fund_p: np.ndarray = np.memmap(
            fund_mm_path, dtype=np.float32, mode="w+", shape=(t_len, n_assets, 78)
        )
        fund_p[:] = 0.0
    else:
        fund_p = np.zeros((t_len, n_assets, 78), dtype=np.float32)

    macro_p = np.full((t_len, 420), np.nan, dtype=np.float32)
    macro_filled = np.zeros(t_len, dtype=bool)

    prep_base = materialization_root / "prepared-v3.1" / "prepared" / market
    enc_base = materialization_root / "encoded-history-v3.1" / "samples" / market
    tgt_base = materialization_root / "targets-v3.1" / "labels" / market

    n_sealed_2024_dropped = 0
    n_sealed_2024_targets_dropped = 0
    n_prices_loaded = 0
    n_encoded_loaded = 0
    chunk_sz = max(1, int(asset_chunk_size))

    for c0 in range(0, n_assets, chunk_sz):
        c1 = min(c0 + chunk_sz, n_assets)
        c_len = c1 - c0
        # Buffers acotados al bloque `[T, c_len, D]` para vectorizar deriva en streaming real!
        news_chunk = np.zeros((t_len, c_len, 384), dtype=np.float32)
        news_ok_chunk = np.zeros((t_len, c_len), dtype=bool)
        charts_chunk = np.zeros((t_len, c_len, 512), dtype=np.float32)
        charts_ok_chunk = np.zeros((t_len, c_len), dtype=bool)
        any_enc_in_chunk = False

        for loc_j, j in enumerate(range(c0, c1)):
            tkr = tickers[j]
            tkr_sealed_2024 = 0
            p_file = prep_base / tkr / "prices.parquet"
            if p_file.is_file():
                ptbl = pq.read_table(
                    p_file, columns=["session", "open", "high", "low", "close", "volume"]
                )
                s_vals = ptbl["session"].to_pylist()
                o_arr = ptbl["open"].to_numpy(zero_copy_only=False)
                h_arr = ptbl["high"].to_numpy(zero_copy_only=False)
                l_arr = ptbl["low"].to_numpy(zero_copy_only=False)
                c_arr = ptbl["close"].to_numpy(zero_copy_only=False)
                v_arr = ptbl["volume"].to_numpy(zero_copy_only=False)
                rows_src: list[int] = []
                rows_dst: list[int] = []
                for r_i, s_str in enumerate(s_vals):
                    if s_str is None:
                        continue
                    s10 = str(s_str)[:10]
                    if s10 >= PRE_2024_DATE_LIMIT:
                        tkr_sealed_2024 += 1
                        continue
                    d_idx = date_to_idx.get(s10)
                    if d_idx is not None:
                        rows_src.append(r_i)
                        rows_dst.append(d_idx)
                n_sealed_2024_dropped += tkr_sealed_2024
                if rows_src:
                    src_i = np.asarray(rows_src, dtype=np.int64)
                    dst_i = np.asarray(rows_dst, dtype=np.int64)
                    open_p[dst_i, j] = o_arr[src_i].astype(np.float32, copy=False)
                    high_p[dst_i, j] = h_arr[src_i].astype(np.float32, copy=False)
                    low_p[dst_i, j] = l_arr[src_i].astype(np.float32, copy=False)
                    close_p[dst_i, j] = c_arr[src_i].astype(np.float32, copy=False)
                    vol_p[dst_i, j] = np.nan_to_num(v_arr[src_i], nan=0.0).astype(
                        np.float32, copy=False
                    )
                    n_prices_loaded += 1

            e_file = enc_base / tkr / "samples.parquet"
            if e_file.is_file():
                etbl = pq.read_table(
                    e_file,
                    columns=["session", "presence", "fundamentals", "macro", "news", "charts"],
                )
                es_vals = etbl["session"].to_pylist()
                er_src: list[int] = []
                er_dst: list[int] = []
                for r_i, s_str in enumerate(es_vals):
                    if s_str is None:
                        continue
                    s10 = str(s_str)[:10]
                    if s10 >= PRE_2024_DATE_LIMIT:
                        continue
                    d_idx = date_to_idx.get(s10)
                    if d_idx is not None:
                        er_src.append(r_i)
                        er_dst.append(d_idx)
                if er_src:
                    any_enc_in_chunk = True
                    src_i = np.asarray(er_src, dtype=np.int64)
                    dst_i = np.asarray(er_dst, dtype=np.int64)
                    pres_enc = _extract_fixed_bool_2d(etbl["presence"], 5)[src_i]
                    pres_mod = pres_enc[:, list(ENCODED_TO_MODEL_MODALITY_INDEX)]
                    f_ok = pres_mod[:, 1]
                    m_ok = pres_mod[:, 2]
                    n_ok = pres_mod[:, 3]
                    c_ok = pres_mod[:, 4]

                    fund_mask_p[dst_i, j] = f_ok
                    macro_mask_p[dst_i, j] = m_ok
                    news_ok_chunk[dst_i, loc_j] = n_ok
                    charts_ok_chunk[dst_i, loc_j] = c_ok

                    fund_raw = _extract_fixed_list_2d(etbl["fundamentals"], 78)[src_i]
                    fund_p[dst_i, j] = fund_raw

                    need_m = m_ok & (~macro_filled[dst_i])
                    if bool(need_m.any()):
                        mac_raw = _extract_fixed_list_2d(etbl["macro"], 420)[src_i]
                        fill_dst = dst_i[need_m]
                        macro_p[fill_dst] = mac_raw[need_m]
                        macro_filled[fill_dst] = True

                    news_chunk[dst_i, loc_j] = _extract_fixed_list_2d(etbl["news"], 384)[src_i]
                    charts_chunk[dst_i, loc_j] = _extract_fixed_list_2d(etbl["charts"], 512)[src_i]

                    if include_targets:
                        t_file = tgt_base / tkr / "labels.parquet"
                        if t_file.is_file():
                            t_schema_names = set(pq.read_schema(t_file).names)
                            has_avail = "target_available_at" in t_schema_names
                            cols_t = ["sample_row", "target"] + (
                                ["target_available_at"] if has_avail else []
                            )
                            ttbl = pq.read_table(t_file, columns=cols_t)
                            s_rows = ttbl["sample_row"].to_numpy(zero_copy_only=False)
                            t_vals = ttbl["target"].to_numpy(zero_copy_only=False).astype(
                                np.float32, copy=False
                            )
                            avail_list = (
                                ttbl["target_available_at"].to_pylist()
                                if has_avail
                                else [None] * len(s_rows)
                            )
                            row_map = {
                                int(sr): int(di) for sr, di in zip(er_src, er_dst, strict=True)
                            }
                            last_pre_dst = int(dst_i.max())
                            for sr_i, tv, av_ts in zip(s_rows, t_vals, avail_list, strict=True):
                                d_i = row_map.get(int(sr_i))
                                if d_i is None or not np.isfinite(tv):
                                    continue
                                # P0-1: Sello estricto sobre `target_available_at >= 2024-01-01`
                                if has_avail:
                                    if av_ts is None or _timestamp_is_sealed_2024(av_ts):
                                        n_sealed_2024_targets_dropped += 1
                                        continue
                                elif tkr_sealed_2024 > 0 and d_i == last_pre_dst:
                                    n_sealed_2024_targets_dropped += 1
                                    continue
                                target_p[d_i, j] = float(tv)
                    n_encoded_loaded += 1

        if any_enc_in_chunk:
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=False):
                nt = torch.from_numpy(news_chunk).to(device=device)
                nm = torch.from_numpy(news_ok_chunk).to(device=device)
                fin_n = torch.isfinite(nt)
                n_valid = nm & fin_n.any(dim=-1)
                n_drift = _causal_embedding_drift(nt, n_valid, halflife=5.0)
                n_dir = torch.where(fin_n, nt, 0.0).mean(dim=-1)
                n_sig = torch.where(n_valid, torch.sign(n_dir) * n_drift, 0.0)
                news_sig_p[:, c0:c1] = n_sig.detach().cpu().numpy()
                # P1-3: escribir máscara real `n_valid` (descarta embeddings todo-NaN!)
                news_mask_p[:, c0:c1] = n_valid.detach().cpu().numpy()

                ct = torch.from_numpy(charts_chunk).to(device=device)
                cm = torch.from_numpy(charts_ok_chunk).to(device=device)
                c_valid = cm & torch.isfinite(ct).any(dim=-1)
                c_drift = _causal_embedding_drift(ct, c_valid, halflife=10.0)
                c_sig = torch.where(c_valid, c_drift, 0.0)
                chart_drift_p[:, c0:c1] = c_sig.detach().cpu().numpy()
                charts_mask_p[:, c0:c1] = c_valid.detach().cpu().numpy()

    # Fallback estrictamente pre-2024 cuando falten labels (última fila siempre NaN)
    if include_targets and not bool(np.isfinite(target_p).any()) and t_len > 1:
        c_ok = np.isfinite(close_p) & (close_p > 0.0)
        target_p[:-1] = np.where(
            c_ok[1:] & c_ok[:-1],
            np.log(np.maximum(close_p[1:], 1e-12) / np.maximum(close_p[:-1], 1e-12)),
            np.nan,
        )

    return dict(
        open=open_p,
        high=high_p,
        low=low_p,
        close=close_p,
        volume=vol_p,
        target=target_p,
        fundamentals=fund_p,
        fundamentals_mask=fund_mask_p,
        macro=macro_p,
        macro_mask=macro_mask_p,
        news_sig=news_sig_p,
        news_mask=news_mask_p,
        chart_drift=chart_drift_p,
        charts_mask=charts_mask_p,
        n_sealed_2024_rows_dropped=n_sealed_2024_dropped,
        n_sealed_2024_targets_dropped=n_sealed_2024_targets_dropped,
        n_prices_loaded=n_prices_loaded,
        n_encoded_loaded=n_encoded_loaded,
        fund_uses_memmap=use_fund_memmap,
    )


def _enrich_with_fp32_fundamental_slabs(
    panel: Mapping[str, Any],
    *,
    device: torch.device,
    fund_slab_mb: float = 512.0,
    staleness_halflife_days: float = 63.0,
) -> np.ndarray:
    """Calcula las `ENRICHED_SOTA_DIM` señales SOTA en FP32 exacto con fundamentales por slabs."""
    o_p = panel["open"]
    h_p = panel["high"]
    l_p = panel["low"]
    c_p = panel["close"]
    v_p = panel["volume"]
    t_len, n_assets = c_p.shape

    base_mod_kw: dict[str, Any] = dict(
        fundamentals=None,
        fundamentals_mask=None,
        news=panel["news_sig"],
        news_mask=panel["news_mask"],
        charts=panel["chart_drift"],
        charts_mask=panel["charts_mask"],
        macro=panel["macro"],
        macro_mask=panel["macro_mask"],
        staleness_halflife_days=staleness_halflife_days,
    )
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=False):
        enriched_t = enrich_full_dataset_sota_torch(
            torch.from_numpy(np.ascontiguousarray(o_p)).to(device=device),
            torch.from_numpy(np.ascontiguousarray(h_p)).to(device=device),
            torch.from_numpy(np.ascontiguousarray(l_p)).to(device=device),
            torch.from_numpy(np.ascontiguousarray(c_p)).to(device=device),
            torch.from_numpy(np.ascontiguousarray(v_p)).to(device=device),
            **base_mod_kw,
        )
    enriched_f32 = enriched_t.detach().cpu().numpy().astype(np.float32, copy=False)

    fund_src = panel["fundamentals"]
    if fund_src is None or t_len == 0 or n_assets == 0:
        return enriched_f32

    fund_mask_src = panel["fundamentals_mask"]
    staleness_scale = math.log1p(max(staleness_halflife_days, 1.0)) / math.log(2.0)

    # Precomputar `traded`, `log_c` y `ret_63` en FP32 exactamente igual a `dataset_enrichment_sota`
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=False):
        c_t = torch.from_numpy(np.ascontiguousarray(c_p)).to(device=device, dtype=torch.float32)
        v_t = torch.from_numpy(np.ascontiguousarray(v_p)).to(device=device, dtype=torch.float32)
        px_ok = torch.isfinite(c_t) & (c_t > 0.0)
        traded = px_ok & torch.isfinite(v_t) & (v_t > 0.0)
        c_safe = torch.where(px_ok, c_t, 1e-12).clamp_min(1e-12)
        log_c = torch.log(c_safe)
        ret_63 = torch.full((t_len, n_assets), float("nan"), dtype=torch.float32, device=device)
        if t_len > 63:
            ok_63 = traded[63:] & traded[:-63]
            ret_63[63:] = torch.where(ok_63, log_c[63:] - log_c[:-63], float("nan"))

        budget_bytes = max(1, int(float(fund_slab_mb) * 1024 * 1024))
        bytes_per_sess = max(n_assets * 78 * 4, 1)
        slab_t = max(1, min(t_len, budget_bytes // bytes_per_sess))

        idx_gap = ENRICHED_SOTA_NAMES.index("price_vs_fund_quality_gap")
        idx_qual = ENRICHED_SOTA_NAMES.index("pit_fund_quality")
        idx_val = ENRICHED_SOTA_NAMES.index("pit_fund_value")
        idx_prof = ENRICHED_SOTA_NAMES.index("pit_fund_profitability")
        idx_avail = ENRICHED_SOTA_NAMES.index("avail_surprise_21")
        fund_valid_all = torch.zeros((t_len, n_assets), dtype=torch.bool, device=device)

        for t0 in range(0, t_len, slab_t):
            t1 = min(t0 + slab_t, t_len)
            f_slab = torch.from_numpy(np.ascontiguousarray(fund_src[t0:t1])).to(
                device=device, dtype=torch.float32
            )
            fm_slab = (
                torch.from_numpy(np.ascontiguousarray(fund_mask_src[t0:t1])).to(device=device)
                if fund_mask_src is not None
                else None
            )
            pit_q, pit_v, pit_p, f_valid, f_fresh = _compute_pit_fundamental_composites(
                f_slab,
                fundamentals_mask=fm_slab,
                fundamentals_age=None,
                log_close=log_c[t0:t1],
                fundamental_layout="encoded_v31_78",
                staleness_scale=staleness_scale,
                compute_dtype=torch.float32,
            )
            fund_valid_all[t0:t1] = f_valid
            both_pf = traded[t0:t1] & torch.isfinite(ret_63[t0:t1]) & f_valid
            g_r63 = gaussian_rank_features(
                torch.where(both_pf, ret_63[t0:t1], 0.0).unsqueeze(-1),
                mask=both_pf.unsqueeze(-1),
                clip_val=LIMIT,
            ).squeeze(-1)
            pf_gap = torch.where(both_pf, g_r63 * f_fresh - pit_q, 0.0)

            for col_i, ten in (
                (idx_gap, pf_gap),
                (idx_qual, pit_q),
                (idx_val, pit_v),
                (idx_prof, pit_p),
            ):
                clean = torch.nan_to_num(ten, nan=0.0, posinf=LIMIT, neginf=-LIMIT).clamp(
                    -LIMIT, LIMIT
                )
                enriched_f32[t0:t1, :, col_i] = clean.detach().cpu().numpy()

        # Recalcular `avail_surprise_21` con `fund_valid_all` real para paridad bit-exact 24/24!
        news_ok_t: torch.Tensor | None = None
        if panel["news_sig"] is not None:
            ns_t = torch.from_numpy(np.ascontiguousarray(panel["news_sig"])).to(
                device=device, dtype=torch.float32
            )
            nm_t = (
                torch.from_numpy(np.ascontiguousarray(panel["news_mask"])).to(
                    device=device, dtype=torch.bool
                )
                if panel["news_mask"] is not None
                else torch.ones((t_len, n_assets), dtype=torch.bool, device=device)
            )
            news_ok_t = nm_t & torch.isfinite(ns_t)
        ch_ok_t: torch.Tensor | None = None
        if panel["chart_drift"] is not None:
            cd_t = torch.from_numpy(np.ascontiguousarray(panel["chart_drift"])).to(
                device=device, dtype=torch.float32
            )
            cm_t = (
                torch.from_numpy(np.ascontiguousarray(panel["charts_mask"])).to(
                    device=device, dtype=torch.bool
                )
                if panel["charts_mask"] is not None
                else torch.ones((t_len, n_assets), dtype=torch.bool, device=device)
            )
            ch_ok_t = cm_t & torch.isfinite(cd_t)
        avail_s = _compute_avail_surprise_21(
            fund_valid_all,
            news_ok_t,
            ch_ok_t,
            traded=traded,
            compute_dtype=torch.float32,
        )
        avail_clean = torch.nan_to_num(avail_s, nan=0.0, posinf=LIMIT, neginf=-LIMIT).clamp(
            -LIMIT, LIMIT
        )
        enriched_f32[:, :, idx_avail] = avail_clean.detach().cpu().numpy()

    return enriched_f32


def _derive_panel_from_lab_x_only(
    cache_dir: Path,
    *,
    tickers: Sequence[str],
    asset_indices_kept: np.ndarray,
    session_epochs: np.ndarray,
) -> dict[str, Any]:
    """Respaldo causal cuando `lab_cache` se ejecuta sin `materialization-root` (tests aislados)."""
    t_len = len(session_epochs)
    n_assets = len(tickers)
    epoch_to_t = {int(e): i for i, e in enumerate(session_epochs)}
    asset_to_col = {int(a): i for i, a in enumerate(asset_indices_kept)}

    days = np.load(cache_dir / "day.npy", mmap_mode="r")
    assets = np.load(cache_dir / "asset.npy", mmap_mode="r")
    y_mm = np.load(cache_dir / "y.npy", mmap_mode="r")
    x_mm = np.load(cache_dir / "X.npy", mmap_mode="r")
    nj = json.loads((cache_dir / "names.json").read_text(encoding="utf-8"))
    base_names: list[str] = list(nj.get("base", nj.get("names", [])))

    def _find_col(name: str, fallback: int) -> int:
        return base_names.index(name) if name in base_names else min(fallback, x_mm.shape[1] - 1)

    c_ret1 = _find_col("return_1", 0)
    c_range = _find_col("range_1", 1)
    c_dvol = _find_col("log_dollar_volume_21", 2)

    ret1_p = np.full((t_len, n_assets), np.nan, dtype=np.float32)
    rng_p = np.full((t_len, n_assets), 0.015, dtype=np.float32)
    dvol_p = np.full((t_len, n_assets), 13.0, dtype=np.float32)
    target_p = np.full((t_len, n_assets), np.nan, dtype=np.float32)

    keep_mask = np.isin(assets, asset_indices_kept) & np.isin(days, session_epochs)
    idx_rows = np.flatnonzero(keep_mask)
    if idx_rows.size:
        t_pos = np.asarray([epoch_to_t[int(d)] for d in days[idx_rows]], dtype=np.int64)
        a_pos = np.asarray([asset_to_col[int(a)] for a in assets[idx_rows]], dtype=np.int64)
        x_sel = np.asarray(x_mm[idx_rows], dtype=np.float32)
        ret1_p[t_pos, a_pos] = x_sel[:, c_ret1]
        rng_p[t_pos, a_pos] = np.clip(
            np.abs(np.nan_to_num(x_sel[:, c_range], nan=0.015)), 0.001, 0.15
        )
        dvol_p[t_pos, a_pos] = np.clip(np.nan_to_num(x_sel[:, c_dvol], nan=13.0), 4.0, 22.0)
        target_p[t_pos, a_pos] = np.asarray(y_mm[idx_rows], dtype=np.float32)

    valid = np.isfinite(ret1_p)
    r_clean = np.where(valid, np.clip(ret1_p, -0.25, 0.25), 0.0)
    close_p = np.where(valid, 100.0 * np.exp(np.cumsum(r_clean, axis=0)), np.nan).astype(np.float32)
    open_p = close_p.copy()
    if t_len > 1:
        open_p[1:] = np.where(valid[1:] & valid[:-1], close_p[:-1], close_p[1:])
    high_p = (np.maximum(open_p, close_p) * np.exp(0.5 * rng_p)).astype(np.float32)
    low_p = (np.minimum(open_p, close_p) * np.exp(-0.5 * rng_p)).astype(np.float32)
    c_denom = np.maximum(np.nan_to_num(close_p, nan=100.0), 1.0)
    vol_p = np.where(valid, np.exp(dvol_p) / c_denom, 0.0).astype(np.float32)
    return dict(
        open=open_p,
        high=high_p,
        low=low_p,
        close=close_p,
        volume=vol_p,
        target=target_p,
        fundamentals=None,
        fundamentals_mask=None,
        macro=None,
        macro_mask=None,
        news_sig=None,
        news_mask=None,
        chart_drift=None,
        charts_mask=None,
        n_sealed_2024_rows_dropped=0,
        n_sealed_2024_targets_dropped=0,
        n_prices_loaded=n_assets,
        n_encoded_loaded=0,
        fund_uses_memmap=False,
    )


def _discover_full_history_calendar(
    *,
    market: str,
    tickers: Sequence[str],
    materialization_root: Path,
    asset_chunk_size: int,
) -> list[str]:
    """Descubre la unión cronológica estrictamente `< 2024-01-01` por bloques de activos."""
    prep_base = materialization_root / "prepared-v3.1" / "prepared" / market
    date_set: set[str] = set()
    chunk_sz = max(1, int(asset_chunk_size))
    for c0 in range(0, len(tickers), chunk_sz):
        for tkr in tickers[c0 : min(c0 + chunk_sz, len(tickers))]:
            p_file = prep_base / tkr / "prices.parquet"
            if not p_file.is_file():
                continue
            tbl = pq.read_table(p_file, columns=["session"])
            for s_val in tbl["session"].to_pylist():
                if s_val is not None:
                    s10 = str(s_val)[:10]
                    if s10 < PRE_2024_DATE_LIMIT:
                        date_set.add(s10)
    return sorted(date_set)


def run_enrichment_pipeline(
    *,
    mode: str,
    market: str,
    lab_cache_dir: str | Path | None = None,
    materialization_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    device: str = "cpu",
    max_assets: int = 0,
    max_sessions: int = 0,
    asset_chunk_size: int = 128,
    fund_slab_mb: float = 512.0,
    dtype: str = "fp32",
    dry_run: bool = False,
    audit_only: bool = False,
    allow_failed_audit: bool = False,
    include_market_macro_indices: bool = False,
) -> dict[str, Any]:
    """Ejecuta enriquecimiento SOTA causal, auditoría de calidad y emisión fail-closed."""
    t0 = time.perf_counter()
    mkt = str(market).strip().upper()
    if mkt not in VALID_MARKETS:
        raise ValueError(f"FM-8: --market inválido {market!r}; válidos: {VALID_MARKETS}")
    if mode not in VALID_MODES:
        raise ValueError(f"--mode inválido {mode!r}; válidos: {VALID_MODES}")
    if dtype not in ("fp32", "fp16"):
        raise ValueError(f"--dtype inválido {dtype!r}; válidos: ('fp32', 'fp16')")

    out_path = Path(output_dir) if output_dir is not None else None
    if not dry_run and out_path is None:
        raise ValueError("--output-dir es obligatorio salvo en --dry-run")

    mat_root = (
        Path(materialization_root)
        if materialization_root is not None
        else DEFAULT_MATERIALIZATION_ROOT
    )
    lab_dir = (
        Path(lab_cache_dir)
        if lab_cache_dir is not None
        else (DEFAULT_LAB_CACHES[mkt] if mode == "lab_cache" else None)
    )
    _guard_output_not_inside_sources(out_path, (lab_dir, mat_root))

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    lab_cache_audit: dict[str, Any] | None = None
    tag_name = mkt.lower()

    with tempfile.TemporaryDirectory(prefix="mars_enr_slab_") as tmp_dir_str:
        scratch_dir = Path(tmp_dir_str)
        if mode == "lab_cache":
            assert lab_dir is not None
            if not lab_dir.is_dir():
                raise FileNotFoundError(f"No existe --lab-cache: {lab_dir}")
            nj = json.loads((lab_dir / "names.json").read_text(encoding="utf-8"))
            cache_mkt = str(nj.get("market", "")).strip().upper()
            if cache_mkt and cache_mkt != mkt:
                raise ValueError(
                    f"FM-8 violado: caché {lab_dir} pertenece a market={cache_mkt!r}, "
                    f"pero se solicitó --market {mkt!r}"
                )
            lab_cache_audit = _audit_lab_cache_general(lab_dir)
            tag_name = lab_dir.name

            days_mm = np.load(lab_dir / "day.npy", mmap_mode="r")
            assets_mm = np.load(lab_dir / "asset.npy", mmap_mode="r")
            y_mm = np.load(lab_dir / "y.npy", mmap_mode="r")
            raw_asset_names: list[str] = list(nj.get("assets", []))
            if not raw_asset_names:
                raise ValueError(f"names.json sin lista `assets` en {lab_dir}")

            uniq_epochs = np.unique(days_mm)
            if int(max_sessions) > 0 and len(uniq_epochs) > int(max_sessions):
                uniq_epochs = uniq_epochs[-int(max_sessions) :]
            session_dates = [_epoch_day_to_iso(int(e)) for e in uniq_epochs]

            # P1-4: muestreo determinista por stride de ID (cero sesgo de supervivencia!)
            in_sess = np.isin(days_mm, uniq_epochs)
            all_asset_ids = np.sort(np.unique(assets_mm[in_sess]))
            if int(max_assets) > 0 and len(all_asset_ids) > int(max_assets):
                step_a = max(1, len(all_asset_ids) // int(max_assets))
                kept_asset_ids = all_asset_ids[::step_a][: int(max_assets)]
            else:
                kept_asset_ids = all_asset_ids
            tickers = [_strip_market_prefix(raw_asset_names[int(a)], mkt) for a in kept_asset_ids]

            has_mat = (mat_root / "prepared-v3.1" / "prepared" / mkt).is_dir()
            if has_mat:
                panel = _ingest_tickers_streaming(
                    market=mkt,
                    tickers=tickers,
                    session_dates=session_dates,
                    materialization_root=mat_root,
                    asset_chunk_size=asset_chunk_size,
                    device=dev,
                    include_targets=False,
                    fund_slab_mb=fund_slab_mb,
                    scratch_dir=scratch_dir,
                )
                t_len, n_a = len(uniq_epochs), len(kept_asset_ids)
                y_lab = np.full((t_len, n_a), np.nan, dtype=np.float32)
                row_sel = np.flatnonzero(in_sess & np.isin(assets_mm, kept_asset_ids))
                if row_sel.size:
                    t_pos = np.searchsorted(uniq_epochs, days_mm[row_sel])
                    a_pos = np.searchsorted(kept_asset_ids, assets_mm[row_sel])
                    y_lab[t_pos, a_pos] = np.asarray(y_mm[row_sel], dtype=np.float32)
                panel["target"] = y_lab
            elif lab_cache_audit["has_x_npy"]:
                panel = _derive_panel_from_lab_x_only(
                    lab_dir,
                    tickers=tickers,
                    asset_indices_kept=kept_asset_ids,
                    session_epochs=uniq_epochs,
                )
            else:
                raise FileNotFoundError(
                    f"Caché {lab_dir} no contiene X.npy y falta materialization-root={mat_root}"
                )
        else:
            prep_dir = mat_root / "prepared-v3.1" / "prepared" / mkt
            if not prep_dir.is_dir():
                raise FileNotFoundError(f"No existe prepared-v3.1 para {mkt}: {prep_dir}")
            all_tickers = sorted(p.name for p in prep_dir.iterdir() if p.is_dir())
            if int(max_assets) > 0 and len(all_tickers) > int(max_assets):
                step_a = max(1, len(all_tickers) // int(max_assets))
                tickers = all_tickers[::step_a][: int(max_assets)]
            else:
                tickers = all_tickers
            session_dates = _discover_full_history_calendar(
                market=mkt,
                tickers=tickers,
                materialization_root=mat_root,
                asset_chunk_size=asset_chunk_size,
            )
            if int(max_sessions) > 0 and len(session_dates) > int(max_sessions):
                session_dates = session_dates[-int(max_sessions) :]
            if not session_dates:
                raise ValueError(f"Sin sesiones pre-2024 en {prep_dir}")
            panel = _ingest_tickers_streaming(
                market=mkt,
                tickers=tickers,
                session_dates=session_dates,
                materialization_root=mat_root,
                asset_chunk_size=asset_chunk_size,
                device=dev,
                include_targets=True,
                fund_slab_mb=fund_slab_mb,
                scratch_dir=scratch_dir,
            )

        assert all(d < PRE_2024_DATE_LIMIT for d in session_dates)
        t_len = len(session_dates)
        n_assets = len(tickers)

        base_enriched_f32 = _enrich_with_fp32_fundamental_slabs(
            panel, device=dev, fund_slab_mb=fund_slab_mb
        )
        market_regime_f32: np.ndarray | None = None
        factor_returns_f32: np.ndarray | None = None
        if include_market_macro_indices:
            with torch.no_grad(), torch.amp.autocast(device_type=dev.type, enabled=False):
                mims_out = compute_market_indices_and_macro_sota_torch(
                    torch.from_numpy(np.ascontiguousarray(panel["open"])).to(device=dev),
                    torch.from_numpy(np.ascontiguousarray(panel["high"])).to(device=dev),
                    torch.from_numpy(np.ascontiguousarray(panel["low"])).to(device=dev),
                    torch.from_numpy(np.ascontiguousarray(panel["close"])).to(device=dev),
                    torch.from_numpy(np.ascontiguousarray(panel["volume"])).to(device=dev),
                    macro=panel["macro"],
                    macro_mask=panel["macro_mask"],
                    macro_layout="encoded_v31_420",
                )
            keep_asset_f32 = (
                select_keep_asset_index_macro_features(mims_out["asset_index_macro_features"])
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            market_regime_f32 = (
                mims_out["market_regime_indices"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            factor_returns_f32 = (
                mims_out["factor_returns"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            enriched_f32 = np.concatenate([base_enriched_f32, keep_asset_f32], axis=-1)
            feature_names_out: tuple[str, ...] = ENRICHED_SOTA_NAMES + KEEP_ASSET_INDEX_MACRO_NAMES
        else:
            enriched_f32 = base_enriched_f32
            feature_names_out = ENRICHED_SOTA_NAMES

        # Validar `MissingnessResilientFeaturizer` sobre la última sesión con tripletas v3.1
        featurizer = MissingnessResilientFeaturizer().eval()
        featurizer_check: dict[str, Any] = dict(evaluated=False, all_finite=True)
        if t_len > 0 and n_assets > 0:
            last_batch: dict[str, torch.Tensor] = {
                "prices": torch.from_numpy(np.ascontiguousarray(enriched_f32[-1])),
            }
            m_last = np.ones((n_assets, 5), dtype=np.float32)
            if panel["fundamentals"] is not None and panel["fundamentals_mask"] is not None:
                last_batch["fundamentals"] = torch.from_numpy(
                    np.ascontiguousarray(panel["fundamentals"][-1], dtype=np.float32)
                )
                m_last[:, 1] = panel["fundamentals_mask"][-1].astype(np.float32)
            if panel["macro"] is not None and panel["macro_mask"] is not None:
                last_batch["macro"] = torch.from_numpy(
                    np.broadcast_to(panel["macro"][-1][None, :], (n_assets, 420)).copy()
                )
                m_last[:, 2] = panel["macro_mask"][-1].astype(np.float32)
            last_batch["modality_mask"] = torch.from_numpy(m_last)
            with torch.no_grad():
                f_out = featurizer.featurize_batch(
                    last_batch, fundamental_layout="encoded_v31_78", macro_layout="encoded_v31_420"
                )
            featurizer_check = dict(
                evaluated=True,
                all_finite=all(
                    bool(torch.isfinite(v).all())
                    for v in f_out.values()
                    if isinstance(v, torch.Tensor) and v.dtype.is_floating_point
                ),
                has_modality_freshness="modality_freshness" in f_out,
            )

        def _causality_fn() -> tuple[torch.Tensor, torch.Tensor, int]:
            win = min(t_len, 320)
            s0 = t_len - win
            t_cut = max(1, (2 * win) // 3)
            sub_p: dict[str, Any] = {
                "open": np.ascontiguousarray(panel["open"][s0:]),
                "high": np.ascontiguousarray(panel["high"][s0:]),
                "low": np.ascontiguousarray(panel["low"][s0:]),
                "close": np.ascontiguousarray(panel["close"][s0:]),
                "volume": np.ascontiguousarray(panel["volume"][s0:]),
                "fundamentals": (
                    np.ascontiguousarray(panel["fundamentals"][s0:])
                    if panel["fundamentals"] is not None
                    else None
                ),
                "fundamentals_mask": (
                    np.ascontiguousarray(panel["fundamentals_mask"][s0:])
                    if panel["fundamentals_mask"] is not None
                    else None
                ),
                "news_sig": (
                    np.ascontiguousarray(panel["news_sig"][s0:])
                    if panel["news_sig"] is not None
                    else None
                ),
                "news_mask": (
                    np.ascontiguousarray(panel["news_mask"][s0:])
                    if panel["news_mask"] is not None
                    else None
                ),
                "chart_drift": (
                    np.ascontiguousarray(panel["chart_drift"][s0:])
                    if panel["chart_drift"] is not None
                    else None
                ),
                "charts_mask": (
                    np.ascontiguousarray(panel["charts_mask"][s0:])
                    if panel["charts_mask"] is not None
                    else None
                ),
                "macro": (
                    np.ascontiguousarray(panel["macro"][s0:])
                    if panel["macro"] is not None
                    else None
                ),
                "macro_mask": (
                    np.ascontiguousarray(panel["macro_mask"][s0:])
                    if panel["macro_mask"] is not None
                    else None
                ),
            }
            sub_mut: dict[str, Any] = {
                k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in sub_p.items()
            }
            sub_mut["close"][t_cut + 1 :] *= 3.5
            sub_mut["high"][t_cut + 1 :] *= 4.0
            sub_mut["low"][t_cut + 1 :] *= 0.3
            sub_mut["volume"][t_cut + 1 :] *= 12.0
            for mk in (
                "fundamentals",
                "fundamentals_mask",
                "news_sig",
                "news_mask",
                "chart_drift",
                "charts_mask",
                "macro",
                "macro_mask",
            ):
                mv = sub_mut.get(mk)
                if isinstance(mv, np.ndarray):
                    if mv.dtype == np.bool_:
                        mv[t_cut + 1 :] = ~mv[t_cut + 1 :]
                    else:
                        mv[t_cut + 1 :] = -mv[t_cut + 1 :] * 2.5 - 3.0
            fa_base = torch.from_numpy(
                _enrich_with_fp32_fundamental_slabs(
                    sub_p, device=torch.device("cpu"), fund_slab_mb=fund_slab_mb
                )
            )
            fb_base = torch.from_numpy(
                _enrich_with_fp32_fundamental_slabs(
                    sub_mut, device=torch.device("cpu"), fund_slab_mb=fund_slab_mb
                )
            )
            if include_market_macro_indices:
                with torch.no_grad(), torch.amp.autocast(device_type="cpu", enabled=False):
                    ma = compute_market_indices_and_macro_sota_torch(
                        sub_p["open"],
                        sub_p["high"],
                        sub_p["low"],
                        sub_p["close"],
                        sub_p["volume"],
                        macro=sub_p["macro"],
                        macro_mask=sub_p["macro_mask"],
                        macro_layout="encoded_v31_420",
                    )
                    mb = compute_market_indices_and_macro_sota_torch(
                        sub_mut["open"],
                        sub_mut["high"],
                        sub_mut["low"],
                        sub_mut["close"],
                        sub_mut["volume"],
                        macro=sub_mut["macro"],
                        macro_mask=sub_mut["macro_mask"],
                        macro_layout="encoded_v31_420",
                    )
                n_w = fa_base.shape[1]
                fa = torch.cat(
                    [
                        fa_base,
                        select_keep_asset_index_macro_features(ma["asset_index_macro_features"]),
                        ma["market_regime_indices"].unsqueeze(1).expand(win, n_w, -1),
                    ],
                    dim=-1,
                )
                fb = torch.cat(
                    [
                        fb_base,
                        select_keep_asset_index_macro_features(mb["asset_index_macro_features"]),
                        mb["market_regime_indices"].unsqueeze(1).expand(win, n_w, -1),
                    ],
                    dim=-1,
                )
                return fa, fb, t_cut
            return fa_base, fb_base, t_cut

        quality_audit = audit_dataset_enrichment_quality(
            enriched_f32,
            panel["target"],
            feature_names=feature_names_out,
            causality_fn=_causality_fn if t_len >= 4 else None,
            run_stress=True,
        )
        market_macro_audit: dict[str, Any] | None = None
        if include_market_macro_indices:
            market_macro_audit = audit_market_indices_macro_value(
                panel["open"],
                panel["high"],
                panel["low"],
                panel["close"],
                panel["volume"],
                panel["target"],
                macro=panel["macro"],
                macro_mask=panel["macro_mask"],
                macro_layout="encoded_v31_420",
                existing_enriched=base_enriched_f32,
                market_label=mkt,
            )

    out_np_dtype = np.float16 if dtype == "fp16" else np.float32
    enriched_emit = np.ascontiguousarray(enriched_f32.astype(out_np_dtype, copy=False))
    enriched_all_finite = bool(np.isfinite(enriched_emit).all())
    mkt_regime_emit = (
        np.ascontiguousarray(market_regime_f32.astype(out_np_dtype, copy=False))
        if market_regime_f32 is not None
        else None
    )
    fac_ret_emit = (
        np.ascontiguousarray(factor_returns_f32.astype(out_np_dtype, copy=False))
        if factor_returns_f32 is not None
        else None
    )
    mkt_macro_ok = True
    if include_market_macro_indices:
        mkt_macro_ok = bool(
            mkt_regime_emit is not None
            and fac_ret_emit is not None
            and np.isfinite(mkt_regime_emit).all()
            and np.isfinite(fac_ret_emit).all()
            and market_macro_audit is not None
            and market_macro_audit["overall_passed"]
        )
    gates_passed = bool(
        enriched_all_finite
        and quality_audit["leakage_screen_passed"]
        and quality_audit["causality_check"]["bit_exact_causal"]
        and quality_audit["missingness_stress"]["all_passed"]
        and featurizer_check["all_finite"]
        and mkt_macro_ok
    )

    h_ctx = hashlib.sha256()
    h_ctx.update(f"{mode}|{mkt}|{dtype}|".encode())
    h_ctx.update(json.dumps(list(feature_names_out)).encode("utf-8"))
    h_ctx.update(json.dumps(list(session_dates)).encode("utf-8"))
    h_ctx.update(json.dumps(list(tickers)).encode("utf-8"))
    h_ctx.update(enriched_emit.tobytes())
    if mkt_regime_emit is not None and fac_ret_emit is not None:
        h_ctx.update(json.dumps(list(MARKET_REGIME_INDEX_NAMES)).encode("utf-8"))
        h_ctx.update(mkt_regime_emit.tobytes())
        h_ctx.update(fac_ret_emit.tobytes())
    sha256_contract = h_ctx.hexdigest()

    sidecar_path: Path | None = None
    report_path: Path | None = None
    if not dry_run and out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        report_path = out_path / f"enrichment_report_{tag_name}_{mkt}.json.gz"
        if (not audit_only) and (gates_passed or allow_failed_audit):
            sidecar_path = out_path / f"enriched_sota_{tag_name}_{mkt}_{dtype}.npz"
            npz_payload: dict[str, Any] = dict(
                enriched_sota=enriched_emit,
                feature_names=np.asarray(feature_names_out),
                session_dates=np.asarray(session_dates),
                session_epochs=np.asarray(
                    [_iso_to_epoch_day(d) for d in session_dates], np.int32
                ),
                tickers=np.asarray(tickers),
                sha256_contract=np.asarray(sha256_contract),
            )
            if mkt_regime_emit is not None and fac_ret_emit is not None:
                npz_payload["market_regime_indices"] = mkt_regime_emit
                npz_payload["market_regime_index_names"] = np.asarray(MARKET_REGIME_INDEX_NAMES)
                npz_payload["factor_returns"] = fac_ret_emit
                npz_payload["factor_portfolio_names"] = np.asarray(FACTOR_PORTFOLIO_NAMES)
                npz_payload["keep_asset_index_macro_names"] = np.asarray(
                    KEEP_ASSET_INDEX_MACRO_NAMES
                )
            np.savez(sidecar_path, **npz_payload)

    summary: dict[str, Any] = dict(
        mode=mode,
        market=mkt,
        tag=tag_name,
        dtype=dtype,
        dry_run=bool(dry_run),
        audit_only=bool(audit_only),
        include_market_macro_indices=bool(include_market_macro_indices),
        strictly_pre_2024=True,
        min_session_date=session_dates[0],
        max_session_date=session_dates[-1],
        n_sessions=t_len,
        n_assets=n_assets,
        n_enriched_features=len(feature_names_out),
        enriched_sota_names=list(feature_names_out),
        n_prices_loaded=int(panel["n_prices_loaded"]),
        n_encoded_loaded=int(panel["n_encoded_loaded"]),
        n_sealed_2024_rows_dropped=int(panel["n_sealed_2024_rows_dropped"]),
        n_sealed_2024_targets_dropped=int(panel["n_sealed_2024_targets_dropped"]),
        fund_uses_memmap=bool(panel["fund_uses_memmap"]),
        enriched_all_finite=enriched_all_finite,
        gates_passed=gates_passed,
        featurizer_check=featurizer_check,
        lab_cache_audit=lab_cache_audit,
        quality_audit=quality_audit,
        sha256_contract=sha256_contract,
        sidecar_path=str(sidecar_path) if sidecar_path is not None else None,
        report_path=str(report_path) if report_path is not None else None,
        elapsed_s=round(time.perf_counter() - t0, 3),
    )
    if include_market_macro_indices:
        summary["keep_asset_index_macro_names"] = list(KEEP_ASSET_INDEX_MACRO_NAMES)
        summary["market_regime_index_dim"] = int(MARKET_REGIME_INDEX_DIM)
        summary["market_regime_index_names"] = list(MARKET_REGIME_INDEX_NAMES)
        summary["market_macro_audit"] = market_macro_audit

    if report_path is not None:
        payload = json.dumps(
            _to_json_safe(summary), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        tmp_gz = report_path.with_suffix(".gz.tmp")
        tmp_gz.write_bytes(gzip.compress(payload, compresslevel=6))
        tmp_gz.replace(report_path)

    if not gates_passed and not allow_failed_audit:
        raise RuntimeError(
            f"Enrichment quality gate FAILED for {mode}/{mkt}/{tag_name}: "
            f"finite={enriched_all_finite}, "
            f"leak_pass={quality_audit['leakage_screen_passed']} "
            f"(max|RankIC|={quality_audit['max_abs_rank_ic']:.4f}, "
            f"max|PearsonIC|={quality_audit['max_abs_pearson_ic']:.4f}), "
            f"causal={quality_audit['causality_check']['bit_exact_causal']}, "
            f"stress={quality_audit['missingness_stress']['all_passed']}, "
            f"featurizer={featurizer_check['all_finite']}"
        )

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materializador y auditor oficial SOTA pre-2024 (US/CN separados, FM-8)."
    )
    parser.add_argument("--mode", choices=VALID_MODES, required=True)
    parser.add_argument("--market", choices=VALID_MARKETS, required=True)
    parser.add_argument("--lab-cache", type=Path, default=None)
    parser.add_argument("--materialization-root", type=Path, default=DEFAULT_MATERIALIZATION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-assets", type=int, default=0)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--asset-chunk-size", type=int, default=128)
    parser.add_argument("--fund-slab-mb", type=float, default=512.0)
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--allow-failed-audit", action="store_true")
    parser.add_argument("--include-market-macro-indices", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_enrichment_pipeline(
            mode=args.mode,
            market=args.market,
            lab_cache_dir=args.lab_cache,
            materialization_root=args.materialization_root,
            output_dir=args.output_dir,
            device=args.device,
            max_assets=args.max_assets,
            max_sessions=args.max_sessions,
            asset_chunk_size=args.asset_chunk_size,
            fund_slab_mb=args.fund_slab_mb,
            dtype=args.dtype,
            dry_run=args.dry_run,
            audit_only=args.audit_only,
            allow_failed_audit=args.allow_failed_audit,
            include_market_macro_indices=args.include_market_macro_indices,
        )
    except RuntimeError as exc:
        print(json.dumps({"gates_passed": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1

    compact = dict(
        mode=summary["mode"],
        market=summary["market"],
        tag=summary["tag"],
        n_sessions=summary["n_sessions"],
        n_assets=summary["n_assets"],
        date_range=f"{summary['min_session_date']}..{summary['max_session_date']}",
        strictly_pre_2024=summary["strictly_pre_2024"],
        n_sealed_2024_rows_dropped=summary["n_sealed_2024_rows_dropped"],
        n_sealed_2024_targets_dropped=summary["n_sealed_2024_targets_dropped"],
        enriched_all_finite=summary["enriched_all_finite"],
        gates_passed=summary["gates_passed"],
        leakage_screen_passed=summary["quality_audit"]["leakage_screen_passed"],
        max_abs_rank_ic=round(float(summary["quality_audit"]["max_abs_rank_ic"]), 5),
        max_abs_pearson_ic=round(float(summary["quality_audit"]["max_abs_pearson_ic"]), 5),
        bit_exact_causal=summary["quality_audit"]["causality_check"]["bit_exact_causal"],
        causality_max_abs_diff=summary["quality_audit"]["causality_check"]["max_abs_diff"],
        missingness_stress_passed=summary["quality_audit"]["missingness_stress"]["all_passed"],
        sha256_contract=summary["sha256_contract"],
        sidecar_path=summary["sidecar_path"],
        report_path=summary["report_path"],
        elapsed_s=summary["elapsed_s"],
    )
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
