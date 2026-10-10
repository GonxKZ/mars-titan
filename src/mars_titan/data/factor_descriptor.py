"""Descriptor de los factores de mercado de una edición y auditoría de sus precios.

El factor residual de Estados Unidos es SPY tal como lo deja la preparación de la edición. Si una
revisión reescribe esos precios, como hace la v3.1 al leer cada número de forma exacta, el
descriptor anterior deja de describir el archivo y la edición necesita uno propio con otra huella.
El factor chino es el CSI300 reconstruido aparte a partir de los boletines mensuales. La revisión
no lo toca, así que se conserva con la huella de su informe.

La auditoría repite las comprobaciones que fijaron el descriptor v2: huella del archivo, precios
finitos y positivos, OHLC coherente, sesiones crecientes del calendario histórico anterior a 2024
y disponibilidad igual a la decisión de cada sesión. La única diferencia es que una fila de SPY con
un desorden OHLC de redondeo se admite si la auditoría de precios de la preparación declaró esa
tolerancia, y entonces queda listada. Ninguna comprobación abre el test final.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .cohort_files import read_manifest
from .prices import check_ordering_rtol, ordering_excess
from .storage import atomic_json, sha256
from .temporal import MarketClock

_FACTOR_BYTES = 64 * 1024**2
_COLUMNS = ["session", "open", "high", "low", "close", "available_at"]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_factor_prices(path, signature, market, *, ordering_rtol=0.0):
    """Auditar los precios de un factor sobre el calendario histórico cerrado en 2023."""
    check_ordering_rtol(ordering_rtol)
    path = Path(path)
    _require(
        not path.is_symlink()
        and path.is_file()
        and path.stat().st_size <= _FACTOR_BYTES
        and sha256(path) == signature,
        "El precio del factor no coincide con la fuente cerrada",
    )
    clock = MarketClock(market, "2000-01-01", "2023-12-31")
    sessions = {
        day.isoformat(): moment for day, moment in zip(clock.days, clock.decisions, strict=True)
    }
    observed, rounded = [], []
    with pq.ParquetFile(path) as file:
        for batch in file.iter_batches(batch_size=256, columns=_COLUMNS, use_threads=False):
            prices = np.column_stack(
                [batch.column(name).to_numpy() for name in ("open", "high", "low", "close")]
            )
            _require(
                np.isfinite(prices).all() and (prices > 0).all(),
                "Los precios del factor deben ser finitos y positivos",
            )
            opening, high, low, close = prices.T
            ordered = (low <= np.minimum(opening, close)) & (np.maximum(opening, close) <= high)
            admitted = ordered.copy()
            if ordering_rtol:
                # Un desorden se admite solo como el redondeo que declaró la auditoría de precios.
                admitted |= ordering_excess(opening, high, low, close) <= ordering_rtol
            _require(admitted.all(), "El factor tiene una fila OHLC incoherente")
            days = batch.column("session").to_pylist()
            rounded.extend(day for day, tidy in zip(days, ordered, strict=True) if not tidy)
            for day, stamp in zip(days, batch.column("available_at").to_pylist(), strict=True):
                _require(
                    "2000-01-01" <= day < "2024-01-01" and day in sessions,
                    "La sesión del factor queda fuera del calendario histórico",
                )
                _require(
                    stamp == sessions[day] and stamp.year < 2024,
                    "La disponibilidad del factor no es la decisión de su sesión",
                )
                _require(not observed or day > observed[-1], "Las sesiones no son crecientes")
                observed.append(day)
    _require(observed and sha256(path) == signature, "El factor está vacío o cambió al leerlo")
    full = sorted(sessions)
    within = [day for day in full if observed[0] <= day <= observed[-1]]
    return dict(
        rows=len(observed),
        first_session=observed[0],
        last_session=observed[-1],
        historical_calendar_sessions=len(full),
        calendar_sessions_within_span=len(within),
        missing_within_span=sorted(set(within) - set(observed)),
        uncovered_before_first=sum(day < observed[0] for day in full),
        uncovered_after_last=sum(day > observed[-1] for day in full),
        ordering_rtol=ordering_rtol,
        rounded_ordering_sessions=rounded,
        session_list_sha256=hashlib.sha256(json.dumps(observed).encode()).hexdigest(),
        source_bytes=path.stat().st_size,
        prices_sha256=signature,
        all_availability_matches_retrospective_calendar=True,
        final_test_rows=0,
        point_in_time_verified=False,
    )


def describe_market_factors(preparation, cn_factor, destination, *, previous):
    """Fijar el descriptor de factores de una preparación y su auditoría, sin sobrescribir.

    `preparation` es el manifiesto de la preparación de la edición, `cn_factor` la carpeta del
    CSI300 con `market-factors.json` y `report.json`, y `previous` el descriptor que se sustituye.
    De este último solo se registra la huella, sin abrir sus precios.
    """
    from mars_titan.training.target_factors import _factors

    preparation, cn_factor = Path(preparation), Path(cn_factor)
    destination, previous = Path(destination), Path(previous)
    audit_path = destination.with_name(f"{destination.stem}-audit.json")
    _require(
        not destination.exists() and not audit_path.exists(),
        "El descriptor ya existe y no se reescribe",
    )
    parent, parent_hash = read_manifest(preparation)
    source = Path(parent["prepared_root"]) / "US" / "SPY"
    candidates = [a for a in parent["assets"] if (a["market"], a["symbol"]) == ("US", "SPY")]
    _require(len(candidates) == 1, "La preparación no contiene SPY una sola vez")
    (candidate,) = candidates
    spy, spy_hash = read_manifest(source / "manifest.json")
    _require(spy_hash == candidate["manifest_sha256"], "El manifiesto de SPY no es el preparado")
    cn_descriptor, cn_hash = read_manifest(cn_factor / "market-factors.json")
    cn_report, cn_report_hash = read_manifest(cn_factor / "report.json")
    _require(
        cn_hash == cn_report["artifacts"]["market-factors.json"],
        "El descriptor del CSI300 no es el de su informe",
    )
    revision = parent.get("price_revision") or {}
    rtol = spy.get("price_audit", {}).get("ordering_rtol", 0.0)
    records = {
        "US": dict(
            market="US",
            symbol="SPY",
            prices_path=str(source / "prices.parquet"),
            prices_sha256=spy["artifacts"]["prices.parquet"],
            return_convention="close_over_open_minus_one",
            point_in_time_verified=False,
            adjustments="as_distributed_retrospective_adjustments_not_reconstructed",
            number_parsing=revision.get("number_parsing"),
            ordering_rtol=rtol,
            base_preparation_sha256=parent_hash,
            base_asset_manifest_sha256=spy_hash,
        ),
        "CN": {**cn_descriptor["CN"], "base_factor_report_sha256": cn_report_hash},
    }
    audits = {
        market: audit_factor_prices(
            item["prices_path"],
            item["prices_sha256"],
            market,
            ordering_rtol=rtol if market == "US" else 0.0,
        )
        for market, item in records.items()
    }
    # La misma validación que aplica la vinculación de etiquetas a un descriptor.
    _factors(records, records, {"US", "CN"})
    previous_hash = sha256(previous)
    report = dict(
        schema_version=3,
        recorded_at=datetime.now(UTC).isoformat(),
        scope="Factores de una preparación revisada, sin supervisión calculada.",
        previous_descriptor_sha256=previous_hash,
        previous_price_files_opened=False,
        base_preparation_sha256=parent_hash,
        price_revision=revision,
        factor_audit=audits,
        origins=dict(
            US=dict(prepared_manifest_sha256=spy_hash),
            CN=dict(descriptor_sha256=cn_hash, report_sha256=cn_report_hash),
        ),
        code_sha256=sha256(Path(__file__)),
        labels_materialized=False,
        corpus_written=False,
        training_ready=False,
        final_test_opened=False,
        limits=[
            "La fecha retrospectiva de sesión no acredita captura contemporánea.",
            "SPY conserva los ajustes distribuidos, sin reconstrucción de acciones corporativas.",
            "CSI300 comienza en 2006. Los periodos anteriores no se completan.",
            "Los factores no intervienen en los vectores de entrada. La supervisión posterior "
            "requiere su propio contrato.",
        ],
    )
    atomic_json(destination, records)
    report["descriptor_sha256"] = sha256(destination)
    atomic_json(audit_path, report)
    _require(sha256(previous) == previous_hash, "El descriptor anterior cambió durante el proceso")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True, help="Manifiesto preparado")
    parser.add_argument("--cn-factor", type=Path, required=True, help="Carpeta del CSI300")
    parser.add_argument("--previous", type=Path, required=True, help="Descriptor sustituido")
    parser.add_argument("--output", type=Path, required=True, help="Descriptor nuevo")
    args = parser.parse_args()
    report = describe_market_factors(
        args.preparation, args.cn_factor, args.output, previous=args.previous
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
