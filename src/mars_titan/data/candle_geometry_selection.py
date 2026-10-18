"""Selección explícita y presupuesto de disco para derivar representaciones de velas."""

import math
import re
import shutil
from copy import deepcopy
from pathlib import Path

import pyarrow.parquet as pq

from .cohort_files import read_manifest, safe_destination
from .cohort_samples import _digest
from .storage import sha256

DEFAULT_RESERVE_BYTES = 16 * 1024**3


def _keys(assets):
    if (
        not isinstance(assets, list)
        or not 1 <= len(assets) <= 10_000
        or any(
            not isinstance(asset, dict)
            or set(asset) != {"market", "symbol"}
            or any(not isinstance(value, str) for value in asset.values())
            for asset in assets
        )
    ):
        raise ValueError("La selección requiere identidades de activos explícitas")
    result = [(asset["market"], asset["symbol"]) for asset in assets]
    if len(result) != len(set(result)):
        raise ValueError("La selección contiene activos duplicados")
    return sorted(result)


def selection_record(request, request_hash, manifest, manifest_hash, config_hash):
    """Vincular una solicitud a un origen completo sin atribuirle validación científica."""
    if (
        not isinstance(request, dict)
        or not isinstance(request_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", request_hash)
    ):
        raise ValueError("La selección necesita un documento con su lista de activos")
    keys = _keys(request.get("assets"))
    available = {(row["market"], row["symbol"]) for row in manifest["coverage"]}
    if not set(keys) <= available:
        raise ValueError("La selección contiene activos ajenos al origen")
    assets = [dict(market=market, symbol=symbol) for market, symbol in keys]
    return dict(
        version=1,
        assets=assets,
        assets_sha256=_digest(assets),
        request=request,
        request_sha256=request_hash,
        source_manifest_sha256=manifest_hash,
        source_configuration_sha256=config_hash,
    )


def project_manifest(manifest, selection):
    """Conservar todas las filas declaradas de cada candidato solicitado, sin filtrar fechas."""
    if selection is None:
        return manifest
    keys = set(_keys(selection["assets"]))
    result = deepcopy(manifest)
    result["coverage"] = [
        row for row in manifest["coverage"] if (row["market"], row["symbol"]) in keys
    ]
    result["assets"] = [row for row in manifest["assets"] if (row["market"], row["symbol"]) in keys]
    result.update(
        scope="development_snapshot",
        cohort_complete=False,
        training_ready=False,
        candidate_count=len(keys),
        samples=sum(row.get("samples", 0) for row in result["coverage"]),
        markets=sorted({market for market, _ in keys}),
        selection=selection,
        selection_complete=True,
    )
    result["calendar_start"] = {
        market: manifest["calendar_start"][market] for market in result["markets"]
    }
    return result


def comparison_population(manifests, configurations):
    """Cotejar la población completa o una selección que identifica ese mismo origen."""
    before, after = [item[0] for item in manifests]
    if before.get("cohort_complete") is not True:
        raise ValueError("La comparación necesita un origen completo")
    selection = after.get("selection")
    if selection is None:
        if after.get("cohort_complete") is not True:
            raise ValueError(
                "La comparación necesita una edición completa o su selección explícita"
            )
        expected = before
    else:
        if not isinstance(selection, dict):
            raise ValueError("La selección necesita su identidad completa")
        record = selection_record(
            selection.get("request"),
            selection.get("request_sha256"),
            before,
            manifests[0][1],
            configurations[0][1],
        )
        expected = project_manifest(before, record)
        fields = (
            "selection",
            "selection_complete",
            "scope",
            "cohort_complete",
            "training_ready",
            "final_test_opened",
            "candidate_count",
            "markets",
            "samples",
        )
        if any(after.get(field) != expected.get(field) for field in fields):
            raise ValueError("La selección no acredita su origen, población o completitud")

        def coverage(rows):
            return [{k: v for k, v in row.items() if k != "fingerprint"} for row in rows]

        if coverage(after["coverage"]) != coverage(expected["coverage"]):
            raise ValueError("La selección ha cambiado la cobertura del origen")
    assets = [
        sorted((a["market"], a["symbol"]) for a in item["assets"]) for item in (expected, after)
    ]
    if assets[0] != assets[1] or len(set(assets[0])) != len(assets[0]):
        raise ValueError("Las representaciones no conservan la misma población de activos")
    return assets[0], expected["samples"]


def free_disk_bytes(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def storage_plan(source, output, manifest, *, batch_rows, reserve_bytes):
    """Estimar escritura desde pies Parquet. No leer filas ni verificar aún sus hashes."""
    if type(reserve_bytes) is not int or reserve_bytes < 1:
        raise ValueError("La reserva de disco debe ser un número positivo de bytes")
    assets = []
    for row in manifest["coverage"]:
        if row["state"] != "encoded":
            continue
        folder = source / "samples" / row["market"] / row["symbol"]
        safe_destination(folder)
        receipt, _ = read_manifest(folder / "manifest.json")
        path = folder / "samples.parquet"
        safe_destination(path)
        with pq.ParquetFile(path) as file:
            if file.metadata.num_rows != row["samples"] or receipt.get("samples") != row["samples"]:
                raise ValueError("Las filas de la estimación no coinciden con el manifiesto")
            uncompressed = sum(
                file.metadata.row_group(i).total_byte_size for i in range(file.num_row_groups)
            )
        factors = folder / "company-factors.parquet"
        factor_bytes = 0
        if receipt.get("company_factors_sha256") is not None:
            safe_destination(factors)
            factor_bytes = factors.stat().st_size
        # Añadir los nuevos vectores completos no presupone que compriman como los anteriores.
        payload = max(path.stat().st_size, uncompressed) + row["samples"] * (512 * 4 + 64 + 8)
        metadata = max(1, math.ceil(row["samples"] / batch_rows)) * 64 * 1024 + 1024**2
        assets.append(
            dict(
                market=row["market"],
                symbol=row["symbol"],
                samples=row["samples"],
                source_file_bytes=path.stat().st_size,
                uncompressed_bytes=uncompressed,
                factor_bytes=factor_bytes,
                estimated_output_bytes=math.ceil(payload * 1.2) + metadata + factor_bytes,
            )
        )
    metadata_bytes = 3 * sum(
        (source / name).stat().st_size for name in ("manifest.json", "configuration.json")
    )
    expected = sum(row["estimated_output_bytes"] for row in assets) + metadata_bytes
    free = free_disk_bytes(output)
    return dict(
        schema_version=1,
        kind="geometry_storage_estimate",
        selected_candidates=manifest["candidate_count"],
        samples=manifest["samples"],
        source_samples_file_bytes=sum(row["source_file_bytes"] for row in assets),
        source_uncompressed_bytes=sum(row["uncompressed_bytes"] for row in assets),
        estimated_output_bytes=expected,
        root_metadata_bytes=metadata_bytes,
        reserve_bytes=reserve_bytes,
        free_disk_bytes=free,
        fits_in_free_space=free >= expected + reserve_bytes,
        assets=assets,
        estimator_code_sha256=sha256(Path(__file__)),
        rule="1.2*(max(file_bytes,uncompressed_bytes)+rows*2120)+65536*row_groups+1048576_per_asset+factors+3*root_json",
        limits=[
            "Estimación con margen, no cota certificada de compresión o espacio concurrente.",
            "Solo se inspeccionan manifiestos y metadatos, no se admite todavía el contenido.",
        ],
    )
