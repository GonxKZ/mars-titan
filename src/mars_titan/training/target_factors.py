"""Dependencia versionada de etiquetas, separada de la codificación de entradas."""

import json
import re
from pathlib import Path

import pyarrow.parquet as pq

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import sha256

from .cohort_contract import cohort_identity

_JSON_BYTES = 64 * 1024
_MANIFEST_BYTES = 8 * 1024**2
_FACTOR_BYTES = 64 * 1024**2
_RECORD_PATHS = {"prices_path", "prices_sha256"}


def same_json(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("La fuente del factor necesita una huella SHA256 válida")
    return value


def _path(value):
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError("La fuente del factor necesita una ruta explícita y acotada")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("La ruta del factor debe ser absoluta")
    safe_destination(path)
    return path


def confirm_sources(sources):
    """Recomprobar las fuentes fijadas, sin sustituir sus huellas por otras nuevas."""
    for path, (signature, limit) in sources.items():
        safe_destination(path)
        if not path.is_file() or path.stat().st_size > limit or sha256(path) != signature:
            raise ValueError("Ha cambiado una fuente o descriptor del factor residual")


def _complete_parent(meta, input_policy):
    if (
        meta.get("kind") != "materialized_corpus"
        or meta.get("scope") != "full_corpus"
        or meta.get("cohort_complete") is not True
        or type(meta.get("failed_assets")) is not int
        or meta["failed_assets"] != 0
        or type(meta.get("samples")) is not int
        or meta["samples"] <= 0
        or meta.get("final_test_opened") is not False
        or not meta.get("assets")
    ):
        raise ValueError("La revisión del factor necesita un censo completo confirmado")
    if cohort_identity(meta, input_policy=input_policy) is None:
        raise ValueError("La revisión del factor necesita una población con cohorte explícita")
    configuration = meta.get("configuration")
    if not isinstance(configuration, dict) or not same_json(
        meta.get("market_factors"), configuration.get("market_factors")
    ):
        raise ValueError("El factor original no concilia con la configuración de codificación")


def _sample_counts(meta):
    """Contrastar el censo declarado con los recibos y las filas confirmadas."""
    root = _path(meta.get("samples_root"))
    sources = {}
    for row in meta["coverage"]:
        if row["state"] != "encoded":
            continue
        folder = root / row["market"] / row["symbol"]
        safe_destination(folder)
        if not folder.resolve().is_relative_to(root.resolve()):
            raise ValueError("El activo de la cobertura queda fuera del origen")
        receipt_path, table_path = folder / "manifest.json", folder / "samples.parquet"
        receipt, signature = read_manifest(receipt_path, _MANIFEST_BYTES)
        if (
            type(receipt.get("samples")) is not int
            or receipt["samples"] != row["samples"]
            or _digest(receipt.get("fingerprint")) != _digest(row.get("fingerprint"))
            or receipt.get("market") != row["market"]
            or receipt.get("symbol") != row["symbol"]
        ):
            raise ValueError("El recibo de muestras no concilia con la cobertura del padre")
        safe_destination(table_path)
        if not table_path.is_file() or table_path.stat().st_size < 8:
            raise ValueError("Falta la tabla de muestras del censo")
        with table_path.open("rb") as stream:
            stream.seek(-8, 2)
            footer = stream.read(8)
        if footer[4:] != b"PAR1" or int.from_bytes(footer[:4], "little") > _MANIFEST_BYTES:
            raise ValueError("El metadato de muestras supera su presupuesto o está dañado")
        with pq.ParquetFile(table_path) as file:
            if file.metadata.num_rows != row["samples"]:
                raise ValueError("Las filas de muestras no concilian con la cobertura")
        sources[receipt_path] = (signature, _MANIFEST_BYTES)
    return sources


def _factors(original, effective, markets):
    if (
        not isinstance(original, dict)
        or not isinstance(effective, dict)
        or set(original) != markets
        or set(effective) != markets
        or not markets <= {"US", "CN"}
    ):
        raise ValueError("Los factores deben identificar los mismos mercados del censo")
    sources = {}
    for market in sorted(markets):
        before, after = original[market], effective[market]
        for record in (before, after):
            if (
                not isinstance(record, dict)
                or record.get("market") != market
                or not isinstance(record.get("symbol"), str)
                or not 1 <= len(record["symbol"]) <= 64
            ):
                raise ValueError("El factor no identifica su mercado e instrumento")
            path = _path(record.get("prices_path"))
            value = (_digest(record.get("prices_sha256")), _FACTOR_BYTES)
            if path in sources and sources[path] != value:
                raise ValueError("La revisión no puede sustituir una fuente previa del factor")
            sources[path] = value
        if not same_json(
            {k: v for k, v in before.items() if k not in _RECORD_PATHS},
            {k: v for k, v in after.items() if k not in _RECORD_PATHS},
        ):
            raise ValueError("La revisión no conserva la semántica del factor residual")
    confirm_sources(sources)
    return sources


def bind_target_factors(manifest, meta, manifest_hash, descriptor, *, input_policy):
    """Fijar un descriptor completo sin modificar el manifiesto codificado original."""
    _complete_parent(meta, input_policy)
    descriptor = _path(str(Path(descriptor).absolute()))
    effective, signature = read_manifest(descriptor, _JSON_BYTES)
    markets = {row["market"] for row in meta["coverage"]}
    sources = _factors(meta.get("market_factors"), effective, markets)
    sources.update(_sample_counts(meta))
    sources.update(
        {
            Path(manifest).absolute(): (_digest(manifest_hash), _MANIFEST_BYTES),
            descriptor: (signature, _JSON_BYTES),
        }
    )
    confirm_sources(sources)
    revision = dict(
        schema_version=1,
        source_manifest=str(Path(manifest).absolute()),
        source_manifest_sha256=manifest_hash,
        descriptor_path=str(descriptor),
        descriptor_sha256=signature,
        contract_sha256=sha256(Path(__file__)),
    )
    return effective, revision, sources


def revision_sources(meta, *, input_policy):
    """Verificar la dependencia también en vistas que conservan las mismas entradas."""
    configuration = meta.get("configuration", {})
    if "target_factor_revision" not in configuration:
        return {}
    revision = configuration["target_factor_revision"]
    if (
        not isinstance(revision, dict)
        or set(revision)
        != {
            "schema_version",
            "source_manifest",
            "source_manifest_sha256",
            "descriptor_path",
            "descriptor_sha256",
            "contract_sha256",
        }
        or type(revision.get("schema_version")) is not int
        or revision["schema_version"] != 1
    ):
        raise ValueError("La revisión del factor no conserva su contrato versionado")
    for name in ("source_manifest_sha256", "descriptor_sha256", "contract_sha256"):
        _digest(revision[name])
    path = _path(revision["source_manifest"])
    parent, signature = read_manifest(path, _MANIFEST_BYTES)
    if signature != revision["source_manifest_sha256"] or signature != configuration.get(
        "source_manifest_sha256"
    ):
        raise ValueError("Ha cambiado el padre de la revisión del factor")
    effective, observed, sources = bind_target_factors(
        path, parent, signature, _path(revision["descriptor_path"]), input_policy=input_policy
    )
    if observed["descriptor_sha256"] != revision["descriptor_sha256"] or not same_json(
        effective, meta.get("market_factors")
    ):
        raise ValueError("Ha cambiado el descriptor efectivo de factores de las etiquetas")
    if (
        meta.get("context_sessions") != parent.get("context_sessions")
        or meta.get("cohort_id") != parent.get("cohort_id")
        or meta.get("roots", {}).get("samples") != parent.get("samples_root")
        or meta.get("roots", {}).get("prepared")
        != parent.get("configuration", {}).get("prepared_root")
        or not {(a["market"], a["symbol"]) for a in meta["assets"]}
        <= {(a["market"], a["symbol"]) for a in parent["assets"]}
    ):
        raise ValueError("La revisión del factor no conserva las entradas del corpus padre")
    return sources
