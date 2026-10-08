"""Enriquecer preparados históricos con cuentas CN revisadas, sin recortar el censo."""

import argparse
import fcntl
import re
import time
from pathlib import Path

from .accounting_catalog import HISTORICAL_ACCOUNTING, JOINT_CONCEPTS
from .china_preparation import _fact_editions, derive_chinese_preparation
from .chinese_corpus import _has, _read, _same
from .cohort_files import read_manifest, safe_destination
from .cohort_samples import _digest, _prepared
from .corpus_preparation import STARTS
from .input_policy import HISTORICAL_MASKED, policy_identity
from .joint_corpus import _copy
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock

_MAX_ASSETS = 10_000
_CODE = (
    "historical_accounting.py",
    "accounting_catalog.py",
    "china_preparation.py",
    "china_fundamentals.py",
    "china_sources.py",
    "chinese_corpus.py",
    "cohort_samples.py",
    "corpus_preparation.py",
    "cohort_news.py",
    "cohort_files.py",
    "joint_corpus.py",
    "currency_samples.py",
    "input_policy.py",
    "storage.py",
    "temporal.py",
)


def _verify(sources):
    for path, digest in sources.items():
        safe_destination(path)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError("Una fuente o un artefacto ha cambiado de huella")


def reviewed_cn_catalog(reviewed_root):
    """Leer referencias de hechos desde la unión revisada, sin interpretar sus muestras."""
    root, sources = Path(reviewed_root).absolute(), {}
    safe_destination(root)
    report = _read(root / "report.json", sources)
    if not _has(
        report,
        schema_version=1,
        kind="chinese_corpus_union",
        status="completed",
        training_ready=False,
        final_test_opened=False,
    ):
        raise ValueError("El catálogo necesita una unión CN revisada y confirmada")
    config = _read(root / "configuration.json", sources, report.get("configuration_sha256"))
    if config.get("policy") != "copied_reviewed_chinese_editions_v1" or not isinstance(
        config.get("sources"), dict
    ):
        raise ValueError("La unión no conserva su política ni fuentes")
    lineage = report.get("lineage")
    if (
        not isinstance(lineage, list)
        or not 1 <= len(lineage) <= _MAX_ASSETS
        or not _same(lineage, config.get("lineage"))
        or not _same(report.get("candidate_count"), len(lineage))
    ):
        raise ValueError("El censo revisado no concilia con su procedencia")
    artifacts = report.get("artifacts", {})
    if set(artifacts) != {"encoded/manifest.json", "supervised/manifest.json"}:
        raise ValueError("La unión no identifica sus manifiestos")
    manifests = [_read(root / name, sources, digest) for name, digest in artifacts.items()]
    catalog = {}
    for item in lineage:
        symbol, edition = item.get("symbol"), item.get("edition")
        if (
            not isinstance(symbol, str)
            or not re.fullmatch(r"\d{6}\.(?:SZ|SS|SH)", symbol)
            or symbol in catalog
            or not isinstance(edition, str)
        ):
            raise ValueError("La procedencia CN repite o no identifica un emisor")
        original = Path(edition) / "prepared/CN" / symbol / "manifest.json"
        safe_destination(original)
        expected = config["sources"].get(str(original.resolve()))
        parent = _read(root / "prepared/CN" / symbol / "manifest.json", sources, expected)
        if not _has(
            parent,
            schema_version=3,
            kind="derived_prepared_asset",
            market="CN",
            symbol=symbol,
            cohort_id="original_audited",
            training_ready=False,
            final_test_opened=False,
        ):
            raise ValueError("El preparado revisado no corresponde al emisor y contrato CN")
        parents = parent.get("parents", {})
        references = [parents.get("facts"), *parents.get("additional_facts", [])]
        if not 1 <= len(references) <= 16 or any(not isinstance(ref, dict) for ref in references):
            raise ValueError("La historia revisada no identifica sus ediciones contables")
        facts = []
        for ref in references:
            path = Path(ref.get("path", ""))
            if path.name != "report.json":
                raise ValueError("La fuente contable no identifica un recibo")
            record = _read(path, sources, ref.get("sha256"))
            if (
                not _has(
                    record,
                    schema_version=1,
                    kind="reconciled_chinese_fundamentals",
                    training_ready=False,
                    final_test_opened=False,
                )
                or not _same(record.get("identity"), ref.get("identity"))
                or record.get("sha256") != ref.get("artifact_sha256")
                or record.get("identity", {}).get("symbol") != symbol
            ):
                raise ValueError("El recibo de hechos no conserva su identidad revisada")
            facts.append(
                dict(
                    path=str(path.parent.resolve()),
                    manifest_sha256=ref["sha256"],
                    artifact_sha256=ref["artifact_sha256"],
                )
            )
        _fact_editions(facts[0]["path"], [ref["path"] for ref in facts[1:]])
        catalog[symbol] = facts
    for manifest in manifests:
        assets = manifest.get("assets")
        if (
            not isinstance(assets, list)
            or not _same(manifest.get("candidate_count"), len(catalog))
            or len(assets) != len(catalog)
            or {(row.get("market"), row.get("symbol")) for row in assets}
            != {("CN", symbol) for symbol in catalog}
        ):
            raise ValueError("Los manifiestos no conservan el mismo censo CN")
    _verify(sources)
    return catalog, sources


def _parent(path):
    sources = {}
    parent = _read(path, sources)
    assets = parent.get("assets")
    if (
        not _has(
            parent,
            schema_version=2,
            kind="prepared_cohort",
            status="completed",
            failed_assets=0,
            training_ready=False,
            **policy_identity(HISTORICAL_MASKED),
        )
        or not isinstance(assets, list)
        or not 1 <= len(assets) <= _MAX_ASSETS
        or not _same(parent.get("candidate_count"), len(assets))
    ):
        raise ValueError("La preparación histórica debe estar completa y sin errores")
    identities = set()
    for row in assets:
        key = row.get("market"), row.get("symbol")
        if (
            key[0] not in STARTS
            or not isinstance(key[1], str)
            or key[1] in {".", ".."}
            or not re.fullmatch(r"[A-Z0-9.^_=\-]{1,64}", key[1])
            or key in identities
            or row.get("state") not in {"prepared", "missing_required_prices"}
        ):
            raise ValueError("El censo histórico no conserva identidades y estados únicos")
        identities.add(key)
    return parent, sources


def _copy_prepared(source, output, clock, cohort, expected):
    origin, _, signature = _prepared(source, clock, cohort, HISTORICAL_MASKED)
    if signature != expected:
        raise ValueError("El activo no conserva la huella de su preparación padre")
    marker = output / "manifest.json"
    if marker.exists():
        _, _, actual = _prepared(output, clock, cohort, HISTORICAL_MASKED)
        if actual != signature:
            raise ValueError("El preparado copiado pertenece a otra edición")
        return origin
    for name, digest in origin["artifacts"].items():
        _copy(source / name, output / name, digest)
    _copy(source / "manifest.json", marker, signature)
    return origin


def prepare_historical_accounting(
    preparation, reviewed_root, output, *, clocks=None, stop_after_assets=None
):
    """Conservar el censo y enriquecer solo activos CN acreditados en un destino nuevo."""
    preparation, reviewed_root, output = map(
        lambda p: Path(p).absolute(), (preparation, reviewed_root, output)
    )
    parent, sources = _parent(preparation)
    catalog, reviewed_sources = reviewed_cn_catalog(reviewed_root)
    sources.update(reviewed_sources)
    if stop_after_assets is not None and (
        type(stop_after_assets) is not int or stop_after_assets < 1
    ):
        raise ValueError("La pausa debe indicar un número positivo de activos")
    prepared = Path(parent["prepared_root"])
    safe_destination(prepared)
    eligible = {
        row["symbol"]
        for row in parent["assets"]
        if row["market"] == "CN" and row["state"] == "prepared"
    }
    if not set(catalog) <= eligible:
        raise ValueError("El censo histórico no contiene todos los emisores CN revisados")
    for refs in catalog.values():
        for ref in refs:
            sources[Path(ref["path"]) / "fundamentals.parquet"] = ref["artifact_sha256"]
    for path in (prepared, preparation, reviewed_root, Path("dataset"), *sources):
        outside_source(path, output)
        outside_source(output, path)
    safe_destination(output)
    markets = sorted({row["market"] for row in parent["assets"]})
    clocks = clocks or {m: MarketClock(m, STARTS[m], "2026-01-01") for m in markets}
    if not set(markets) <= clocks.keys() or any(clocks[m].market != m for m in markets):
        raise ValueError("Falta el calendario correspondiente al mercado")
    code = {name: sha256(Path(__file__).with_name(name)) for name in _CODE}
    identity = dict(
        policy=HISTORICAL_ACCOUNTING,
        input_policy=HISTORICAL_MASKED,
        parent_sha256=sources[preparation],
        source_root=str(prepared.resolve()),
        sources={str(path.resolve()): digest for path, digest in sources.items()},
        reviewed_assets=catalog,
        fundamental_concepts=list(JOINT_CONCEPTS),
        calendars={
            market: _digest([value.isoformat() for value in clocks[market].decisions])
            for market in markets
        },
        code=code,
    )
    _verify(sources)
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        ".edition.lock",
        "configuration.json",
        "manifest.json",
        "progress.json",
        "prepared",
    ):
        safe_destination(output / name)
    with (output / ".edition.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = output / "configuration.json"
        if not config.exists():
            if any(path.name != ".edition.lock" for path in output.iterdir()):
                raise ValueError("El destino contiene una edición no identificada")
            atomic_json(config, identity)
        configured, config_hash = read_manifest(config)
        if not _same(configured, identity):
            raise ValueError("La configuración corresponde a otra edición contable")
        report = dict(
            {
                key: value
                for key, value in parent.items()
                if key not in {"elapsed_seconds", "reused_assets"}
            },
            status="running",
            assets=[],
            failed_assets=0,
            prepared_root=str(output / "prepared"),
            accounting_policy=HISTORICAL_ACCOUNTING,
            configuration=dict(parent.get("configuration", {}), accounting_sha256=config_hash),
            parent_preparation=dict(path=str(preparation), sha256=sources[preparation]),
            reviewed_accounting_assets=sorted(catalog),
            training_ready=False,
        )
        started = time.perf_counter()
        for row in parent["assets"]:
            if stop_after_assets is not None and len(report["assets"]) >= stop_after_assets:
                report["status"] = "paused"
                break
            current = dict(row)
            if row["state"] == "prepared":
                market, symbol = row["market"], row["symbol"]
                source, destination = (
                    prepared / market / symbol,
                    output / "prepared" / market / symbol,
                )
                safe_destination(source)
                safe_destination(destination)
                if sha256(source / "manifest.json") != row["manifest_sha256"]:
                    raise ValueError("Ha cambiado la huella de la preparación padre")
                if market == "CN" and symbol in catalog:
                    refs = catalog[symbol]
                    receipt = derive_chinese_preparation(
                        source / "manifest.json",
                        refs[0]["path"],
                        destination,
                        clock=clocks[market],
                        additional_facts=[ref["path"] for ref in refs[1:]],
                        input_policy=HISTORICAL_MASKED,
                    )
                else:
                    receipt = _copy_prepared(
                        source,
                        destination,
                        clocks[market],
                        parent["cohort_id"],
                        row["manifest_sha256"],
                    )
                current.update(
                    counts=receipt["counts"], manifest_sha256=sha256(destination / "manifest.json")
                )
            report["assets"].append(current)
            atomic_json(output / "progress.json", report)
        else:
            report["status"] = "completed"
        _verify(sources)
        if code != {name: sha256(Path(__file__).with_name(name)) for name in code}:
            raise ValueError("El código cambió durante el enriquecimiento")
        final = output / "manifest.json"
        if report["status"] == "completed":
            if final.exists():
                prior, _ = read_manifest(final)
                if not _same(prior, report):
                    raise ValueError("El manifiesto no conserva el censo y sus identidades")
            else:
                atomic_json(final, report)
        return dict(report, elapsed_seconds=time.perf_counter() - started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "reviewed-cn", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--stop-after-assets", type=int)
    args = parser.parse_args(argv)
    report = prepare_historical_accounting(
        args.prepared, args.reviewed_cn, args.output, stop_after_assets=args.stop_after_assets
    )
    print(
        f"Preparación contable: {report['status']}, {len(report['assets'])} candidatos recorridos."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
