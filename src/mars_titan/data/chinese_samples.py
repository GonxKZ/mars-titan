"""Codificar un activo chino revisado sin declararlo un corpus completo."""

import argparse
import fcntl
import json
import re
from pathlib import Path

from mars_titan.training.corpus_targets import prepare_corpus_targets

from .china_preparation import derive_chinese_preparation
from .cohort_files import read_manifest, safe_destination
from .corpus_encoding import encode_corpus
from .currency_samples import _macro_inputs
from .embeddings import FrozenEncoders
from .macro_coverage import _read_catalog
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock

CNY_CONCEPTS = (
    "cn-reported:Assets:CNY",
    "cn-reported:Liabilities:CNY",
    "cn-reported:EquityIncludingNoncontrollingInterest:CNY",
)
_CODE = (
    "chinese_samples.py",
    "china_preparation.py",
    "cohort_samples.py",
    "corpus_encoding.py",
    "currency_samples.py",
    "cohort_contexts.py",
    "cohort_files.py",
    "macro_coverage.py",
    "storage.py",
    "temporal.py",
)


def _parent(path, facts_edition):
    parent, signature = read_manifest(path)
    facts, facts_hash = read_manifest(facts_edition / "report.json", maximum=1024**2)
    symbol = facts.get("identity", {}).get("symbol")
    if (
        parent.get("schema_version") != 1
        or parent.get("kind") != "prepared_cohort"
        or parent.get("status") != "completed"
        or parent.get("failed_assets") != 0
        or parent.get("cohort_id") != "original_audited"
        or not isinstance(parent.get("assets"), list)
        or parent.get("candidate_count") != len(parent["assets"])
        or not isinstance(symbol, str)
        or not re.fullmatch(r"\d{6}\.(?:SS|SH|SZ)", symbol)
    ):
        raise ValueError("El padre y la revisión no identifican un activo preparado")
    keys = [(a.get("market"), a.get("symbol")) for a in parent["assets"]]
    if len(set(keys)) != len(keys):
        raise ValueError("El padre contiene candidatos duplicados")
    selected = [a for a in parent["assets"] if (a.get("market"), a.get("symbol")) == ("CN", symbol)]
    if len(selected) != 1 or selected[0].get("state") != "prepared":
        raise ValueError("El emisor revisado no es un candidato preparado del origen")
    original = Path(parent["prepared_root"]) / "CN" / symbol / "manifest.json"
    safe_destination(original)
    if sha256(original) != selected[0].get("manifest_sha256"):
        raise ValueError("La huella del activo preparado no coincide con su padre")
    return (
        parent,
        symbol,
        original,
        {
            path: signature,
            facts_edition / "report.json": facts_hash,
            original: selected[0]["manifest_sha256"],
        },
    )


def _verify(sources, code):
    for path, digest in sources.items():
        safe_destination(path)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError("Una fuente cambió durante la codificación china")
    if any(sha256(Path(__file__).with_name(name)) != digest for name, digest in code.items()):
        raise ValueError("El código cambió durante la codificación china")


def _confirmed(path, result):
    expected = {key: value for key, value in result.items() if key != "reused_assets"}
    actual, signature = read_manifest(path)
    if actual != expected:
        raise ValueError("El recibo de una etapa no conserva su identidad confirmada")
    return signature


def prepare_chinese_samples(
    parent_preparation,
    facts_edition,
    output,
    *,
    macro_path,
    admission_path,
    catalog_path,
    market_factors,
    encoders=None,
    clock=None,
):
    """Derivar, codificar y etiquetar un activo con las 140 posiciones macro completas.

    Las particiones anuales de la supervisión son preliminares. Esta comprobación
    de datos no autoriza una comparación temporal con una población insuficiente.
    """
    (
        parent_preparation,
        facts_edition,
        output,
        macro_path,
        admission_path,
        catalog_path,
        market_factors,
    ) = map(
        Path,
        (
            parent_preparation,
            facts_edition,
            output,
            macro_path,
            admission_path,
            catalog_path,
            market_factors,
        ),
    )
    safe_destination(output)
    code = {name: sha256(Path(__file__).with_name(name)) for name in _CODE}
    parent, symbol, original, sources = _parent(parent_preparation, facts_edition)
    for protected in (
        parent_preparation,
        Path(parent["prepared_root"]),
        facts_edition,
        macro_path,
        admission_path.parent,
        catalog_path,
        market_factors,
        Path("dataset"),
    ):
        outside_source(protected, output)
        outside_source(output, protected)
    clock = clock or MarketClock("CN", "1990-12-19", "2026-01-01")
    if clock.market != "CN":
        raise ValueError("La muestra china necesita su calendario CN")
    identifiers = sorted(_read_catalog(catalog_path))
    macros, admitted, admission_hash, decisions_hash = _macro_inputs(
        macro_path, admission_path, catalog_path, identifiers, market="CN"
    )
    if any(t not in clock.decisions for t in admitted):
        raise ValueError("La admisión macro incluye decisiones ajenas al calendario")
    factors, factors_hash = read_manifest(market_factors, maximum=1024**2)
    if set(factors) != {"CN"} or factors["CN"].get("market") != "CN":
        raise ValueError("Falta el factor propio del mercado chino")
    factor_path = Path(factors["CN"]["prices_path"])
    outside_source(output, factor_path)
    sources.update(
        {
            macro_path: macros.sha256,
            admission_path: admission_hash,
            admission_path.parent / "complete-decisions.parquet": decisions_hash,
            catalog_path: sha256(catalog_path),
            market_factors: factors_hash,
            factor_path: factors["CN"]["prices_sha256"],
        }
    )
    _verify(sources, code)
    encoders = encoders if encoders is not None else FrozenEncoders()
    identity = dict(
        policy="reviewed_cn_asset_complete_macro_v1",
        symbol=symbol,
        sources={str(path.resolve()): digest for path, digest in sources.items()},
        code=code,
        encoders=encoders.spec,
        fundamental_concepts=list(CNY_CONCEPTS),
        parent_candidate_count=parent["candidate_count"],
        source_unit="CNY",
        context_sessions=64,
    )
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        ".edition.lock",
        "configuration.json",
        "preparation.json",
        "report.json",
        "prepared",
        "encoded",
        "supervised",
    ):
        safe_destination(output / name)
    with (output / ".edition.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configuration = output / "configuration.json"
        if not configuration.exists():
            if any(p.name != ".edition.lock" for p in output.iterdir()):
                raise ValueError("La salida contiene una edición sin identidad")
            atomic_json(configuration, identity)
        configuration_hash = _confirmed(configuration, identity)
        destination = output / "prepared/CN" / symbol
        derive_chinese_preparation(original, facts_edition, destination, clock=clock)
        selection = dict(
            schema_version=1,
            kind="prepared_cohort",
            scope="reviewed_asset_subset",
            status="completed",
            cohort_id="original_audited",
            candidate_count=1,
            failed_assets=0,
            prepared_root=str((output / "prepared").resolve()),
            configuration=dict(markets=["CN"]),
            parent_preparation=dict(
                path=str(parent_preparation.resolve()),
                sha256=sources[parent_preparation],
                candidate_count=parent["candidate_count"],
                selection="reviewed_asset",
                selected_symbol=symbol,
            ),
            assets=[
                dict(
                    market="CN",
                    symbol=symbol,
                    state="prepared",
                    manifest_sha256=sha256(destination / "manifest.json"),
                )
            ],
        )
        selection_path = output / "preparation.json"
        if not selection_path.exists():
            atomic_json(selection_path, selection)
        selection_hash = _confirmed(selection_path, selection)
        encoded = encode_corpus(
            selection_path,
            output / "encoded",
            macros={"CN": macro_path},
            market_factors=factors,
            encoders=encoders,
            clocks={"CN": clock},
            fundamental_concepts=CNY_CONCEPTS,
            source_unit="CNY",
            company_factors=False,
            admitted_decisions={"CN": admitted},
        )
        if encoded["failed_assets"] or not encoded["samples"]:
            raise ValueError("El activo no ha producido muestras multimodales completas")
        encoded_path = output / "encoded/manifest.json"
        encoded_hash = _confirmed(encoded_path, encoded)
        supervised = prepare_corpus_targets(
            encoded_path, output / "prepared", output / "supervised"
        )
        supervised_path = output / "supervised/manifest.json"
        supervised_hash = _confirmed(supervised_path, supervised)
        _verify(
            {
                **sources,
                configuration: configuration_hash,
                selection_path: selection_hash,
                encoded_path: encoded_hash,
                supervised_path: supervised_hash,
            },
            code,
        )
        report = dict(
            schema_version=1,
            status="completed",
            symbol=symbol,
            scope="development_snapshot",
            cohort_complete=False,
            parent_candidate_count=parent["candidate_count"],
            samples=encoded["samples"],
            counts=supervised["counts"],
            configuration_sha256=configuration_hash,
            artifacts={
                "preparation.json": selection_hash,
                "encoded/manifest.json": encoded_hash,
                "supervised/manifest.json": supervised_hash,
            },
            training_ready=False,
            final_test_opened=False,
        )
        if (output / "report.json").exists():
            if read_manifest(output / "report.json")[0] != report:
                raise ValueError("El recibo no conserva el resultado de la edición")
        else:
            atomic_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "parent-preparation",
        "facts-edition",
        "output",
        "macro-path",
        "admission-path",
        "catalog-path",
        "market-factors",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    import torch

    torch.set_num_threads(4)
    report = prepare_chinese_samples(**vars(args))
    print(
        json.dumps(
            {k: report[k] for k in ("symbol", "samples", "counts", "scope", "training_ready")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
