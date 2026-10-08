"""Codificación recuperable del universo preparado, sin recortar activos o filas."""

import argparse
import fcntl
import json
import re
from pathlib import Path

from .accounting_catalog import HISTORICAL_ACCOUNTING, JOINT_CONCEPTS, historical_accounting_context
from .cohort_contexts import MacroVectors
from .cohort_files import read_manifest as _read
from .cohort_files import safe_destination
from .cohort_news import COHORT_POLICIES
from .cohort_samples import materialize_cohort_asset
from .corpus_preparation import STARTS
from .embeddings import EmbeddingCache, FrozenEncoders
from .input_policy import INPUT_POLICIES, STRICT_INPUTS, masked_inputs, policy_identity
from .samples import FUNDAMENTAL_CONCEPTS
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock, aware


def encode_corpus(
    preparation,
    output,
    *,
    macros,
    market_factors=None,
    cache_path=None,
    encoders=None,
    clocks=None,
    context=64,
    fundamental_concepts=FUNDAMENTAL_CONCEPTS,
    source_unit="USD",
    company_factors=True,
    admitted_decisions=None,
    input_policy=STRICT_INPUTS,
    macro_indicators=None,
    accounting_policy=None,
):
    """Procesar todos los candidatos y publicar solo una cobertura completa sin errores."""
    masked = masked_inputs(input_policy)
    if accounting_policy is not None and (
        accounting_policy != HISTORICAL_ACCOUNTING
        or not masked
        or tuple(fundamental_concepts) != FUNDAMENTAL_CONCEPTS
        or source_unit != "USD"
        or company_factors is not True
    ):
        raise ValueError(
            "La política contable conjunta requiere entradas históricas y sus opciones acordadas"
        )
    if masked and (context != 64 or admitted_decisions is not None):
        raise ValueError(
            "La política histórica requiere 64 sesiones y no filtra por completitud macro"
        )
    preparation, output = Path(preparation), Path(output)
    meta, preparation_hash = _read(preparation)
    if meta.get("accounting_policy") is not None and meta["accounting_policy"] != accounting_policy:
        raise ValueError("La preparación requiere su política contable explícita")
    cohort = meta.get("cohort_id")
    if (
        meta.get("schema_version") != (2 if masked else 1)
        or any(meta.get(k) != v for k, v in policy_identity(input_policy).items())
        or meta.get("kind") != "prepared_cohort"
        or meta.get("scope")
        not in {None, "full_corpus", "market_projection", "reviewed_asset_subset"}
        or meta.get("status") != "completed"
        or meta.get("failed_assets") != 0
        or cohort not in COHORT_POLICIES
        or not isinstance(meta.get("assets"), list)
        or not meta["assets"]
        or meta.get("candidate_count") != len(meta["assets"])
    ):
        raise ValueError(
            "La preparación completa debe reconciliar todos sus candidatos sin errores"
        )
    identities = set()
    for asset in meta["assets"]:
        market, symbol = asset.get("market"), asset.get("symbol")
        if (
            market not in STARTS
            or not isinstance(symbol, str)
            or symbol in {".", ".."}
            or not re.fullmatch(r"[A-Z0-9.^_=\-]{1,64}", symbol)
            or (market, symbol) in identities
            or asset.get("state")
            not in (
                {"prepared", "missing_required_prices"}
                if masked
                else {"prepared", "missing_modalities"}
            )
        ):
            raise ValueError("La identidad o el estado de un candidato no es válido")
        identities.add((market, symbol))
    markets = sorted({market for market, _ in identities})
    if not isinstance(macros, dict):
        raise ValueError("Los contextos macro deben identificar sus mercados")
    if not masked and not set(markets) <= set(macros):
        raise ValueError("Falta el contexto macro de un mercado solicitado")
    if type(context) is not int or not 2 <= context <= 512:
        raise ValueError("El contexto debe contener entre 2 y 512 sesiones")
    concepts = tuple(fundamental_concepts)
    if (
        not concepts
        or len(concepts) > 1024
        or len(set(concepts)) != len(concepts)
        or any(not isinstance(name, str) or not name for name in concepts)
        or source_unit not in {"USD", "CAD", "CNY"}
        or type(company_factors) is not bool
    ):
        raise ValueError("La representación contable solicitada no es válida")
    admitted = None
    if admitted_decisions is not None:
        if set(admitted_decisions) != set(markets):
            raise ValueError("La admisión debe identificar todos los mercados seleccionados")
        admitted = {
            market: frozenset(aware(t) for t in moments)
            for market, moments in admitted_decisions.items()
        }
    if sha256(preparation) != preparation_hash:
        raise ValueError("La preparación cambió después de su lectura")
    prepared = Path(meta["prepared_root"]).resolve()
    for source in (prepared, Path("dataset").resolve(), preparation.resolve()):
        outside_source(source, output)
        outside_source(output, source)
    if output.is_symlink():
        raise ValueError("El destino no puede ser un enlace")
    clocks = clocks or {m: MarketClock(m, STARTS[m], "2026-01-01") for m in markets}
    if not set(markets) <= set(clocks) or any(clocks[m].market != m for m in markets):
        raise ValueError("Falta el calendario correspondiente a cada mercado")
    if admitted is not None and any(
        moment.year >= 2024 or moment not in clocks[market].decisions
        for market, moments in admitted.items()
        for moment in moments
    ):
        raise ValueError("La admisión debe pertenecer al calendario anterior a 2024")
    contexts = {
        m: MacroVectors(macros.get(m), input_policy=input_policy, indicators=macro_indicators)
        for m in markets
    }
    factors = market_factors or {}
    for market, item in factors.items():
        if market not in markets or item.get("market") != market:
            raise ValueError("El factor no corresponde al mercado declarado")
        path = Path(item["prices_path"])
        if path.is_symlink() or not path.is_file() or sha256(path) != item["prices_sha256"]:
            raise ValueError("Ha cambiado la fuente del factor de mercado")
    encoders = encoders if encoders is not None else FrozenEncoders()
    identity = dict(
        **policy_identity(input_policy),
        preparation_sha256=preparation_hash,
        prepared_root=str(prepared),
        cohort_id=cohort,
        context_sessions=context,
        encoders=encoders.spec,
        macro_sha256={m: contexts[m].sha256 for m in markets},
        macro_indicators={m: contexts[m].indicators for m in markets},
        market_factors=factors,
        fundamental_concepts=list(JOINT_CONCEPTS if accounting_policy else concepts),
        source_unit=source_unit,
        company_factors=company_factors,
        admitted_decisions={m: sorted(t.isoformat() for t in admitted[m]) for m in markets}
        if admitted is not None
        else None,
        code={
            name: sha256(Path(__file__).with_name(name))
            for name in (
                "corpus_encoding.py",
                "cohort_samples.py",
                "cohort_contexts.py",
                "cohort_files.py",
                "input_policy.py",
            )
        },
    )
    if accounting_policy:
        identity["accounting_policy"] = accounting_policy
        identity["accounting_contexts"] = {
            market: {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in historical_accounting_context(market).items()
            }
            for market in markets
        }
        identity["code"]["accounting_catalog.py"] = sha256(
            Path(__file__).with_name("accounting_catalog.py")
        )
    for name in (
        ".edition.lock",
        "configuration.json",
        "manifest.json",
        "progress.json",
        "samples",
    ):
        if (output / name).is_symlink():
            raise ValueError("Un artefacto de salida es un enlace")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".edition.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = output / "configuration.json"
        if config.exists():
            if _read(config)[0] != identity:
                raise ValueError("La configuración pertenece a otra edición")
        elif any(p.name != ".edition.lock" for p in output.iterdir()):
            raise ValueError("El destino contiene una edición no identificada")
        else:
            atomic_json(config, identity)
        cache_path = Path(cache_path) if cache_path else output / "embeddings.sqlite"
        outside_source(Path("dataset"), cache_path)
        outside_source(prepared, cache_path)
        if cache_path.is_symlink():
            raise ValueError("La caché no puede ser un enlace")
        cache = EmbeddingCache(cache_path)
        result = dict(
            **policy_identity(input_policy),
            schema_version=3 if masked else 2,
            markets=markets,
            preparation_scope=meta.get("scope", "full_corpus"),
            parent_preparation=meta.get("parent_preparation"),
            kind="materialized_corpus",
            cohort_id=cohort,
            news_content_policy=COHORT_POLICIES[cohort],
            scope="development_snapshot",
            cohort_complete=False,
            context_sessions=context,
            samples_root=str((output / "samples").resolve()),
            calendar_start={m: clocks[m].days[0].isoformat() for m in markets},
            assets=[],
            coverage=[],
            samples=0,
            candidate_count=len(meta["assets"]),
            failed_assets=0,
            market_factors=factors,
            configuration=identity,
            final_test_opened=False,
            training_ready=False,
        )
        reused = 0
        try:
            for asset in meta["assets"]:
                market, symbol = asset["market"], asset["symbol"]
                if asset["state"] in {"missing_modalities", "missing_required_prices"}:
                    result["coverage"].append(dict(asset))
                else:
                    source = prepared / market / symbol
                    try:
                        safe_destination(output / "samples" / market / symbol)
                        if not source.resolve().is_relative_to(prepared) or source.is_symlink():
                            raise ValueError("El activo no pertenece al origen declarado")
                        if sha256(source / "manifest.json") != asset["manifest_sha256"]:
                            raise ValueError("Ha cambiado el manifiesto del activo preparado")
                        accounting = (
                            historical_accounting_context(market)
                            if accounting_policy
                            else dict(
                                fundamental_concepts=concepts,
                                source_unit=source_unit,
                                company_factors=company_factors,
                            )
                        )
                        receipt = materialize_cohort_asset(
                            source,
                            output / "samples" / market / symbol,
                            clocks[market],
                            contexts[market],
                            encoders,
                            cache,
                            cohort=cohort,
                            context=context,
                            **accounting,
                            admitted_decisions=admitted[market] if admitted is not None else None,
                            input_policy=input_policy,
                        )
                        if receipt["symbol"] != symbol:
                            raise ValueError("El recibo pertenece a otro activo")
                        reused += int(receipt["reused"])
                        result["coverage"].append(
                            dict(
                                market=market,
                                symbol=symbol,
                                state="encoded",
                                samples=receipt["samples"],
                                fingerprint=receipt["fingerprint"],
                            )
                        )
                        if receipt["samples"]:
                            result["assets"].append(
                                dict(market=market, symbol=symbol, cohort_id=cohort)
                            )
                        result["samples"] += receipt["samples"]
                    except (OSError, ValueError) as error:
                        result["failed_assets"] += 1
                        result["coverage"].append(
                            dict(market=market, symbol=symbol, state="failed", detail=str(error))
                        )
                atomic_json(output / "progress.json", result)
        finally:
            cache.close()
        if sha256(preparation) != identity["preparation_sha256"]:
            raise ValueError("La preparación cambió durante el recorrido")
        if not result["failed_assets"]:
            if meta.get("scope") != "reviewed_asset_subset":
                result.update(scope="full_corpus", cohort_complete=True)
            existing = output / "manifest.json"
            if existing.exists() and _read(existing)[0] != result:
                raise ValueError("El manifiesto confirmado no coincide con el recorrido")
            if not existing.exists():
                atomic_json(existing, result)
        atomic_json(output / "progress.json", result)
        return {**result, "reused_assets": reused}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared", type=Path, required=True, help="Manifiesto completo de preparación"
    )
    parser.add_argument("--output", type=Path, required=True, help="Nueva edición de vectores")
    parser.add_argument("--macro-us", type=Path, help="Contextos macro estadounidenses")
    parser.add_argument("--macro-cn", type=Path, help="Contextos macro chinos")
    parser.add_argument(
        "--market-factors", type=Path, help="Identidades de los factores residuales"
    )
    parser.add_argument("--cache", type=Path, help="Caché persistente de vectores")
    parser.add_argument("--context", type=int, default=64, help="Sesiones de contexto")
    parser.add_argument("--input-policy", choices=INPUT_POLICIES, default=STRICT_INPUTS)
    parser.add_argument("--accounting-policy", choices=(HISTORICAL_ACCOUNTING,))
    parser.add_argument(
        "--macro-catalog", type=Path, help="Catálogo explícito para conservar indicadores ausentes"
    )
    args = parser.parse_args()
    import torch

    from .macro_coverage import _read_catalog

    torch.set_num_threads(4)
    result = encode_corpus(
        args.prepared,
        args.output,
        macros={m: p for m, p in (("US", args.macro_us), ("CN", args.macro_cn)) if p},
        market_factors=_read(args.market_factors)[0] if args.market_factors else None,
        cache_path=args.cache,
        context=args.context,
        input_policy=args.input_policy,
        accounting_policy=args.accounting_policy,
        macro_indicators=sorted(_read_catalog(args.macro_catalog)) if args.macro_catalog else None,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "cohort_id",
                    "candidate_count",
                    "samples",
                    "failed_assets",
                    "cohort_complete",
                    "reused_assets",
                )
            },
            ensure_ascii=False,
        )
    )
    return int(result["failed_assets"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
