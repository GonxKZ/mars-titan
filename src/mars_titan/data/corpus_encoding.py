"""Codificación recuperable del universo preparado, sin recortar activos o filas."""

import argparse
import fcntl
import json
import re
import shutil
from functools import partial
from pathlib import Path

from .accounting_catalog import HISTORICAL_ACCOUNTING, JOINT_CONCEPTS, historical_accounting_context
from .cohort_contexts import MacroVectors
from .cohort_files import read_manifest as _read
from .cohort_files import safe_destination
from .cohort_news import COHORT_POLICIES
from .cohort_samples import _digest, materialize_cohort_asset
from .corpus_preparation import STARTS
from .embeddings import EmbeddingCache, FrozenEncoders, encoder_spec, strict_fp32
from .input_policy import INPUT_POLICIES, STRICT_INPUTS, masked_inputs, policy_identity
from .pretraining_free_encoders import PretrainingFreeEncoders
from .price_windows import calendar_digest, check_price_window_contract
from .samples import FUNDAMENTAL_CONCEPTS
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock, aware
from .vector_carry import (
    COMPUTED,
    CarriedTexts,
    CarriedVectors,
    CollectingEncoders,
    CollectingVectors,
    MissingVector,
    PendingVectors,
    ReuseOnlyEncoders,
    encode_pending,
    release_vectors,
    strict_fp32_spec,
    text_carry_identity,
)


def _free_disk_bytes(path):
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def _same_json(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def encode_corpus(
    preparation,
    output,
    *,
    macros,
    market_factors=None,
    cache_path=None,
    encoders=None,
    encoder_options=None,
    clocks=None,
    context=64,
    fundamental_concepts=FUNDAMENTAL_CONCEPTS,
    source_unit="USD",
    company_factors=True,
    admitted_decisions=None,
    input_policy=STRICT_INPUTS,
    macro_indicators=None,
    accounting_policy=None,
    cache_charts=True,
    max_new_assets=None,
    min_free_disk_bytes=0,
    price_window=None,
    vector_carry=None,
    shard=None,
    on_confirmed=None,
    text_carry=None,
):
    """Procesar todos los candidatos y publicar solo una cobertura completa sin errores.

    `price_window` declara las ventanas por sesión del calendario con bit de presencia de la
    edición v3.1. Sin él, las ventanas siguen exigiendo 64 filas consecutivas. `vector_carry`
    nombra una edición anterior con el mismo codificador en FP32 estricto cuyos gráficos y
    textos se reutilizan cuando la entrada es idéntica. `shard=(k, n)` procesa solo los
    candidatos con posición `i % n == k`, comparte el candado con los demás fragmentos y no
    publica el manifiesto. La pasada sin fragmentos reutiliza después cada activo confirmado y
    publica la edición.
    Con `CollectingEncoders`, cada activo se materializa en `collect/` y solo se confirma si no
    le falta ningún vector. Si falta alguno, se descarta y sus entradas quedan pendientes de GPU.
    `on_confirmed(market, symbol)` se llama tras cada activo confirmado, fuera del registro de
    fallos por activo, de modo que cualquier error suyo detiene el recorrido.
    `text_carry` nombra una edición cuyos textos se reutilizan aunque su codificador registre
    otra precisión, siempre que superen el contraste por activo de `encode_pending`. Los gráficos
    nunca se heredan por esta vía.
    """
    if shard is not None and (
        not isinstance(shard, tuple)
        or len(shard) != 2
        or any(type(v) is not int for v in shard)
        or not 0 <= shard[0] < shard[1] <= 16
    ):
        raise ValueError("El fragmento debe ser (k, n) con 0 <= k < n <= 16")
    masked = masked_inputs(input_policy)
    if encoder_options is not None and (
        not isinstance(encoder_options, dict) or encoders is not None
    ):
        raise ValueError("Las opciones requieren construir un codificador nuevo")
    if (
        type(cache_charts) is not bool
        or max_new_assets is not None
        and (type(max_new_assets) is not int or max_new_assets < 1)
        or type(min_free_disk_bytes) is not int
        or min_free_disk_bytes < 0
    ):
        raise ValueError("La política de caché y los límites de ejecución no son válidos")
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
    if min_free_disk_bytes and _free_disk_bytes(output) < min_free_disk_bytes:
        raise OSError("El disco libre no alcanza la reserva de codificación")
    encoders = encoders if encoders is not None else FrozenEncoders(**(encoder_options or {}))
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
    if not cache_charts:
        identity["cache_charts"] = False
    if price_window is not None:
        check_price_window_contract(price_window)
        if set(price_window["calendars"]) != set(markets) or any(
            price_window["calendars"][m]
            != dict(
                start=clocks[m].days[0].isoformat(),
                end=clocks[m].days[-1].isoformat(),
                decisions_sha256=calendar_digest(clocks[m]),
            )
            for m in markets
        ):
            raise ValueError("El contrato de ventanas no declara los calendarios de la edición")
        identity["price_window"] = price_window
    previous = None
    collecting = isinstance(encoders, CollectingEncoders)
    if vector_carry is not None:
        previous = Path(vector_carry).resolve()
        configured, configured_hash = _read(previous / "configuration.json")
        if not strict_fp32_spec(configured.get("encoders")):
            raise ValueError("La edición anterior no registra FP32 estricto y no se hereda")
        if configured.get("encoders") != encoders.spec:
            raise ValueError("La edición anterior usó otro codificador y sus vectores no sirven")
        outside_source(previous, output)
        outside_source(output, previous)
        identity["vector_carry"] = dict(
            edition=str(previous),
            configuration_sha256=configured_hash,
            rule="same_strict_fp32_encoder_and_identical_png_or_text_identity",
        )
    if text_carry is not None:
        identity["text_carry"] = text_carry_identity(text_carry, encoders.spec)
        outside_source(Path(identity["text_carry"]["edition"]), output)
        outside_source(output, Path(identity["text_carry"]["edition"]))
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
        fcntl.flock(lock, (fcntl.LOCK_SH if shard else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        config = output / "configuration.json"
        if shard and not config.exists():
            raise ValueError("Los fragmentos necesitan la configuración ya publicada")
        if config.exists():
            if not _same_json(_read(config)[0], identity):
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
        cache = EmbeddingCache(cache_path, cache_charts=cache_charts)
        # Los vectores calculados en GPU para esta edición y después los de la anterior.
        fallbacks = []
        if (output / COMPUTED).exists():
            fallbacks.append(EmbeddingCache(output / COMPUTED, read_only=True))
        if previous is not None:
            fallbacks.append(EmbeddingCache(previous / "embeddings.sqlite", read_only=True))
        texts = (
            CarriedTexts(output, identity["text_carry"], _digest(encoders.spec))
            if text_carry is not None
            else None
        )
        selecting = bool(fallbacks) or collecting or texts is not None
        if selecting:
            cache = CarriedVectors(
                cache, previous, _digest(encoders.spec), fallbacks=fallbacks, texts=texts
            )
            if collecting:
                name = (
                    f"pending-vectors-{shard[0]}-of-{shard[1]}.sqlite"
                    if shard
                    else "pending-vectors.sqlite"
                )
                cache = CollectingVectors(
                    cache, encoders, PendingVectors(output / name), texts=texts
                )
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
        new_assets = 0
        progress = (
            output / f"progress-{shard[0]}-of-{shard[1]}.json"
            if shard
            else output / "progress.json"
        )
        try:
            for position, asset in enumerate(meta["assets"]):
                market, symbol = asset["market"], asset["symbol"]
                if shard and position % shard[1] != shard[0]:
                    continue
                encoded = False
                if asset["state"] in {"missing_modalities", "missing_required_prices"}:
                    result["coverage"].append(dict(asset))
                else:
                    confirmed = output / "samples" / market / symbol / "manifest.json"
                    if (
                        max_new_assets is not None
                        and new_assets >= max_new_assets
                        and not confirmed.exists()
                    ):
                        result["stop_reason"] = "new_asset_limit"
                        break
                    if min_free_disk_bytes and _free_disk_bytes(output) < min_free_disk_bytes:
                        result["stop_reason"] = "disk_reserve"
                        break
                    new_assets += int(not confirmed.exists())
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
                        if selecting:
                            cache.select(market, symbol, encoders.spec)
                        target = output / "samples" / market / symbol
                        staging = output / "collect" / market / symbol
                        if collecting and not confirmed.exists():
                            # Nunca confirmado: es un resto de una recogida interrumpida.
                            safe_destination(staging)
                            if staging.exists():
                                shutil.rmtree(staging)
                            added = cache.added
                        receipt = materialize_cohort_asset(
                            source,
                            staging if collecting and not confirmed.exists() else target,
                            clocks[market],
                            contexts[market],
                            encoders,
                            cache,
                            cohort=cohort,
                            context=context,
                            **accounting,
                            admitted_decisions=admitted[market] if admitted is not None else None,
                            input_policy=input_policy,
                            price_window=price_window,
                        )
                        if receipt["symbol"] != symbol:
                            raise ValueError("El recibo pertenece a otro activo")
                        if collecting and not confirmed.exists():
                            _confirm_collected(staging, target, cache.added - added)
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
                        encoded = True
                    except (OSError, ValueError) as error:
                        result["failed_assets"] += 1
                        result["coverage"].append(
                            dict(market=market, symbol=symbol, state="failed", detail=str(error))
                        )
                atomic_json(progress, result)
                if encoded and on_confirmed is not None:
                    on_confirmed(market, symbol)
        finally:
            cache.close()
        if sha256(preparation) != identity["preparation_sha256"]:
            raise ValueError("La preparación cambió durante el recorrido")
        if shard:
            return {**result, "reused_assets": reused, "shard": list(shard)}
        if not result["failed_assets"] and len(result["coverage"]) == len(meta["assets"]):
            if meta.get("scope") != "reviewed_asset_subset":
                result.update(scope="full_corpus", cohort_complete=True)
            existing = output / "manifest.json"
            if existing.exists() and not _same_json(_read(existing)[0], result):
                raise ValueError("El manifiesto confirmado no coincide con el recorrido")
            if not existing.exists():
                atomic_json(existing, result)
        atomic_json(output / "progress.json", result)
        return {**result, "reused_assets": reused}


def _confirm_collected(staging, target, pending):
    """Mover un activo completo a su destino o descartarlo si le faltan vectores."""
    if pending:
        shutil.rmtree(staging)
        raise MissingVector(f"{pending} entradas pendientes del codificador en GPU")
    if (target / "manifest.json").exists():
        raise ValueError("El activo ya estaba confirmado")
    if target.exists():
        # Solo quedan la configuración y el candado de un intento anterior sin recibo.
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, help="Manifiesto completo de preparación")
    parser.add_argument("--output", type=Path, required=True, help="Nueva edición de vectores")
    parser.add_argument("--macro-us", type=Path, help="Contextos macro estadounidenses")
    parser.add_argument("--macro-cn", type=Path, help="Contextos macro chinos")
    parser.add_argument(
        "--market-factors", type=Path, help="Identidades de los factores residuales"
    )
    parser.add_argument("--cache", type=Path, help="Caché persistente de vectores")
    parser.add_argument(
        "--cache-charts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Conservar una copia de los gráficos en caché además del Parquet",
    )
    parser.add_argument(
        "--max-new-assets", type=int, help="Pausar tras este número de activos nuevos"
    )
    parser.add_argument(
        "--min-free-disk-bytes", type=int, default=0, help="Reserva de disco entre activos"
    )
    parser.add_argument(
        "--cuda-memory-bytes",
        type=int,
        default=6 * 1024**3,
        help="Límite del asignador Torch, sin incluir el contexto CUDA",
    )
    parser.add_argument(
        "--min-free-cuda-bytes",
        type=int,
        default=0,
        help="Memoria CUDA libre requerida antes de cargar pesos",
    )
    parser.add_argument(
        "--text-batch-size", type=int, default=32, help="Fragmentos por lote de texto"
    )
    parser.add_argument("--image-batch-size", type=int, default=64, help="Gráficos por lote CUDA")
    parser.add_argument(
        "--word-embedding-placement",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Ubicación de la tabla de palabras FP32, con el resto del codificador en CUDA",
    )
    parser.add_argument("--context", type=int, default=64, help="Sesiones de contexto")
    parser.add_argument("--input-policy", choices=INPUT_POLICIES, default=STRICT_INPUTS)
    parser.add_argument("--accounting-policy", choices=(HISTORICAL_ACCOUNTING,))
    parser.add_argument(
        "--macro-catalog", type=Path, help="Catálogo explícito para conservar indicadores ausentes"
    )
    parser.add_argument(
        "--price-window", type=Path, help="Contrato JSON de ventanas por sesión del calendario"
    )
    parser.add_argument(
        "--vector-carry", type=Path, help="Edición anterior con el mismo codificador"
    )
    parser.add_argument(
        "--reuse-only",
        action="store_true",
        help="Solo CPU con vectores existentes. Un vector nuevo deja el activo pendiente",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Solo CPU. Confirma los activos completos y anota los vectores que faltan",
    )
    parser.add_argument(
        "--encode-pending",
        action="store_true",
        help="Solo GPU. Codifica las entradas anotadas por la recogida y termina",
    )
    parser.add_argument("--max-pending", type=int, help="Entradas por tramo de GPU")
    parser.add_argument(
        "--release-vectors",
        action="store_true",
        help="Borra los PNG pendientes y los gráficos calculados de activos ya confirmados",
    )
    parser.add_argument(
        "--substitute-previous",
        type=Path,
        help="Edición anterior cuyas muestras se borran tras verificar cada activo nuevo",
    )
    parser.add_argument(
        "--substitution-records", type=Path, help="Registros de la sustitución activo a activo"
    )
    parser.add_argument(
        "--text-carry",
        type=Path,
        help="Edición cuyos textos se reutilizan tras contrastar una muestra de cada activo",
    )
    parser.add_argument("--shard", type=int, nargs=2, metavar=("K", "N"))
    parser.add_argument(
        "--encoders",
        choices=("frozen", "pretraining_free"),
        default="frozen",
        help="MiniLM y ResNet18 congelados en CUDA o el control sin preentrenamiento en CPU",
    )
    args = parser.parse_args()
    control = args.encoders == "pretraining_free"
    if control and (args.reuse_only or args.collect or args.encode_pending or args.text_carry):
        parser.error("El control sin preentrenamiento calcula sus vectores en una sola pasada")
    if args.reuse_only and args.collect:
        parser.error("--reuse-only y --collect son pasadas distintas")
    if args.prepared is None and not (args.encode_pending or args.release_vectors):
        parser.error("La codificación necesita --prepared")
    if (args.substitute_previous is None) != (args.substitution_records is None):
        parser.error("La sustitución necesita la edición anterior y la carpeta de registros")
    import torch

    from .edition_substitution import substitute_asset
    from .macro_coverage import _read_catalog

    torch.set_num_threads(4)
    # Todas las pasadas nombran o calculan vectores en FP32 estricto, sin TF32.
    strict_fp32()
    if args.release_vectors:
        print(json.dumps(release_vectors(args.output)))
        return 0
    options = dict(
        cuda_memory_bytes=args.cuda_memory_bytes,
        min_free_cuda_bytes=args.min_free_cuda_bytes,
        text_batch_size=args.text_batch_size,
        image_batch_size=args.image_batch_size,
        word_embedding_placement=args.word_embedding_placement,
    )
    if args.encode_pending:
        with (args.output / ".edition.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            counts = encode_pending(
                args.output, FrozenEncoders(**options), max_items=args.max_pending
            )
        # Memoria del asignador de Torch, sin el contexto CUDA que mide nvidia-smi.
        counts["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        counts["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
        print(json.dumps(counts))
        return int(counts["remaining"] > 0)
    spec = (
        encoder_spec(
            text_batch_size=args.text_batch_size,
            image_batch_size=args.image_batch_size,
            word_embedding_placement=args.word_embedding_placement,
        )
        if args.reuse_only or args.collect
        else None
    )
    reuse = (
        ReuseOnlyEncoders(spec)
        if args.reuse_only
        else CollectingEncoders(spec)
        if args.collect
        else None
    )
    substitution = (
        partial(substitute_asset, args.substitute_previous, args.output, args.substitution_records)
        if args.substitute_previous
        else None
    )
    result = encode_corpus(
        args.prepared,
        args.output,
        macros={m: p for m, p in (("US", args.macro_us), ("CN", args.macro_cn)) if p},
        market_factors=_read(args.market_factors)[0] if args.market_factors else None,
        cache_path=args.cache,
        encoders=PretrainingFreeEncoders() if control else reuse,
        encoder_options=None if reuse or control else options,
        cache_charts=args.cache_charts,
        max_new_assets=args.max_new_assets,
        min_free_disk_bytes=args.min_free_disk_bytes,
        context=args.context,
        input_policy=args.input_policy,
        accounting_policy=args.accounting_policy,
        macro_indicators=sorted(_read_catalog(args.macro_catalog)) if args.macro_catalog else None,
        price_window=_read(args.price_window)[0] if args.price_window else None,
        vector_carry=args.vector_carry,
        shard=tuple(args.shard) if args.shard else None,
        on_confirmed=substitution,
        text_carry=args.text_carry,
    )
    summary = {
        k: result[k]
        for k in (
            "cohort_id",
            "candidate_count",
            "samples",
            "failed_assets",
            "cohort_complete",
            "reused_assets",
        )
    }
    if "stop_reason" in result:
        summary["stop_reason"] = result["stop_reason"]
    print(json.dumps(summary, ensure_ascii=False))
    return int(result["failed_assets"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
