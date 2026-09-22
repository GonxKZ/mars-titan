"""Órdenes locales para preparar y verificar datos, sin entrenar modelos por defecto."""

import argparse
import json
import sys
from pathlib import Path

from .inventory import inventory
from .storage import atomic_json, sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("inventory", help="Inventariar toda la copia sin modificarla")
    audit.add_argument("--source", type=Path, default=Path("dataset"))
    audit.add_argument(
        "--database", type=Path, default=Path("data/interim/source-inventory.sqlite")
    )
    audit.add_argument("--report", type=Path)
    audit.add_argument("--verify", action="store_true", help="Recalcular todas las huellas")
    corpus = commands.add_parser(
        "corpus-index", help="Indexar todas las fuentes textuales por bloques"
    )
    corpus.add_argument("--source", type=Path, default=Path("dataset"))
    corpus.add_argument("--inventory", type=Path, required=True)
    corpus.add_argument("--database", type=Path, required=True)
    corpus.add_argument("--market", choices=["US", "CN", "all"], default="all")
    corpus.add_argument("--cutoff", default="2023-12-31")
    corpus.add_argument("--report", type=Path)
    state = commands.add_parser("corpus-status", help="Consultar el avance confirmado del índice")
    state.add_argument("--database", type=Path, required=True)
    queue = commands.add_parser("news-queue", help="Crear la cola editorial de todo el índice")
    queue.add_argument("--catalog", type=Path, required=True)
    queue.add_argument("--database", type=Path, required=True)
    queue.add_argument("--manual-reviews", type=Path)
    news_state = commands.add_parser("news-status", help="Consultar verificaciones y pendientes")
    news_state.add_argument("--database", type=Path, required=True)
    news_verify = commands.add_parser(
        "news-verify", help="Contrastar cuerpos sin abrir el test final"
    )
    news_verify.add_argument("--database", type=Path, required=True)
    news_verify.add_argument("--evidence", type=Path, required=True)
    news_verify.add_argument("--stop-after", type=int, help="Corte operativo, no límite del corpus")
    news_verify.add_argument(
        "--network", action="store_true", help="Consultar el editor autorizado"
    )
    news_verify.add_argument("--quota-bytes", type=int, default=2 * 1024**3)
    panels = commands.add_parser("select", help="Seleccionar un panel técnico por sector y volumen")
    panels.add_argument("--source", type=Path, default=Path("dataset"))
    panels.add_argument(
        "--database", type=Path, default=Path("data/interim/source-inventory.sqlite")
    )
    panels.add_argument("--market", choices=["US", "CN"], required=True)
    panels.add_argument("--per-sector", type=int, default=2)
    panels.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser(
        "prepare", help="Normalizar el panel, sin afirmar aptitud de entrenamiento"
    )
    prepare.add_argument("--source", type=Path, default=Path("dataset"))
    prepare.add_argument("--panel", type=Path, required=True)
    prepare.add_argument("--output", type=Path, default=Path("data/processed/phase1"))
    prepare.add_argument("--limit", type=int)
    prepare.add_argument("--start")
    prepare.add_argument("--end", default="2026-01-01")
    prepare.add_argument(
        "--news-reviews",
        type=Path,
        default=Path("data/manifests/news-reviews.json"),
        help="Manifiesto de noticias completas contrastadas",
    )
    prepare.add_argument(
        "--unreviewed-profile",
        action="store_true",
        help="Solo diagnóstico de coste, sin admitir contenido para investigación",
    )
    prices = commands.add_parser("audit-prices", help="Auditar todos los precios inventariados")
    prices.add_argument("--source", type=Path, default=Path("dataset"))
    prices.add_argument(
        "--database", type=Path, default=Path("data/interim/source-inventory.sqlite")
    )
    prices.add_argument("--state", type=Path, default=Path("data/interim/price-audit-state.json"))
    prices.add_argument("--report", type=Path, default=Path("reports/data/price-audit.json"))
    prices.add_argument("--details", type=Path, help="Parquet por activo en un directorio separado")
    macro = commands.add_parser(
        "macro", help="Calcular el contexto macro con unidades y versiones históricas"
    )
    macro.add_argument("--source", type=Path, default=Path("data/external/phase1-macro"))
    macro.add_argument("--catalog", type=Path, default=Path("data/catalogs/macro-indicators.csv"))
    macro.add_argument("--market", choices=["US", "CN"], required=True)
    macro.add_argument("--start", default="2000-01-01")
    macro.add_argument("--end", default="2025-03-31")
    encode = commands.add_parser(
        "encode", help="Materializar cuatro modalidades y el contexto macro en CUDA"
    )
    encode.add_argument("--panel", type=Path, required=True)
    encode.add_argument("--prepared", type=Path, default=Path("data/processed/phase1"))
    encode.add_argument("--output", type=Path, default=Path("data/processed/phase1/samples"))
    encode.add_argument("--start")
    encode.add_argument("--end", default="2026-01-01")
    encode.add_argument(
        "--company-factors",
        action="store_true",
        help="Ampliar el contexto contable con siete ratios de balance auditados",
    )
    encode.add_argument(
        "--cache", type=Path, default=Path("data/embeddings/phase1/representations.sqlite")
    )
    profile = commands.add_parser("profile", help="Medir sondas de coste, sin evaluar predicciones")
    profile.add_argument("--samples", type=Path, default=Path("data/processed/phase1/samples/US"))
    profile.add_argument("--prepared", type=Path, default=Path("data/processed/phase1"))
    profile.add_argument("--report", type=Path, default=Path("reports/data/cost-profile.json"))
    profile.add_argument("--steps", type=int, default=50)
    profile.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    from .storage import outside_source

    if args.command == "news-queue":
        from .news_registry import initialize_verification

        result = initialize_verification(
            args.catalog, args.database, manual_reviews=args.manual_reviews
        )
    elif args.command == "news-status":
        from .news_registry import verification_status

        result = verification_status(args.database)
    elif args.command == "news-verify":
        from .news_registry import verify_pending

        result = verify_pending(
            args.database,
            args.evidence,
            stop_after=args.stop_after,
            network=args.network,
            quota_bytes=args.quota_bytes,
        )
    elif args.command == "corpus-status":
        from .corpus_catalog import corpus_status

        result = corpus_status(args.database)
    elif args.command == "corpus-index":
        from .corpus_catalog import corpus_candidates, index_news

        if args.database.resolve() == args.inventory.resolve():
            raise ValueError("El índice no puede sobrescribir el inventario")
        if args.report:
            outside_source(args.source, args.report)
            if args.report.exists() or args.report.resolve() in {
                args.database.resolve(),
                args.inventory.resolve(),
            }:
                raise ValueError("El informe necesita una ruta nueva y separada de las bases")
        markets = ("US", "CN") if args.market == "all" else (args.market,)
        assets = [
            asset
            for market in markets
            for asset in corpus_candidates(args.source, args.inventory, market)
        ]
        result = index_news(args.source, assets, args.database, cutoff=args.cutoff)
        if args.report:
            atomic_json(args.report, result)
    elif args.command == "inventory":
        result = inventory(args.source, args.database, verify=args.verify)
        if args.report:
            outside_source(args.source, args.report)
            atomic_json(args.report, result)
    elif args.command == "select":
        from .preparation import candidates, select_panel

        outside_source(args.source, args.output)
        pool = candidates(args.source, args.database, args.market)
        selected = select_panel(pool, per_sector=args.per_sector)
        result = {
            "market": args.market,
            "purpose": "engineering_profile_only",
            "seed": 42,
            "selection": "median_then_90th_percentile_source_bytes_per_current_sector",
            "candidate_assets": len(pool),
            "assets": selected,
            "source_database_sha256": sha256(args.database),
        }
        atomic_json(args.output, result)
    elif args.command == "prepare":
        from .news_reviews import load_reviews
        from .preparation import prepare_asset
        from .temporal import MarketClock

        panel = json.loads(args.panel.read_text())
        reviews = None if args.unreviewed_profile else load_reviews(args.news_reviews)
        start = args.start or ("1990-01-01" if panel["market"] == "US" else "2000-01-01")
        clock = MarketClock(panel["market"], start, args.end)
        assets = panel["assets"][: args.limit] if args.limit is not None else panel["assets"]
        if args.limit is not None and args.limit < 1:
            raise ValueError("El límite debe ser positivo")
        summaries = []
        for asset in assets:
            item = prepare_asset(args.source, args.output, asset, clock, news_reviews=reviews)
            brief = {k: item[k] for k in ("symbol", "counts", "elapsed_seconds", "reused")}
            summaries.append(brief)
            print(json.dumps(brief), file=sys.stderr, flush=True)
        result = {"market": panel["market"], "prepared": summaries, "training_ready": False}
    elif args.command == "audit-prices":
        from .audit import audit_prices

        outside_source(args.source, args.report)
        if args.report.resolve() in {args.database.resolve(), args.state.resolve()}:
            raise ValueError("El informe no puede sobrescribir el inventario ni el estado")
        if args.details is not None:
            outside_source(args.details, args.report)
        result = audit_prices(args.source, args.database, args.state, details_root=args.details)
        atomic_json(args.report, result)
    elif args.command == "macro":
        from .macro_preparation import prepare_macro

        result = prepare_macro(
            args.source,
            args.catalog,
            Path(f"data/processed/phase1/macro-{args.market}.parquet"),
            Path(f"reports/data/macro-{args.market}-calculation.json"),
            market=args.market,
            start=args.start,
            end=args.end,
        )
    elif args.command == "encode":
        from .embeddings import EmbeddingCache, FrozenEncoders
        from .samples import materialize_samples, validate_sample_inputs
        from .temporal import MarketClock

        outside_source(Path("dataset"), args.cache)
        panel = json.loads(args.panel.read_text())
        start = args.start or ("1990-01-01" if panel["market"] == "US" else "2000-01-01")
        clock = MarketClock(panel["market"], start, args.end)
        validate_sample_inputs(args.prepared, args.output, panel, clock)
        encoders = FrozenEncoders()
        cache = EmbeddingCache(args.cache)
        try:
            result = materialize_samples(
                args.prepared,
                args.prepared / f"macro-{panel['market']}.parquet",
                args.output,
                panel,
                clock,
                encoders,
                cache,
                company_factors=args.company_factors,
            )
            atomic_json(Path(f"reports/data/encoded-panel-{panel['market'].lower()}.json"), result)
        finally:
            cache.close()
    else:
        import subprocess

        import pyarrow.parquet as pq

        from mars_titan.profiling import profile_grid

        outside_source(Path("dataset"), args.report)
        listing = subprocess.run(
            ["rg", "--files", "--no-ignore", str(args.samples), "-g", "samples.parquet"],
            capture_output=True,
            text=True,
            check=False,
        )
        if listing.returncode not in (0, 1):
            raise ValueError(listing.stderr)
        paths = [
            Path(name)
            for name in listing.stdout.splitlines()
            if pq.ParquetFile(name).metadata.num_rows
        ]
        result = profile_grid(
            paths, args.prepared, args.report, steps=args.steps, repeats=args.repeats
        )
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 2 if result.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
