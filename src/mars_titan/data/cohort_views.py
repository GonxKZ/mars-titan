"""Vistas inmutables por mercado sobre una preparación terminada e identificada."""

import argparse
import fcntl
import json
import re
from pathlib import Path

from .cohort_files import read_manifest, safe_destination
from .cohort_news import COHORT_POLICIES
from .storage import atomic_json, outside_source, sha256


def project_prepared_cohort(source, output, *, markets):
    """Conservar todos los candidatos seleccionados y los límites del origen completo."""
    source, output = Path(source), Path(output)
    if not markets or len(set(markets)) != len(markets) or not set(markets) <= {"US", "CN"}:
        raise ValueError("La selección de mercados no es válida")
    meta, digest = read_manifest(source)
    cohort, assets = meta.get("cohort_id"), meta.get("assets")
    if (
        meta.get("schema_version") != 1
        or meta.get("kind") != "prepared_cohort"
        or meta.get("scope") not in {None, "full_corpus"}
        or meta.get("status") not in {"completed", "completed_with_errors"}
        or cohort not in COHORT_POLICIES
        or not isinstance(assets, list)
        or not assets
        or type(meta.get("candidate_count")) is not int
        or meta["candidate_count"] != len(assets)
    ):
        raise ValueError("La preparación de origen no está terminada o no concilia sus candidatos")
    keys, failures = set(), 0
    for item in assets:
        market, symbol = item.get("market"), item.get("symbol")
        if (
            market not in {"US", "CN"}
            or not isinstance(symbol, str)
            or symbol in {".", ".."}
            or not re.fullmatch(r"[A-Z0-9.^_=\-]{1,64}", symbol)
            or (market, symbol) in keys
            or item.get("state") not in {"prepared", "failed", "missing_modalities"}
        ):
            raise ValueError("El origen contiene candidatos duplicados o no válidos")
        keys.add((market, symbol))
        failures += item["state"] == "failed"
    available = {market for market, _ in keys}
    if (
        type(meta.get("failed_assets")) is not int
        or meta["failed_assets"] != failures
        or (meta["status"] == "completed" and failures)
        or set(meta.get("configuration", {}).get("markets", [])) != available
        or not set(markets) <= available
    ):
        raise ValueError("Los mercados y recuentos del origen no concilian")
    selected = [a for a in assets if a["market"] in markets]
    if any(a["state"] == "failed" for a in selected):
        raise ValueError("La selección contiene candidatos fallidos y necesita corregir sus fallos")
    prepared = Path(meta["prepared_root"]).resolve()
    safe_destination(output)
    for protected in (source.resolve(), prepared, Path("dataset").resolve()):
        outside_source(protected, output)
    for item in selected:
        if item["state"] != "prepared":
            continue
        path = prepared / item["market"] / item["symbol"] / "manifest.json"
        if not path.resolve().is_relative_to(prepared):
            raise ValueError("El activo está fuera de la preparación declarada")
        receipt, actual = read_manifest(path)
        if actual != item.get("manifest_sha256"):
            raise ValueError("Ha cambiado la huella del manifiesto preparado")
        if any(
            receipt.get(k) != v
            for k, v in dict(market=item["market"], symbol=item["symbol"], cohort_id=cohort).items()
        ):
            raise ValueError("La identidad del activo no coincide con el origen")
    result = dict(
        schema_version=1,
        kind="prepared_cohort",
        scope="market_projection",
        status="completed",
        cohort_id=cohort,
        candidate_count=len(selected),
        failed_assets=0,
        prepared_root=str(prepared),
        assets=selected,
        training_ready=False,
        snapshot_sha256=meta.get("snapshot_sha256"),
        configuration={
            **meta["configuration"],
            "markets": sorted(markets),
            "projection_code_sha256": sha256(Path(__file__)),
            "manifest_reader_sha256": sha256(Path(__file__).with_name("cohort_files.py")),
        },
        parent_preparation=dict(
            path=str(source.resolve()),
            sha256=digest,
            status=meta["status"],
            candidate_count=meta["candidate_count"],
            failed_assets=failures,
            markets=sorted(available),
            excluded_markets=sorted(available - set(markets)),
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(f".{output.name}.lock")
    safe_destination(lock_path)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if sha256(source) != digest:
            raise ValueError("La preparación de origen cambió durante la selección")
        if output.exists():
            if read_manifest(output)[0] != result:
                raise ValueError("El destino pertenece a otra edición o identidad")
        else:
            atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Manifiesto terminado del origen"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Nuevo manifiesto de la selección"
    )
    parser.add_argument(
        "--market",
        choices=["US", "CN", "all"],
        required=True,
        help="Mercados completos que se seleccionan",
    )
    args = parser.parse_args()
    result = project_prepared_cohort(
        args.source, args.output, markets=("US", "CN") if args.market == "all" else (args.market,)
    )
    print(
        json.dumps(
            {k: result[k] for k in ("cohort_id", "candidate_count", "failed_assets", "scope")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
