"""Vista de medida con uno de cada N activos, enlazados en duro y verificados.

Solo para medir memoria con menos filas que la vista completa. Recorre los activos en el
orden del manifiesto y toma los de posición múltiplo de N cuyas tres tablas (precios,
muestras y etiquetas) existen y conservan la huella SHA-256 del manifiesto. Sus muestras se
enlazan en duro en SALIDA/samples, con el mismo inodo y sin copiar datos. Los precios y las
etiquetas se leen de sus raíces originales. El manifiesto nuevo conserva los activos
elegidos, sus recuentos y su cobertura, y el resto de campos sin cambios.

Un enlace duro mantiene vivo el archivo aunque la edición original lo sustituya, así que la
vista se borra al terminar la medida.

Uso: view_subset.py MANIFIESTO SALIDA --every N
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--every", type=int, required=True)
    args = parser.parse_args()
    if args.every < 1:
        raise SystemExit("--every necesita un entero positivo")
    if args.output.exists():
        raise SystemExit("La vista necesita un directorio nuevo")
    manifest = json.loads(args.manifest.read_text())
    roots = {key: Path(value) for key, value in manifest["roots"].items()}
    samples_root = args.output / "samples"
    chosen, skipped = [], {}
    for position, asset in enumerate(manifest["assets"]):
        if position % args.every:
            continue
        reason = None
        for kind in ("prices", "samples", "labels"):
            root = roots["prepared" if kind == "prices" else kind]
            path = root / asset["market"] / asset["symbol"] / f"{kind}.parquet"
            if not path.is_file():
                reason = f"{kind}_missing"
            elif sha256(path) != asset[f"{kind}_sha256"]:
                reason = f"{kind}_changed"
            if reason:
                break
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        source = roots["samples"] / asset["market"] / asset["symbol"] / "samples.parquet"
        target = samples_root / asset["market"] / asset["symbol"] / "samples.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
        chosen.append(asset)
    if not chosen:
        raise SystemExit("Ningún activo elegido conserva sus tablas")
    identities = {(asset["market"], asset["symbol"]) for asset in chosen}
    view = copy.deepcopy(manifest)
    view["roots"] = dict(manifest["roots"], samples=str(samples_root.resolve()))
    view["assets"] = chosen
    view["counts"] = {
        partition: sum(asset["counts"][partition] for asset in chosen)
        for partition in manifest["counts"]
    }
    view["coverage"] = [
        row for row in manifest["coverage"] if (row["market"], row["symbol"]) in identities
    ]
    view["candidate_count"] = len(view["coverage"])
    view["samples"] = sum(asset["samples"] for asset in chosen)
    (args.output / "manifest.json").write_text(json.dumps(view, indent=1) + "\n")
    summary = dict(
        every=args.every,
        considered=len(range(0, len(manifest["assets"]), args.every)),
        assets=len(chosen),
        skipped=skipped,
        markets={
            market: sum(asset["market"] == market for asset in chosen) for market in ("US", "CN")
        },
        counts=view["counts"],
        source_counts=manifest["counts"],
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
