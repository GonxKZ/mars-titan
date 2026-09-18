"""Descargar el catálogo desde sus fuentes y dejar un manifiesto de cada resultado."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from reference_library import download_reference, read_catalogs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=Path("docs/references/library"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/references/download-log.json"))
    args = parser.parse_args()
    catalogs = args.catalog or [
        Path("docs/references/finance-sources.json"),
        Path("docs/references/neural-sources.json"),
        Path("docs/references/book-sources.json"),
    ]
    entries = read_catalogs(catalogs)
    with ThreadPoolExecutor(max_workers=3) as executor:
        records = list(executor.map(lambda entry: download_reference(entry, args.output), entries))
    manifest = {"retrieved_at": datetime.now(UTC).isoformat(), "records": records}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for record in records:
        print(f"{record['id']}: {record['status']} {record.get('error', '')}")
    print(f"Manifiesto: {args.manifest}")
    return int(any(record["status"] in {"failed", "cached_unverified"} for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
