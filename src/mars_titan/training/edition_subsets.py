"""Subconjuntos declarados del catálogo de la edición, reproducibles bit a bit.

Una definición elige filas del catálogo con reglas fijas y sin mirar ningún objetivo:

- `markets`: mercados incluidos,
- `years`: primer y último año de la sesión, ambos incluidos,
- `labels`: `usable` conserva las filas cuyo objetivo puede entrar en una vista (aceptado o
  recuperado en el corte anual) y `all` todas las muestras,
- `modalities.required`: bloques que deben estar presentes en la fila,
- `liquidity`: universo de liquidez de cada año, decidido en su corte,
- `max_per_asset`: máximo de filas por activo,
- `sample`: fracción por estrato (`market`, `year` o `target_partition`) con semilla.

El orden de las reglas es el de la lista. El máximo por activo y el muestreo conservan las
filas con menor clave de orden, una mezcla `splitmix64` de la semilla, del activo y del
instante de la predicción (`row_order_v1`). La semilla es la del muestreo, o 0 sin él. La
clave no depende del orden de lectura ni de otras filas, así que la misma definición sobre
el mismo catálogo da los mismos identificadores. El número de filas de cada estrato es
`floor(fracción · n + 1/2)`.

**Liquidez.** El corte de un año es el 1 de enero a las 00:00 UTC. Cada activo recibe en el
corte el estrato de `evaluation.liquidity_strata` con la edición sin ajustar: el último
cierre negociado verificado y la mediana del volumen de las `volume_sessions` filas que
terminan en la última sesión publicada antes del corte, con los umbrales declarados. Las
filas cuya sesión cae en ese año se conservan si el estrato del activo está en `keep`. Así
el universo de un año solo depende de sesiones anteriores al año, y por tanto anteriores a
la predicción de cada fila y a cualquier corte del walk-forward que empiece en él o después.
Un activo sin sesiones verificadas suficientes antes del corte queda `unclassified`, igual
que en la evaluación. La definición se declara con su fecha y la identidad de la edición
antes de mirar ningún resultado, y no usa objetivos ni predicciones.

El resultado es `ids.parquet` con (`market`, `asset_id`, `prediction_at`) ordenados y su
manifiesto con la definición, la huella del catálogo y la de los identificadores. Esa
identidad entra en las vistas y en la campaña que los usan.
"""

import argparse
import hashlib
import json
import math
import tempfile
import time
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import MODALITIES
from mars_titan.data.macro_coverage import _publish_directory
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation import liquidity_strata as ls

from .edition_catalog import KEYS, STAMP, read_catalog_manifest

SCHEMA_VERSION = 1
DEFINITION = "edition_subset_definition"
KIND = "edition_subset"
IDS_FILE = "ids.parquet"
ROW_ORDER = "row_order_v1"
LABELS = ("usable", "all")
STRATA = ("market", "year", "target_partition")
IDS_SCHEMA = pa.schema(
    [("market", pa.string()), ("asset_id", pa.string()), ("prediction_at", STAMP)]
)
_FIELDS = {"schema_version", "kind", "name", "markets", "years", "labels"}
_OPTIONAL = {"modalities", "liquidity", "max_per_asset", "sample", "notes"}
# Corte del universo de liquidez: el comienzo del año natural de la sesión, en UTC.
LIQUIDITY_CUT = "calendar_year_start_utc_v1"
_LIQUIDITY_FIELDS = {
    "declared_at",
    "source",
    "edition_id",
    "cut",
    "price_below",
    "volume_below",
    "volume_sessions",
    "keep",
}
_COLUMNS = [
    *KEYS,
    "session",
    "label_reason",
    "annual_boundary_label",
    "target_partition",
    *[f"has_{name}" for name in MODALITIES],
]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def validate_definition(definition):
    """Comprobar la definición completa antes de leer el catálogo."""
    _require(isinstance(definition, dict), "La definición del subconjunto debe ser un objeto")
    keys = set(definition)
    _require(
        _FIELDS <= keys <= _FIELDS | _OPTIONAL
        and definition["schema_version"] == SCHEMA_VERSION
        and definition["kind"] == DEFINITION,
        "La definición del subconjunto no cumple su contrato",
    )
    name = definition["name"]
    _require(
        isinstance(name, str) and 1 <= len(name) <= 96 and name.isascii(),
        "El subconjunto necesita un nombre ASCII corto",
    )
    markets = definition["markets"]
    _require(
        isinstance(markets, list)
        and markets
        and markets == sorted(set(markets))
        and set(markets) <= {"CN", "US"},
        "Los mercados deben ser CN o US, ordenados y sin repetir",
    )
    years = definition["years"]
    _require(
        isinstance(years, dict)
        and set(years) == {"first", "last"}
        and _integer(years["first"], 2000, 2023)
        and _integer(years["last"], years["first"], 2023),
        "Los años deben estar entre 2000 y 2023 y en orden",
    )
    _require(definition["labels"] in LABELS, f"labels debe ser uno de {LABELS}")
    modalities = definition.get("modalities", {"required": []})
    _require(
        isinstance(modalities, dict)
        and set(modalities) == {"required"}
        and isinstance(modalities["required"], list)
        and modalities["required"] == [m for m in MODALITIES if m in modalities["required"]]
        and len(set(modalities["required"])) == len(modalities["required"]),
        "Las modalidades requeridas deben seguir el orden de la máscara sin repetir",
    )
    if "liquidity" in definition:
        validate_liquidity(definition["liquidity"])
    limit = definition.get("max_per_asset")
    _require(limit is None or _integer(limit, 1, 1_000_000), "Máximo por activo inválido")
    sample = definition.get("sample")
    if sample is not None:
        _require(
            isinstance(sample, dict)
            and set(sample) == {"fraction", "strata", "seed"}
            and type(sample["fraction"]) is float
            and 0.0 < sample["fraction"] < 1.0
            and isinstance(sample["strata"], list)
            and sample["strata"] == [s for s in STRATA if s in sample["strata"]]
            and len(set(sample["strata"])) == len(sample["strata"])
            and _integer(sample["seed"], 0, 2**63 - 1),
            "El muestreo necesita fracción en (0, 1), estratos declarados y semilla",
        )
    _require(
        "notes" not in definition or isinstance(definition["notes"], str),
        "Las notas deben ser texto",
    )
    return definition


def _iso_date(value):
    try:
        return isinstance(value, str) and date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def validate_liquidity(section):
    """Declaración del universo de liquidez, con los umbrales y la ventana de la evaluación."""
    _require(
        isinstance(section, dict) and set(section) == _LIQUIDITY_FIELDS,
        "La liquidez del subconjunto no tiene exactamente sus campos",
    )
    _require(
        _iso_date(section["declared_at"])
        and section["source"] == ls.SOURCE
        and section["cut"] == LIQUIDITY_CUT
        and isinstance(section["edition_id"], str)
        and len(section["edition_id"]) == 64
        and all(c in "0123456789abcdef" for c in section["edition_id"]),
        "La liquidez se declara con su fecha, su fuente, su corte y la edición sin ajustar",
    )
    price = section["price_below"]
    _require(
        type(price) in (int, float) and 0 < price <= 1_000,
        "El umbral de precio debe ser positivo y fijado de antemano",
    )
    _require(
        _integer(section["volume_below"], 1, ls.MAX_THRESHOLD)
        and _integer(section["volume_sessions"], 1, ls.MAX_VOLUME_SESSIONS),
        "El umbral de volumen y su ventana deben ser enteros positivos acotados",
    )
    keep = section["keep"]
    _require(
        isinstance(keep, list)
        and keep
        and keep == [name for name in ls.NAMES if name in keep]
        and len(set(keep)) == len(keep),
        "Los estratos conservados deben ser de la evaluación, en su orden y sin repetir",
    )
    return section


def definition_sha256(definition):
    return hashlib.sha256(
        json.dumps(validate_definition(definition), sort_keys=True).encode()
    ).hexdigest()


def _mix(values):
    """splitmix64 sobre enteros sin signo de 64 bits, con desbordamiento modular.

    La aritmética de `uint64` ya es módulo 2**64, así que se opera en el sitio sobre una sola
    copia de la entrada.
    """
    with np.errstate(over="ignore"):
        z = values + np.uint64(0x9E3779B97F4A7C15)
        z ^= z >> np.uint64(30)
        z *= np.uint64(0xBF58476D1CE4E5B9)
        z ^= z >> np.uint64(27)
        z *= np.uint64(0x94D049BB133111EB)
        z ^= z >> np.uint64(31)
        return z


def _asset_digest(market, asset_id):
    payload = f"{market}/{asset_id}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def row_order(seed, assets, prediction):
    """Clave de orden de cada fila: la semilla, el activo y el instante, nada más."""
    seeded = _mix(np.array([seed], dtype=np.uint64))[0]
    moments = _mix(prediction.astype(np.int64, copy=False).view(np.uint64) ^ seeded)
    moments ^= assets
    return _mix(moments)


class _Codes:
    """Códigos enteros estables dentro de una lectura para textos repetidos."""

    def __init__(self, names=()):
        self.names, self.index = list(names), {name: n for n, name in enumerate(names)}

    def encode(self, column):
        encoded = column.combine_chunks().dictionary_encode()
        local = [
            self.index.setdefault(name, len(self.index)) for name in encoded.dictionary.to_pylist()
        ]
        self.names.extend(list(self.index)[len(self.names) :])
        return np.asarray(local, dtype=np.int32)[encoded.indices.to_numpy()]


def _selected_rows(catalog, definition):
    """Filas que pasan los filtros por fila, leídas por grupos y sin vectores.

    Cada fila guarda solo códigos compactos (activo, año y partición) y su instante. El
    mercado y la huella de cada activo se guardan una vez por activo.
    """
    years = definition["years"]
    required = definition.get("modalities", {"required": []})["required"]
    assets, partitions = _Codes(), _Codes(("none",))
    parts = []
    with pq.ParquetFile(catalog) as file:
        for group in range(file.num_row_groups):
            table = file.read_row_group(group, columns=_COLUMNS, use_threads=False)
            year = pc.year(table["session"])
            keep = pc.and_(
                pc.is_in(table["market"], pa.array(definition["markets"])),
                pc.and_(pc.greater_equal(year, years["first"]), pc.less_equal(year, years["last"])),
            )
            if definition["labels"] == "usable":
                usable = pc.or_(
                    pc.equal(table["label_reason"], "accepted"), table["annual_boundary_label"]
                )
                keep = pc.and_(keep, pc.fill_null(usable, False))
            for name in required:
                keep = pc.and_(keep, table[f"has_{name}"])
            table = table.filter(keep)
            if not table.num_rows:
                continue
            parts.append(
                dict(
                    asset=assets.encode(
                        pc.binary_join_element_wise(table["market"], table["asset_id"], "/")
                    ),
                    prediction=table["prediction_at"].cast(pa.int64()).to_numpy(),
                    year=pc.year(table["session"]).cast(pa.int16()).to_numpy(),
                    partition=partitions.encode(table["target_partition"].fill_null("none")),
                )
            )
    if not parts:
        return None
    rows = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    keys = [name.split("/", 1) for name in assets.names]
    rows["asset_market"] = np.array([market == "US" for market, _ in keys], dtype=np.int8)
    rows["asset_digest"] = np.array([_asset_digest(*key) for key in keys], dtype=np.uint64)
    rows["partition_names"] = partitions.names
    rows["keys"] = keys
    return rows


def _first_by_group(groups, order, limit_of):
    """Posiciones con menor clave dentro de cada grupo, con el límite que da `limit_of`.

    `groups` son enteros no negativos. Solo importa qué filas comparten grupo, así que se
    ordenan en el tipo sin signo más pequeño que los contiene, que numpy ordena por radix.
    `lexsort` es estable y los empates de la clave siguen el orden de las filas.
    """
    if not len(groups):
        return np.empty(0, dtype=np.int64)
    groups = groups.astype(np.min_scalar_type(int(groups.max())), copy=False)
    ranked = np.lexsort((order, groups))
    sorted_groups = groups[ranked]
    starts = np.flatnonzero(np.r_[True, sorted_groups[1:] != sorted_groups[:-1]])
    sizes = np.diff(np.r_[starts, len(ranked)])
    rank = np.arange(len(ranked)) - np.repeat(starts, sizes)
    limits = np.repeat([limit_of(int(size)) for size in sizes], sizes)
    # Marcar y recorrer da las posiciones ya ordenadas sin un segundo sort.
    chosen = np.zeros(len(groups), dtype=bool)
    chosen[ranked[rank < limits]] = True
    return np.flatnonzero(chosen)


def _year_span(rows):
    first = int(rows["year"].min())
    return first, int(rows["year"].max()) - first + 1


def _strata(rows, keep, names):
    """Estrato de cada fila conservada, por mercado, año o partición.

    Es una clave entera en base mixta: filas con la misma clave comparten estrato y su valor
    no interviene en la selección.
    """
    first, span = _year_span(rows)
    columns = dict(
        market=lambda: (rows["asset_market"][rows["asset"][keep]], 2),
        year=lambda: (rows["year"][keep] - first, span),
        target_partition=lambda: (rows["partition"][keep], len(rows["partition_names"])),
    )
    stratum = np.zeros(len(keep), dtype=np.int64)
    for name in names:
        values, radix = columns[name]()
        stratum *= radix
        stratum += values
    return stratum


def _present(key, size):
    """`np.unique(key, return_inverse=True)` para enteros en [0, size) sin ordenar las filas."""
    present = np.zeros(size, dtype=bool)
    present[key] = True
    position = np.cumsum(present, dtype=np.int64) - 1
    return np.flatnonzero(present), position[key]


def year_start(year):
    """Corte de un año: el 1 de enero a las 00:00 UTC, en microsegundos."""
    return int(np.datetime64(f"{year:04d}-01-01", "us").astype(np.int64))


def liquidity_edition(section, root):
    """Asignador de estratos de la edición sin ajustar que nombra la definición."""
    _require(root is not None, "La liquidez del subconjunto necesita la edición sin ajustar")
    assigner = ls.EditionLiquidity(root, section)
    _require(
        assigner.manifest["edition_id"] == section["edition_id"],
        "La edición sin ajustar no es la que declara el subconjunto",
    )
    return assigner


def _universe(rows, section, assigner):
    """Estrato de cada (activo, año) en el corte del año y si sus filas se conservan.

    Los pares salen ordenados por activo y año, el orden de `np.unique` sobre las dos columnas.
    """
    first, span = _year_span(rows)
    pairs, inverse = _present(
        rows["asset"].astype(np.int64) * span + (rows["year"] - first), len(rows["keys"]) * span
    )
    keys = [rows["keys"][code] for code in (pairs // span).tolist()]
    years = (pairs % span + first).tolist()
    markets = np.array([market for market, _ in keys], dtype=object)
    assets = np.array([f"{market}/{asset}" for market, asset in keys], dtype=object)
    cuts = np.array([year_start(year) for year in years], dtype=np.int64)
    codes, _ = assigner.assign(markets, assets, cuts)
    kept = np.isin(codes, [ls.NAMES.index(name) for name in section["keep"]])
    universe = {}
    for (market, _), year, code in zip(keys, years, codes.tolist(), strict=True):
        counts = universe.setdefault(market, {}).setdefault(str(year), dict.fromkeys(ls.NAMES, 0))
        counts[ls.NAMES[code]] += 1
    return kept[inverse], universe


def select(catalog, definition, liquidity=None, *, universe=None):
    """Tabla de identificadores del subconjunto, ordenada por mercado, activo e instante.

    `liquidity` es el asignador de `liquidity_edition` cuando la definición declara un
    universo de liquidez. Si se pasa `universe`, un diccionario, recibe los activos de cada
    estrato por mercado y año.
    """
    definition = validate_definition(definition)
    _require(
        ("liquidity" in definition) == (liquidity is not None),
        "Solo una definición con liquidez usa la edición sin ajustar",
    )
    rows = _selected_rows(catalog, definition)
    if rows is None:
        return IDS_SCHEMA.empty_table()
    sample = definition.get("sample")
    digests = rows["asset_digest"][rows["asset"]]
    order = row_order((sample or {}).get("seed", 0), digests, rows["prediction"])
    del digests
    keep = np.arange(len(order))
    if liquidity is not None:
        _require(
            liquidity.manifest["edition_id"] == definition["liquidity"]["edition_id"]
            and liquidity.section == definition["liquidity"],
            "El asignador de liquidez no corresponde a la definición",
        )
        kept, counts = _universe(rows, definition["liquidity"], liquidity)
        keep = keep[kept]
        if universe is not None:
            universe.update(counts)
    if definition.get("max_per_asset") is not None:
        limit = definition["max_per_asset"]
        keep = keep[_first_by_group(rows["asset"][keep], order[keep], lambda _size: limit)]
    if sample is not None:
        fraction = sample["fraction"]
        groups = _strata(rows, keep, sample["strata"])
        chosen = _first_by_group(groups, order[keep], lambda n: math.floor(fraction * n + 0.5))
        keep = keep[chosen]
    codes = pa.array(rows["asset"][keep])
    table = pa.table(
        dict(
            market=pa.array([market for market, _ in rows["keys"]]).take(codes),
            asset_id=pa.array([asset for _, asset in rows["keys"]]).take(codes),
            prediction_at=pa.array(rows["prediction"][keep], pa.int64()).cast(STAMP),
        ),
        schema=IDS_SCHEMA,
    )
    return table.sort_by([(name, "ascending") for name in KEYS])


def _counts(table):
    """Filas por mercado y año de la predicción y activos del subconjunto."""
    years = pc.year(table["prediction_at"])
    grouped = (
        pa.table(dict(market=table["market"], year=years))
        .group_by(["market", "year"])
        .aggregate([([], "count_all")])
        .sort_by([("market", "ascending"), ("year", "ascending")])
    )
    by_year = {}
    for market, year, total in zip(
        *(grouped[c].to_pylist() for c in ("market", "year", "count_all")), strict=True
    ):
        by_year.setdefault(market, {})[str(year)] = total
    assets = (
        pa.table(dict(market=table["market"], asset_id=table["asset_id"]))
        .group_by(["market", "asset_id"])
        .aggregate([])
    )
    return dict(
        rows=table.num_rows,
        assets=assets.num_rows,
        markets={m: sum(v.values()) for m, v in sorted(by_year.items())},
        prediction_years=by_year,
    )


def materialize(catalog_manifest, definition_path, output, *, liquidity_root=None):
    """Escribir los identificadores y su manifiesto en una carpeta nueva.

    `liquidity_root` es la carpeta de la edición sin ajustar que nombra una definición con
    universo de liquidez. El manifiesto guarda su ruta y los activos de cada estrato en cada
    corte, para que `verify_subset` lo repita.
    """
    began = time.perf_counter()
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("El subconjunto ya existe")
    safe_destination(output)
    manifest, manifest_sha256, catalog = read_catalog_manifest(catalog_manifest)
    definition, declared_sha256 = read_manifest(Path(definition_path), 1024**2)
    validate_definition(definition)
    assigner, universe = None, {}
    if "liquidity" in definition:
        assigner = liquidity_edition(definition["liquidity"], liquidity_root)
    else:
        _require(liquidity_root is None, "La definición no declara un universo de liquidez")
    table = select(catalog, definition, assigner, universe=universe)
    _require(table.num_rows > 0, "La definición no conserva ninguna fila")
    _require(sha256(catalog) == manifest["catalog"]["sha256"], "El catálogo cambió al leerlo")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        stage = Path(temporary) / "subset"
        stage.mkdir()
        path = stage / IDS_FILE
        pq.write_table(table, path, compression="zstd", row_group_size=1 << 20)
        document = dict(
            schema_version=SCHEMA_VERSION,
            kind=KIND,
            row_order=ROW_ORDER,
            definition=definition,
            definition_sha256=definition_sha256(definition),
            definition_file_sha256=declared_sha256,
            catalog=dict(
                manifest=str(Path(catalog_manifest).resolve()),
                manifest_sha256=manifest_sha256,
                sha256=manifest["catalog"]["sha256"],
                rows=manifest["catalog"]["rows"],
            ),
            ids=dict(file=IDS_FILE, sha256=sha256(path), bytes=path.stat().st_size),
            counts=_counts(table),
            elapsed_seconds=time.perf_counter() - began,
        )
        if assigner is not None:
            document["liquidity"] = dict(
                edition=str(Path(liquidity_root).resolve()),
                **assigner.identity(),
                universe=universe,
            )
        atomic_json(stage / "manifest.json", document)
        _publish_directory(stage, output)
    return document


def subset_identity(document):
    """Lo que identifica al subconjunto en las vistas y en la campaña."""
    return dict(
        name=document["definition"]["name"],
        definition_sha256=document["definition_sha256"],
        catalog_sha256=document["catalog"]["sha256"],
        ids_sha256=document["ids"]["sha256"],
        rows=document["counts"]["rows"],
    )


def read_subset(path, *, rows=True):
    """Manifiesto del subconjunto con los identificadores comprobados frente a su huella.

    Con `rows=False` solo se comprueba la huella del archivo y no se devuelven las filas.
    """
    path = Path(path)
    document, digest = read_manifest(path)
    _require(
        document.get("schema_version") == SCHEMA_VERSION
        and document.get("kind") == KIND
        and document.get("row_order") == ROW_ORDER
        and definition_sha256(document.get("definition")) == document.get("definition_sha256"),
        "El manifiesto del subconjunto no cumple su contrato",
    )
    ids = path.parent / document["ids"]["file"]
    _require(
        ids.is_file() and not ids.is_symlink() and sha256(ids) == document["ids"]["sha256"],
        "Los identificadores del subconjunto no corresponden a su manifiesto",
    )
    if not rows:
        return document, digest, None
    table = pq.read_table(ids)
    _require(
        table.schema == IDS_SCHEMA and table.num_rows == document["counts"]["rows"],
        "Los identificadores del subconjunto cambiaron de forma",
    )
    return document, digest, table


def verify_subset(path):
    """Volver a seleccionar desde el catálogo y exigir los mismos identificadores."""
    document, digest, table = read_subset(path)
    manifest, manifest_sha256, catalog = read_catalog_manifest(document["catalog"]["manifest"])
    _require(
        manifest_sha256 == document["catalog"]["manifest_sha256"]
        and manifest["catalog"]["sha256"] == document["catalog"]["sha256"],
        "El subconjunto no corresponde al catálogo declarado",
    )
    definition, assigner, universe = document["definition"], None, {}
    if "liquidity" in definition:
        assigner = liquidity_edition(definition["liquidity"], document["liquidity"]["edition"])
    again = select(catalog, definition, assigner, universe=universe)
    _require(again.equals(table), "La definición ya no reproduce los mismos identificadores")
    _require(
        assigner is None or universe == document["liquidity"]["universe"],
        "El universo de liquidez ya no es el publicado",
    )
    return dict(status="verified", subset_manifest_sha256=digest, **subset_identity(document))


class SubsetRows:
    """Pertenencia de cada (activo, instante) a un subconjunto, consultada por activo."""

    def __init__(self, table):
        _require(table.schema == IDS_SCHEMA, "Los identificadores no tienen su esquema")
        keys = pc.binary_join_element_wise(table["market"], table["asset_id"], "/")
        encoded = keys.combine_chunks().dictionary_encode()
        codes = encoded.indices.to_numpy()
        moments = table["prediction_at"].cast(pa.int64()).to_numpy()
        starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]]) if len(codes) else []
        bounds = np.r_[starts, len(codes)]
        names = encoded.dictionary.to_pylist()
        self.rows = {}
        for start, end in zip(bounds[:-1], bounds[1:], strict=True):
            name = names[codes[start]]
            _require(name not in self.rows, "Los identificadores no están ordenados por activo")
            values = moments[start:end]
            _require(np.all(np.diff(values) > 0), "Un activo repite o desordena sus instantes")
            self.rows[name] = values

    def member(self, market, asset_id, prediction):
        """Máscara de las decisiones del activo que pertenecen al subconjunto."""
        selected = self.rows.get(f"{market}/{asset_id}")
        if selected is None:
            return np.zeros(len(prediction), dtype=bool)
        return np.isin(prediction, selected, assume_unique=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("materialize", help="Escribir un subconjunto declarado")
    build.add_argument("--catalog", type=Path, required=True, help="Manifiesto del catálogo")
    build.add_argument("--definition", type=Path, required=True, help="Definición JSON")
    build.add_argument("--output", type=Path, required=True, help="Carpeta nueva")
    build.add_argument(
        "--liquidity-edition", type=Path, help="Edición sin ajustar de una definición con liquidez"
    )
    verify = commands.add_parser("verify", help="Reproducir un subconjunto desde el catálogo")
    verify.add_argument("--subset", type=Path, required=True, help="Manifiesto del subconjunto")
    args = parser.parse_args(argv)
    if args.command == "materialize":
        document = materialize(
            args.catalog, args.definition, args.output, liquidity_root=args.liquidity_edition
        )
        print(json.dumps(subset_identity(document)))
    else:
        print(json.dumps(verify_subset(args.subset)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
