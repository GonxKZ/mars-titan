"""Sector de cada activo según las fichas de FinMultiTime, para el control sectorial de PT4.

Las fichas `sp500stock_data_description.csv` (columna `Sector`) y
`hs300stock_data_description.csv` (columna `申万一级行业`, nivel 1 de Shenwan) describen el
universo en la fecha en que se construyó el conjunto. Es una clasificación actual, no una
serie histórica: una empresa reclasificada después (como en la revisión GICS de 2018) aparece
con su sector posterior en todos los años. Los activos sin sector (`N/A`, `Client Error`,
vacío o ausentes de la ficha) van a un grupo `unknown` de su mercado, nunca a un sector
inventado. La identidad guarda la huella de los dos archivos y la del reparto.
"""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

UNKNOWN = "unknown"
_SOURCES = {
    "US": ("sp500stock_data_description.csv", "utf-8-sig", "stock_name", "Sector"),
    "CN": ("hs300stock_data_description.csv", "gb18030", "证券代码", "申万一级行业"),
}
_MISSING = {"", "N/A", "Client Error"}
_MAX_BYTES = 8 * 1024**2


def _flow(market, symbol):
    symbol = symbol.strip().upper()
    # La preparación del corpus usa el sufijo .SS de Yahoo para Shanghái.
    return f"{market}/{symbol.replace('.SH', '.SS')}" if market == "CN" else f"US/{symbol}"


@dataclass(frozen=True)
class SectorMap:
    """Reparto fijo de flujos en grupos (mercado, sector), con el grupo desconocido aparte."""

    groups: tuple[str, ...]
    assignments: tuple[tuple[str, int], ...]
    sources_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self):
        flows = [flow for flow, _ in self.assignments]
        if (
            len(set(self.groups)) != len(self.groups)
            or len(set(flows)) != len(flows)
            or any(
                type(index) is not int or not 0 <= index < len(self.groups)
                for _, index in self.assignments
            )
            or any(f"{market}/{UNKNOWN}" not in self.groups for market in _SOURCES)
        ):
            raise ValueError("El reparto sectorial no es coherente")
        object.__setattr__(self, "_table", dict(self.assignments))
        # La identidad entra en la del predictor, que se calcula en cada decisión. Con miles
        # de activos serializar el reparto cada vez dominaba el coste del control sectorial.
        text = json.dumps(
            [self.groups, self.assignments], ensure_ascii=False, separators=(",", ":")
        )
        object.__setattr__(
            self,
            "_identity",
            dict(
                schema_version=1,
                source="finmultitime_current_descriptions_v1",
                sources_sha256=dict(self.sources_sha256),
                groups=len(self.groups),
                assigned_flows=len(self.assignments),
                assignment_sha256=hashlib.sha256(text.encode()).hexdigest(),
                point_in_time=False,
            ),
        )

    @classmethod
    def from_finmultitime(cls, root):
        """Leer las dos fichas del conjunto y asignar cada símbolo a su grupo."""
        root = Path(root)
        names, sectors, sources = [], {}, []
        for market, (file, encoding, symbol_field, sector_field) in _SOURCES.items():
            path = root / file
            if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_BYTES:
                raise ValueError(f"La ficha {file} no es un archivo regular de hasta 8 MiB")
            data = path.read_bytes()
            sources.append((file, hashlib.sha256(data).hexdigest()))
            rows = csv.DictReader(data.decode(encoding).splitlines())
            if rows.fieldnames is None or {symbol_field, sector_field} - set(rows.fieldnames):
                raise ValueError(f"La ficha {file} no tiene las columnas esperadas")
            for row in rows:
                symbol, sector = row[symbol_field] or "", (row[sector_field] or "").strip()
                if not symbol.strip():
                    continue
                flow = _flow(market, symbol)
                group = f"{market}/{UNKNOWN if sector in _MISSING else sector}"
                if sectors.get(flow, group) != group:
                    raise ValueError(f"{flow} figura con dos sectores en la ficha")
                sectors[flow] = group
            names.append(f"{market}/{UNKNOWN}")
        names = sorted(set(names) | set(sectors.values()))
        index = {name: position for position, name in enumerate(names)}
        return cls(
            tuple(names),
            tuple(sorted((flow, index[group]) for flow, group in sectors.items())),
            tuple(sources),
        )

    def group(self, flow):
        """Índice del grupo de un flujo. Sin ficha, el grupo desconocido de su mercado."""
        if flow in self._table:
            return self._table[flow]
        market = flow.split("/", 1)[0]
        if market not in _SOURCES:
            raise ValueError("El flujo no pertenece a US ni a CN")
        return self.groups.index(f"{market}/{UNKNOWN}")

    def identity(self):
        identity = dict(self._identity)
        identity["sources_sha256"] = dict(identity["sources_sha256"])
        return identity
