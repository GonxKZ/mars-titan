"""Registro de trazas con cadencia fija, presupuesto de bytes y recuperación sin duplicados.

Las estadísticas de cada tensor se calculan en su dispositivo sobre una copia separada del
grafo (`detach`), sin usar ningún generador aleatorio. Mientras no se vuelca, nada se copia
al host. En cada volcado hay una sola transferencia de todas las estadísticas pendientes.

Cada volcado escribe un archivo Parquet numerado (`parts/part-00000001.parquet`) con
escritura atómica. El cursor que se guarda en el checkpoint es el número de la última
parte confirmada. Al reanudar, `load_state_dict` borra las partes posteriores, que se
escribieron después del último estado guardado, y así no hay eventos duplicados ni perdidos.

Si una parte haría superar `max_bytes`, no se escribe. El registrador queda agotado y el
manifiesto lo declara con el paso en el que ocurrió.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mars_titan.data.storage import atomic_json

SCHEMA_VERSION = 1
KIND = "learning_traces"
STATS = ("mean", "std", "min", "max", "abs_max", "l2", "finite_fraction")
SCHEMA = pa.schema(
    [
        ("step", pa.int64()),
        ("phase", pa.string()),
        ("metric", pa.string()),
        ("group", pa.string()),
        ("stat", pa.string()),
        ("value", pa.float64()),
    ]
)
_NAME = re.compile(r"[a-z][a-z0-9_.:/-]{0,95}")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class TraceConfig:
    every: int
    max_bytes: int
    enabled: bool = True

    def __post_init__(self):
        _require(type(self.every) is int and self.every >= 1, "La cadencia es un entero positivo")
        _require(
            type(self.max_bytes) is int and self.max_bytes >= 1024,
            "El presupuesto de bytes debe ser un entero de al menos 1024",
        )
        _require(type(self.enabled) is bool, "enabled es booleano")


def tensor_stats(value):
    """Estadísticas en float64 sobre los valores finitos, calculadas en el dispositivo.

    Devuelve un tensor con una entrada por nombre de `STATS`. Sin valores finitos, todas
    valen NaN salvo `finite_fraction`. Un tensor vacío da NaN en todas.
    """
    x = value.detach().reshape(-1).to(torch.float64)
    if x.numel() == 0:
        return torch.full((len(STATS),), float("nan"), dtype=torch.float64, device=x.device)
    finite = torch.isfinite(x)
    count = finite.sum().to(torch.float64)
    zero = torch.zeros((), dtype=torch.float64, device=x.device)
    values = torch.where(finite, x, zero)
    n = count.clamp(min=1.0)
    mean = values.sum() / n
    centered = torch.where(finite, x - mean, zero)
    std = (centered.square().sum() / n).sqrt()
    inf = torch.full((), float("inf"), dtype=torch.float64, device=x.device)
    minimum = torch.where(finite, x, inf).min()
    maximum = torch.where(finite, x, -inf).max()
    abs_max = torch.where(finite, x.abs(), zero).max()
    l2 = values.square().sum().sqrt()
    stats = torch.stack([mean, std, minimum, maximum, abs_max, l2])
    stats = torch.where(count > 0, stats, torch.full_like(stats, float("nan")))
    return torch.cat([stats, (count / x.numel()).reshape(1)])


def _to_host(tensors):
    """Copiar al host con una sola transferencia por dispositivo, conservando el orden."""
    result, by_device = [None] * len(tensors), {}
    for index, value in enumerate(tensors):
        by_device.setdefault(value.device, []).append(index)
    for indices in by_device.values():
        host = torch.stack([tensors[i] for i in indices]).to("cpu").tolist()
        for index, value in zip(indices, host, strict=True):
            result[index] = value
    return result


class TraceRecorder:
    """Acumula estadísticas pendientes y las vuelca por partes numeradas."""

    def __init__(self, folder, config, identity):
        _require(isinstance(config, TraceConfig), "Falta la configuración de trazas")
        _require(isinstance(identity, dict) and identity, "Las trazas necesitan una identidad")
        self.folder, self.config, self.identity = Path(folder), config, identity
        self.parts = self.folder / "parts"
        self.sequence, self.bytes, self.exhausted_at = 0, 0, None
        self._pending, self._scalars = [], []
        if config.enabled:
            self.parts.mkdir(parents=True, exist_ok=True)
            self._write_manifest()

    def _manifest(self):
        return dict(
            schema_version=SCHEMA_VERSION,
            kind=KIND,
            identity=self.identity,
            config=asdict(self.config),
            stats=list(STATS),
            columns=SCHEMA.names,
            exhausted_at_step=self.exhausted_at,
        )

    def _write_manifest(self):
        atomic_json(self.folder / "manifest.json", self._manifest())

    @property
    def active(self):
        return self.config.enabled and self.exhausted_at is None

    def due(self, step):
        """Si el paso toca registro. Solo compara enteros: no sincroniza el dispositivo."""
        return self.active and step % self.config.every == 0

    def tensor(self, metric, value, *, group="all"):
        """Guardar las estadísticas de un tensor hasta el siguiente volcado."""
        if not self.active:
            return
        _require(_NAME.fullmatch(metric) and _NAME.fullmatch(group), "Nombre de traza inválido")
        self._pending.append((metric, group, tensor_stats(value)))

    def scalar(self, metric, value, *, group="all"):
        """Guardar un valor. Un tensor de un elemento se queda en su dispositivo."""
        if not self.active:
            return
        _require(_NAME.fullmatch(metric) and _NAME.fullmatch(group), "Nombre de traza inválido")
        if torch.is_tensor(value):
            _require(value.numel() == 1, "Un escalar de traza tiene un solo elemento")
            value = value.detach().reshape(()).to(torch.float64)
        else:
            value = float(value)
        self._scalars.append((metric, group, value))

    def flush(self, step, phase):
        """Volcar lo pendiente en una parte nueva. Devuelve las filas escritas."""
        _require(type(step) is int and step >= 0, "El paso es un entero no negativo")
        _require(_NAME.fullmatch(phase), "Fase de traza inválida")
        pending, scalars = self._pending, self._scalars
        self._pending, self._scalars = [], []
        if not self.active or not (pending or scalars):
            return 0
        rows = dict(metric=[], group=[], stat=[], value=[])
        for (metric, group, _), values in zip(
            pending, _to_host([stats for _, _, stats in pending]), strict=True
        ):
            for stat, value in zip(STATS, values, strict=True):
                rows["metric"].append(metric)
                rows["group"].append(group)
                rows["stat"].append(stat)
                rows["value"].append(value)
        devices = [value for _, _, value in scalars if torch.is_tensor(value)]
        host = iter(_to_host(devices))
        for metric, group, value in scalars:
            rows["metric"].append(metric)
            rows["group"].append(group)
            rows["stat"].append("value")
            rows["value"].append(next(host) if torch.is_tensor(value) else value)
        count = len(rows["value"])
        table = pa.table(dict(step=[step] * count, phase=[phase] * count, **rows), schema=SCHEMA)
        sequence = self.sequence + 1
        target = self.parts / f"part-{sequence:08d}.parquet"
        temporary = self.parts / f".part-{sequence:08d}.parquet.tmp"
        pq.write_table(table, temporary)
        size = temporary.stat().st_size
        if self.bytes + size > self.config.max_bytes:
            temporary.unlink()
            self.exhausted_at = step
            self._write_manifest()
            return 0
        temporary.replace(target)
        self.sequence, self.bytes = sequence, self.bytes + size
        return count

    def state_dict(self):
        """Cursor que se guarda en el checkpoint junto al resto del estado."""
        return dict(sequence=self.sequence, bytes=self.bytes, exhausted_at=self.exhausted_at)

    def load_state_dict(self, state):
        """Volver al cursor guardado y borrar las partes escritas después."""
        _require(
            set(state) == {"sequence", "bytes", "exhausted_at"}
            and type(state["sequence"]) is int
            and state["sequence"] >= 0,
            "El cursor de trazas no es válido",
        )
        if not self.config.enabled:
            return
        for path in sorted(self.parts.glob("part-*.parquet")):
            if int(path.stem.split("-")[1]) > state["sequence"]:
                path.unlink()
        for path in self.parts.glob(".part-*.tmp"):
            path.unlink()
        sizes = [
            p.stat().st_size
            for p in self.parts.glob("part-*.parquet")
            if int(p.stem.split("-")[1]) <= state["sequence"]
        ]
        _require(
            len(sizes) == state["sequence"] and sum(sizes) == state["bytes"],
            "Faltan partes de trazas confirmadas o cambiaron de tamaño",
        )
        self.sequence, self.bytes = state["sequence"], state["bytes"]
        self.exhausted_at = state["exhausted_at"]
        self._pending, self._scalars = [], []
        self._write_manifest()


def read_traces(folder):
    """Leer el manifiesto y todas las partes en orden, comprobando esquema y numeración."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    _require(
        manifest.get("kind") == KIND and manifest.get("schema_version") == SCHEMA_VERSION,
        "El manifiesto no es de trazas de aprendizaje",
    )
    paths = sorted((folder / "parts").glob("part-*.parquet"))
    numbers = [int(path.stem.split("-")[1]) for path in paths]
    _require(numbers == list(range(1, len(paths) + 1)), "La numeración de las partes tiene huecos")
    tables = [pq.read_table(path) for path in paths]
    _require(all(table.schema.equals(SCHEMA) for table in tables), "Una parte tiene otro esquema")
    return manifest, pa.concat_tables(tables) if tables else SCHEMA.empty_table()
