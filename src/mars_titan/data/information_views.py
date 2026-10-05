"""Vistas cerradas sobre dependencias y contratos comunes a sus consumidores."""

import copy
import hashlib
import json
import math
import re
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

import numpy as np

from mars_titan.models.baselines.inputs import MODALITIES

from .cohort_files import read_manifest, safe_destination
from .storage import atomic_json, sha256

ROLES = {"parent", "memory", "hmm", "calibration", "router", "normalization"}
METADATA = {
    "target",
    "sample_ids",
    "asset_ids",
    "market",
    "prediction_at",
    "target_available_at",
    "input_available_at",
    "available_at",
    "weight",
    "cohort_id",
    "confirmed_cursor",
    "origin",
    "information_view_sha256",
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _names(values, *, maximum=4096):
    return (
        isinstance(values, list)
        and len(values) <= maximum
        and all(isinstance(v, str) and 0 < len(v) <= 256 for v in values)
        and len(set(values)) == len(values)
    )


def _shape_sizes(shapes):
    if not isinstance(shapes, dict) or set(shapes) != set(MODALITIES):
        raise ValueError("La vista debe describir las dimensiones de todas las modalidades")
    for shape in shapes.values():
        if (
            not isinstance(shape, list)
            or not 1 <= len(shape) <= 2
            or any(type(n) is not int or not 1 <= n <= 4096 for n in shape)
        ):
            raise ValueError("Las dimensiones de la vista no son válidas")
    sizes = {key: math.prod(shape) for key, shape in shapes.items()}
    if sum(sizes.values()) > 16384:
        raise ValueError("Las dimensiones de la vista exceden el presupuesto")
    return sizes


def _variable_columns(row, sizes):
    fields = {
        "name",
        "source",
        "block",
        "value",
        "mask",
        "age",
        "dependencies",
        "transformation",
        "history",
    }
    if (
        not isinstance(row, dict)
        or set(row) != fields
        or not _names([row["name"]])
        or row["source"] not in MODALITIES
        or row["block"] not in {*MODALITIES, None}
        or not _names(row["dependencies"])
        or any(
            not isinstance(row[k], str) or not 1 <= len(row[k]) <= 2048
            for k in ("transformation", "history")
        )
    ):
        raise ValueError("La variable o sus dependencias no cumplen el contrato")
    columns = []
    for kind in ("value", "mask", "age"):
        indices = row[kind]
        if (
            not isinstance(indices, list)
            or len(indices) > 16384
            or any(type(i) is not int or i < 0 for i in indices)
        ):
            raise ValueError("La variable necesita índices enteros acotados")
        columns.extend(indices)
    block = row["block"]
    if block is None:
        if columns:
            raise ValueError("Una dependencia sin tensor no puede declarar columnas")
    elif (
        not row["value"]
        or len(set(columns)) != len(columns)
        or any(i >= sizes[block] for i in columns)
    ):
        raise ValueError("Las columnas de la variable se solapan o exceden sus dimensiones")
    return set(columns)


def _layout(spec):
    sizes, variables = _shape_sizes(spec["shapes"]), spec["variables"]
    if not isinstance(variables, list) or not 1 <= len(variables) <= 4096:
        raise ValueError("La vista necesita un catálogo acotado de variables")
    covered, nodes = {name: set() for name in sizes}, {}
    for row in variables:
        columns = _variable_columns(row, sizes)
        if row["name"] in nodes:
            raise ValueError("El catálogo de variables contiene una identidad duplicada")
        block = row["block"]
        if block is not None:
            if covered[block] & columns:
                raise ValueError("Las columnas de distintas variables se solapan")
            covered[block].update(columns)
        nodes[row["name"]] = row
    if any(covered[block] != set(range(size)) for block, size in sizes.items()):
        raise ValueError("La vista deja valores, máscaras o edades sin identificar")
    graph = {key: set(row["dependencies"]) for key, row in nodes.items()}
    if any(not dependencies <= nodes.keys() for dependencies in graph.values()):
        raise ValueError("La vista contiene una dependencia desconocida")
    try:
        tuple(TopologicalSorter(graph).static_order())
    except CycleError as error:
        raise ValueError("Las dependencias de la vista contienen un ciclo") from error
    return nodes


class InformationView:
    """Conservar dimensiones y población, anulando cada variable con sus máscaras y edades."""

    def __init__(self, manifest):
        required = {
            "schema_version",
            "kind",
            "cohort_sha256",
            "representation_sha256",
            "shapes",
            "variables",
            "allowed_sources",
            "allowed_variables",
        }
        if (
            not isinstance(manifest, dict)
            or set(manifest) != required
            or type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != 1
            or manifest["kind"] != "information_view"
            or not all(_hash(manifest[k]) for k in ("cohort_sha256", "representation_sha256"))
            or len(json.dumps(manifest, allow_nan=False).encode()) > 2 * 1024**2
        ):
            raise ValueError("El manifiesto de la vista no cumple su contrato")
        self._manifest = copy.deepcopy(manifest)
        self._nodes = _layout(self._manifest)
        sources, allowed = manifest["allowed_sources"], manifest["allowed_variables"]
        if (
            not _names(sources)
            or not set(sources) <= set(MODALITIES)
            or not _names(allowed)
            or not set(allowed) <= self._nodes.keys()
        ):
            raise ValueError("La vista permite fuentes o variables desconocidas")
        for key in allowed:
            row = self._nodes[key]
            if row["source"] not in sources or not set(row["dependencies"]) <= set(allowed):
                raise ValueError("La vista conserva una ruta indirecta de información retirada")
        self.identity = _digest(self._manifest)
        self._excluded = {name: [] for name in MODALITIES}
        for key, row in self._nodes.items():
            if key not in allowed and row["block"] is not None:
                self._excluded[row["block"]].extend(row["value"] + row["mask"] + row["age"])

    @property
    def manifest(self):
        return copy.deepcopy(self._manifest)

    def without(self, *, sources=(), variables=()):
        if (
            not _names(list(sources))
            or not set(sources) <= set(MODALITIES)
            or not _names(list(variables))
            or not set(variables) <= self._nodes.keys()
        ):
            raise ValueError("La retirada contiene una fuente o variable desconocida")
        spec = self.manifest
        spec["allowed_sources"] = [s for s in spec["allowed_sources"] if s not in sources]
        allowed = set(spec["allowed_variables"]) - set(variables)
        while True:
            current = {
                key
                for key in allowed
                if self._nodes[key]["source"] in spec["allowed_sources"]
                and set(self._nodes[key]["dependencies"]) <= allowed
            }
            if current == allowed:
                break
            allowed = current
        spec["allowed_variables"] = [
            row["name"] for row in spec["variables"] if row["name"] in allowed
        ]
        return InformationView(spec)

    def apply(self, batch, *, max_bytes=64 * 1024**2):
        if not isinstance(batch, dict) or set(batch) - METADATA - {"inputs"}:
            raise ValueError("El lote contiene una ruta indirecta sin contrato de vista")
        marker = batch.get("information_view_sha256", self.identity)
        if marker != self.identity:
            raise ValueError("El lote ya pertenece a otra vista")
        inputs = batch.get("inputs")
        if not isinstance(inputs, dict) or set(inputs) != set(MODALITIES):
            raise ValueError("La vista requiere las dimensiones de las modalidades completas")
        if type(max_bytes) is not int or not 1 <= max_bytes <= 64 * 1024**2:
            raise ValueError("El presupuesto de la vista no es válido")
        rows, size = None, 0
        for name, shape in self._manifest["shapes"].items():
            value = inputs[name]
            if not isinstance(value, np.ndarray) or value.dtype != np.float32:
                raise ValueError("Las entradas deben ser tensores NumPy float32")
            if rows is None:
                rows = len(value) if value.ndim else 0
            if not 1 <= rows <= 4096 or value.shape != (rows, *shape):
                raise ValueError("Las dimensiones no coinciden con la vista")
            size += value.nbytes
        if size > max_bytes:
            raise ValueError("El lote supera el presupuesto de la vista")
        for values in inputs.values():
            if not np.isfinite(values).all():
                raise ValueError("La cohorte completa debe contener valores finitos")
        for key in (
            "sample_ids",
            "asset_ids",
            "target",
            "market",
            "weight",
            "prediction_at",
            "target_available_at",
            "input_available_at",
            "available_at",
        ):
            if key == "prediction_at" and key in batch and np.isscalar(batch[key]):
                continue
            if key in batch and np.shape(batch[key]) != (rows,):
                raise ValueError("Las entradas y metadatos no están alineados")
        result = {key: copy.deepcopy(value) for key, value in batch.items() if key != "inputs"}
        result["inputs"] = {name: value.copy(order="C") for name, value in inputs.items()}
        for name, columns in self._excluded.items():
            result["inputs"][name].reshape(rows, -1)[:, columns] = 0
        result["information_view_sha256"] = self.identity
        return result

    def save(self, path):
        path = Path(path)
        safe_destination(path)
        if path.exists():
            raise ValueError("La vista ya tiene un archivo en ese destino")
        atomic_json(path, self._manifest)

    @classmethod
    def load(cls, path):
        return cls(read_manifest(Path(path), 2 * 1024**2)[0])

    def artifact(self, role, artifact_sha256, *, dependencies=()):
        """Construir la declaración que debe guardar el productor junto a su artefacto."""
        if role not in ROLES or not _hash(artifact_sha256) or len(dependencies) > 64:
            raise ValueError("El artefacto necesita un rol, una huella y dependencias acotadas")
        for dependency in dependencies:
            self.check_artifact(dependency)
        return dict(
            schema_version=1,
            kind="information_artifact",
            role=role,
            artifact_sha256=artifact_sha256,
            information_view_sha256=self.identity,
            fit_view_sha256=self.identity,
            cohort_sha256=self._manifest["cohort_sha256"],
            representation_sha256=self._manifest["representation_sha256"],
            dependencies=sorted(_digest(d) for d in dependencies),
        )

    def check_artifact(self, record):
        fields = {
            "schema_version",
            "kind",
            "role",
            "artifact_sha256",
            "information_view_sha256",
            "fit_view_sha256",
            "cohort_sha256",
            "representation_sha256",
            "dependencies",
        }
        if (
            not isinstance(record, dict)
            or set(record) != fields
            or record["schema_version"] != 1
            or record["kind"] != "information_artifact"
            or record["role"] not in ROLES
            or not _hash(record["artifact_sha256"])
            or record["information_view_sha256"] != self.identity
            or record["fit_view_sha256"] != self.identity
            or any(
                record[k] != self._manifest[k] for k in ("cohort_sha256", "representation_sha256")
            )
            or not _names(record["dependencies"], maximum=64)
            or any(not _hash(v) for v in record["dependencies"])
        ):
            raise ValueError("El artefacto no acredita la misma vista, cohorte y representación")


class ViewConsumer:
    """Comprobar dependencias antes de entregar exclusivamente tensores de la vista."""

    def __init__(self, view, artifact, consume, *, dependencies=(), artifact_path=None):
        if not callable(consume) or len(dependencies) > 64:
            raise ValueError("El consumidor o el presupuesto de dependencias no es válido")
        records = [artifact, *dependencies]
        for record in records:
            view.check_artifact(record)
        graph = {_digest(row): set(row["dependencies"]) for row in records}
        if len(graph) != len(records) or any(not deps <= graph.keys() for deps in graph.values()):
            raise ValueError("Falta una dependencia del artefacto o está duplicada")
        try:
            tuple(TopologicalSorter(graph).static_order())
        except CycleError as error:
            raise ValueError("Las dependencias del consumidor contienen un ciclo") from error
        self.view, self.consume, self.role = view, consume, artifact["role"]
        self.artifact = copy.deepcopy(artifact)
        self.path = None if artifact_path is None else Path(artifact_path)
        self._signature = None
        self._verify()

    def _verify(self):
        self.view.check_artifact(self.artifact)
        if self.path is None:
            return
        safe_destination(self.path)
        if not self.path.is_file() or not 0 < self.path.stat().st_size <= 512 * 1024**2:
            raise ValueError("El artefacto no es regular o excede su presupuesto")
        stat = self.path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if signature != self._signature:
            observed = sha256(self.path)
            after = self.path.stat()
            if observed != self.artifact["artifact_sha256"] or signature != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ValueError("La huella del artefacto ha cambiado")
            self._signature = signature

    def __call__(self, batch):
        self._verify()
        if batch.get("information_view_sha256") != self.view.identity:
            raise ValueError("El consumidor requiere un lote confirmado de su vista")
        inputs = self.view.apply(batch)["inputs"]
        return self.consume(inputs)


def attach_parent(batch, parent):
    """Construir el vector residual común después de comprobar y recalcular el padre."""
    if parent.role != "parent":
        raise ValueError("La característica adicional requiere el contrato de un padre")
    values = np.asarray(parent(batch), dtype=np.float64)
    result = parent.view.apply(batch)
    count = len(result["inputs"]["prices"])
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError("El padre no devuelve una predicción finita por muestra")
    with np.errstate(over="raise", invalid="raise"):
        features = np.concatenate(
            [result["inputs"][k].reshape(count, -1) for k in MODALITIES]
            + [values[:, None].astype(np.float32)],
            axis=1,
        )
    result.update(parent=values.copy(), features=features)
    return result
