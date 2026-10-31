"""Capturar S2 desde el enlace financiero y una fila verificada, antes de sus etiquetas.

La clave procede del paso de codificación de la sesión. Se comprueban su contrato
y la huella de las entradas. Este productor no vuelve a ejecutar el predictor ni
acredita por sí solo la procedencia fuera de muestra del manifiesto de ajuste.
"""

import csv
import hashlib
import io
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np

from mars_titan.data.input_policy import MODALITIES, masked_inputs, validate_historical_vectors
from mars_titan.models.candidate.episode_codec import FrozenCandidateCodec
from mars_titan.models.titans.episodic_snapshot import _digest_id
from mars_titan.models.titans.holographic_network import (
    CATALOG,
    MACRO_FAMILIES,
    HolographicNetwork,
    area_layout,
)

from .episodic_session import DecisionRow, _canonical
from .failure_context import MAX_MACRO_INDICATORS, FailureContext, FailureContextLayout
from .financial_consumers import CandidateBinding, TitansBinding
from .financial_mechanisms import MechanismBinding
from .spaced_session import SpacedTitansBinding

MAX_CATALOG_BYTES = 2 * 1024**2


def _categories(path, expected):
    _digest_id(expected, "El catálogo macro de S2")
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_CATALOG_BYTES + 1)
    if len(raw) > MAX_CATALOG_BYTES or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("El catálogo macro no conserva su huella o supera su límite")
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), strict=True)
        fields = reader.fieldnames or []
        rows = list(islice(reader, MAX_MACRO_INDICATORS + 1))
    except (csv.Error, UnicodeError) as error:
        raise ValueError("El catálogo macro no es un CSV legible") from error
    if (
        not {"id", "category"} <= set(fields)
        or len(fields) != len(set(fields))
        or not 1 <= len(rows) <= MAX_MACRO_INDICATORS
        or any(
            type(r.get("id")) is not str
            or not 1 <= len(r["id"]) <= 128
            or not r["id"].isprintable()
            or r.get("category") not in MACRO_FAMILIES
            for r in rows
        )
        or len({r["id"] for r in rows}) != len(rows)
    ):
        raise ValueError(
            "El catálogo macro necesita indicadores únicos y sus doce familias válidas"
        )
    return {r["id"]: r["category"] for r in rows}


def _areas(network, specification, categories):
    if network is None:
        return (), ()
    if type(network) is not HolographicNetwork:
        raise ValueError("La actividad S2 necesita la estructura H1 del consumidor")
    layout, config = network.layout, network.config
    if layout != area_layout(config.areas, specification.representation, masked=True):
        raise ValueError("La estructura de áreas no corresponde a la representación financiera")
    if network.family_blocks:
        recorded = dict(layout.families)
        indicators = specification.representation["macro_indicators"]
        if any(
            recorded.get(f"macro_{family}")
            != tuple(i for i in indicators if categories[i] == family)
            for family in MACRO_FAMILIES
        ):
            raise ValueError("La estructura macro y el catálogo de S2 no coinciden")
    modalities = [name for _, name, _ in layout.blocks]
    expected = np.eye(len(MODALITIES))[[MODALITIES.index(name) for name in modalities]]
    observed = network.block_modality.detach().cpu().numpy()
    if not np.array_equal(observed, expected):
        raise ValueError("La estructura de presencia del grafo está alterada")
    names, owners = [], []
    if config.catalog_areas:
        indices = [b for b, members in enumerate(layout.members) for _ in members]
        if not np.array_equal(network.area_owner.detach().cpu().numpy(), indices):
            raise ValueError("La estructura de pertenencia del grafo está alterada")
        names.extend(layout.areas)
        owners.extend(modalities[i] for i in indices)
    names.extend(f"slot_{i}" for i in range(config.hidden_groups))
    owners.extend(None for _ in range(config.hidden_groups))
    return tuple(names), tuple(owners)


def _checked_row(row, specification):
    if (
        type(row) is not DecisionRow
        or not isinstance(row.inputs, Mapping)
        or set(row.inputs) != set(MODALITIES)
        or not isinstance(row.presence, np.ndarray)
        or row.presence.dtype != np.bool_
        or row.presence.shape != (len(MODALITIES),)
    ):
        raise ValueError("El productor S2 necesita una fila financiera con sus cinco modalidades")
    dimensions = specification.dimensions
    for name in MODALITIES:
        value = row.inputs[name]
        shape = (
            (specification.context, dimensions[name]) if name == "prices" else (dimensions[name],)
        )
        if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != shape:
            raise ValueError("La fila S2 no conserva la forma y el tipo de sus entradas")
    validate_historical_vectors(
        {name: row.inputs[name][None] for name in MODALITIES},
        row.presence[None],
        specification.representation,
    )
    metadata = row.metadata()
    metadata.pop("input_sha256")
    metadata["input_contract_id"] = specification.fingerprint()
    digest = hashlib.sha256(_canonical(metadata).encode())
    for name in MODALITIES:
        digest.update(row.inputs[name].tobytes())
    digest.update(row.presence.tobytes())
    if digest.hexdigest() != row.input_sha256:
        raise ValueError("La huella de la fila no conserva las entradas declaradas")


@dataclass(frozen=True, init=False)
class FailureContextProducer:
    layout: FailureContextLayout
    _specification: object

    def __init__(self, binding, *, macro_catalog_sha256, catalog=CATALOG):
        if type(binding) not in (
            TitansBinding,
            SpacedTitansBinding,
            CandidateBinding,
            MechanismBinding,
        ):
            raise ValueError("El productor S2 necesita un enlace financiero admitido")
        binding.check()
        specification, codec = binding.input_spec, binding.codec
        if not masked_inputs(specification.input_policy):
            raise ValueError("El contexto S2 requiere máscaras y edades históricas explícitas")
        categories = _categories(catalog, macro_catalog_sha256)
        indicators = tuple(specification.representation["macro_indicators"])
        if any(i not in categories for i in indicators):
            raise ValueError("Un indicador de la representación no aparece en el catálogo macro")
        network = (
            None
            if type(binding) is CandidateBinding
            else binding.consumer.predictor.holographic_network
        )
        names, owners = _areas(network, specification, categories)
        layout = FailureContextLayout(
            input_id=specification.fingerprint(),
            codec_id=codec.fingerprint(),
            macro_catalog_id=macro_catalog_sha256,
            macro_indicators=indicators,
            macro_families=tuple(categories[i] for i in indicators),
            family_names=MACRO_FAMILIES,
            area_names=names,
            area_modalities=owners,
            key_width=codec.identity()["key_width"],
            allow_zero_key=type(codec) is FrozenCandidateCodec,
        )
        binding.consumer.verify(strong=True)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "_specification", specification)

    def identity(self):
        return dict(
            schema_version=1,
            kind="financial_failure_context_producer_v1",
            layout=self.layout.identity(),
            row_contract="decision_row_numpy_float32_sha256_v1",
        )

    def capture(self, row, *, action, macro_available_at=None):
        _checked_row(row, self._specification)
        return FailureContext(
            self.layout,
            **row.metadata(),
            key_inputs=row.key_inputs,
            macro=row.inputs["macro"],
            presence=row.presence,
            action=action,
            macro_available_at=macro_available_at,
        )
