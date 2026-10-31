"""Contexto N2 de un evento y vistas inmutables vinculadas a sus bloques CPU."""

import hashlib
import re
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from .config import canonical
from .dendritic_regime import context_arrays
from .financial_inputs import CPUDecisionBatch, DecisionBatch, device_tensor

ENTRY_FIELDS = (
    "flow_id",
    "sample_id",
    "prediction_at",
    "input_available_at",
    "prices_sha256",
    "route",
    "source_slot",
    "market_input_available_at",
)


def _key(config, seed):
    return hashlib.sha256(
        canonical(dict(configuration=config.identity(), seed=seed)).encode()
    ).hexdigest()


def _price_hash(prices):
    return hashlib.sha256(prices.tobytes()).hexdigest()


@dataclass(frozen=True, init=False)
class DendriticView:
    """Una asignación del evento que solo puede usarse con el bloque al que se vinculó."""

    configuration: str
    context_id: str
    input_digest: str
    flows: tuple[str, ...]
    at: int
    routes: tuple[int, ...]
    source_id: str
    _sealed: tuple = field(repr=False, compare=False)

    def _signature(self):
        return (
            self.configuration,
            self.context_id,
            self.input_digest,
            self.flows,
            self.at,
            self.routes,
            self.source_id,
        )

    def validate(self, batch, config, seed, *, context_id=None):
        if (
            self._signature() != self._sealed
            or type(batch) is not DecisionBatch
            or batch.input_digest != self.input_digest
            or batch.flow_ids != self.flows
            or any(at != self.at for at in batch.prediction_at)
            or _key(config, seed) != self.configuration
            or (context_id is not None and context_id != self.context_id)
        ):
            raise ValueError("La vista N2 no corresponde al bloque, contexto o configuración")

    def tensor(self, batch, config, seed):
        self.validate(batch, config, seed)
        array = np.eye(5)[list(self.routes), 1:]
        return device_tensor(array, batch.inputs["prices"].device, batch.inputs["prices"].dtype)


@dataclass(frozen=True, init=False)
class DendriticPlan:
    """Asignación de toda la cohorte, sin tensores de dispositivo ni historial de autograd."""

    configuration: str
    input_contract: str
    context_id: str
    entries: tuple[tuple, ...] = field(repr=False)
    source_id: str
    _lookup: MappingProxyType = field(repr=False, compare=False)
    _sealed: tuple = field(repr=False, compare=False)

    @classmethod
    def create(cls, config, rows, *, seed, input_contract, context_id):
        if (
            type(rows) not in (tuple, list)
            or not 1 <= len(rows) <= 8192
            or any(
                type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in (input_contract, context_id)
            )
        ):
            raise ValueError("El evento N2 necesita filas y huellas acotadas")
        for row in rows:
            prices = row.inputs["prices"]
            if (
                not isinstance(prices, np.ndarray)
                or prices.shape != (64, 6)
                or prices.dtype != np.float32
                or type(row.sample_id) is not str
                or len(row.sample_id) > 128
                or row.sample_id != f"{row.flow_id}/{row.prediction_at}"
            ):
                raise ValueError("La fuente de precios N2 no conserva sus campos y geometría")
        prices = np.stack([row.inputs["prices"] for row in rows])
        flows = tuple(row.flow_id for row in rows)
        moments = tuple(row.prediction_at for row in rows)
        available = tuple(row.input_available_at for row in rows)
        routes, slots, availability = context_arrays(
            config, prices, flows, moments, available, seed=seed
        )
        entries = tuple(
            (
                row.flow_id,
                row.sample_id,
                row.prediction_at,
                row.input_available_at,
                _price_hash(row.inputs["prices"]),
                route,
                slot,
                received,
            )
            for row, route, slot, received in zip(rows, routes, slots, availability, strict=True)
        )
        result = object.__new__(cls)
        for name, value in dict(
            configuration=_key(config, seed),
            input_contract=input_contract,
            context_id=context_id,
            entries=entries,
        ).items():
            object.__setattr__(result, name, value)
        source_id = hashlib.sha256(canonical(result.record()).encode()).hexdigest()
        object.__setattr__(result, "source_id", source_id)
        object.__setattr__(result, "_lookup", MappingProxyType({row[0]: row for row in entries}))
        object.__setattr__(result, "_sealed", result._signature())
        return result

    def _signature(self):
        return (
            self.configuration,
            self.input_contract,
            self.context_id,
            self.entries,
            self.source_id,
            id(self._lookup),
        )

    def record(self):
        if hasattr(self, "_sealed") and self._signature() != self._sealed:
            raise ValueError("La asignación del evento N2 cambió después de publicarse")
        return dict(
            schema_version=1,
            kind="dendritic_event_context_v1",
            columns=list(ENTRY_FIELDS),
            configuration=self.configuration,
            input_contract=self.input_contract,
            context_id=self.context_id,
            entries=[list(row) for row in self.entries],
        )

    def validate_event(self, config, seed, *, input_contract, context_id, flows, at):
        """Vincular una vez el origen completo con la cohorte de propuestas."""
        self.record()
        if (
            self.configuration != _key(config, seed)
            or self.input_contract != input_contract
            or self.context_id != context_id
            or sorted(row[0] for row in self.entries) != sorted(flows)
            or any(row[2] != at for row in self.entries)
        ):
            raise ValueError("El origen N2 no corresponde a la configuración y cohorte")
        return self.source_id

    def view(self, batch):
        if type(batch) is not CPUDecisionBatch or self._signature() != self._sealed:
            raise ValueError("La asignación N2 necesita un bloque CPU y una fuente intactos")
        batch.verify()
        if batch.input_contract_id != self.input_contract:
            raise ValueError("El bloque N2 procede de otro contrato de entradas")
        routes = []
        for index, flow in enumerate(batch.flow_ids):
            row = self._lookup.get(flow)
            actual = (
                flow,
                batch.sample_ids[index],
                batch.prediction_at[index],
                batch.input_available_at[index],
                _price_hash(batch.inputs["prices"][index]),
            )
            if row is None or actual != row[:5]:
                raise ValueError("Los precios o metadatos N2 no pertenecen al evento verificado")
            routes.append(row[5])
        view = object.__new__(DendriticView)
        for name, value in dict(
            configuration=self.configuration,
            context_id=self.context_id,
            input_digest=batch.input_digest,
            flows=batch.flow_ids,
            at=batch.prediction_at[0],
            routes=tuple(routes),
            source_id=self.source_id,
        ).items():
            object.__setattr__(view, name, value)
        object.__setattr__(view, "_sealed", view._signature())
        return view
