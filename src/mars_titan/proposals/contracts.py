"""Registro finito de recetas, sin construir modelos ni autorizar su despliegue."""

import hashlib
import json
from dataclasses import dataclass, field

from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.episodic_readout import EpisodicReadoutConfig
from mars_titan.models.titans.episodic_snapshot import _digest_id
from mars_titan.models.titans.financial import FinancialConfig
from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog

MAX_REGISTRY_BYTES = 1024**2
MAX_CONFIGURATIONS = 8


def _bounded_identity(value):
    encoded = canonical(value)
    if len(encoded.encode()) > MAX_REGISTRY_BYTES:
        raise ValueError("La identidad del registro supera 1 MiB")
    return encoded


def _fingerprint(encoded):
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class RegisteredConfiguration:
    """Receta fija identificada, distinta de los pesos y del consumidor efectivo.

    J conserva su campo en la recurrencia del puente. K y el programa anterior a MAC
    siguen en sus configuraciones originales. Los objetos recibidos se contrastan
    con el sello al consultar la receta, incluso si se alteran sus campos internos.
    """

    predictor: FinancialConfig
    readout: EpisodicReadoutConfig | None
    action: MechanismAction
    catalog: MechanismCatalog
    dtype: str
    _sealed: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self._validate()
        object.__setattr__(self, "_sealed", _bounded_identity(self._current_identity()))

    def _validate(self):
        if (
            type(self.predictor) is not FinancialConfig
            or type(self.catalog) is not MechanismCatalog
            or type(self.action) is not MechanismAction
            or (self.readout is not None and type(self.readout) is not EpisodicReadoutConfig)
            or type(self.dtype) is not str
            or self.dtype not in ("torch.float32", "torch.float64")
        ):
            raise ValueError("La propuesta necesita configuraciones tipadas y precisión explícita")
        for config in (self.predictor, self.catalog, self.action, self.readout):
            if config is not None:
                config.__post_init__()
        if self.action.calibration != "none":
            raise ValueError("El registro inicial no admite calibradores nuevos")
        if self.readout is not None and (
            self.readout.refinements != 1 or self.readout.hidden_size != self.predictor.hidden_size
        ):
            raise ValueError("El lector base necesita K1 y la dimensión del predictor")
        program, registered = self.predictor.regional_execution, self.catalog.regional_execution
        if canonical(None if program is None else program.identity()) != canonical(
            None if registered is None else registered.identity()
        ):
            raise ValueError("El catálogo no corresponde al programa regional")
        bridge = None if self.readout is None else self.readout.evidence_bridge
        reader_contract = "legacy" if bridge is None else bridge.identity()["kind"]
        if self.catalog.reader_contract != reader_contract:
            raise ValueError("El catálogo no corresponde al contrato del lector")

    def _current_identity(self):
        return dict(
            schema_version=1,
            kind="registered_financial_configuration_v1",
            predictor=self.predictor.identity(),
            readout=None if self.readout is None else self.readout.identity(),
            action=dict(
                context=self.action.context,
                route=self.action.route,
                calibration=self.action.calibration,
            ),
            catalog=self.catalog.identity(),
            dtype=self.dtype,
            policy=None,
            calibration=None,
        )

    def identity(self):
        """Devolver una copia del sello tras comprobar las configuraciones recibidas."""
        try:
            current = _bounded_identity(self._current_identity())
        except (AttributeError, KeyError, TypeError) as error:
            raise ValueError("La configuración registrada ha cambiado") from error
        if current != self._sealed:
            raise ValueError("La configuración registrada no conserva su sello")
        self._validate()
        return json.loads(self._sealed)

    @property
    def configuration_id(self):
        self.identity()
        return _fingerprint(self._sealed)


@dataclass(frozen=True)
class ProposalRegistry:
    """Conjunto explícito de recetas admisibles, sin generación ni selección implícitas."""

    entries: tuple[RegisteredConfiguration, ...]
    baseline_id: str
    _sealed: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self._validate_entries()
        _digest_id(self.baseline_id, "La configuración de referencia")
        identifiers = tuple(entry.configuration_id for entry in self.entries)
        if len(set(identifiers)) != len(identifiers) or self.baseline_id not in identifiers:
            raise ValueError("El registro repite configuraciones o no contiene la referencia")
        object.__setattr__(self, "_sealed", _bounded_identity(self._current_identity()))

    def _validate_entries(self):
        if (
            type(self.entries) is not tuple
            or not 1 <= len(self.entries) <= MAX_CONFIGURATIONS
            or any(type(entry) is not RegisteredConfiguration for entry in self.entries)
        ):
            raise ValueError("El registro necesita entre una y ocho configuraciones explícitas")

    def _current_identity(self):
        return dict(
            schema_version=1,
            kind="finite_financial_proposal_registry_v1",
            baseline_id=self.baseline_id,
            entries=[entry.identity() for entry in self.entries],
        )

    def identity(self):
        self._validate_entries()
        if _bounded_identity(self._current_identity()) != self._sealed:
            raise ValueError("El registro no conserva su sello")
        return json.loads(self._sealed)

    @property
    def registry_id(self):
        self.identity()
        return _fingerprint(self._sealed)

    def resolve(self, configuration_id):
        """Resolver solo una receta registrada, sin crear ni sustituir un consumidor."""
        _digest_id(configuration_id, "La configuración solicitada")
        self.identity()
        for entry in self.entries:
            if entry.configuration_id == configuration_id:
                return entry
        raise ValueError("La configuración solicitada no pertenece al registro")
