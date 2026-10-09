"""Variante MARS-TITAN con ampliaciones construida desde su declaración.

`configs/titans/mars-titan-extensions.json` fija los componentes, sus valores, su control y
si están conectados. Este módulo construye solo las combinaciones de componentes conectados y
rechaza el resto con el motivo declarado. Con todos apagados la variante es el brazo
Titans-MAC `mac_online` seleccionado, sin lector, con admisión M0 y sin control local, y su
identidad es la del núcleo. Cada combinación activa tiene una identidad propia.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.financial import FinancialPredictor
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

from .associative_memory import CORRECTION_KEYS, AssociativeMemoryConfig, MatureCorrection
from .financial_consumers import bank_retention

DECLARATION = Path("configs/titans/mars-titan-extensions.json")
VARIANT = "mars_titan_extensions_v1"
CONNECTED = ("episodic_bank", "refinements", "refinement_episodes", "associative_memory")
M3_MOTIVE = (
    "M3 no está definida: faltan a_norm y r_norm acreditados, sus escalas y sus umbrales, "
    "así que no se construye ninguna variante con M3"
)
_BASE_FIELDS = {"architecture", "configuration", "dtype", "parameters_sha256", "recipe"}
# Opciones del lector que fija la combinación de componentes y no la receta.
_READOUT_FIELDS = {"mode", "refinements", "episode_selection", "codec_id"}


def load_declaration(path=DECLARATION):
    """Leer la declaración y exigir que su conexión coincida con este constructor."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 256 * 1024:
        raise ValueError("La declaración no es un archivo regular de hasta 256 KiB")
    document = json.loads(path.read_text(encoding="utf-8"))
    components = document.get("components") if isinstance(document, dict) else None
    if (
        not isinstance(components, dict)
        or document.get("schema_version") != 1
        or document.get("variant") != VARIANT
        or document.get("final_test_opened") is not False
        or any(c.get("enabled") is not False for c in components.values())
    ):
        raise ValueError("La declaración de MARS-TITAN no conserva su esquema apagado")
    connected = tuple(name for name, c in components.items() if c.get("connection") == "connected")
    if connected != CONNECTED or any(
        c.get("connection") not in ("connected", "declared") for c in components.values()
    ):
        raise ValueError("La conexión declarada no coincide con la del constructor")
    return document


def core_identity(predictor, recipe):
    """Identidad del brazo Titans-MAC de partida: configuración, parámetros y receta."""
    if type(predictor) is not FinancialPredictor:
        raise ValueError("El núcleo de MARS-TITAN es un FinancialPredictor de Titans-MAC")
    return dict(
        architecture="titans_mac",
        configuration=predictor.config.identity(),
        dtype=str(predictor.head.weight.dtype),
        parameters_sha256=predictor._parameter_id,
        recipe=recipe,
    )


def _associative(value, allowed):
    fields = {"rule", "key", "rate", "forgetting"}
    if not isinstance(value, dict) or set(value) != fields or value["rule"] not in allowed:
        raise ValueError("associative_memory declara rule, key, rate y forgetting permitidos")
    if value["key"] not in CORRECTION_KEYS:
        raise ValueError("La clave de B6 debe ser codec o constant")
    memory = AssociativeMemoryConfig(
        value["rule"],
        key_size=64 if value["key"] == "codec" else 1,
        rate=value["rate"],
        forgetting=value["forgetting"],
    )
    return MatureCorrection(memory, key=value["key"])


def select_variant(declaration, components, *, base):
    """Validar la combinación pedida contra la declaración y devolver su variante."""
    if not isinstance(components, dict) or not isinstance(base, dict):
        raise ValueError("La variante necesita componentes y la identidad del núcleo")
    if base.get("architecture") != "titans_mac" or set(base) != _BASE_FIELDS:
        raise ValueError("La base debe ser la identidad de un núcleo Titans-MAC")
    if base["configuration"].get("variant") != "mac_online":
        raise ValueError("MARS-TITAN parte de Titans-MAC mac_online")
    canonical(base)
    declared = declaration["components"]
    for name, value in components.items():
        component = declared.get(name)
        if component is None:
            raise ValueError(f"El componente {name} no está declarado en MARS-TITAN")
        if component["connection"] != "connected":
            raise ValueError(
                f"El componente {name} está declarado sin conexión: {component['pending']}"
            )
        if name == "episodic_bank" and value == "m3":
            raise ValueError(M3_MOTIVE)
        if value == component["disabled_value"]:
            raise ValueError(f"El valor apagado de {name} se expresa omitiendo el componente")
        if name != "associative_memory" and not any(
            type(value) is type(allowed) and value == allowed for allowed in component["allowed"]
        ):
            raise ValueError(f"{name} no admite el valor {value!r}")
        required = component.get("requires")
        if required is not None and required not in components:
            raise ValueError(f"{name} necesita {required} activo")
    if "refinement_episodes" in components and components.get("refinements", 1) == 1:
        raise ValueError("Los episodios fijos solo cambian el cálculo con K mayor que 1")
    components, correction = dict(sorted(components.items())), None
    if "associative_memory" in components:
        if "episodic_bank" in components:
            # El error del banco y la escritura de A mezclarían dos memorias en una emisión.
            raise ValueError("B6 se contrasta sin banco episódico")
        correction = _associative(
            components["associative_memory"], declared["associative_memory"]["allowed"]
        )
        memory = correction.memory
        # La identidad usa los valores normalizados, así 1 y 1.0 no crean dos brazos.
        components["associative_memory"] = dict(
            rule=memory.rule, key=correction.key, rate=memory.rate, forgetting=memory.forgetting
        )
    canonical(components)
    return MarsTitanVariant(base=base, components=components, correction=correction)


@dataclass(frozen=True)
class MarsTitanVariant:
    """Combinación de componentes sobre un núcleo Titans-MAC identificado."""

    base: dict
    components: dict
    correction: MatureCorrection | None = None

    @property
    def admission(self):
        bank = self.components.get("episodic_bank")
        return bank if bank in ("m1", "m2") else "m0"

    @property
    def readout_mode(self):
        """None sin lector, `no_bank` para M0 con refinador y `bank` con escritura."""
        bank = self.components.get("episodic_bank")
        return None if bank is None else "no_bank" if bank == "m0_no_bank" else "bank"

    @property
    def refinements(self):
        return self.components.get("refinements", 1)

    @property
    def episode_selection(self):
        return self.components.get("refinement_episodes", "per_step")

    def identity(self):
        if not self.components:
            return self.base
        return dict(schema_version=1, variant=VARIANT, base=self.base, components=self.components)

    def fingerprint(self):
        return hashlib.sha256(canonical(self.identity()).encode()).hexdigest()

    def readout_config(self, codec_id, **options):
        """Configuración del lector con los campos que fija la combinación. None sin banco."""
        if _READOUT_FIELDS & set(options):
            raise ValueError("Modo, K y episodios pertenecen a la combinación de componentes")
        if self.readout_mode is None:
            return None
        return EpisodicReadoutConfig(
            codec_id,
            mode=self.readout_mode,
            refinements=self.refinements,
            episode_selection=self.episode_selection,
            **options,
        )

    def consumer(self, predictor, readout=None):
        """Consumidor congelado del núcleo identificado y del lector de esta combinación."""
        if core_identity(predictor, self.base["recipe"]) != self.base:
            raise ValueError("El predictor no es el núcleo Titans-MAC de la variante")
        if predictor.local_control is not None:
            raise ValueError("El control local pertenece a CM-v1, no a esta variante")
        if self.readout_mode is None:
            if readout is not None:
                raise ValueError("Sin banco episódico la variante no lleva lector")
        elif type(readout) is not EpisodicReadout or any(
            getattr(readout.config, name) != value
            for name, value in (
                ("mode", self.readout_mode),
                ("refinements", self.refinements),
                ("episode_selection", self.episode_selection),
            )
        ):
            raise ValueError("El lector no corresponde a la combinación de componentes")
        return FrozenFinancialConsumer(predictor, readout=readout)

    def session_options(self, *, capacity, seed, policy="reservoir", **retention):
        """Admisión, retención y corrección B6 para `FinancialSession`."""
        options = dict(
            admission=self.admission,
            retention=bank_retention(
                self.admission, capacity=capacity, seed=seed, policy=policy, **retention
            ),
        )
        if self.correction is not None:
            options["associative"] = self.correction
        return options
