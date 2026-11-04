"""Variante MARS-TITAN con ampliaciones construida desde su declaración.

`configs/titans/mars-titan-extensions.json` fija los componentes, sus valores, su control y
si están conectados. Este módulo construye solo las combinaciones de componentes conectados y
rechaza el resto con el motivo declarado. Con todos apagados la variante es el modelo
Titans-MAC `mac_online` seleccionado, sin lector, con admisión M0 y sin control local, y su
identidad es la del núcleo. Cada combinación activa tiene una identidad propia.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from dataclasses import fields as fields_of
from pathlib import Path

from mars_titan.models.titans.bridge_hierarchy import hierarchy_options
from mars_titan.models.titans.bridge_intervention import intervention_options
from mars_titan.models.titans.bridge_recurrence import BridgeRecurrenceConfig, BridgeStoppingConfig
from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.evidence_bridge import (
    BRIDGE_VIEW_FIELDS,
    EvidenceBridgeConfig,
    SharedEvidenceBridge,
    bridge_view_options,
)
from mars_titan.models.titans.financial import FinancialPredictor
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

from . import (
    area_memory,
    area_query_bridge,
    kernel_memory,
    multiscale_mixture,
    role_binding,
    write_scores,
)
from .area_memory import AreaLayout
from .area_query_bridge import AreaQueryBridgeConfig
from .associative_memory import (
    CORRECTION_KEYS,
    KEY_SIZES,
    AssociativeMemoryConfig,
    KalmanNoise,
    MatureCorrection,
)
from .financial_consumers import bank_retention
from .kernel_memory import KERNEL_RULE, KernelMemoryConfig
from .multiscale_mixture import MultiscaleMixtureConfig
from .role_binding import RoleBindingConfig

DECLARATION = Path("configs/titans/mars-titan-extensions.json")
VARIANT = "mars_titan_extensions_v1"
CONNECTED = (
    "episodic_bank",
    "refinements",
    "refinement_episodes",
    "evidence_bridge",
    "associative_memory",
)
# Escrituras que usan el banco y, por tanto, el lector en modo `bank`.
WRITES = ("m1", "m2", "m3")
_BASE_FIELDS = {"architecture", "configuration", "dtype", "parameters_sha256", "recipe"}
# Opciones del lector que fija la combinación de componentes y no la receta.
_READOUT_FIELDS = {"mode", "refinements", "episode_selection", "codec_id", "evidence_bridge"}
_STOPPING_DECLARATION = dict(
    default=None,
    backends=["adaptive_reference_full", "adaptive_compact"],
    criterion="present_coordinate_linf_symmetric_scale_v1",
    predicate_dtype="torch.float64",
)


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
    if components["episodic_bank"].get("m3") != write_scores.declaration():
        raise ValueError("La definición declarada de M3 no coincide con la del código")
    associative = components["associative_memory"]
    if any(
        (associative.get(field) or {}).get("key") != list(CORRECTION_KEYS)
        for field in ("value", "kalman_value")
    ):
        raise ValueError("Las claves declaradas de B6 no coinciden con las del código")
    if (
        associative.get("area_value") != area_memory.declaration()
        or associative.get("kernel_value") != kernel_memory.declaration()
        or associative.get("query_bridge_value") != area_query_bridge.declaration()
        or associative.get("role_binding_value") != role_binding.declaration()
        or associative.get("multiscale_value") != multiscale_mixture.declaration()
    ):
        raise ValueError("La declaración de B1 y de su control no coincide con el código")
    return document


def core_identity(predictor, recipe):
    """Identidad del modelo Titans-MAC de partida: configuración, parámetros y receta."""
    if type(predictor) is not FinancialPredictor:
        raise ValueError("El núcleo de MARS-TITAN es un FinancialPredictor de Titans-MAC")
    return dict(
        architecture="titans_mac",
        configuration=predictor.config.identity(),
        dtype=str(predictor.head.weight.dtype),
        parameters_sha256=predictor._parameter_id,
        recipe=recipe,
    )


_KERNEL_FIELDS = ("bandwidth", "forgetting", "capacity")
_KALMAN_FIELDS = (
    "process_noise",
    "observation_noise",
    "cohort_correlation",
    "prior_variance",
    "huber_threshold",
)


def _associative(value, component):
    rule = value.get("rule") if isinstance(value, dict) else None
    kalman, kernel = rule == "kalman", rule == KERNEL_RULE
    if kalman:
        # La regla kalman (PT3) sustituye η y λ por las varianzas de su modelo dinámico.
        if set(value) != {"rule", "key", *_KALMAN_FIELDS} or "kalman" not in component["allowed"]:
            raise ValueError("associative_memory kalman declara rule, key y sus cinco varianzas")
    elif kernel:
        # El control de Nadaraya-Watson de B1 sustituye η por su ancho τ y su capacidad.
        if set(value) != {"rule", "key", *_KERNEL_FIELDS} or "kernel_value" not in component:
            raise ValueError(
                "associative_memory nadaraya_watson declara rule, key, τ, λ y capacidad"
            )
    else:
        fields = {"rule", "key", "rate", "forgetting"}
        if (
            not isinstance(value, dict)
            or set(value) - {"areas", "query_bridge", "role_binding", "multiscale"} != fields
            or rule not in component["allowed"]
        ):
            raise ValueError("associative_memory declara rule, key, rate y forgetting permitidos")
    if value["key"] not in CORRECTION_KEYS:
        raise ValueError("La clave de B6 debe ser una de " + ", ".join(CORRECTION_KEYS))
    if kernel:
        memory = KernelMemoryConfig(**{name: value[name] for name in _KERNEL_FIELDS})
        return MatureCorrection(memory, key=value["key"])
    options = (
        dict(kalman=KalmanNoise(**{name: value[name] for name in _KALMAN_FIELDS}))
        if kalman
        else dict(rate=value["rate"], forgetting=value["forgetting"])
    )
    memory = AssociativeMemoryConfig(value["rule"], key_size=KEY_SIZES[value["key"]], **options)
    areas = value.get("areas")
    if areas is not None:
        # B1 declara todos los campos de su reparto, sin valores implícitos.
        if not isinstance(areas, dict) or set(areas) != {f.name for f in fields_of(AreaLayout)}:
            raise ValueError("Las áreas de B1 declaran areas, shifts, permutation, seed y labels")
        areas = AreaLayout(**areas)
    bridge = value.get("query_bridge")
    if bridge is not None:
        if not isinstance(bridge, dict) or set(bridge) != {
            f.name for f in fields_of(AreaQueryBridgeConfig)
        }:
            raise ValueError("El puente B2 declara gain, steps, targets y seed")
        bridge = AreaQueryBridgeConfig(**bridge)
    binding = value.get("role_binding")
    if binding is not None:
        if not isinstance(binding, dict) or set(binding) != {
            f.name for f in fields_of(RoleBindingConfig)
        }:
            raise ValueError("La ligadura B3 declara source, mode y seed")
        binding = RoleBindingConfig(**binding)
    multiscale = value.get("multiscale")
    if multiscale is not None:
        # E1 declara todos sus campos. Los periodos de escritura de R2 (#786) son opcionales.
        required = {f.name for f in fields_of(MultiscaleMixtureConfig)} - {"write_periods"}
        if not isinstance(multiscale, dict) or set(multiscale) - {"write_periods"} != required:
            raise ValueError(
                "E1 declara half_lives_days, weighting, weight_rate y loss_scale, y R2 además "
                "write_periods"
            )
        multiscale = MultiscaleMixtureConfig(**multiscale)
    return MatureCorrection(
        memory,
        key=value["key"],
        areas=areas,
        query_bridge=bridge,
        role_binding=binding,
        multiscale=multiscale,
    )


def _normalized(correction):
    # La identidad usa los valores normalizados, así 1 y 1.0 no crean dos modelos.
    memory = correction.memory
    if memory.rule == "kalman":
        return dict(rule=memory.rule, key=correction.key, **asdict(memory.kalman))
    if memory.rule == KERNEL_RULE:
        values = {name: getattr(memory, name) for name in _KERNEL_FIELDS}
        return dict(rule=memory.rule, key=correction.key, **values)
    normalized = dict(
        rule=memory.rule, key=correction.key, rate=memory.rate, forgetting=memory.forgetting
    )
    if correction.areas is not None:
        normalized["areas"] = asdict(correction.areas)
    if correction.query_bridge is not None:
        normalized["query_bridge"] = asdict(correction.query_bridge)
    if correction.role_binding is not None:
        normalized["role_binding"] = asdict(correction.role_binding)
    if correction.multiscale is not None:
        multiscale = asdict(correction.multiscale)
        # Sin los periodos de R2 la identidad normalizada de E1 no cambia.
        if multiscale["write_periods"] is None:
            multiscale.pop("write_periods")
        normalized["multiscale"] = multiscale
    return normalized


def check_components(declaration, components):
    """Validar una combinación sin núcleo: componentes normalizados y corrección B6."""
    if not isinstance(components, dict):
        raise ValueError("La variante necesita sus componentes declarados")
    declared = declaration["components"]
    for name, value in components.items():
        component = declared.get(name)
        if component is None:
            raise ValueError(f"El componente {name} no está declarado en MARS-TITAN")
        if component["connection"] != "connected":
            raise ValueError(
                f"El componente {name} está declarado sin conexión: {component['pending']}"
            )
        if value == component["disabled_value"]:
            raise ValueError(f"El valor apagado de {name} se expresa omitiendo el componente")
        if name not in ("associative_memory", "evidence_bridge") and not any(
            type(value) is type(allowed) and value == allowed for allowed in component["allowed"]
        ):
            raise ValueError(f"{name} no admite el valor {value!r}")
        required = component.get("requires")
        if required is not None and required not in components:
            raise ValueError(f"{name} necesita {required} activo")
    components, correction = dict(sorted(components.items())), None
    if "evidence_bridge" in components:
        bridge = bridge_view_options(components["evidence_bridge"])
        if bridge["topology"] not in declared["evidence_bridge"]["allowed"]:
            raise ValueError("La topología del puente no está declarada")
        if "recurrence" in bridge:
            recurrence = declared["evidence_bridge"].get("recurrence")
            if type(recurrence) is not dict or {
                name: value for name, value in recurrence.items() if name != "stopping"
            } != dict(
                steps=[1, 2, 4], max_node_updates_per_row=1216, requires_cycles=0, default=None
            ):
                raise ValueError("La declaración no admite el contrato recurrente acotado")
            if (
                "stopping" in bridge["recurrence"]
                and recurrence.get("stopping") != _STOPPING_DECLARATION
            ):
                raise ValueError("La declaración no admite el contrato de parada adaptativa")
        if "intervention" in bridge and declared["evidence_bridge"].get("intervention") != dict(
            default=None, kind="fixed_reader_mask_and_directed_edge_cuts_v1"
        ):
            raise ValueError("La declaración no admite las intervenciones regionales fijas")
        if "hierarchy" in bridge and declared["evidence_bridge"].get("hierarchy") != dict(
            default=None, kind="fixed_two_level_hierarchy_v1"
        ):
            raise ValueError("La declaración no admite la jerarquía temporal fija")
        components["evidence_bridge"] = bridge
    if "associative_memory" in components:
        if "episodic_bank" in components:
            # El error del banco y la escritura de A mezclarían dos memorias en una emisión.
            raise ValueError("B6 se contrasta sin banco episódico")
        correction = _associative(components["associative_memory"], declared["associative_memory"])
        components["associative_memory"] = _normalized(correction)
    canonical(components)
    return components, correction


def select_variant(declaration, components, *, base):
    """Validar la combinación pedida contra la declaración y devolver su variante."""
    if not isinstance(base, dict):
        raise ValueError("La variante necesita la identidad del núcleo")
    if base.get("architecture") != "titans_mac" or set(base) != _BASE_FIELDS:
        raise ValueError("La base debe ser la identidad de un núcleo Titans-MAC")
    if base["configuration"].get("variant") != "mac_online":
        raise ValueError("MARS-TITAN parte de Titans-MAC mac_online")
    canonical(base)
    components, correction = check_components(declaration, components)
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
        return bank if bank in WRITES else "m0"

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
        config = EpisodicReadoutConfig(
            codec_id,
            mode=self.readout_mode,
            refinements=self.refinements,
            episode_selection=self.episode_selection,
            **options,
        )
        if "evidence_bridge" in self.components:
            geometry = dict(self.components["evidence_bridge"])
            if "recurrence" in geometry:
                recurrence = dict(geometry["recurrence"])
                if "stopping" in recurrence:
                    recurrence["stopping"] = BridgeStoppingConfig(**recurrence["stopping"])
                geometry["recurrence"] = BridgeRecurrenceConfig(**recurrence)
            if "intervention" in geometry:
                geometry["intervention"] = intervention_options(geometry["intervention"])
            if "hierarchy" in geometry:
                geometry["hierarchy"] = hierarchy_options(geometry["hierarchy"])
            bridge = EvidenceBridgeConfig(
                codec_id,
                **geometry,
                **{
                    name: getattr(config, name)
                    for name in (
                        "hidden_size",
                        "neighbors",
                        "temperature",
                        "max_batch",
                        "max_working_bytes",
                    )
                },
            )
            config = replace(config, evidence_bridge=bridge)
        return config

    def consumer(self, predictor, readout=None):
        """Consumidor congelado del núcleo identificado y del lector de esta combinación."""
        if core_identity(predictor, self.base["recipe"]) != self.base:
            raise ValueError("El predictor no es el núcleo Titans-MAC de la variante")
        if predictor.local_control is not None:
            raise ValueError("El control local pertenece a CM-v1, no a esta variante")
        if self.readout_mode is None:
            if readout is not None:
                raise ValueError("Sin banco episódico la variante no lleva lector")
        elif (
            type(readout) is not EpisodicReadout
            or self._bridge_options(readout) != self.components.get("evidence_bridge")
            or any(
                getattr(readout.config, name) != value
                for name, value in (
                    ("mode", self.readout_mode),
                    ("refinements", self.refinements),
                    ("episode_selection", self.episode_selection),
                )
            )
        ):
            raise ValueError("El lector no corresponde a la combinación de componentes")
        return FrozenFinancialConsumer(predictor, readout=readout)

    @staticmethod
    def _bridge_options(readout):
        readout.config.__post_init__()
        bridge = readout.config.evidence_bridge
        module = readout.evidence_bridge
        if bridge is None:
            if module is not None:
                raise ValueError("El lector no declara el módulo del puente de la combinación")
            return None
        if (
            type(module) is not SharedEvidenceBridge
            or type(module.config) is not EvidenceBridgeConfig
        ):
            raise ValueError("El módulo del puente no corresponde a la combinación")
        module.config.__post_init__()
        if canonical(module.config.identity()) != canonical(bridge.identity()):
            raise ValueError("Los metadatos y el módulo del puente no conservan la combinación")
        geometry = {name: getattr(bridge, name) for name in BRIDGE_VIEW_FIELDS}
        if bridge.recurrence is not None:
            geometry["recurrence"] = bridge.recurrence.options()
        if bridge.intervention is not None:
            geometry["intervention"] = asdict(bridge.intervention)
        if bridge.hierarchy is not None:
            geometry["hierarchy"] = asdict(bridge.hierarchy)
        return bridge_view_options(geometry)

    def session_options(self, *, capacity, seed, policy="reservoir", **retention):
        """Admisión, retención y corrección B6 para `FinancialSession`.

        M3 recibe en `retention` las escalas `scalers` del tramo de entrenamiento.
        """
        options = dict(
            admission=self.admission,
            retention=bank_retention(
                self.admission, capacity=capacity, seed=seed, policy=policy, **retention
            ),
        )
        if self.correction is not None:
            options["associative"] = self.correction
        return options
