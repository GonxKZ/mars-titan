"""Materializar recetas registradas con pesos emparejados y estado temporal nuevo."""

import hashlib
import json
from dataclasses import dataclass, replace

import numpy as np
import torch

from mars_titan.data.input_policy import MODALITIES
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.financial import FinancialPredictor, _tensor_digest
from mars_titan.models.titans.financial_inputs import FinancialInputSpec
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.models.titans.regional_execution import RegionalExecutionConfig
from mars_titan.models.titans.state import check_finite

from .contracts import ProposalRegistry, RegisteredConfiguration
from .mature_review import MAX_REVIEW_BYTES, encoded, fingerprint, require


def _codec_identity(codec, specification):
    require(type(codec) is FrozenEpisodeCodec, "Falta el codec episódico congelado")
    codec._verify_projection()
    identity = codec.identity()
    widths = tuple(
        specification.dimensions[name] * (specification.context if name == "prices" else 1)
        for name in MODALITIES
    )
    require(
        identity["input_specification"] == specification.identity()
        and identity["input_contract_id"] == codec.input_contract_id == specification.fingerprint()
        and identity["projection_sha256"] == hashlib.sha256(codec._projection).hexdigest()
        and codec._input_widths == widths
        and list(codec._modality_weights) == identity["modality_weights"]
        and codec._presence_weight == identity["presence_weight"]
        and codec._key_constant == identity["key_constant"]
        and codec.max_buffer_bytes == identity["max_buffer_bytes"]
        and np.array_equal(codec._transposed, codec._projection.T),
        "El codec no conserva sus entradas o proyección",
    )
    return identity


def _fixed_identity(consumer):
    require(
        type(consumer) is FrozenMechanismConsumer, "Falta un consumidor de mecanismos congelado"
    )
    consumer.verify(strong=True)
    identity = consumer.identity()
    require(
        consumer.model_id == fingerprint(identity)
        and identity["fixed_controls"] is True
        and "selector" not in identity
        and identity["calibration"] is None
        and "local_control" not in identity["predictor"],
        "El padre necesita una receta fija sin selector, calibración o control local",
    )
    return identity


def _parent_configuration(registry, parent, parent_codec):
    identity = _fixed_identity(parent)
    baseline = registry.resolve(registry.baseline_id)
    registered = baseline.identity()
    require(
        registered["predictor"] == identity["predictor"]["configuration"]
        and registered["dtype"] == identity["predictor"]["dtype"]
        and registered["action"] == identity["action"]
        and registered["catalog"] == identity["catalog"]
        and (
            identity["readout"] is None
            if registered["readout"] is None
            else identity["readout"] is not None
            and registered["readout"] == identity["readout"]["configuration"]
            and registered["dtype"] == identity["readout"]["dtype"]
        ),
        "La referencia registrada no corresponde al consumidor padre",
    )
    _codec_identity(parent_codec, baseline.predictor.inputs)
    if baseline.readout is not None:
        require(
            baseline.readout.codec_id == parent_codec.fingerprint(),
            "El lector padre pertenece a otro codec",
        )
    return baseline


def _check_candidate(baseline, candidate):
    predictor = baseline.predictor
    program, requested = predictor.regional_execution, candidate.predictor.regional_execution
    if type(program) is RegionalExecutionConfig and type(requested) is RegionalExecutionConfig:
        predictor = replace(
            predictor, regional_execution=replace(program, program=requested.program)
        )
    reader = baseline.readout
    if reader is not None and candidate.readout is not None:
        bridge, requested_bridge = reader.evidence_bridge, candidate.readout.evidence_bridge
        if (
            bridge is not None
            and requested_bridge is not None
            and bridge.recurrence is not None
            and requested_bridge.recurrence is not None
        ):
            recurrence = replace(
                bridge.recurrence,
                steps=requested_bridge.recurrence.steps,
                max_node_updates_per_row=requested_bridge.recurrence.max_node_updates_per_row,
            )
            reader = replace(reader, evidence_bridge=replace(bridge, recurrence=recurrence))
    expected = replace(
        baseline,
        predictor=predictor,
        readout=reader,
        catalog=replace(baseline.catalog, regional_execution=predictor.regional_execution),
    )
    require(
        expected.identity() == candidate.identity(),
        "La candidata cambia opciones distintas del programa regional o J",
    )


def _phase_inputs(original, specification):
    require(type(specification) is FinancialInputSpec, "Faltan las entradas tipadas de la fase")
    source, target = original.identity(), specification.identity()
    require(
        {key: value for key, value in source.items() if key != "view_sha256"}
        == {key: value for key, value in target.items() if key != "view_sha256"},
        "La fase cambia corpus, representación, dimensiones, contexto o política",
    )


def _phase_codec(parent_codec, specification):
    identity = parent_codec.identity()
    codec = FrozenEpisodeCodec(
        specification,
        **{
            name: identity[name]
            for name in (
                "seed",
                "modality_weights",
                "presence_weight",
                "key_constant",
                "max_buffer_bytes",
            )
        },
    )
    current = _codec_identity(codec, specification)
    excluded = {"input_specification", "input_contract_id"}
    require(
        {key: value for key, value in identity.items() if key not in excluded}
        == {key: value for key, value in current.items() if key not in excluded}
        and not np.shares_memory(parent_codec._projection, codec._projection),
        "El codec de fase cambió la proyección o comparte su memoria",
    )
    return codec


def _effective_configuration(candidate, specification, codec):
    reader = candidate.readout
    if reader is not None:
        bridge = reader.evidence_bridge
        if bridge is not None:
            bridge = replace(bridge, codec_id=codec.fingerprint())
        reader = replace(reader, codec_id=codec.fingerprint(), evidence_bridge=bridge)
    return replace(
        candidate, predictor=replace(candidate.predictor, inputs=specification), readout=reader
    )


def _tensors(predictor, readout):
    result = {}
    for prefix, module in (("predictor", predictor), ("readout", readout)):
        if module is None:
            continue
        for kind, values in (
            ("parameter", module.named_parameters()),
            ("buffer", module.named_buffers()),
        ):
            for name, value in values:
                result[f"{prefix}.{name}"] = kind, value
    return result


def _pairs(parent, predictor, readout):
    original, destination = _tensors(parent.predictor, parent.readout), _tensors(predictor, readout)
    require(original.keys() == destination.keys(), "La candidata cambia los tensores compartidos")
    for name in sorted(original):
        kind, source = original[name]
        target_kind, target = destination[name]
        require(
            kind == target_kind
            and source.layout == target.layout == torch.strided
            and source.shape == target.shape
            and source.dtype == target.dtype
            and source.device == target.device
            and (
                source.numel() == 0
                or source.untyped_storage().data_ptr() != target.untyped_storage().data_ptr()
            ),
            "La candidata cambia geometría, precisión, dispositivo o comparte un tensor",
        )
        check_finite(source, "Los tensores del padre")
        yield name, kind, source, target


def _runtime_body(runtime):
    require(
        type(runtime._registry) is ProposalRegistry
        and type(runtime.effective_configuration) is RegisteredConfiguration,
        "La materialización perdió sus configuraciones tipadas",
    )
    baseline = _parent_configuration(runtime._registry, runtime._parent, runtime._parent_codec)
    candidate = runtime._registry.resolve(runtime.configuration_id)
    _check_candidate(baseline, candidate)
    _fixed_identity(runtime.consumer)
    specification = runtime.effective_configuration.predictor.inputs
    _phase_inputs(baseline.predictor.inputs, specification)
    source_codec = _codec_identity(runtime._parent_codec, baseline.predictor.inputs)
    target_codec = _codec_identity(runtime.codec, specification)
    data_fields = {"input_specification", "input_contract_id"}
    require(
        {key: value for key, value in source_codec.items() if key not in data_fields}
        == {key: value for key, value in target_codec.items() if key not in data_fields}
        and not np.shares_memory(runtime._parent_codec._projection, runtime.codec._projection),
        "El codec efectivo cambió la proyección o comparte la memoria del padre",
    )
    expected = _effective_configuration(candidate, specification, runtime.codec)
    require(
        expected.identity() == runtime.effective_configuration.identity(),
        "La configuración efectiva cambió fuera de la reasignación de vista",
    )
    actual = RegisteredConfiguration(
        predictor=runtime.consumer.predictor.config,
        readout=None if runtime.consumer.readout is None else runtime.consumer.readout.config,
        action=runtime.consumer.action,
        catalog=runtime.consumer.catalog,
        dtype=str(runtime.consumer.predictor.head.weight.dtype),
    )
    require(actual.identity() == expected.identity(), "El consumidor efectivo cambió de receta")
    transfers = {}
    for name, kind, source, target in _pairs(
        runtime._parent, runtime.consumer.predictor, runtime.consumer.readout
    ):
        check_finite(target, "Los tensores efectivos")
        before, after = _tensor_digest(source), _tensor_digest(target)
        require(before == after, "La transferencia cambió los bytes de un tensor")
        transfers[name] = dict(
            kind=kind,
            shape=list(target.shape),
            dtype=str(target.dtype),
            bytes=target.numel() * target.element_size(),
            source_sha256=before,
            target_sha256=after,
        )
    return dict(
        schema_version=1,
        kind="paired_registered_candidate_runtime_v1",
        mode="paired_parent_weights_fresh_state_v1",
        registry_id=runtime._registry.registry_id,
        configuration_id=candidate.configuration_id,
        effective_configuration_id=expected.configuration_id,
        registered_configuration=candidate.identity(),
        effective_configuration=expected.identity(),
        parent_consumer_id=runtime._parent.model_id,
        parent_parameter_id=runtime._parent.predictor._parameter_id,
        consumer_id=runtime.consumer.model_id,
        parameter_id=runtime.consumer.predictor._parameter_id,
        parent_codec_id=runtime._parent_codec.fingerprint(),
        codec_id=runtime.codec.fingerprint(),
        parent_codec=source_codec,
        codec=target_codec,
        tensors=transfers,
        runtime_state_transferred=False,
        parameter_aliases=False,
        codec_aliases=False,
        rng_preserved=True,
        selection_performed=False,
        deployment_performed=False,
    )


@dataclass(frozen=True, init=False)
class CandidateRuntime:
    """Consumidor efectivo verificable, sin sellar silenciosamente cambios posteriores."""

    consumer: FrozenMechanismConsumer
    codec: FrozenEpisodeCodec
    configuration_id: str
    effective_configuration: RegisteredConfiguration
    _registry: ProposalRegistry
    _parent: FrozenMechanismConsumer
    _parent_codec: FrozenEpisodeCodec
    _json: str
    _sha256: str

    def __init__(
        self,
        consumer,
        codec,
        configuration_id,
        effective_configuration,
        *,
        registry,
        parent,
        parent_codec,
    ):
        for name, value in dict(
            consumer=consumer,
            codec=codec,
            configuration_id=configuration_id,
            effective_configuration=effective_configuration,
            _registry=registry,
            _parent=parent,
            _parent_codec=parent_codec,
        ).items():
            object.__setattr__(self, name, value)
        content = encoded(_runtime_body(self), MAX_REVIEW_BYTES)
        object.__setattr__(self, "_json", content)
        object.__setattr__(self, "_sha256", hashlib.sha256(content.encode()).hexdigest())

    def record(self):
        require(
            hashlib.sha256(self._json.encode()).hexdigest() == self._sha256
            and encoded(_runtime_body(self), MAX_REVIEW_BYTES) == self._json,
            "La materialización no conserva el recibo de sus objetos efectivos",
        )
        return dict(json.loads(self._json), runtime_id=self._sha256)


def materialize_registered_candidate(
    registry, configuration_id, *, parent, parent_codec, input_spec, device="cuda:0"
):
    """Copiar pesos del padre a una receta permitida y reasignar su vista de fase."""
    require(type(registry) is ProposalRegistry, "Falta el registro tipado de candidatas")
    require(
        type(device) is str and device == "cuda:0", "La materialización requiere cuda:0 explícita"
    )
    candidate = registry.resolve(configuration_id)
    baseline = _parent_configuration(registry, parent, parent_codec)
    _check_candidate(baseline, candidate)
    _phase_inputs(baseline.predictor.inputs, input_spec)
    require(str(parent.predictor.head.weight.device) == device, "El padre usa otro dispositivo")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible para materializar la candidata")
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone()
    with torch.random.fork_rng(devices=[0]):
        codec = _phase_codec(parent_codec, input_spec)
        effective = _effective_configuration(candidate, input_spec, codec)
        dtype = {"torch.float32": torch.float32, "torch.float64": torch.float64}[effective.dtype]
        predictor = (
            FinancialPredictor(effective.predictor, device=device, dtype=dtype)
            .eval()
            .requires_grad_(False)
        )
        reader = (
            None
            if effective.readout is None
            else EpisodicReadout(effective.readout, device=device, dtype=dtype)
            .eval()
            .requires_grad_(False)
        )
        pairs = list(_pairs(parent, predictor, reader))
        with torch.no_grad():
            for _, _, source, target in pairs:
                target.copy_(source)
        predictor._seal_parameters()
        consumer = FrozenMechanismConsumer(
            predictor, readout=reader, action=effective.action, catalog=effective.catalog
        )
        runtime = CandidateRuntime(
            consumer,
            codec,
            configuration_id,
            effective,
            registry=registry,
            parent=parent,
            parent_codec=parent_codec,
        )
    require(
        torch.equal(cpu_rng, torch.get_rng_state())
        and torch.equal(cuda_rng, torch.cuda.get_rng_state(0)),
        "La materialización cambió el RNG global de CPU o CUDA",
    )
    return runtime
