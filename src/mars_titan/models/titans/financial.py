"""Predictor financiero por decisión, sin banco ni publicación de estado."""

import hashlib
import re
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields, replace

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from mars_titan.data.input_policy import MODALITIES, masked_inputs
from mars_titan.models.baselines.multimodal import (
    HEADS,
    SCALAR_HEAD,
    MultimodalReference,
    validate_architecture,
)
from mars_titan.models.quantile_head import CONTRACT, QUANTILE_HEAD, QuantileHead, median

from .config import (
    GateBias,
    MACConfig,
    MemoryConfig,
    bounded_integer,
    canonical,
    require_identity,
)
from .financial_inputs import FINAL_TEST_US, HISTORICAL_START_US, DecisionBatch, FinancialInputSpec
from .local_control import MACProjectionConfig, MACProjectionControl, ProjectedMACResult
from .mac import TitansMAC
from .state import MACState, check_differentiable, check_finite

VARIANTS = ("transformer_direct", "mac_disabled", "mac_frozen", "mac_online")


@dataclass(frozen=True)
class FinancialConfig:
    inputs: FinancialInputSpec
    variant: str = "transformer_direct"
    hidden_size: int = 64
    layers: int = 1
    seed: int = 42
    persistent_tokens: int = 4
    max_batch: int = 256
    max_state_bytes: int = 64 * 1024**2
    bank_policy: str = "disabled"
    refinements: int = 1
    head: str = SCALAR_HEAD
    gate_bias: GateBias | None = None

    def __post_init__(self):
        if not isinstance(self.inputs, FinancialInputSpec) or self.variant not in VARIANTS:
            raise ValueError("El predictor necesita una entrada y un control identificados")
        if not isinstance(self.head, str) or self.head not in HEADS:
            raise ValueError("La cabeza de salida no pertenece al contrato del predictor")
        if isinstance(self.gate_bias, dict):
            # Las recetas JSON declaran los tres valores de forma explícita.
            if set(self.gate_bias) != {field.name for field in fields(GateBias)}:
                raise ValueError("gate_bias debe declarar alpha_half_life, eta y theta")
            object.__setattr__(self, "gate_bias", GateBias(**self.gate_bias))
        if self.gate_bias is not None:
            if not isinstance(self.gate_bias, GateBias):
                raise ValueError("gate_bias debe ser GateBias, sus tres valores o None")
            self.gate_bias.logits(MemoryConfig.theta_max)
        validate_architecture(self.hidden_size, self.layers, 0.0)
        bounded_integer(self.seed, "semilla", 0, 2**32 - 1)
        bounded_integer(self.persistent_tokens, "prefijo", 0, 64)
        bounded_integer(self.max_batch, "lote", 1, 256)
        bounded_integer(self.max_state_bytes, "bytes de estado", 1, 256 * 1024**2)
        if (
            self.bank_policy != "disabled"
            or type(self.refinements) is not int
            or self.refinements != 1
        ):
            raise ValueError("Esta pieza admite únicamente banco desactivado y K=1")

    def identity(self):
        result = dict(
            schema_version=1,
            inputs=self.inputs.identity(),
            variant=self.variant,
            hidden_size=self.hidden_size,
            layers=self.layers,
            seed=self.seed,
            persistent_tokens=self.persistent_tokens,
            max_batch=self.max_batch,
            max_state_bytes=self.max_state_bytes,
            bank_policy=self.bank_policy,
            refinements=self.refinements,
            segment_length=1,
            dropout=0.0,
            heads=4,
            feedforward_multiplier=2,
            output="scalar_corpus_target",
            token_policy="one_fused_token_per_new_decision",
            mask_fusion="zero_after_projection_then_concat_presence"
            if masked_inputs(self.inputs.input_policy)
            else "strict_original",
        )
        # La identidad escalar no cambia. La cabeza de cuantiles sustituye la salida.
        if self.head == QUANTILE_HEAD:
            result.update(output=QUANTILE_HEAD, output_head=dict(CONTRACT))
        # Sin bias declarado la identidad no cambia. Se registra en todas las variantes para
        # que el emparejamiento desde mac_online compare la misma configuración.
        if self.gate_bias is not None:
            result.update(memory_gate_bias=asdict(self.gate_bias))
        return result


@dataclass(frozen=True)
class FinancialState:
    config_id: str
    parameter_id: str
    flow_ids: tuple[str, ...]
    last_sample_ids: tuple[str | None, ...]
    last_prediction_at: tuple[int, ...]
    observed_steps: torch.Tensor
    mac: MACState | None


@dataclass(frozen=True)
class PreparedDecisions:
    point_predictions: torch.Tensor
    detached_tokens: torch.Tensor
    next_state: FinancialState
    representation_id: str
    input_digest: str
    working_state: torch.Tensor | None = None
    local_control: ProjectedMACResult | None = None
    # Solo con `quantile_head_v1`: [flujos, 5]. La predicción puntual es su mediana.
    quantiles: torch.Tensor | None = None


def _tensor_digest(value):
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


class FinancialPredictor(nn.Module):
    """Preparar una decisión por flujo. El llamante confirma o descarta la propuesta."""

    def __init__(self, config, *, local_control=None, device="cpu", dtype=torch.float32):
        super().__init__()
        if torch.is_inference_mode_enabled():
            raise ValueError(
                "inference_mode es incompatible con esta interfaz de gradientes explícitos"
            )
        if not isinstance(config, FinancialConfig) or dtype not in (torch.float32, torch.float64):
            raise ValueError("La configuración o precisión financiera no es válida")
        if local_control is not None and (
            not isinstance(local_control, MACProjectionConfig)
            or config.variant != "mac_online"
            or config.hidden_size > 64
        ):
            raise ValueError("El control local requiere MAC online de dimensión hasta 64")
        self.config = config
        self.masked = masked_inputs(config.inputs.input_policy)
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            reference = MultimodalReference(
                "transformer",
                config.inputs.dimensions,
                context=config.inputs.context,
                hidden_size=config.hidden_size,
                layers=config.layers,
                dropout=0.0,
            )
            self.price_encoder = reference.price_encoder
            self.encoders, self.fusion, self.head = (
                reference.encoders,
                reference.fusion,
                reference.head,
            )
            if self.masked:
                self.fusion[0] = nn.Linear(5 * config.hidden_size + 5, config.hidden_size)
            # La cabeza de cuantiles se crea al final. El tronco recibe los mismos pesos
            # iniciales que la variante escalar con la misma semilla.
            if config.head == QUANTILE_HEAD:
                self.head = QuantileHead(config.hidden_size)
        self.mac = None
        if config.variant != "transformer_direct":
            self.mac = TitansMAC(
                MACConfig(
                    memory=MemoryConfig(
                        dim=config.hidden_size,
                        depth=2,
                        max_batch=config.max_batch,
                        max_tokens=1,
                        max_state_bytes=config.max_state_bytes,
                        parameter_seed=config.seed,
                        gate_bias=config.gate_bias,
                    ),
                    heads=4,
                    persistent_tokens=config.persistent_tokens,
                    max_segment=1,
                    memory_mode=config.variant.removeprefix("mac_"),
                    parameter_seed=config.seed,
                ),
                dtype=dtype,
            )
        self.local_control = (
            MACProjectionControl(local_control, self.mac.config, dtype=dtype)
            if local_control is not None
            else None
        )
        self.to(device=device, dtype=dtype)
        self._seal_parameters()
        self.register_load_state_dict_post_hook(self._after_load)

    def _signature(self):
        return tuple(
            (name, id(value), value._version, value.dtype, value.device)
            for name, value in (*self.named_parameters(), *self.named_buffers())
        )

    def _seal_parameters(self):
        if self.head.weight.dtype not in (torch.float32, torch.float64):
            raise ValueError("El predictor admite float32 y float64")
        self._parameter_id = self._parameter_digest()
        self._parameter_signature = self._signature()

    def _parameter_digest(self):
        fingerprints = {
            name: _tensor_digest(value)
            for name, value in (*self.named_parameters(), *self.named_buffers())
        }
        return hashlib.sha256(canonical(fingerprints).encode()).hexdigest()

    def verify_parameter_identity(self):
        """Comprobar bytes en una frontera de sesión o recuperación, fuera de prepare."""
        self._check_parameters()
        if self._parameter_digest() != self._parameter_id:
            raise ValueError("La huella de los parámetros no coincide con su identidad confirmada")

    @staticmethod
    def _after_load(module, incompatible_keys):
        module._seal_parameters()

    def _apply(self, function, recurse=True):
        result = super()._apply(function, recurse=recurse)
        if hasattr(self, "_parameter_signature"):
            self._seal_parameters()
        return result

    def _check_parameters(self):
        if self._signature() != self._parameter_signature:
            raise ValueError("Los parámetros cambiaron fuera de una carga o copia identificada")

    def get_extra_state(self):
        result = dict(
            configuration=self.config.identity(),
            dtype=str(self.head.weight.dtype),
            backbone_contract=self.price_encoder.configuration,
            mac_contract=self.mac.config.identity() if self.mac else None,
            training=self.training,
        )
        if self.local_control is not None:
            result["local_control"] = self.local_control.get_extra_state()
        return result

    def _validate_extra_state(self, state):
        expected = self.get_extra_state()
        if (
            not isinstance(state, dict)
            or set(state) != set(expected)
            or type(state["training"]) is not bool
        ):
            raise ValueError("El contrato necesita un modo train/eval explícito")
        require_identity(state, expected | {"training": state["training"]})

    def set_extra_state(self, state):
        self._validate_extra_state(state)
        self.train(state["training"])

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        expected = self.state_dict()
        if self.local_control is not None:
            self.local_control.validate_payload(state_dict, prefix + "local_control.")
        actual = {
            name.removeprefix(prefix): value
            for name, value in state_dict.items()
            if name.startswith(prefix)
        }
        if set(actual) != set(expected):
            raise ValueError("El archivo no contiene el contrato completo del predictor")
        for name, value in expected.items():
            supplied = actual[name]
            if isinstance(value, torch.Tensor):
                if (
                    not isinstance(supplied, torch.Tensor)
                    or supplied.shape != value.shape
                    or supplied.dtype != value.dtype
                    or supplied.layout != torch.strided
                ):
                    raise ValueError("Los parámetros no tienen forma o precisión compatibles")
                check_finite(supplied, "Los parámetros guardados")
            else:
                if name == "_extra_state":
                    self._validate_extra_state(supplied)
                else:
                    require_identity(supplied, value)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _config_id(self):
        return hashlib.sha256(canonical(self.get_extra_state()).encode()).hexdigest()

    def _tensor_bytes_per_flow(self):
        fast = (
            0
            if self.mac is None
            else 4 * self.config.hidden_size**2 * self.head.weight.element_size() + 8
        )
        return fast + 8

    def _metadata(self, flows, ids, moments):
        return dict(
            schema_version=1,
            configuration=self.get_extra_state(),
            parameter_id=self._parameter_id,
            flow_ids=flows,
            last_sample_ids=ids,
            last_prediction_at=moments,
            mac=dict(
                schema_version=1,
                configuration=self.mac.config.identity(),
                memory=dict(schema_version=1, configuration=self.mac.memory.config.identity()),
            )
            if self.mac
            else None,
        )

    def _check_bytes(self, count):
        if count > self.config.max_state_bytes:
            raise ValueError("El estado financiero supera el presupuesto de bytes")

    def _state_size(self, flows, ids, moments):
        metadata = len(canonical(self._metadata(flows, ids, moments)).encode())
        self._check_bytes(self._tensor_bytes_per_flow() * len(flows) + metadata)

    @staticmethod
    def _storage_bytes(tensors):
        storages = set()
        total = 0
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
                raise ValueError("El estado necesita tensores densos")
            storage = tensor.untyped_storage()
            key = (tensor.device, storage.data_ptr())
            if key in storages:
                raise ValueError("Los campos del estado no pueden compartir almacenamiento")
            storages.add(key)
            total += storage.nbytes()
        return total

    def _usage(self, state):
        tensors = [state.observed_steps]
        if state.mac:
            tensors.extend(
                (*state.mac.memory.weights, *state.mac.memory.momentum, state.mac.memory.steps)
            )
        tensor_bytes = self._storage_bytes(tensors)
        metadata = len(
            canonical(
                self._metadata(state.flow_ids, state.last_sample_ids, state.last_prediction_at)
            ).encode()
        )
        return dict(
            flow_count=len(state.flow_ids),
            tensor_bytes_per_flow=self._tensor_bytes_per_flow(),
            tensor_bytes=tensor_bytes,
            metadata_bytes=metadata,
            total_bytes=tensor_bytes + metadata,
        )

    def state_usage(self, state):
        """Contar almacenamiento de tensores y metadatos JSON, sin estimar el RSS."""
        self._validate_state(state)
        return self._usage(state)

    def initial_state(self, flow_ids, *, differentiable=False):
        check_differentiable(differentiable)
        self.verify_parameter_identity()
        flow_ids = tuple(flow_ids)
        self._check_flows(flow_ids)
        size = len(flow_ids)
        self._state_size(flow_ids, (None,) * size, (-1,) * size)
        return FinancialState(
            self._config_id(),
            self._parameter_id,
            flow_ids,
            (None,) * size,
            (-1,) * size,
            torch.zeros(size, dtype=torch.int64, device="cpu"),
            self.mac.initial_state(size, differentiable=differentiable) if self.mac else None,
        )

    def _check_flows(self, ids):
        bounded_integer(len(ids), "lote de flujos", 1, self.config.max_batch)
        if len(set(ids)) != len(ids) or any(
            not isinstance(value, str)
            or re.fullmatch(r"(?:US|CN)/[A-Z0-9.^_=\-]{1,64}", value) is None
            for value in ids
        ):
            raise ValueError("Los flujos deben ser identificadores únicos de mercado y activo")

    def _validate_state(self, state, *, device=None):
        self._validate_cursor(state)
        if self.mac:
            if not isinstance(state.mac, MACState):
                raise ValueError("Falta el estado MAC identificado")
            self.mac.memory.validate_state(state.mac.memory, device=device)
            if state.mac.config_id != self.mac.config.fingerprint():
                raise ValueError("El contrato MAC del estado no coincide")
            expected = (
                state.observed_steps
                if self.config.variant == "mac_online"
                else torch.zeros_like(state.observed_steps)
            )
            if not torch.equal(state.mac.memory.steps, expected.to(state.mac.memory.steps.device)):
                raise ValueError("Las escrituras MAC no coinciden con las observaciones")
        elif state.mac is not None:
            raise ValueError("El control directo no admite estado MAC")
        self._check_bytes(self._usage(state)["total_bytes"])

    def _validate_cursor(self, state):
        self._check_parameters()
        if (
            not isinstance(state, FinancialState)
            or state.config_id != self._config_id()
            or state.parameter_id != self._parameter_id
        ):
            raise ValueError("La identidad de estado o parámetros no corresponde al predictor")
        self._check_flows(state.flow_ids)
        size = len(state.flow_ids)
        steps = state.observed_steps
        if (
            not isinstance(steps, torch.Tensor)
            or steps.device.type != "cpu"
            or steps.dtype != torch.int64
            or steps.shape != (size,)
            or steps.layout != torch.strided
            or not steps.is_contiguous()
            or (steps < 0).any()
            or len(state.last_sample_ids) != size
            or len(state.last_prediction_at) != size
        ):
            raise ValueError("Los contadores del estado no corresponden a sus flujos")
        lower = HISTORICAL_START_US if self.masked else 0
        for flow, identifier, at, count in zip(
            state.flow_ids,
            state.last_sample_ids,
            state.last_prediction_at,
            steps.tolist(),
            strict=True,
        ):
            if type(at) is not int or (
                (count == 0 and (identifier is not None or at != -1))
                or (count > 0 and (not lower <= at < FINAL_TEST_US or identifier != f"{flow}/{at}"))
            ):
                raise ValueError("El cursor no corresponde al último token del flujo")

    def select_state(self, state, flow_ids):
        self._validate_state(state)
        self._check_flows(flow_ids)
        if not set(flow_ids) <= set(state.flow_ids):
            raise ValueError("La selección contiene un flujo desconocido")
        indices = [state.flow_ids.index(flow) for flow in flow_ids]
        mac = state.mac
        if mac:
            memory = mac.memory
            mac = replace(
                mac,
                memory=replace(
                    memory,
                    weights=tuple(w[indices] for w in memory.weights),
                    momentum=tuple(m[indices] for m in memory.momentum),
                    steps=memory.steps[indices],
                ),
            )
        return replace(
            state,
            flow_ids=tuple(flow_ids),
            last_sample_ids=tuple(state.last_sample_ids[i] for i in indices),
            last_prediction_at=tuple(state.last_prediction_at[i] for i in indices),
            observed_steps=state.observed_steps[indices],
            mac=mac,
        )

    def prepare(
        self, batch, state, *, differentiable=False, control_selection=None, control_context_id=None
    ):
        check_differentiable(differentiable)
        if torch.is_inference_mode_enabled():
            raise ValueError("inference_mode impide la actualización asociativa, utilice no_grad")
        self._validate_state(state)
        if (
            not isinstance(batch, DecisionBatch)
            or batch.input_contract_id != self.config.inputs.fingerprint()
        ):
            raise ValueError("La identidad de entradas no corresponde al predictor")
        batch.verify()
        if batch.flow_ids != state.flow_ids:
            raise ValueError("Las entradas y el estado no conservan el mismo orden de flujos")
        if any(
            now <= previous
            for now, previous in zip(batch.prediction_at, state.last_prediction_at, strict=True)
        ):
            raise ValueError("El token está repetido o no es posterior al cursor del flujo")
        if (state.observed_steps == torch.iinfo(torch.int64).max).any():
            raise ValueError("El contador de observaciones desbordaría int64")
        self._state_size(batch.flow_ids, batch.sample_ids, batch.prediction_at)
        for value in (*batch.inputs.values(), batch.presence):
            if value.device != self.head.weight.device:
                raise ValueError("Las entradas y los parámetros necesitan el mismo dispositivo")
        if any(value.dtype != self.head.weight.dtype for value in batch.inputs.values()):
            raise ValueError("Las entradas y los parámetros necesitan la misma precisión")
        if self.local_control is None:
            if control_selection is not None or control_context_id is not None:
                raise ValueError("El predictor anterior no admite una selección de C")
        elif self.local_control.config.mode != "disabled":
            self.local_control._resolve_selection(
                control_selection, control_context_id, batch.flow_ids, state.observed_steps
            )
        backend = sdpa_kernel(SDPBackend.MATH) if self.local_control is not None else nullcontext()
        local_result = None
        with torch.set_grad_enabled(differentiable), backend:
            representations = [self.price_encoder(batch.inputs["prices"])]
            for index, name in enumerate(MODALITIES[1:], 1):
                projected = self.encoders[name](batch.inputs[name])
                if self.masked:
                    projected = projected * batch.presence[:, index : index + 1]
                representations.append(projected)
            if self.masked:
                representations.append(batch.presence.to(dtype=self.head.weight.dtype))
            token = self.fusion(torch.cat(representations, dim=-1))
            check_finite(token, "El token fusionado")
            if self.mac:
                encoded, next_mac = self.mac(
                    token.unsqueeze(1), state.mac, differentiable=differentiable
                )
            else:
                encoded, next_mac = token, None
            if self.local_control is not None:
                local_result = self.local_control(
                    self.mac,
                    token.unsqueeze(1),
                    state.mac,
                    flow_ids=batch.flow_ids,
                    observed_steps=state.observed_steps,
                    selection=control_selection,
                    context_id=control_context_id,
                    differentiable=differentiable,
                )
            quantiles = None
            if self.config.head == QUANTILE_HEAD:
                quantiles = self.head(encoded)
                check_finite(quantiles, "Los cuantiles")
                point = median(quantiles)
            else:
                point = self.head(encoded).squeeze(-1)
            check_finite(point, "La predicción")
            working = encoded.clone() if differentiable else encoded.detach().clone()
        next_state = replace(
            state,
            last_sample_ids=batch.sample_ids,
            last_prediction_at=batch.prediction_at,
            observed_steps=state.observed_steps + 1,
            mac=next_mac,
        )
        return PreparedDecisions(
            point,
            token.detach().clone(),
            next_state,
            hashlib.sha256(canonical([self._config_id(), self._parameter_id]).encode()).hexdigest(),
            batch.input_digest,
            working,
            local_result,
            quantiles,
        )

    def export_state(self, state):
        self.verify_parameter_identity()
        self._validate_state(state)
        return dict(
            schema_version=1,
            configuration=self.get_extra_state(),
            parameter_id=state.parameter_id,
            flow_ids=state.flow_ids,
            last_sample_ids=state.last_sample_ids,
            last_prediction_at=state.last_prediction_at,
            observed_steps=state.observed_steps.clone(),
            mac=self.mac.export_state(state.mac) if self.mac else None,
        )

    def export_state_cpu(self, state):
        """Exportar copias CPU sin grafo para artefactos de recuperación por bloques."""
        from .financial_blocks import export_state_cpu

        return export_state_cpu(self, state)

    def restore_state(self, payload, *, device=None):
        from .financial_blocks import restore_state

        return restore_state(self, payload, device=device)

    def gather_state(self, blocks, flow_ids, *, max_source_bytes=512 * 1024**2):
        """Reunir filas confirmadas en CPU y transferir solo el resultado al predictor."""
        from .financial_blocks import gather_state

        return gather_state(self, blocks, flow_ids, max_source_bytes=max_source_bytes)


def copy_paired_parameters(source, target):
    """Copiar parámetros de controles emparejados sin transferir su estado rápido."""
    if not isinstance(source, FinancialPredictor) or not isinstance(target, FinancialPredictor):
        raise ValueError("El emparejamiento requiere dos predictores financieros")
    source.verify_parameter_identity()
    target.verify_parameter_identity()
    if source.training != target.training:
        raise ValueError("El emparejamiento requiere el mismo modo train/eval")

    def identity(model):
        return {
            key: value
            for key, value in model.config.identity().items()
            if key not in {"variant", "seed"}
        }

    require_identity(identity(source), identity(target))
    control_receipt = None
    if source.local_control is not None or target.local_control is not None:
        if source.local_control is None or target.local_control is None:
            raise ValueError("El nuevo factorial requiere Math en ambos controles")
        left, right = source.local_control, target.local_control
        require_identity(
            {
                key: value
                for key, value in left.config.identity().items()
                if key not in {"mode", "weight"}
            },
            {
                key: value
                for key, value in right.config.identity().items()
                if key not in {"mode", "weight"}
            },
        )
        if left.get_extra_state()["basis_sha256"] != right.get_extra_state()["basis_sha256"]:
            raise ValueError("Los controles emparejados necesitan la misma base de proyección")
        control_receipt = dict(
            source_id=left.fingerprint(),
            target_id=right.fingerprint(),
            basis_sha256=left.get_extra_state()["basis_sha256"],
            basis_transferred=False,
            mac_sdpa_backend="math",
        )
    original, destination = dict(source.named_parameters()), dict(target.named_parameters())
    copied = sorted(original.keys() & destination.keys())
    for name in copied:
        if (
            original[name].shape != destination[name].shape
            or original[name].dtype != destination[name].dtype
        ):
            raise ValueError("Los parámetros emparejados tienen formas o precisiones diferentes")
        check_finite(original[name], "Los parámetros de origen")
    before = target._parameter_id
    with torch.no_grad():
        for name in copied:
            destination[name].copy_(original[name])
    transfers = {
        name: dict(
            source_sha256=_tensor_digest(original[name]),
            target_sha256=_tensor_digest(destination[name]),
            shape=list(destination[name].shape),
            dtype=str(destination[name].dtype),
        )
        for name in copied
    }
    if any(record["source_sha256"] != record["target_sha256"] for record in transfers.values()):
        raise ValueError("La copia emparejada no conserva los bytes de origen")
    target._seal_parameters()
    receipt = dict(
        schema_version=1,
        source_variant=source.config.variant,
        target_variant=target.config.variant,
        source_parameter_id=source._parameter_id,
        target_before=before,
        target_after=target._parameter_id,
        runtime_state_transferred=False,
        initialized_only=sorted(destination.keys() - original.keys()),
        copied_parameters=transfers,
    )
    if control_receipt is not None:
        receipt["local_control"] = control_receipt
    return receipt
