"""MAC con salida al cierre del segmento y estado explícito por flujo."""

from dataclasses import replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .causal_convolution import CausalDepthwiseConvolution
from .config import RAW_OUTPUT_QUERY, MACConfig, require_identity
from .differential_attention import DifferentialMap
from .key_competition import competitive_normalize
from .neural_memory import NeuralMemory
from .state import (
    MACState,
    check_differentiable,
    check_finite,
    copy_state,
    has_deferred_checks,
    require_payload,
)


class TitansMAC(nn.Module):
    """Atiende [P, h, S] y escribe únicamente las posiciones correspondientes a S."""

    def __init__(
        self,
        config: MACConfig,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if not isinstance(config, MACConfig):
            raise ValueError("Se necesita MACConfig")
        self.config = config
        self.memory = NeuralMemory(config.memory, dtype=dtype)
        dim = config.memory.dim
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.parameter_seed)
            self.query_projection = nn.Linear(dim, dim, bias=False, dtype=dtype)
            self.attention = nn.MultiheadAttention(
                dim, config.heads, dropout=0.0, bias=False, batch_first=True, dtype=dtype
            )
            self.persistent = nn.Parameter(torch.empty(config.persistent_tokens, dim, dtype=dtype))
            nn.init.normal_(self.persistent, std=dim**-0.5)
            if config.memory.qkv_convolution:
                # Se sortea después de los parámetros anteriores para que estos conserven sus
                # valores con la misma semilla.
                self.query_convolution = CausalDepthwiseConvolution(
                    dim, config.memory.qkv_convolution, dtype=dtype
                )
        # U1 de #786 sortea sus proyecciones con una semilla derivada, fuera del generador global.
        self.differential = (
            DifferentialMap(dim, config.heads, seed=config.parameter_seed, dtype=dtype)
            if config.differential_attention is not None
            else None
        )
        self.to(device=device)

    def initial_state(self, batch_size: int, *, differentiable: bool = False) -> MACState:
        return MACState(
            self.memory.initial_state(batch_size, differentiable=differentiable),
            self.config.fingerprint(),
            self.memory.initial_windows(batch_size, 1),
        )

    def _validate_state(
        self, state: MACState, *, defer_slots: list[int | None] | None = None
    ) -> None:
        if not isinstance(state, MACState) or state.config_id != self.config.fingerprint():
            raise ValueError("El estado MAC pertenece a otro contrato de configuración")
        self.memory.validate_state(state.memory, defer_slots=defer_slots)
        self.validate_query_window(state)

    def validate_query_window(self, state: MACState, *, device=None) -> None:
        """La ventana de q no comparte almacenamiento con el estado de la memoria."""
        memory = state.memory
        storage_ids = {
            (value.device, value.untyped_storage().data_ptr())
            for value in (
                *memory.weights,
                *memory.momentum,
                memory.steps,
                *memory.convolution,
                *memory.levels,
                *memory.modulation,
            )
        }
        self.memory.validate_windows(
            state.convolution,
            1,
            state.memory.steps.shape[0],
            device=device,
            storage_ids=storage_ids,
        )

    def _output_query(self, output: Tensor) -> Tensor:
        """Consulta de la puerta de salida: cruda o proyectada y normalizada L2."""
        if self.config.output_query == RAW_OUTPUT_QUERY:
            return output
        projected = self.query_projection(output)
        check_finite(projected, "La consulta de salida")
        return F.normalize(projected, dim=-1, eps=1e-12)

    def forward(
        self,
        segment: Tensor,
        state: MACState,
        *,
        differentiable: bool = False,
        gate_shift: dict | None = None,
        surprise: list | None = None,
        input_slots: list[int | None] | None = None,
        # --- memory_cross_sectional_surprise ---
        session_groups=None,
        # --- fin memory_cross_sectional_surprise ---
    ) -> tuple[Tensor, MACState]:
        """`gate_shift` y `surprise` llegan a la escritura de la memoria, que solo existe online."""
        check_differentiable(differentiable)
        if (gate_shift is not None or surprise is not None) and self.config.memory_mode != "online":
            raise ValueError("Las puertas moduladas necesitan la memoria online")
        # --- memory_cross_sectional_surprise ---
        if session_groups is not None and self.config.memory_mode != "online":
            raise ValueError("Las puertas moduladas necesitan la memoria online")
        if session_groups is not None and self.memory.cross_sectional_surprise is None:
            raise ValueError("session_groups requiere cross_sectional_surprise declarado")
        # --- fin memory_cross_sectional_surprise ---
        can_fuse_write_read = (
            self.config.memory_mode == "online"
            and self.config.memory_mode != "disabled"
            and isinstance(segment, Tensor)
            and segment.ndim == 3
            and segment.shape[1] == 1
            and segment.is_cuda
            and self.memory.supports_fused_write_and_read()
        )
        if input_slots is None and can_fuse_write_read and has_deferred_checks():
            input_slots = []
            self._validate_state(state, defer_slots=input_slots)
        else:
            self._validate_state(state)
        self.memory.validate_input(segment, state.memory)
        batch, length, _ = segment.shape
        if length > self.config.max_segment:
            raise ValueError("El segmento supera el presupuesto de tokens de MAC")
        enabled = self.config.memory_mode != "disabled"
        packed_length = 2 * length + self.config.persistent_tokens if enabled else length
        elements = batch * self.config.heads * packed_length**2
        if self.differential is not None:
            # El segundo mapa de U1 solo tiene las filas de las consultas del segmento.
            elements += batch * self.config.heads * length * packed_length
        if elements > self.config.max_attention_elements:
            raise ValueError("La atención supera el presupuesto de elementos")
        windows = state.convolution
        with torch.set_grad_enabled(differentiable):
            use_fused_attn = (
                enabled
                and length == 1
                and self.differential is None
                and segment.is_cuda
                and segment.dtype in (torch.float32, torch.float64)
                and not torch.is_autocast_enabled("cuda")
                and self.config.memory.dim in (32, 64, 128)
                and self.config.heads in (1, 2, 4, 8)
                and (self.config.memory.dim // self.config.heads) in (8, 16, 32, 64, 128)
            )
            if enabled:
                queries, window = self.memory.project(
                    self.query_projection(segment),
                    getattr(self, "query_convolution", None),
                    windows[0] if windows else None,
                )
                windows = () if window is None else (window,)
                if self.config.memory.key_competition is not None:
                    queries = competitive_normalize(queries, self.config.memory.key_competition)
                elif self.config.memory.normalize_qk:
                    queries = F.normalize(queries, dim=-1, eps=1e-12)
                retrieved = self.memory.read(queries, state.memory)
                if not use_fused_attn:
                    prefix = self.persistent.unsqueeze(0).expand(batch, -1, -1)
                    packed = torch.cat((prefix, retrieved, segment), dim=1)
            else:
                packed = segment
            if use_fused_attn:
                from .fused_memory_write import fused_mac_last_token_attention

                check_finite(self.persistent, "La entrada de atención")
                check_finite(retrieved, "La entrada de atención")
                check_finite(segment, "La entrada de atención")
                dim = self.config.memory.dim
                in_weight = self.attention.in_proj_weight
                kv_weight = in_weight[dim:]
                kv_prefix = F.linear(self.persistent, kv_weight)
                kv_ret = F.linear(retrieved.squeeze(1), kv_weight)
                qkv_seg = F.linear(segment.squeeze(1), in_weight)
                mixed = fused_mac_last_token_attention(
                    qkv_seg[:, :dim], kv_prefix, kv_ret, qkv_seg[:, dim:], self.config.heads
                )
                observed = self.attention.out_proj(mixed).unsqueeze(1)
            else:
                check_finite(packed, "La entrada de atención")
                mask = torch.ones(
                    packed_length, packed_length, dtype=torch.bool, device=segment.device
                ).triu(1)
                if self.config.attention_layout == "sequence_contiguous_v1":
                    # MultiheadAttention transpone antes de QKV. Hacer contigua esa vista evita
                    # que matmul elija mm o bmm según requires_grad en la proyección sin sesgo.
                    packed = packed.transpose(0, 1).contiguous().transpose(0, 1)
                attended, _ = self.attention(
                    packed, packed, packed, attn_mask=mask, need_weights=False
                )
                observed = attended[:, -length:]
                if self.differential is not None:
                    # U1 resta el segundo mapa con los valores de la MHA. Con λ nulo resta cero.
                    values_weight = self.attention.in_proj_weight[2 * self.config.memory.dim :]
                    observed = observed - self.differential(
                        packed, values_weight, self.attention.out_proj, mask, length
                    )
            check_finite(observed, "La salida de atención")
            output = observed[:, -1]
            post_query = self._output_query(output) if enabled else output
            if (
                self.config.memory_mode == "online"
                and enabled
                and length == 1
                and segment.is_cuda
                and self.memory.supports_fused_write_and_read()
            ):
                next_memory, read_after = self.memory.update_and_read(
                    observed,
                    state.memory,
                    post_query,
                    differentiable=differentiable,
                    gate_shift=gate_shift,
                    surprise=surprise,
                    input_slots=input_slots,
                )
                output = output * read_after
            else:
                if self.config.memory_mode == "online":
                    next_memory = self.memory.update(
                        observed,
                        state.memory,
                        differentiable=differentiable,
                        gate_shift=gate_shift,
                        surprise=surprise,
                        # --- memory_cross_sectional_surprise ---
                        session_groups=session_groups,
                        # --- fin memory_cross_sectional_surprise ---
                    )
                else:
                    next_memory = copy_state(state.memory, differentiable=differentiable)
                if enabled:
                    output = output * self.memory.read(
                        post_query.unsqueeze(1), next_memory
                    ).squeeze(1)
                else:
                    output = output.clone()
                    # Sin memoria no se calculan consultas y la ventana conserva su valor.
                    windows = tuple(window.clone() for window in windows)
            check_finite(output, "La salida de MAC")
        if not differentiable:
            output = output.detach()
        return output, MACState(next_memory, state.config_id, windows)

    def get_extra_state(self) -> dict:
        return {"configuration": self.config.identity(), "dtype": str(self.persistent.dtype)}

    def set_extra_state(self, state: object) -> None:
        require_identity(state, self.get_extra_state())

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        require_identity(state_dict.get(prefix + "_extra_state"), self.get_extra_state())
        require_identity(
            state_dict.get(prefix + "memory._extra_state"), self.memory.get_extra_state()
        )
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def export_state(self, state: MACState) -> dict:
        self._validate_state(state)
        payload = {
            "schema_version": 1,
            "configuration": self.config.identity(),
            "memory": self.memory.export_state(state.memory),
        }
        if self.config.memory.window:
            payload["convolution"] = tuple(window.detach().clone() for window in state.convolution)
        return payload

    def payload_fields(self) -> set[str]:
        fields = {"schema_version", "configuration", "memory"}
        return fields | {"convolution"} if self.config.memory.window else fields

    def restore_state(self, payload: object) -> MACState:
        value = require_payload(payload, self.payload_fields())
        require_identity(value["configuration"], self.config.identity())
        state = MACState(
            self.memory.restore_state(value["memory"]),
            self.config.fingerprint(),
            value.get("convolution", ()),
        )
        self.validate_query_window(state)
        return replace(state, convolution=tuple(w.detach().clone() for w in state.convolution))
