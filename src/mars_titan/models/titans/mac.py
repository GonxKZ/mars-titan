"""MAC con salida al cierre del segmento y estado explícito por flujo."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import MACConfig, require_identity
from .neural_memory import NeuralMemory
from .state import MACState, check_differentiable, check_finite, copy_state, require_payload


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
        self.to(device=device)

    def initial_state(self, batch_size: int, *, differentiable: bool = False) -> MACState:
        return MACState(
            self.memory.initial_state(batch_size, differentiable=differentiable),
            self.config.fingerprint(),
        )

    def _validate_state(self, state: MACState) -> None:
        if not isinstance(state, MACState) or state.config_id != self.config.fingerprint():
            raise ValueError("El estado MAC pertenece a otro contrato de configuración")
        self.memory.validate_state(state.memory)

    def forward(
        self, segment: Tensor, state: MACState, *, differentiable: bool = False
    ) -> tuple[Tensor, MACState]:
        check_differentiable(differentiable)
        self._validate_state(state)
        self.memory.validate_input(segment, state.memory)
        batch, length, _ = segment.shape
        if length > self.config.max_segment:
            raise ValueError("El segmento supera el presupuesto de tokens de MAC")
        enabled = self.config.memory_mode != "disabled"
        packed_length = 2 * length + self.config.persistent_tokens if enabled else length
        if batch * self.config.heads * packed_length**2 > self.config.max_attention_elements:
            raise ValueError("La atención supera el presupuesto de elementos")
        with torch.set_grad_enabled(differentiable):
            if enabled:
                queries = self.query_projection(segment)
                if self.config.memory.normalize_qk:
                    queries = F.normalize(queries, dim=-1, eps=1e-12)
                retrieved = self.memory.read(queries, state.memory)
                prefix = self.persistent.unsqueeze(0).expand(batch, -1, -1)
                packed = torch.cat((prefix, retrieved, segment), dim=1)
            else:
                packed = segment
            check_finite(packed, "La entrada de atención")
            mask = torch.ones(
                packed_length, packed_length, dtype=torch.bool, device=segment.device
            ).triu(1)
            attended, _ = self.attention(packed, packed, packed, attn_mask=mask, need_weights=False)
            observed = attended[:, -length:]
            check_finite(observed, "La salida de atención")
            if self.config.memory_mode == "online":
                next_memory = self.memory.update(
                    observed, state.memory, differentiable=differentiable
                )
            else:
                next_memory = copy_state(state.memory, differentiable=differentiable)
            output = observed[:, -1]
            if enabled:
                output = output * self.memory.read(output.unsqueeze(1), next_memory).squeeze(1)
            else:
                output = output.clone()
            check_finite(output, "La salida de MAC")
        if not differentiable:
            output = output.detach()
        return output, MACState(next_memory, state.config_id)

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
        return {
            "schema_version": 1,
            "configuration": self.config.identity(),
            "memory": self.memory.export_state(state.memory),
        }

    def restore_state(self, payload: object) -> MACState:
        value = require_payload(payload, {"schema_version", "configuration", "memory"})
        require_identity(value["configuration"], self.config.identity())
        return MACState(self.memory.restore_state(value["memory"]), self.config.fingerprint())
