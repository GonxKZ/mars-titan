"""Contexto compartido entre los flujos de un mercado, escrito una vez por sesión (PT4).

El núcleo trata cada activo como un flujo independiente. PT4 añade un contexto común a todos
los flujos de un mercado con dos formas:

- `market_memory_v1`, la propuesta: una segunda memoria neuronal de Titans por mercado. Cada
  flujo la lee con su propia consulta y la escritura llega una sola vez por sesión, después
  de emitir todas sus predicciones, con la media de los gradientes internos de los tokens de
  esa sesión (`NeuralMemory.update_shared`).
- `sector_mean_v1`, el control obligatorio: la media de los tokens fusionados de la sesión
  anterior de los flujos de su mismo sector, sin memoria ni olvido aprendido.

En las dos, la lectura de la sesión t usa solo lo confirmado al cerrar sesiones anteriores,
así que ninguna predicción depende de otra fila de su misma sesión. Los tokens de la sesión se
guardan pendientes hasta `commit`, que los ordena por flujo antes de escribir: la partición en
bloques físicos no cambia la escritura. El contexto entra en el token con una proyección que
empieza en cero, de modo que con la misma semilla el predictor arranca igual que su control.
"""

import hashlib
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import MemoryConfig, canonical
from .key_competition import competitive_normalize
from .neural_memory import MAX_SHARED_ROWS, NeuralMemory
from .state import NeuralMemoryState, check_finite, map_memory_rows

MARKET_MEMORY = "market_memory_v1"
SECTOR_MEAN = "sector_mean_v1"
SESSION_CONTEXTS = (MARKET_MEMORY, SECTOR_MEAN)
MARKETS = ("US", "CN")


@dataclass(frozen=True)
class SessionContextState:
    """Estado del contexto: lo confirmado por mercado y los tokens pendientes de su sesión."""

    config_id: str
    parameter_id: str
    committed_at: tuple[int, ...]
    pending_at: tuple[int, ...]
    pending_flows: tuple[tuple[str, ...], ...]
    pending_tokens: tuple[Tensor | None, ...]
    # Memoria de mercado: un estado de lote 1 por mercado. Control sectorial: medias [G, H].
    memory: tuple[NeuralMemoryState, ...] | None = None
    means: Tensor | None = None


def _moved(value, device):
    """El mismo payload con sus tensores copiados al dispositivo pedido."""
    if isinstance(value, Tensor):
        return value.detach().to(device=device, copy=True)
    if isinstance(value, tuple):
        return tuple(_moved(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _moved(item, device) for key, item in value.items()}
    return value


def _seed(seed, name):
    digest = hashlib.sha256(f"{name}/{seed}".encode()).digest()
    return int.from_bytes(digest[:7], "little")


class SessionContext(nn.Module):
    """Lectura, acumulación por sesión y escritura confirmada del contexto de PT4."""

    def __init__(self, kind, hidden_size, *, seed, memory_options=None, sector_map=None):
        super().__init__()
        if kind not in SESSION_CONTEXTS:
            raise ValueError("El contexto de sesión no pertenece a PT4")
        if (kind == SECTOR_MEAN) != (sector_map is not None):
            raise ValueError("Solo el control sectorial necesita el reparto de sectores")
        self.kind, self.hidden_size, self.sector_map = kind, hidden_size, sector_map
        if kind == MARKET_MEMORY:
            options = dict(memory_options or {})
            self.memory = NeuralMemory(
                MemoryConfig(
                    dim=hidden_size,
                    depth=2,
                    max_batch=1,
                    max_tokens=1,
                    parameter_seed=_seed(seed, MARKET_MEMORY),
                    **options,
                )
            )
            # La consulta se sortea con un generador propio y no toca el generador global.
            generator = torch.Generator(device="cpu").manual_seed(_seed(seed, "market_query"))
            self.query = nn.Parameter(torch.empty(hidden_size, hidden_size))
            nn.init.kaiming_uniform_(self.query, a=5**0.5, generator=generator)
        # La lectura entra en el token con una ganancia que empieza en cero.
        self.output = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        # La identidad no cambia después de construir y se valida en cada bloque.
        self._fingerprint = hashlib.sha256(canonical(self.identity()).encode()).hexdigest()

    def identity(self):
        result = dict(kind=self.kind, read="previous_sessions_only", output_initial="zero")
        if self.kind == MARKET_MEMORY:
            result.update(
                memory=self.memory.config.identity(),
                write="once_per_session_after_emission_mean_inner_gradient",
                gates="mean_of_session_logits",
                query="own_linear_then_l2",
            )
        else:
            result.update(
                sectors=self.sector_map.identity(),
                write="mean_token_of_the_sector_in_its_last_session",
            )
        return result

    def fingerprint(self):
        return self._fingerprint

    def initial_state(self, parameter_id, *, differentiable=False):
        reference = self.output
        memory, means = None, None
        if self.kind == MARKET_MEMORY:
            memory = tuple(
                self.memory.initial_state(1, differentiable=differentiable) for _ in MARKETS
            )
        else:
            means = torch.zeros(
                len(self.sector_map.groups),
                self.hidden_size,
                dtype=reference.dtype,
                device=reference.device,
            )
        empty = (-1,) * len(MARKETS)
        return SessionContextState(
            self.fingerprint(),
            parameter_id,
            empty,
            empty,
            ((),) * len(MARKETS),
            (None,) * len(MARKETS),
            memory,
            means,
        )

    def validate(self, state, parameter_id):
        if (
            not isinstance(state, SessionContextState)
            or state.config_id != self.fingerprint()
            or state.parameter_id != parameter_id
            or len(state.committed_at) != len(MARKETS)
            or len(state.pending_at) != len(MARKETS)
            or (state.memory is None) != (self.kind != MARKET_MEMORY)
            or (state.means is None) != (self.kind != SECTOR_MEAN)
        ):
            raise ValueError("El contexto de sesión no corresponde al predictor")
        if state.memory is not None:
            for value in state.memory:
                self.memory.validate_state(value)
        if state.means is not None:
            check_finite(state.means, "Las medias sectoriales")

    def _market(self, flow):
        return MARKETS.index(flow.split("/", 1)[0])

    def _session_rows(self, tokens, flow_ids, prediction_at, state):
        if (
            not isinstance(tokens, Tensor)
            or tokens.layout != torch.strided
            or tokens.ndim != 2
            or tokens.shape != (len(flow_ids), self.hidden_size)
            or len(flow_ids) != len(prediction_at)
            or not flow_ids
            or tokens.dtype != self.output.dtype
            or tokens.device != self.output.device
        ):
            raise ValueError("Los tokens y las filas del contexto no corresponden al módulo")
        check_finite(tokens, "Los tokens del contexto")
        markets = [self._market(flow) for flow in flow_ids]
        for market in sorted(set(markets)):
            rows = [i for i, value in enumerate(markets) if value == market]
            moments = {prediction_at[i] for i in rows}
            if len(moments) != 1:
                raise ValueError("Las filas de un mercado en un bloque deben ser de una sesión")
            (moment,) = moments
            if moment <= state.committed_at[market]:
                raise ValueError("La sesión ya se confirmó: no puede leer ni escribir de nuevo")
            if state.pending_at[market] not in (-1, moment):
                raise ValueError("La sesión anterior sigue pendiente de confirmar")
            names = tuple(flow_ids[i] for i in rows)
            yield market, rows, moment, names

    def read_confirmed(self, tokens, flow_ids, prediction_at, state):
        """Leer el snapshot confirmado sin preparar tokens ni consumir flujos pendientes."""
        sessions = tuple(self._session_rows(tokens, flow_ids, prediction_at, state))
        context = torch.zeros_like(tokens)
        for market, rows, _, names in sessions:
            index = torch.tensor(rows, device=tokens.device)
            selected = tokens.index_select(0, index)
            context = context.index_copy(0, index, self._read(selected, names, market, state))
        return context @ self.output.T

    def stage_tokens(self, tokens, flow_ids, prediction_at, state):
        """Preparar una sola alta por flujo y sesión, sin leer ni confirmar la memoria."""
        pending_at = list(state.pending_at)
        flows, pending = list(state.pending_flows), list(state.pending_tokens)
        for market, rows, moment, names in self._session_rows(
            tokens, flow_ids, prediction_at, state
        ):
            if len(set(names) | set(flows[market])) != len(names) + len(flows[market]):
                raise ValueError("Un flujo no puede aparecer dos veces en la misma sesión")
            index = torch.tensor(rows, device=tokens.device)
            selected = tokens.index_select(0, index)
            pending_at[market] = moment
            flows[market] = flows[market] + names
            pending[market] = (
                selected if pending[market] is None else torch.cat((pending[market], selected))
            )
        return replace(
            state,
            pending_at=tuple(pending_at),
            pending_flows=tuple(flows),
            pending_tokens=tuple(pending),
        )

    def read(self, tokens, flow_ids, prediction_at, state):
        """Leer el contexto y preparar sus tokens, conservando el contrato anterior."""
        pending = self.stage_tokens(tokens, flow_ids, prediction_at, state)
        return self.read_confirmed(tokens, flow_ids, prediction_at, state), pending

    def _read(self, tokens, flows, market, state):
        if self.kind == MARKET_MEMORY:
            projected = tokens @ self.query.T
            query = (
                F.normalize(projected, dim=-1, eps=1e-12)
                if self.memory.config.key_competition is None
                else competitive_normalize(projected, self.memory.config.key_competition)
            )
            return self.memory.read_shared(query, state.memory[market])
        groups = torch.tensor([self.sector_map.group(flow) for flow in flows], device=tokens.device)
        return state.means.index_select(0, groups)

    def commit(self, state, *, differentiable=False):
        """Escribir una vez cada mercado con sesión pendiente, en orden canónico de flujos."""
        memory = None if state.memory is None else list(state.memory)
        means = state.means
        committed = list(state.committed_at)
        for market, tokens in enumerate(state.pending_tokens):
            if tokens is None:
                continue
            flows = state.pending_flows[market]
            order = sorted(range(len(flows)), key=flows.__getitem__)
            if len(order) > MAX_SHARED_ROWS:
                raise ValueError("La sesión supera las filas admitidas por la escritura")
            ordered = tokens.index_select(0, torch.tensor(order, device=tokens.device))
            if not differentiable:
                ordered = ordered.detach()
            if self.kind == MARKET_MEMORY:
                memory[market] = self.memory.update_shared(
                    ordered, memory[market], differentiable=differentiable
                )
            else:
                means = self._sector_means(ordered, [flows[i] for i in order], means)
            committed[market] = state.pending_at[market]
        empty = (-1,) * len(MARKETS)
        return replace(
            state,
            committed_at=tuple(committed),
            pending_at=empty,
            pending_flows=((),) * len(MARKETS),
            pending_tokens=(None,) * len(MARKETS),
            memory=None if memory is None else tuple(memory),
            means=means,
        )

    def _sector_means(self, tokens, flows, means):
        """Media de cada sector con miembros en la sesión. Los demás conservan su valor."""
        groups = torch.tensor([self.sector_map.group(flow) for flow in flows], device=tokens.device)
        present = torch.unique(groups)
        sums = torch.zeros_like(means).index_add(0, groups, tokens)
        counts = torch.zeros(len(means), dtype=tokens.dtype, device=tokens.device).index_add(
            0, groups, torch.ones_like(groups, dtype=tokens.dtype)
        )
        updated = sums.index_select(0, present) / counts.index_select(0, present).unsqueeze(-1)
        return means.index_copy(0, present, updated)

    def detach(self, state, parameter_id):
        """Cortar el grafo del contexto en una barrera y confirmar los parámetros nuevos."""
        if any(tokens is not None for tokens in state.pending_tokens):
            raise ValueError("Solo se corta el grafo con todas las sesiones confirmadas")
        memory = state.memory
        if memory is not None:
            memory = tuple(map_memory_rows(value, lambda t: t.detach()) for value in memory)
        means = None if state.means is None else state.means.detach()
        return replace(state, parameter_id=parameter_id, memory=memory, means=means)

    def export_state(self, state):
        """Copias CPU sin grafo, solo con todas las sesiones confirmadas."""
        state = self.detach(state, state.parameter_id)
        return dict(
            schema_version=1,
            configuration=self.identity(),
            committed_at=list(state.committed_at),
            memory=None
            if state.memory is None
            else [_moved(self.memory.export_state(value), "cpu") for value in state.memory],
            means=None if state.means is None else state.means.detach().cpu().clone(),
        )

    def restore_state(self, payload, parameter_id, *, device):
        if (
            not isinstance(payload, dict)
            or set(payload)
            != {"schema_version", "configuration", "committed_at", "memory", "means"}
            or payload["schema_version"] != 1
            or canonical(payload["configuration"]) != canonical(self.identity())
        ):
            raise ValueError("El contexto guardado no corresponde al predictor")
        empty = (-1,) * len(MARKETS)
        memory = None
        if self.kind == MARKET_MEMORY:
            memory = tuple(
                self.memory.restore_state(_moved(value, device)) for value in payload["memory"]
            )
        means = None if payload["means"] is None else payload["means"].to(device).clone()
        state = SessionContextState(
            self.fingerprint(),
            parameter_id,
            tuple(int(value) for value in payload["committed_at"]),
            empty,
            ((),) * len(MARKETS),
            (None,) * len(MARKETS),
            memory,
            means,
        )
        self.validate(state, parameter_id)
        return state
