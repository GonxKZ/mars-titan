"""Crédito por activaciones y pullback local, sin backward ni optimizador."""

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from mars_titan.models.baselines.transformer import validate_attention_budget
from mars_titan.models.titans.config import canonical

from .config import DustConfig
from .financial import FinancialDustAdapter, finite, layer_norm_pullback, linear_pullback


def center_rewards(rewards):
    """Centrar la población completa y corregir que la media incluye cada draw."""
    if not isinstance(rewards, torch.Tensor) or rewards.ndim < 1 or len(rewards) < 2:
        raise ValueError("El centrado Dust necesita al menos dos draws")
    finite(rewards)
    population = len(rewards)
    return (rewards - rewards.mean(dim=0, keepdim=True)) * (population / (population - 1))


@dataclass(frozen=True)
class DustEstimate:
    """Resultado privado del cálculo. El llamante decide si aplica sus gradientes."""

    loss: torch.Tensor
    gradients: dict
    counters: dict


class DustEstimator:
    """Estimar todos los parámetros del adaptador con un generador independiente."""

    def __init__(self, adapter, config, *, loss="mae", huber_delta=0.01):
        if type(adapter) is not FinancialDustAdapter or type(config) is not DustConfig:
            raise ValueError("Dust necesita su adaptador y configuración explícitos")
        if (
            not isinstance(loss, str)
            or loss not in {"mae", "mse", "huber"}
            or type(huber_delta) not in (int, float)
            or not math.isfinite(huber_delta)
            or huber_delta <= 0
        ):
            raise ValueError("Dust solo admite pérdidas residuales escalares declaradas")
        self.adapter, self.config = adapter, config
        self.loss, self.huber_delta = loss, huber_delta
        self.generator = torch.Generator(device=adapter.device).manual_seed(config.seed)
        self.steps = 0
        self.totals = dict(
            full_forward_calls=0,
            local_attention_calls=0,
            head_loss_calls=0,
            full_forward_rows=0,
            logical_rows=0,
        )

    def identity(self):
        return dict(
            **self.config.identity(),
            adapter=self.adapter.identity(),
            loss=self.loss,
            reduction="mean_over_financial_rows_once_after_local_pullbacks",
            huber_delta=self.huber_delta,
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )

    def state_dict(self):
        """Guardar el generador privado solo en una barrera de estimación completa."""
        return dict(
            schema_version=1,
            identity=self.identity(),
            rng=self.generator.get_state().clone(),
            steps=self.steps,
            totals=dict(self.totals),
        )

    def load_state_dict(self, state):
        if (
            not isinstance(state, dict)
            or set(state) != {"schema_version", "identity", "rng", "steps", "totals"}
            or type(state["schema_version"]) is not int
            or state["schema_version"] != 1
            or canonical(state["identity"]) != canonical(self.identity())
            or type(state["steps"]) is not int
            or not 0 <= state["steps"] < 2**63
            or not isinstance(state["totals"], dict)
            or set(state["totals"]) != set(self.totals)
            or any(type(v) is not int or v < 0 for v in state["totals"].values())
        ):
            raise ValueError("El estado Dust no conserva su identidad y contadores")
        rng = state["rng"]
        if (
            not isinstance(rng, torch.Tensor)
            or rng.dtype != torch.uint8
            or rng.device.type != "cpu"
            or rng.ndim != 1
            or not 0 < rng.numel() <= 1024**2
        ):
            raise ValueError("El generador Dust guardado no tiene un estado válido")
        try:
            candidate = torch.Generator(device=self.adapter.device)
            candidate.set_state(rng)
        except (RuntimeError, TypeError) as error:
            raise ValueError("El estado aleatorio Dust no corresponde al dispositivo") from error
        self.generator.set_state(candidate.get_state())
        self.steps, self.totals = state["steps"], dict(state["totals"])

    def _per_row(self, prediction, target):
        if self.loss == "mae":
            result = (prediction - target).abs()
        elif self.loss == "mse":
            result = (prediction - target).square()
        else:
            result = F.huber_loss(prediction, target, reduction="none", delta=self.huber_delta)
        finite(result)
        return result

    def _budget(self, inputs, rows):
        config, adapter = self.config, self.adapter
        chunks = (config.population + config.chunk_size - 1) // config.chunk_size
        forwards = 2 + (6 + 7 * adapter.layers) * chunks
        physical = min(config.population, config.chunk_size) * rows
        if physical > adapter.max_batch:
            raise ValueError("El presupuesto Dust de réplicas supera el lote del predictor")
        validate_attention_budget(
            physical,
            context=adapter.context,
            heads=adapter.predictor.price_encoder._settings["heads"],
            layers=adapter.layers,
            max_batch=adapter.max_batch,
        )
        # Es una cota de tensores lógicos. No certifica los temporales internos de SDPA.
        h, t, layers, p, n = (
            adapter.hidden_size,
            adapter.context,
            adapter.layers,
            config.population,
            config.chunk_size,
        )
        heads = adapter.predictor.price_encoder._settings["heads"]
        elements = (
            64 * rows * t * h * layers
            + 8 * n * rows * t * h * layers
            + 8 * p * rows * t * h
            + 16 * n * rows * heads * t * t * layers
            + 4 * n * sum(value.numel() for value in inputs.values())
            + 4 * sum(value.numel() for value in adapter.parameters())
            + 8 * p * rows
        )
        workspace = elements * adapter.predictor.head.weight.element_size()
        if forwards > config.max_forward_calls or workspace > config.max_workspace_bytes:
            raise ValueError("El presupuesto de llamadas o memoria de Dust es insuficiente")
        return dict(
            full_forward_calls=0,
            full_forward_rows=0,
            local_attention_calls=0,
            head_loss_calls=0,
            workspace_bound_bytes=workspace,
            population=config.population,
            logical_rows=rows,
        ), forwards

    def _noise(self, output):
        return torch.randn(
            (self.config.population, *output.shape),
            dtype=output.dtype,
            device=output.device,
            generator=self.generator,
        )

    def _chunks(self):
        for start in range(0, self.config.population, self.config.chunk_size):
            end = min(start + self.config.chunk_size, self.config.population)
            yield start, end

    @staticmethod
    def _repeat(value, count):
        return value.repeat((count,) + (1,) * (value.ndim - 1))

    def _direct_error(self, name, output, clean, inputs, presence, target, counters):
        rows, population = len(target), self.config.population
        noise = self._noise(output)
        rewards = torch.empty((population, rows), dtype=output.dtype, device=output.device)
        for start, end in self._chunks():
            count = end - start
            repeated = {key: self._repeat(value, count) for key, value in inputs.items()}
            flags = None if presence is None else self._repeat(presence, count)
            jitter = self.config.sigma * noise[start:end].reshape(count * rows, *output.shape[1:])
            prediction = self.adapter.instrumented_forward(repeated, flags, {name: jitter})
            counters["full_forward_calls"] += 1
            counters["full_forward_rows"] += count * rows
            loss = self._per_row(prediction, self._repeat(target, count)).reshape(count, rows)
            rewards[start:end] = clean.unsqueeze(0) - loss
        scale = center_rewards(rewards).reshape((population, rows) + (1,) * (noise.ndim - 2))
        return -(scale * noise).sum(dim=0) / (population * self.config.sigma)

    def _head_error(self, output, clean, target, counters):
        noise = self._noise(output)
        rewards = torch.empty(
            (self.config.population, len(target)), dtype=output.dtype, device=output.device
        )
        for start, end in self._chunks():
            prediction = output.unsqueeze(0) + self.config.sigma * noise[start:end]
            loss = self._per_row(
                prediction.squeeze(-1), target.unsqueeze(0).expand(end - start, -1)
            )
            rewards[start:end] = clean.unsqueeze(0) - loss
            counters["head_loss_calls"] += 1
        return -(center_rewards(rewards).unsqueeze(-1) * noise).sum(dim=0) / (
            self.config.population * self.config.sigma
        )

    def _local_errors(self, attention, hub_error, counters):
        adapter, sigma = self.adapter, self.config.sigma
        heads = adapter.predictor.price_encoder._settings["heads"]
        rows, queries, width = (
            attention.q.shape[0],
            attention.q.shape[1],
            adapter.hidden_size // heads,
        )
        baseline = adapter._attention_output(
            attention.q, attention.k, attention.v, last_only=attention.last_only
        )
        counters["local_attention_calls"] += 1
        length = attention.k.shape[1]
        positions = torch.arange(length, device=adapter.device)
        queries_at = positions[-1:] if attention.last_only else positions
        credit = (queries_at[None, :] >= positions[:, None]).to(dtype=adapter.dtype)
        target = hub_error.reshape(rows, queries, heads, width)
        result = {}
        for kind in ("q", "k", "v"):
            original = getattr(attention, kind)
            noise = self._noise(original)
            error = torch.zeros_like(original)
            for start, end in self._chunks():
                count = end - start
                parts = {
                    key: self._repeat(getattr(attention, key), count) for key in ("q", "k", "v")
                }
                parts[kind] = parts[kind] + sigma * noise[start:end].reshape(
                    count * rows, *original.shape[1:]
                )
                changed = adapter._attention_output(**parts, last_only=attention.last_only)
                delta = changed.reshape(count, rows, queries, heads, width) - baseline.reshape(
                    1, rows, queries, heads, width
                )
                finite(*parts.values(), delta)
                score = (target.unsqueeze(0) * delta).sum(dim=-1)
                if kind != "q":
                    score = torch.einsum("nbsh,ts->nbth", score, credit)
                estimate = score.unsqueeze(-1) * noise[start:end].reshape(
                    count, rows, original.shape[1], heads, width
                )
                error += estimate.sum(dim=0).reshape_as(original) / (self.config.population * sigma)
                counters["local_attention_calls"] += 1
            result[attention.prefix + "." + kind] = error
        return result

    @torch.no_grad()
    def estimate(self, inputs, presence, target):
        """Devolver estimaciones completas sin modificar pesos, gradientes ni estado emisor."""
        adapter = self.adapter
        rows = adapter.validate_inputs(inputs, presence)
        if (
            not isinstance(target, torch.Tensor)
            or target.shape != (rows,)
            or target.dtype != adapter.dtype
            or target.device != adapter.device
        ):
            raise ValueError("La etiqueta residual Dust no conserva filas, precisión y dispositivo")
        finite(target)
        counters, expected_calls = self._budget(inputs, rows)
        adapter.validate_model()
        before = self.generator.get_state().clone()
        try:
            reference = adapter.forward(inputs, presence)
            clean, sites, attentions = adapter._instrumented(inputs, presence, None, capture=True)
            tolerance = 2e-6 if adapter.dtype == torch.float32 else 2e-12
            if not torch.allclose(clean, reference, rtol=tolerance, atol=tolerance):
                raise ValueError("El núcleo instrumental Dust no conserva la ruta limpia")
            counters["full_forward_calls"], counters["full_forward_rows"] = 2, 2 * rows
            loss = self._per_row(reference, target)
            errors = {}
            for name, site in sites.items():
                if site.kind == "packed":
                    continue
                errors[name] = (
                    self._head_error(site.output, loss, target, counters)
                    if name == "head"
                    else self._direct_error(
                        name, site.output, loss, inputs, presence, target, counters
                    )
                )
            for attention in attentions:
                name = attention.prefix + ".attention_output"
                hub = self._direct_error(
                    name, attention.output, loss, inputs, presence, target, counters
                )
                errors.update(self._local_errors(attention, hub, counters))
            gradients = {
                name: torch.zeros_like(value) for name, value in adapter.named_parameters()
            }
            covered = set()
            for name, site in sites.items():
                pullback = layer_norm_pullback if site.kind == "norm" else linear_pullback
                weight, bias = pullback(site.inputs, errors[name])
                if site.part is None:
                    gradients[site.weight] += weight / rows
                    gradients[site.bias] += bias / rows
                else:
                    start = site.part * adapter.hidden_size
                    end = start + adapter.hidden_size
                    gradients[site.weight][start:end] += weight / rows
                    gradients[site.bias][start:end] += bias / rows
                covered.update((site.weight, site.bias))
            if covered != set(gradients) or counters["full_forward_calls"] != expected_calls:
                raise ValueError("Dust no completó su cobertura o su contabilidad de llamadas")
            finite(*gradients.values())
            adapter.validate_model()
        except BaseException:
            self.generator.set_state(before)
            raise
        self.steps += 1
        for name in self.totals:
            self.totals[name] += counters[name]
        return DustEstimate(loss.mean().detach(), gradients, counters)
