"""Precios y predicciones inmutables con una admisión explícita de su procedencia."""

import hashlib
import json
import re
from dataclasses import asdict

import numpy as np

from mars_titan.environments.cohorts import FINAL_TEST_START_US, VALIDATION_START_US

from .portfolio import CorporateAction, Quote


def _times(values):
    result = np.asarray(values)
    if (
        result.ndim != 1
        or not len(result)
        or result.dtype.kind not in "iu"
        or int(result.min()) < 0
        or int(result.max()) >= 2**63
    ):
        raise ValueError("Los tiempos deben ser enteros UTC no negativos de 64 bits")
    return result.astype(np.int64, copy=True)


class MarketTape:
    def __init__(
        self,
        prices,
        close_times,
        assets,
        scores,
        *,
        domain,
        currency,
        partition="train",
        prediction_times=None,
        audit=None,
        actions=(),
        parent_id="fixed",
        open_times=None,
        source_identity=None,
    ):
        if domain not in {"synthetic", "real"} or partition not in {"train", "validation"}:
            raise ValueError("La simulación necesita origen y partición de desarrollo")
        if domain == "real" and (
            not isinstance(audit, dict)
            or audit.get("price_basis") != "unadjusted"
            or audit.get("corporate_actions_complete") is not True
            or not re.fullmatch(r"[a-f0-9]{64}", str(audit.get("evidence_sha256", "")))
        ):
            raise ValueError("La procedencia de los ajustes OHLC y acciones no está acreditada")
        if domain == "real" and open_times is None:
            raise ValueError(
                "La simulación histórica necesita aperturas de su calendario acreditado"
            )
        if domain == "real" and prediction_times is None:
            raise ValueError(
                "Las predicciones históricas necesitan tiempos de disponibilidad explícitos"
            )
        prices, scores, times = np.asarray(prices), np.asarray(scores), _times(close_times)
        if (
            prices.ndim != 3
            or prices.shape[2] != 5
            or prices.shape[0] < 2
            or not 1 <= prices.shape[1] <= 4096
            or prices.shape[0] * prices.shape[1] > 1_048_576
            or len(assets) != prices.shape[1]
            or len(set(assets)) != len(assets)
            or any(not isinstance(asset, str) or not 1 <= len(asset) <= 96 for asset in assets)
            or not isinstance(currency, str)
            or not re.fullmatch(r"[A-Z]{3}", currency)
            or scores.shape != prices.shape[:2]
            or times.shape != (len(prices),)
            or times.dtype.kind not in "iu"
            or (np.diff(times) <= 0).any()
            or np.isinf(prices).any()
            or np.isinf(scores).any()
            or (prices[:, :, :4][np.isfinite(prices[:, :, :4])] <= 0).any()
            or (prices[:, :, 4][np.isfinite(prices[:, :, 4])] < 0).any()
        ):
            raise ValueError("Los precios, predicciones o tiempos no cumplen el contrato")
        low, high = (
            (0, VALIDATION_START_US)
            if partition == "train"
            else (VALIDATION_START_US, FINAL_TEST_START_US)
        )
        if domain == "real" and not (times[0] >= low and times[-1] < high):
            raise ValueError("La simulación histórica cruza su partición o el test sellado")
        order = np.argsort(assets)
        self.assets = [assets[i] for i in order]
        self.prices = np.array(prices[:, order], dtype=np.float64, copy=True)
        self.scores = np.array(scores[:, order], dtype=np.float64, copy=True)
        self.close_times = times.astype(np.int64, copy=True)
        self.prediction_times = _times(times if prediction_times is None else prediction_times)
        if self.prediction_times.shape != times.shape or (self.prediction_times > times).any():
            raise ValueError("Las predicciones contienen información posterior al cierre")
        self.open_times = (
            self.close_times - min(23_400_000_000, int(np.diff(times).min()) // 2)
            if open_times is None
            else _times(open_times)
        )
        if (
            self.open_times.shape != times.shape
            or (self.open_times >= self.close_times).any()
            or (self.open_times[1:] <= self.close_times[:-1]).any()
        ):
            raise ValueError("El calendario no separa decisiones y ejecuciones")
        if self.open_times[0] < 0:
            raise ValueError("El calendario de apertura necesita instantes no negativos")
        self.domain, self.currency, self.partition = domain, currency, partition
        self.actions = tuple(actions)
        for action in self.actions:
            if not isinstance(action, CorporateAction):
                raise ValueError("La acción corporativa no tiene un contrato válido")
            action.validate()
            if action.asset not in self.assets or action.effective_at not in self.open_times:
                raise ValueError("La acción corporativa no pertenece al calendario")
        self.identity = dict(
            domain=domain,
            currency=currency,
            partition=partition,
            assets=self.assets,
            parent_id=parent_id,
            audit=audit,
            actions=[vars(action) for action in self.actions],
            source=source_identity,
        )
        digest = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode())
        for value in (
            self.prices,
            self.scores,
            self.close_times,
            self.open_times,
            self.prediction_times,
        ):
            digest.update(value.tobytes())
            value.setflags(write=False)
        self.sha256 = digest.hexdigest()

    def __len__(self):
        return len(self.close_times)

    def quotes(self, position):
        result = {}
        for index, asset in enumerate(self.assets):
            row = self.prices[position, index]
            values = [None if np.isnan(row[i]) else float(row[i]) for i in (0, 3, 4)]
            result[asset] = Quote(*values)
        return result

    @classmethod
    def from_world(
        cls, world, predict, *, parent_id="fixed_signal_reference", check_resources=None
    ):
        if check_resources is not None and not callable(check_resources):
            raise ValueError("La comprobación de recursos debe ser una función")
        raw_world = world.raw_world
        start = world.config.context - 1
        prices = raw_world.prices[start:]
        scores = np.full(prices.shape[:2], np.nan)
        for position in range(len(world) + 1):
            if check_resources is not None:
                check_resources()
            cohort = world.observation_at(start + position)
            values = np.asarray(predict(cohort["inputs"]), dtype=np.float64)
            if values.shape != (len(cohort["asset_ids"]),) or not np.isfinite(values).all():
                raise ValueError("El padre no devuelve una predicción por muestra admitida")
            scores[position, : len(values)] = values
        if check_resources is not None:
            check_resources()
        return cls(
            prices,
            raw_world.times[start:],
            raw_world.asset_ids,
            scores,
            domain="synthetic",
            currency="USD",
            partition=world.partition,
            parent_id=parent_id,
            source_identity=dict(
                source_sha256=world.source_sha256,
                generator=asdict(world.config),
                encoding=world.identity.get("encoding", "analytic"),
            ),
        )
