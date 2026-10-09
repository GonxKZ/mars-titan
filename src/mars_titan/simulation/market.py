"""Precios y predicciones inmutables con una admisión explícita de su procedencia."""

import hashlib
import json
import re
from dataclasses import asdict

import numpy as np

from mars_titan.environments.cohorts import FINAL_TEST_START_US, VALIDATION_START_US
from mars_titan.evaluation.splits import PARTITIONS

from .portfolio import MAX_INSTRUMENTS, CorporateAction, Quote

# 4.200 activos durante un año de 251 sesiones superan el millón de celdas anterior.
# Con 48 bytes por celda (OHLCV y predicción en float64) el máximo ocupa 96 MiB por copia.
MAX_TAPE_CELLS = 2_097_152
RECONSTRUCTED = "unadjusted_reconstructed"
# Tratamiento fijo de la edición reconstruida (#379). Una cinta no puede declarar otro.
RECONSTRUCTED_CONTRACT = dict(
    corporate_actions="provider_events_in_verified_rows",
    corporate_actions_complete=False,
    exit_returns="unavailable",
    population="listed_through_2025_03",
    rows="verified_only",
    non_trading="zero_volume_or_missing_row_without_execution",
    valuation="last_traded_close",
    same_day_split_dividend="lower_cash_without_execution",
    off_grid_open="without_execution",
)
CURRENCIES = {"US": "USD", "CN": "CNY"}
_HEX = re.compile(r"[a-f0-9]{64}")
_SEGMENT = {"receipt_sha256", "fold", "partition", "start", "end", "labels_used_until"}


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


def _reconstructed_contract(audit, currency):
    """Comprobar que la cinta declara las limitaciones de la edición sin suavizarlas."""
    keys = {"price_basis", "market", "edition_id", "evidence_sha256", "walk_forward"}
    keys |= {"prediction_fit_ends", "assumptions", *RECONSTRUCTED_CONTRACT}
    assumptions = audit.get("assumptions")
    if (
        set(audit) != keys
        or any(audit[key] != value for key, value in RECONSTRUCTED_CONTRACT.items())
        or audit["corporate_actions_complete"] is not False
        or CURRENCIES.get(audit["market"]) != currency
        or not _HEX.fullmatch(str(audit["edition_id"]))
        or not _HEX.fullmatch(str(audit["evidence_sha256"]))
        or not isinstance(assumptions, dict)
        or set(assumptions) != {"dividend_payment_lag_sessions"}
        or type(assumptions["dividend_payment_lag_sessions"]) is not int
        or not 0 <= assumptions["dividend_payment_lag_sessions"] <= 252
    ):
        raise ValueError("La cinta reconstruida no declara su tratamiento y sus limitaciones")


def _walk_forward_fits(segments, times):
    """Último dato de ajuste de cada sesión, derivado de los recibos de su ventana."""
    if (
        not isinstance(segments, list)
        or not 1 <= len(segments) <= 128
        or any(
            not isinstance(segment, dict)
            or set(segment) != _SEGMENT
            or not _HEX.fullmatch(str(segment["receipt_sha256"]))
            or not isinstance(segment["fold"], str)
            or segment["partition"] not in PARTITIONS
            or any(type(segment[key]) is not int for key in ("start", "end", "labels_used_until"))
            or not 0 <= segment["start"] < segment["end"] <= FINAL_TEST_START_US
            for segment in segments
        )
        or any(a["end"] > b["start"] for a, b in zip(segments, segments[1:], strict=False))
    ):
        raise ValueError("Los tramos walk-forward de la cinta no son válidos")
    starts = np.array([segment["start"] for segment in segments], dtype=np.int64)
    ends = np.array([segment["end"] for segment in segments], dtype=np.int64)
    owner = np.searchsorted(starts, times, side="right") - 1
    if (owner < 0).any() or (times >= ends[np.maximum(owner, 0)]).any():
        raise ValueError("Una sesión de la cinta queda fuera de sus tramos walk-forward")
    if len(set(owner.tolist())) != len(segments):
        raise ValueError("Un tramo walk-forward declarado no tiene sesiones")
    return [segments[i]["labels_used_until"] for i in owner]


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
        reconstructed = (
            domain == "real"
            and isinstance(audit, dict)
            and audit.get("price_basis") == RECONSTRUCTED
        )
        if reconstructed:
            _reconstructed_contract(audit, currency)
        elif domain == "real" and (
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
            or not 1 <= prices.shape[1] <= MAX_INSTRUMENTS
            or prices.shape[0] * prices.shape[1] > MAX_TAPE_CELLS
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
        if reconstructed:
            # Los cortes salen de los recibos walk-forward y no de las fechas fijas de 2023.
            if audit["prediction_fit_ends"] != _walk_forward_fits(audit["walk_forward"], times):
                raise ValueError(
                    "Las predicciones históricas necesitan un ajuste anterior a cada decisión"
                )
            prefixes = {asset.split("/", 1)[0] for asset in assets}
            if prefixes != {audit["market"]} or not np.isfinite(prices[:, :, 3]).all():
                # Sin retornos de salida, ningún activo puede quedar sin valorar.
                raise ValueError("La cinta reconstruida necesita un cierre valorado por sesión")
        elif domain == "real" and not (times[0] >= low and times[-1] < high):
            raise ValueError("La simulación histórica cruza su partición o el test sellado")
        order = np.argsort(assets)
        self.assets = [assets[i] for i in order]
        self.prices = np.array(prices[:, order], dtype=np.float64, order="C", copy=True)
        self.scores = np.array(scores[:, order], dtype=np.float64, order="C", copy=True)
        self.close_times = times.astype(np.int64, copy=True)
        self.prediction_times = _times(times if prediction_times is None else prediction_times)
        if self.prediction_times.shape != times.shape or (self.prediction_times > times).any():
            raise ValueError("Las predicciones contienen información posterior al cierre")
        if domain == "real":
            # Una predicción dentro de muestra filtraría el ajuste del padre a la política.
            fits = audit.get("prediction_fit_ends")
            if (
                not isinstance(fits, list)
                or any(type(value) is not int for value in fits)
                or _times(fits).shape != times.shape
                or (_times(fits) > self.prediction_times).any()
            ):
                raise ValueError(
                    "Las predicciones históricas necesitan un ajuste anterior a cada decisión"
                )
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
        ids = set()
        for action in self.actions:
            if not isinstance(action, CorporateAction):
                raise ValueError("La acción corporativa no tiene un contrato válido")
            action.validate()
            if action.id in ids:
                raise ValueError("La acción corporativa está duplicada")
            if action.asset not in self.assets or action.effective_at not in self.open_times:
                raise ValueError("La acción corporativa no pertenece al calendario")
            if action.effective_at == self.open_times[0]:
                # El episodio empieza en el primer cierre y no ejecuta esa apertura.
                raise ValueError("La acción corporativa es anterior a la primera decisión")
            ids.add(action.id)
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
