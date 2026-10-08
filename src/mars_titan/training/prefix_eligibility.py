"""Acreditar exclusiones del residual usando únicamente sus pares pasados."""

import hashlib
import json
import math
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import exchange_calendars
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from mars_titan.data.audited_prices import MAX_PRICE_BYTES, read_audited_factor, read_audited_prices
from mars_titan.data.budget_targets import _aligned_returns
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import sha256
from mars_titan.data.temporal import MarketClock

from .corpus_inputs import CorpusDataset

_CUTOFF = "2023-12-31"
_RESERVED = 1_704_067_200_000_000
_HISTORY = 252
_MINIMUM = 126


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _signature(path):
    safe_destination(path)
    stat = path.stat()
    if not path.is_file():
        raise ValueError("La fuente del prefijo no es un archivo regular")
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@dataclass(frozen=True)
class PrefixEvidence:
    policy_id: str
    flow_id: str
    decision_at: int
    history_pairs: int
    market_variance: float | None
    records_sha256: str

    def __post_init__(self):
        if (
            any(
                not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
                for v in (self.policy_id, self.records_sha256)
            )
            or not isinstance(self.flow_id, str)
            or type(self.decision_at) is not int
            or not 0 < self.decision_at < _RESERVED
            or type(self.history_pairs) is not int
            or not 0 <= self.history_pairs <= _HISTORY
            or ((self.market_variance is None) != (self.history_pairs < _MINIMUM))
            or (
                self.market_variance is not None
                and (
                    type(self.market_variance) is not float
                    or not math.isfinite(self.market_variance)
                    or self.market_variance < 0
                )
            )
        ):
            raise ValueError("La evidencia no conserva sus tipos, recuentos o huellas")

    @property
    def reason(self):
        if self.history_pairs < _MINIMUM:
            return "insufficient_pairs"
        if self.market_variance <= np.finfo(float).eps:
            return "zero_market_variance"
        return None

    def fingerprint(self):
        return _digest(asdict(self))


class PrefixTargetVerifier:
    """Vincular precios y factor efectivos al corpus, sin consultar sus labels.

    La caché retiene arrays inmutables y acotados. El prefijo usa el calendario
    original de las etiquetas y su umbral de varianza, sin estimar alpha o beta.
    """

    def __init__(self, dataset, *, source_manifest, cache_bytes=128 * 1024**2):
        if type(dataset) is not CorpusDataset:
            raise ValueError("El prefijo necesita un lector de corpus verificado")
        if type(cache_bytes) is not int or not 0 <= cache_bytes <= 512 * 1024**2:
            raise ValueError("La caché del prefijo debe estar entre cero y 512 MiB")
        self._cache, self._cache_bytes, self._cache_limit = OrderedDict(), 0, cache_bytes
        self._files, self._signatures = {}, {}
        metadata, identity = read_manifest(dataset.path, 8 * 1024**2)
        if identity != dataset.identity or metadata != dataset.manifest:
            raise ValueError("El corpus cambió respecto de su lectura verificada")
        parent_path = Path(source_manifest)
        parent, parent_hash = read_manifest(parent_path, 8 * 1024**2)
        if parent_hash != metadata["configuration"]["source_manifest_sha256"]:
            raise ValueError("El calendario no procede de la edición materializada original")
        if (
            "target_factor_revision" not in metadata["configuration"]
            and metadata["market_factors"] != parent["market_factors"]
        ):
            raise ValueError("Los factores efectivos cambiaron sin una revisión identificada")
        self.source_id = identity
        self._register(dataset.path, identity, 8 * 1024**2)
        self._register(parent_path, parent_hash, 8 * 1024**2)
        self._manifest_paths = (dataset.path, parent_path)
        assets = metadata["assets"]
        if len(assets) > 8192:
            raise ValueError("El prefijo supera el censo de 8192 activos")
        self._clocks, self._calendar, self._positions = {}, {}, {}
        self._prices, self._factors = {}, {}
        for market in sorted({row["market"] for row in assets}):
            start = parent["calendar_start"][market]
            clock = MarketClock(market, start, _CUTOFF)
            moments = np.asarray(
                [int(pd.Timestamp(t).value // 1000) for t in clock.decisions], dtype="<i8"
            )
            if len(moments) > 200_000:
                raise ValueError("El calendario del prefijo supera su presupuesto")
            self._clocks[market] = clock
            self._calendar[market] = [day.isoformat() for day in clock.days]
            self._positions[market] = {int(at): i for i, at in enumerate(moments)}
            factor = metadata["market_factors"][market]
            if factor["market"] != market:
                raise ValueError("El factor pertenece a otro mercado")
            self._factors[market] = self._register(
                Path(factor["prices_path"]), factor["prices_sha256"], MAX_PRICE_BYTES
            )
        prepared = Path(metadata["roots"]["prepared"])
        for row in assets:
            flow = f"{row['market']}/{row['symbol']}"
            self._prices[flow] = self._register(
                prepared / row["market"] / row["symbol"] / "prices.parquet",
                row["prices_sha256"],
                MAX_PRICE_BYTES,
            )
        self._identity = dict(
            schema_version=1,
            recipe="verified_residual_prefix_252_126_v1",
            source_sha256=identity,
            parent_sha256=parent_hash,
            factors=metadata["market_factors"],
            calendar_start=parent["calendar_start"],
            calendars={
                market: _digest([day.isoformat() for day in clock.decisions])
                for market, clock in self._clocks.items()
            },
            cutoff=_CUTOFF,
            history=_HISTORY,
            minimum=_MINIMUM,
            variance="numpy_var_ddof0_fp64_lte_epsilon",
            numpy_version=np.__version__,
            pandas_version=pd.__version__,
            exchange_calendars_version=exchange_calendars.__version__,
            code_sha256=sha256(Path(__file__)),
            price_reader_sha256=sha256(Path(__file__).parents[1] / "data/audited_prices.py"),
            target_reference_sha256=sha256(Path(__file__).parents[1] / "data/budget_targets.py"),
        )
        self.policy_id = _digest(self._identity)

    def _register(self, path, digest, maximum):
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("La fuente necesita una huella de contenido")
        signature = _signature(path)
        if signature[2] > maximum or sha256(path) != digest or _signature(path) != signature:
            raise ValueError("La fuente cambió o supera su presupuesto")
        if path in self._files and self._files[path] != (digest, maximum):
            raise ValueError("La misma fuente mezcla contratos de contenido")
        self._files[path], self._signatures[path] = (digest, maximum), signature
        return path

    def _confirm(self, path):
        signature = _signature(path)
        digest, maximum = self._files[path]
        if signature != self._signatures[path]:
            if signature[2] > maximum or sha256(path) != digest or _signature(path) != signature:
                raise ValueError("La fuente del prefijo cambió desde su confirmación")
            self._signatures[path] = signature

    def identity(self):
        return json.loads(json.dumps(self._identity))

    def _returns(self, path, market, *, factor=False):
        self._confirm(path)
        key = path, market, factor
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        metadata = pq.read_metadata(path)
        record = dict(
            path=str(path),
            sha256=self._files[path][0],
            rows=metadata.num_rows,
            source_sha256=self.source_id,
        )
        reader = read_audited_factor if factor else read_audited_prices
        frame, _, reserved = reader(record, self.source_id, self._clocks[market], _CUTOFF)
        if reserved:
            raise ValueError(
                "La fuente efectiva del prefijo contiene sesiones reservadas posteriores a 2023"
            )
        # Arrow puede conservar la sesión como categoría tras el diccionario Parquet.
        frame["session"] = frame["session"].astype(str)
        returns, available = _aligned_returns(frame, self._calendar[market], _CUTOFF)
        available = available.to_numpy(dtype="datetime64[us]").astype("<i8")
        arrays = tuple(
            np.frombuffer(np.asarray(a).tobytes(), dtype=a.dtype) for a in (returns, available)
        )
        size = sum(a.nbytes for a in arrays)
        if size <= self._cache_limit:
            while self._cache and self._cache_bytes + size > self._cache_limit:
                _, old = self._cache.popitem(last=False)
                self._cache_bytes -= sum(a.nbytes for a in old)
            self._cache[key] = arrays
            self._cache_bytes += size
        return arrays

    def evidence(self, flow_id, decision_at):
        if (
            not isinstance(flow_id, str)
            or flow_id not in self._prices
            or type(decision_at) is not int
            or not 0 < decision_at < _RESERVED
        ):
            raise ValueError("El prefijo necesita un flujo conocido y una decisión anterior a 2024")
        market = flow_id.split("/")[0]
        position = self._positions[market].get(decision_at)
        if position is None:
            raise ValueError("La decisión no pertenece al calendario del prefijo")
        for path in self._manifest_paths:
            self._confirm(path)
        stock, stock_at = self._returns(self._prices[flow_id], market)
        factor, factor_at = self._returns(self._factors[market], market, factor=True)
        window = slice(max(0, position - _HISTORY + 1), position + 1)
        y, x, y_at, x_at = (a[window] for a in (stock, factor, stock_at, factor_at))
        valid = np.isfinite(x) & np.isfinite(y) & (x_at <= decision_at) & (y_at <= decision_at)
        count = int(valid.sum())
        variance = float(np.var(x[valid])) if count >= _MINIMUM else None
        if variance is not None and not math.isfinite(variance):
            raise ValueError("La varianza del prefijo no es finita")
        digest = hashlib.sha256()
        digest.update(_digest([market, self._calendar[market][window]]).encode())
        for values in (y, x, y_at, x_at):
            digest.update(values.tobytes())
        return PrefixEvidence(
            self.policy_id, flow_id, decision_at, count, variance, digest.hexdigest()
        )

    def verify(self, proof, *, flow_id, decision_at):
        if type(proof) is not PrefixEvidence or proof != self.evidence(flow_id, decision_at):
            raise ValueError("La evidencia no corresponde a los registros verificados del prefijo")
        return proof.reason
