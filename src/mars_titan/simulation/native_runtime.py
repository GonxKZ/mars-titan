"""Interfaz C acotada del cálculo contable, sin dependencias de enlace con Python."""

import ctypes as ct
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from mars_titan.data.storage import sha256


class Position(ct.Structure):
    _fields_ = [
        ("quantity", ct.c_double),
        ("target", ct.c_double),
        ("capacity", ct.c_double),
        ("decision_at", ct.c_int64),
    ]


class Account(ct.Structure):
    _fields_ = [(name, ct.c_double) for name in ("cash", "costs", "turnover", "receivable", "nav")]


class Trade(ct.Structure):
    _fields_ = [
        ("quantity", ct.c_double),
        ("price", ct.c_double),
        ("cost", ct.c_double),
        ("reason", ct.c_int32),
        ("reserved", ct.c_uint32),
    ]


class Layout(ct.Structure):
    _fields_ = [
        (name, ct.c_uint32)
        for name in ("abi_version", "position_size", "account_size", "trade_size")
    ]


POSITION = np.dtype(
    [
        ("quantity", np.float64),
        ("target", np.float64),
        ("capacity", np.float64),
        ("decision_at", np.int64),
    ],
    align=True,
)
ACCOUNT = np.dtype(
    [(name, np.float64) for name in ("cash", "costs", "turnover", "receivable", "nav")], align=True
)
TRADE = np.dtype(
    [
        ("quantity", np.float64),
        ("price", np.float64),
        ("cost", np.float64),
        ("reason", np.int32),
        ("reserved", np.uint32),
    ],
    align=True,
)


def _pointer(array, kind):
    return array.ctypes.data_as(ct.POINTER(kind))


def _check(code, error):
    if code:
        message = error.value.decode("utf-8")
        if code == 3:
            raise FloatingPointError(message)
        raise ValueError(message)


class NativeLibrary:
    def __init__(self, path):
        self.path = path
        self.signature = path.stat()
        self.sha256 = sha256(path)
        self.api = ct.CDLL(str(path))
        self.api.mt_simulation_layout_v1.restype = Layout
        self.api.mt_simulation_layout_v1.argtypes = []
        layout = self.api.mt_simulation_layout_v1()
        if layout.abi_version != 1 or any(
            actual != dtype.itemsize or actual != ct.sizeof(kind)
            for actual, dtype, kind in (
                (layout.position_size, POSITION, Position),
                (layout.account_size, ACCOUNT, Account),
                (layout.trade_size, TRADE, Trade),
            )
        ):
            raise ValueError("La biblioteca nativa no comparte el contrato binario")
        self.api.mt_simulation_step_v1.restype = ct.c_int
        self.api.mt_simulation_step_v1.argtypes = [
            ct.c_uint32,
            ct.c_uint32,
            ct.POINTER(ct.c_uint32),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_uint8),
            ct.POINTER(ct.c_double),
            ct.POINTER(Position),
            ct.POINTER(Account),
            ct.c_double,
            ct.c_double,
            ct.c_int64,
            ct.c_int64,
            ct.c_int64,
            ct.POINTER(Position),
            ct.POINTER(Account),
            ct.POINTER(Trade),
            ct.POINTER(ct.c_char),
            ct.c_size_t,
        ]
        self.api.mt_simulation_observation_v1.restype = ct.c_int
        self.api.mt_simulation_observation_v1.argtypes = [
            ct.c_uint32,
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_uint8),
            ct.POINTER(Position),
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ct.POINTER(ct.c_float),
            ct.POINTER(ct.c_char),
            ct.c_size_t,
        ]
        self.check_identity()

    def check_identity(self):
        current = self.path.stat()
        if (current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns) != (
            self.signature.st_ino,
            self.signature.st_size,
            self.signature.st_mtime_ns,
            self.signature.st_ctime_ns,
        ):
            raise RuntimeError("La biblioteca cambió durante el proceso. Inicia otro proceso")

    @staticmethod
    def frame(values, count):
        if (
            not isinstance(values, np.ndarray)
            or values.shape != (count, 5)
            or values.dtype != np.float64
            or not values.flags.c_contiguous
            or not values.flags.aligned
        ):
            raise ValueError("El bloque OHLCV debe ser float64 contiguo con una fila por activo")
        return values

    def step(self, owner, prices, open_at, close_at):
        code = self.api.mt_simulation_step_v1(
            len(owner.assets),
            len(owner.currencies),
            _pointer(owner._currency_ids, ct.c_uint32),
            _pointer(owner._lots, ct.c_double),
            _pointer(owner._retired, ct.c_uint8),
            _pointer(prices, ct.c_double),
            _pointer(owner._positions, Position),
            _pointer(owner._accounts, Account),
            owner.rate,
            owner.participation,
            owner.clock,
            open_at,
            close_at,
            _pointer(owner._next_positions, Position),
            _pointer(owner._next_accounts, Account),
            _pointer(owner._trades, Trade),
            owner._error,
            len(owner._error),
        )
        _check(code, owner._error)

    def observation(self, owner, prices, previous, scores, score_scale):
        count = len(owner.assets)
        self.frame(prices, count)
        self.frame(previous, count)
        if (
            not isinstance(scores, np.ndarray)
            or scores.shape != (count,)
            or scores.dtype != np.float64
            or not scores.flags.c_contiguous
            or not scores.flags.aligned
        ):
            raise ValueError("Las predicciones deben ser float64 contiguo y conservar los activos")
        output = np.empty(count * 6 + 2, dtype=np.float32)
        code = self.api.mt_simulation_observation_v1(
            count,
            _pointer(prices, ct.c_double),
            _pointer(previous, ct.c_double),
            _pointer(scores, ct.c_double),
            _pointer(owner._retired, ct.c_uint8),
            _pointer(owner._positions, Position),
            float(owner._accounts[0]["nav"]),
            float(owner._accounts[0]["cash"]),
            score_scale,
            _pointer(output, ct.c_float),
            owner._error,
            len(owner._error),
        )
        _check(code, owner._error)
        return output


@lru_cache(maxsize=4)
def _load(path):
    return NativeLibrary(path)


def load_library(path=None):
    selected = path or os.environ.get("MARS_TITAN_NATIVE_LIBRARY")
    if selected is None:
        root = Path(__file__).parents[3] / "build" / "native" / "native-release"
        candidates = [
            root / name
            for name in (
                "libmars_titan_simulation.so",
                "libmars_titan_simulation.dylib",
                "mars_titan_simulation.dll",
            )
        ]
        selected = next((item for item in candidates if item.is_file()), None)
    if selected is None:
        raise FileNotFoundError(
            "Compila native con los presets native-release antes de usar el simulador nativo"
        )
    library = _load(Path(selected).resolve(strict=True))
    library.check_identity()
    return library
