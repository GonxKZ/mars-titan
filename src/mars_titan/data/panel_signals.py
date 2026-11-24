"""Artefactos de señales transversales de panel precalculadas por mercado (`panel_signals_v1`).

Las señales que dependen de todo el mercado (por ejemplo, momento residual frente a pares) se
calculan fuera de este repositorio y se entregan como un artefacto por mercado bajo
`<raíz>/<nombre del panel>/<mercado>/` con dos ficheros:

- `panel.npy`: array float32 en orden C de forma `[sesiones, símbolos, señales]`, finito y acotado
  a ±8, con cero donde el activo no tiene dato esa sesión.
- `manifest.json`: metadatos, ejes (`sessions`, `symbols`, `names`), forma, tipo, huella SHA-256
  de `panel.npy`, `final_test_opened: false` y procedencia libre en `source`.

El lector valida en cerrado todo el contrato, comprueba la huella del manifiesto contra la que
declara la vista y verifica la huella de `panel.npy` por bloques una vez por proceso (sin cargar el
fichero entero en memoria), abriéndolo con `mmap_mode="r"`. Las consultas `rows(symbol, days)`
emparejan cada día exacto con `np.searchsorted`, devolviendo ceros cuando el símbolo o el día no
existen en el artefacto (nunca la sesión anterior ni la siguiente).
"""

import os
import re
import tempfile
import threading
from datetime import date
from pathlib import Path

import numpy as np

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.data.technical_features import LIMIT

ENV = "MARS_TITAN_PANEL_SIGNALS"
KIND = "panel_signals_v1"
SCHEMA_VERSION = 1
MARKETS = ("US", "CN")
FINAL_TEST_START = date(2024, 1, 1)

NAME_PRICE_VOLUME_16_V1 = "price_volume_16_v1"
NAMES_PRICE_VOLUME_16_V1 = (
    "bv_21",
    "jump_var_ratio_21",
    "vol_of_vol_63",
    "realized_skew_signed_21",
    "peer_resid_mom_5",
    "peer_resid_mom_21",
    "peer_resid_mom_63",
    "liq_cond_rev_5",
    "signed_abn_turn_21",
    "kaufman_er_21",
    "amihud_asym_up_down_21",
    "rank_vel_5_21",
    "rank_accel_5_21_63",
    "peer_rev_5_x_idio_vol_surge",
    "limit_hit_streak_5",
    "post_halt_age_decay",
)
PANELS = {
    NAME_PRICE_VOLUME_16_V1: NAMES_PRICE_VOLUME_16_V1,
}

_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_ISO_DAY = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "name",
        "market",
        "names",
        "sessions",
        "symbols",
        "shape",
        "dtype",
        "panel_sha256",
        "final_test_opened",
        "source",
    }
)
_MAX_MANIFEST_BYTES = 16 * 1024**2
# Huellas de `panel.npy` comprobadas en este proceso por ruta real y firma de stat.
_PANEL_DIGESTS: dict[tuple[str, tuple[int, int, int, int, int]], str] = {}
_PANEL_DIGEST_LOCK = threading.Lock()
_PANEL_DIGEST_LIMIT = 4096


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _signature(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _is_sha256(value):
    return isinstance(value, str) and _HEX_SHA256.fullmatch(value) is not None


def _panel_sha256(path: Path, signature) -> str:
    """Huella SHA-256 de `panel.npy`, calculada por bloques como mucho una vez por proceso."""
    key = (str(path.resolve()), signature)
    with _PANEL_DIGEST_LOCK:
        cached = _PANEL_DIGESTS.get(key)
    if cached is not None:
        return cached
    digest = sha256(path)
    if _signature(path.stat()) != signature:
        raise ValueError("El archivo panel.npy cambió durante su comprobación")
    with _PANEL_DIGEST_LOCK:
        if len(_PANEL_DIGESTS) >= _PANEL_DIGEST_LIMIT:
            _PANEL_DIGESTS.clear()
        _PANEL_DIGESTS[key] = digest
    return digest


def _checked_sessions(sessions) -> np.ndarray:
    """Validar días ISO estrictamente crecientes y anteriores a 2024-01-01."""
    _require(
        isinstance(sessions, (list, tuple)) and len(sessions) >= 1,
        "Las sesiones del panel deben ser una lista no vacía de fechas ISO",
    )
    previous = None
    for value in sessions:
        _require(
            isinstance(value, str) and _ISO_DAY.fullmatch(value) is not None,
            "Cada sesión del panel debe ser una fecha ISO YYYY-MM-DD",
        )
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            raise ValueError(f"Sesión del panel no es una fecha ISO válida: {value}") from None
        _require(
            parsed.isoformat() == value,
            f"Sesión del panel no conserva el formato ISO YYYY-MM-DD: {value}",
        )
        _require(
            parsed < FINAL_TEST_START,
            f"El test final de 2024 está sellado: se rechaza la sesión {value}",
        )
        if previous is not None:
            _require(
                parsed > previous,
                f"Las sesiones del panel deben ser estrictamente crecientes: {previous} -> {value}",
            )
        previous = parsed
    return np.asarray(sessions, dtype="datetime64[D]")


def _checked_symbols(symbols) -> tuple[str, ...]:
    """Validar lista no vacía de símbolos ordenados y sin repetir."""
    _require(
        isinstance(symbols, (list, tuple)) and len(symbols) >= 1,
        "Los símbolos del panel deben ser una lista no vacía de textos",
    )
    previous = None
    for value in symbols:
        _require(
            isinstance(value, str) and bool(value),
            "Los símbolos del panel deben ser textos no vacíos",
        )
        if previous is not None:
            _require(
                value > previous,
                "Los símbolos del panel deben estar ordenados y sin repetir: "
                f"{previous!r}, {value!r}",
            )
        previous = value
    return tuple(symbols)


def _checked_regular_file(path: Path, root: Path, label: str) -> None:
    try:
        inside = os.path.commonpath((root, os.path.realpath(path))) == str(root)
    except ValueError:
        inside = False
    _require(
        path.is_file() and not path.is_symlink() and inside,
        f"Falta el archivo regular {label} del artefacto del panel dentro de su raíz ({path})",
    )


def _read_npy_header(path: Path):
    """Leer cabecera .npy sin cargar datos para exigir float32 y orden C."""
    try:
        with path.open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                return np.lib.format.read_array_header_1_0(stream)
            if version == (2, 0):
                return np.lib.format.read_array_header_2_0(stream)
            if version == (3, 0):
                return np.lib.format._read_array_header(stream, version)
    except (OSError, ValueError) as error:
        raise ValueError(f"La cabecera de {path.name} no es un archivo .npy válido") from error
    raise ValueError(f"Versión de archivo .npy no admitida en {path.name}")


def panel_root() -> Path:
    """Carpeta raíz de los artefactos de panel, leída de `MARS_TITAN_PANEL_SIGNALS`."""
    raw = os.environ.get(ENV)
    _require(
        bool(raw),
        f"Falta la variable de entorno {ENV} requerida para leer el panel declarado por la vista",
    )
    path = Path(raw)
    _require(
        path.is_absolute() and not path.is_symlink() and path.is_dir(),
        f"{ENV} debe señalar una carpeta absoluta y regular existente ({raw})",
    )
    return path


def checked_panel_spec(panel, *, markets=None) -> dict:
    """Validar la declaración `input_extensions['panel']` y sus mercados."""
    _require(
        isinstance(panel, dict) and set(panel) == {"name", "sha256"},
        "La extensión de entradas panel debe declarar exactamente name y sha256",
    )
    name = panel["name"]
    _require(
        isinstance(name, str) and name in PANELS,
        f"El panel {name!r} no pertenece al catálogo declarado ({', '.join(PANELS)})",
    )
    digests = panel["sha256"]
    _require(
        isinstance(digests, dict)
        and bool(digests)
        and all(
            isinstance(market, str) and market in MARKETS and _is_sha256(digest)
            for market, digest in digests.items()
        ),
        "La extensión de entradas panel.sha256 debe asignar una huella SHA-256 a cada mercado "
        f"({', '.join(MARKETS)})",
    )
    if markets is not None:
        expected = set(markets)
        _require(
            set(digests) == expected,
            f"Los mercados del panel ({sorted(digests)}) deben coincidir exactamente con los "
            f"mercados de la vista ({sorted(expected)})",
        )
    return panel


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_panel(
    root: Path | str,
    *,
    name: str,
    market: str,
    sessions,
    symbols,
    values,
    names=None,
    source: dict | None = None,
) -> tuple[Path, str]:
    """Escribir un artefacto válido en `<root>/<name>/<market>/` y devolver `(manifest, sha256)`."""
    _require(
        isinstance(name, str) and name in PANELS,
        f"El panel {name!r} no pertenece al catálogo declarado ({', '.join(PANELS)})",
    )
    _require(
        isinstance(market, str) and market in MARKETS,
        f"El mercado del panel debe ser uno de {', '.join(MARKETS)}",
    )
    expected_names = list(PANELS[name])
    if names is not None:
        _require(
            isinstance(names, (list, tuple)) and list(names) == expected_names,
            "Los nombres de las señales del panel no coinciden con el catálogo o su orden",
        )
    checked_sessions = _checked_sessions(sessions)
    checked_symbols = _checked_symbols(symbols)
    _require(
        source is None or isinstance(source, dict),
        "El campo source del manifiesto del panel debe ser un diccionario",
    )
    _require(
        isinstance(values, np.ndarray) and values.dtype == np.dtype(np.float32),
        "Los valores del panel deben ser un array numpy de tipo float32",
    )
    expected_shape = (len(checked_sessions), len(checked_symbols), len(expected_names))
    _require(
        values.ndim == 3 and values.shape == expected_shape,
        f"La forma del panel {values.shape} no coincide con los ejes {expected_shape}",
    )
    _require(
        bool(np.isfinite(values).all()),
        "El panel contiene valores no finitos",
    )
    _require(
        float(np.abs(values).max()) <= LIMIT,
        f"El panel contiene valores fuera de la cota ±{LIMIT:g}",
    )
    array = np.ascontiguousarray(values, dtype=np.float32)
    folder = Path(root) / name / market
    safe_destination(folder)
    folder.mkdir(parents=True, exist_ok=True)
    panel_path = folder / "panel.npy"
    manifest_path = folder / "manifest.json"
    _atomic_npy(panel_path, array)
    panel_digest = sha256(panel_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "name": name,
        "market": market,
        "names": expected_names,
        "sessions": list(sessions),
        "symbols": list(checked_symbols),
        "shape": list(expected_shape),
        "dtype": "float32",
        "panel_sha256": panel_digest,
        "final_test_opened": False,
        "source": {} if source is None else dict(source),
    }
    atomic_json(manifest_path, manifest)
    return manifest_path, sha256(manifest_path)


class PanelSignals:
    """Lector memmap verificado de un artefacto `panel_signals_v1` para un mercado."""

    def __init__(self, root: Path | str, *, name: str, market: str, manifest_sha256: str):
        _require(
            isinstance(name, str) and name in PANELS,
            f"El panel {name!r} no pertenece al catálogo declarado ({', '.join(PANELS)})",
        )
        _require(
            isinstance(market, str) and market in MARKETS,
            f"El mercado del panel debe ser uno de {', '.join(MARKETS)}",
        )
        _require(
            _is_sha256(manifest_sha256),
            "La huella SHA-256 esperada del manifiesto del panel no es válida",
        )
        root_path = Path(root)
        _require(
            root_path.is_dir() and not root_path.is_symlink(),
            f"La raíz del panel no es una carpeta regular existente ({root_path})",
        )
        resolved_root = root_path.resolve()
        folder = root_path / name / market
        manifest_path = folder / "manifest.json"
        panel_path = folder / "panel.npy"
        _checked_regular_file(manifest_path, resolved_root, "manifest.json")
        _checked_regular_file(panel_path, resolved_root, "panel.npy")

        manifest, actual_manifest_sha256 = read_manifest(manifest_path, _MAX_MANIFEST_BYTES)
        _require(
            actual_manifest_sha256 == manifest_sha256,
            f"La huella SHA-256 de manifest.json del panel {name}/{market} no coincide con la "
            "declarada por la vista",
        )
        _require(
            isinstance(manifest, dict) and set(manifest) == _MANIFEST_FIELDS,
            "El manifiesto del panel no declara los campos exactos del contrato",
        )
        _require(
            type(manifest["schema_version"]) is int
            and manifest["schema_version"] == SCHEMA_VERSION,
            "La versión de esquema del manifiesto del panel debe ser 1",
        )
        _require(
            manifest["kind"] == KIND,
            f"La clase del manifiesto del panel debe ser {KIND}",
        )
        _require(
            manifest["name"] == name,
            f"El nombre del panel en el manifiesto ({manifest['name']!r}) no coincide con {name!r}",
        )
        _require(
            manifest["market"] == market,
            f"El mercado del panel en el manifiesto ({manifest['market']!r}) no coincide con "
            f"{market!r}",
        )
        expected_names = list(PANELS[name])
        _require(
            isinstance(manifest["names"], list) and manifest["names"] == expected_names,
            "Los nombres de las señales del panel no coinciden con el catálogo o su orden",
        )
        _require(
            manifest["final_test_opened"] is False,
            "El test final debe permanecer cerrado en el manifiesto del panel (final_test_opened)",
        )
        _require(
            manifest["dtype"] == "float32",
            "El tipo de datos del manifiesto del panel debe ser float32",
        )
        _require(
            isinstance(manifest["source"], dict),
            "El campo source del manifiesto del panel debe ser un diccionario",
        )
        _require(
            _is_sha256(manifest["panel_sha256"]),
            "La huella panel_sha256 del manifiesto del panel no es válida",
        )
        sessions = _checked_sessions(manifest["sessions"])
        symbols = _checked_symbols(manifest["symbols"])
        shape = manifest["shape"]
        expected_shape = [len(sessions), len(symbols), len(expected_names)]
        _require(
            isinstance(shape, list)
            and len(shape) == 3
            and all(type(value) is int for value in shape)
            and shape == expected_shape,
            f"La forma declarada en el manifiesto ({shape}) no coincide con los ejes "
            f"({expected_shape})",
        )

        signature = _signature(panel_path.stat())
        actual_panel_sha256 = _panel_sha256(panel_path, signature)
        _require(
            actual_panel_sha256 == manifest["panel_sha256"],
            f"La huella SHA-256 de panel.npy del panel {name}/{market} no coincide con su "
            "manifiesto",
        )
        hdr_shape, fortran_order, hdr_dtype = _read_npy_header(panel_path)
        _require(
            hdr_dtype == np.dtype(np.float32),
            f"El tipo de datos de panel.npy ({hdr_dtype}) debe ser float32",
        )
        _require(
            not fortran_order,
            "panel.npy debe estar almacenado en orden C contiguo",
        )
        _require(
            hdr_shape == tuple(expected_shape),
            f"La forma de panel.npy ({hdr_shape}) no coincide con el manifiesto "
            f"({tuple(expected_shape)})",
        )
        try:
            array = np.load(panel_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"No se pudo abrir panel.npy como memmap ({panel_path})") from error
        _require(
            isinstance(array, np.ndarray)
            and array.shape == tuple(expected_shape)
            and array.dtype == np.dtype(np.float32)
            and array.flags.c_contiguous,
            "panel.npy no conserva tipo float32, orden C contiguo y forma declarada",
        )
        _require(
            _signature(panel_path.stat()) == signature,
            "panel.npy cambió durante su apertura",
        )

        self.root = root_path
        self.folder = folder
        self.manifest_path = manifest_path
        self.panel_path = panel_path
        self.name = name
        self.market = market
        self.names = PANELS[name]
        self.width = len(self.names)
        self.manifest = manifest
        self.manifest_sha256 = actual_manifest_sha256
        self.panel_sha256 = actual_panel_sha256
        self._resolved_root = resolved_root
        self._manifest_signature = _signature(manifest_path.stat())
        self._panel_signature = signature
        self._sessions = sessions
        self.symbols = symbols
        self._symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
        self._data = array

    def verify(self) -> None:
        """Comprobar que `manifest.json` y `panel.npy` no han cambiado desde su apertura."""
        _checked_regular_file(self.manifest_path, self._resolved_root, "manifest.json")
        _checked_regular_file(self.panel_path, self._resolved_root, "panel.npy")
        manifest_sig = _signature(self.manifest_path.stat())
        if manifest_sig != self._manifest_signature:
            _, digest = read_manifest(self.manifest_path, _MAX_MANIFEST_BYTES)
            _require(
                digest == self.manifest_sha256,
                f"manifest.json del panel {self.name}/{self.market} ha cambiado",
            )
            self._manifest_signature = manifest_sig
        panel_sig = _signature(self.panel_path.stat())
        if panel_sig != self._panel_signature:
            _require(
                panel_sig[0] == self._panel_signature[0]
                and panel_sig[1] == self._panel_signature[1]
                and _panel_sha256(self.panel_path, panel_sig) == self.panel_sha256,
                f"panel.npy del panel {self.name}/{self.market} ha cambiado desde su apertura",
            )
            self._panel_signature = panel_sig

    def rows(self, symbol: str, days) -> np.ndarray:
        """Filas float32 `[len(days), K]` para cada día exacto del activo, o ceros si falta."""
        _require(isinstance(symbol, str), "El símbolo debe ser un texto")
        days = np.asarray(days, dtype="datetime64[D]")
        _require(
            days.ndim == 1 and not bool(np.isnat(days).any()),
            "Las sesiones consultadas deben ser un vector 1D de días válidos sin NaT",
        )
        result = np.zeros((len(days), self.width), dtype=np.float32)
        column = self._symbol_index.get(symbol)
        if column is None or len(days) == 0:
            return result
        positions = np.searchsorted(self._sessions, days)
        found = positions < len(self._sessions)
        found[found] = self._sessions[positions[found]] == days[found]
        if found.any():
            selected = np.asarray(self._data[positions[found], column, :], dtype=np.float32)
            _require(
                bool(np.isfinite(selected).all()),
                f"El panel {self.name}/{self.market} contiene valores no finitos para {symbol}",
            )
            _require(
                bool(np.all(np.abs(selected) <= LIMIT)),
                f"El panel {self.name}/{self.market} contiene valores fuera de ±{LIMIT:g} "
                f"para {symbol}",
            )
            result[found] = selected
        return result
