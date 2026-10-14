"""Perturbaciones declaradas de las entradas en la lectura, para estudiar la robustez (#37).

Como la ablación de modalidades (`data.modality_ablation`), cada perturbación se aplica a la
tabla de muestras de un activo después de sus comprobaciones y la tabla resultante vuelve a
pasarlas. El modelo recibe lo mismo que ante datos reales con esa ausencia o ese ruido. Nada
se ajusta con los datos: ni escalas, ni umbrales, ni filas elegidas.

- `burst_absence`: noticias y fundamentales ausentes en ráfagas de días naturales. Los días
  desde el 1 de enero de 2000 se agrupan en bloques de `length_days` y cada bloque de cada
  activo queda ausente con probabilidad `share`. Las filas que ya carecían de la modalidad
  conservan su causa original y las demás reciben la causa `input_perturbation`.
- `multiplicative_noise`: cada valor observado se multiplica por `1 + scale · z`, con `z`
  normal estándar. Las máscaras por concepto, las edades y los bloques ausentes no cambian, y
  un valor cero sigue siendo cero. El ruido del macro depende del mercado y de la fecha, no
  del activo, para que todos los activos de una sesión lean el mismo macro.

Cada número aleatorio es una función de la semilla, el activo (o el mercado), la fecha de la
muestra y la posición en el vector (SplitMix64 y Box-Muller). No depende del orden ni del
agrupamiento de las filas, así que leer un activo entero o grupo a grupo da lo mismo.

Los precios llegan por otra lectura y no se perturban. Las filas duplicadas y la
disponibilidad posterior a la decisión no son perturbaciones que el modelo pueda recibir:
el lector las rechaza, y la declaración de robustez cita las pruebas que lo comprueban.
"""

import hashlib
import math

import numpy as np
import pyarrow as pa

from .input_policy import HISTORICAL_MASKED, MODALITIES, policy_identity
from .modality_ablation import (
    VARIANTS,
    ablate_samples,
    ablated_modalities,
    ablation_identity,
    absent_rows,
    list_column,
)

BURST = "burst_absence"
NOISE = "multiplicative_noise"
REASON = "input_perturbation"
SEED = 20261013
RANDOM = "splitmix64_box_muller"
# Intensidades fijadas en desarrollo, sin mirar resultados: un mes y un trimestre de
# sesiones para las ráfagas y un 10 % y un 50 % de ruido relativo.
PERTURBATIONS = {
    "burst_absence_21d_10pct": dict(
        kind=BURST, modalities=("news", "fundamentals"), length_days=21, share=0.1
    ),
    "burst_absence_63d_25pct": dict(
        kind=BURST, modalities=("news", "fundamentals"), length_days=63, share=0.25
    ),
    "noise_10pct": dict(
        kind=NOISE, modalities=("news", "charts", "fundamentals", "macro"), scale=0.1
    ),
    "noise_50pct": dict(
        kind=NOISE, modalities=("news", "charts", "fundamentals", "macro"), scale=0.5
    ),
}
_DAY_US = 86_400_000_000
_ORIGIN_US = 946_684_800_000_000  # 2000-01-01 00:00 UTC
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
# Bloques numéricos con valor, máscara y edad por concepto. Solo se perturba el valor.
_CATALOGUED = {"fundamentals", "macro"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def check_spec(spec):
    """Exigir una perturbación completa, con modalidades e intensidad válidas."""
    _require(isinstance(spec, dict), "La perturbación debe ser un objeto")
    modalities = spec.get("modalities")
    _require(
        isinstance(modalities, tuple)
        and modalities
        and len(set(modalities)) == len(modalities)
        and set(modalities) <= set(MODALITIES) - {"prices"},
        "La perturbación necesita modalidades declaradas y distintas, sin precios",
    )
    if spec.get("kind") == BURST:
        _require(
            set(spec) == {"kind", "modalities", "length_days", "share"}
            and set(modalities) <= {"news", "fundamentals"}
            and type(spec["length_days"]) is int
            and 1 <= spec["length_days"] <= 366
            and isinstance(spec["share"], float)
            and 0 <= spec["share"] <= 1,
            "La ráfaga necesita noticias o fundamentales, una duración y una proporción",
        )
    else:
        _require(
            spec.get("kind") == NOISE
            and set(spec) == {"kind", "modalities", "scale"}
            and isinstance(spec["scale"], float)
            and 0 <= spec["scale"] <= 1,
            "La perturbación declarada no existe o su intensidad no es válida",
        )
    return spec


def _mix(values):
    """SplitMix64 sobre enteros sin signo de 64 bits, con desbordamiento modular."""
    with np.errstate(over="ignore"):
        values = values + _GOLDEN
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def _key(*parts):
    text = ":".join(str(part) for part in (SEED, *parts))
    return np.uint64(int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big"))


def _uniform(key, counters, width):
    """Uniformes en [0, 1) por contador y posición, sin estado compartido entre filas."""
    rows = _mix(key ^ _mix(np.asarray(counters).astype(np.uint64)))
    with np.errstate(over="ignore"):
        cells = rows[:, None] + np.arange(1, width + 1, dtype=np.uint64)[None, :] * _GOLDEN
    return (_mix(cells) >> np.uint64(11)).astype(np.float64) * 2.0**-53


def _normal(key, counters, width):
    pairs = _uniform(key, counters, 2 * width)
    radius = np.sqrt(-2.0 * np.log1p(-pairs[:, :width]))
    return radius * np.cos(2.0 * math.pi * pairs[:, width:])


def _moments(table):
    column = table["prediction_at"].combine_chunks()
    _require(
        pa.types.is_timestamp(column.type) and column.type.tz and not column.null_count,
        "Las fechas de las muestras necesitan zona y no pueden faltar",
    )
    return column.cast(pa.timestamp("us", tz="UTC")).cast(pa.int64()).to_numpy()


def burst_rows(asset, moments, length_days, share):
    """Filas de un activo dentro de una ráfaga, según el bloque de días de cada fecha."""
    moments = np.asarray(moments, dtype=np.int64)
    _require(not len(moments) or moments.min() >= _ORIGIN_US, "La muestra es anterior a 2000")
    blocks = (moments - _ORIGIN_US) // (_DAY_US * length_days)
    return _uniform(_key(BURST, asset, length_days), blocks, 1)[:, 0] < share


def days_since_burst(asset, moments, length_days, share, horizon_days):
    """Días desde el final de la última ráfaga anterior de cada fila de un activo.

    Una ráfaga ocupa los días `[b · length_days, (b + 1) · length_days)` desde 2000 y el
    primer día limpio cuenta como 0. Las filas dentro de una ráfaga, sin ninguna ráfaga
    anterior desde 2000 o a `horizon_days` días o más de la última reciben NaN. Usa el mismo
    generador que `burst_rows`, así que no hace falta leer las entradas perturbadas.
    """
    moments = np.asarray(moments, dtype=np.int64)
    _require(type(horizon_days) is int and horizon_days >= 1, "El horizonte debe ser positivo")
    if not len(moments):
        return np.zeros(0)
    _require(moments.min() >= _ORIGIN_US, "La muestra es anterior a 2000")
    days = (moments - _ORIGIN_US) // _DAY_US
    blocks = days // length_days
    # Bloques desde el que puede contener la última ráfaga dentro del horizonte.
    first = max(0, int(blocks.min()) - horizon_days // length_days - 1)
    grid = np.arange(first, int(blocks.max()) + 1)
    flags = _uniform(_key(BURST, asset, length_days), grid, 1)[:, 0] < share
    last = np.maximum.accumulate(np.where(flags, grid, -1))
    position = blocks - first
    # Fuera de una ráfaga, la última hasta el bloque actual es la última anterior a él.
    previous = last[position]
    since = (days - (previous + 1) * length_days).astype(np.float64)
    clean = ~flags[position] & (previous >= 0) & (since < horizon_days)
    return np.where(clean, since, np.nan)


def _noisy(table, spec, asset):
    count = len(table)
    moments = _moments(table)
    presence = (
        table["presence"].combine_chunks().flatten().to_numpy(zero_copy_only=False)
    ).reshape(count, len(MODALITIES))
    market = asset.split("/", 1)[0]
    for name in spec["modalities"]:
        vector = table[name].combine_chunks()
        values = vector.flatten().to_numpy(zero_copy_only=False)
        width = len(values) // count
        values = values.reshape(count, width).copy()
        observed = presence[:, MODALITIES.index(name)]
        data = width // 3 if name in _CATALOGUED else width
        noisy = np.repeat(observed[:, None], data, axis=1)
        if name in _CATALOGUED:
            # Valor, máscara y edad: solo los conceptos observados de un bloque presente.
            noisy &= values[:, data : 2 * data] == 1
        key = _key(NOISE, market if name == "macro" else asset, name)
        factor = 1.0 + spec["scale"] * _normal(key, moments, data)
        changed = values[:, :data].astype(np.float64) * factor
        values[:, :data] = np.where(noisy, changed.astype(values.dtype), values[:, :data])
        table = table.set_column(
            table.schema.get_field_index(name),
            name,
            list_column(pa.array(values.ravel()), width, vector.type),
        )
    return table


def perturb_samples(table, spec, asset):
    """Tabla de muestras de `asset` (`mercado/símbolo`) con la perturbación aplicada."""
    check_spec(spec)
    _require(isinstance(asset, str) and "/" in asset, "La perturbación necesita el activo")
    if not len(table):
        return table
    if spec["kind"] == BURST:
        rows = burst_rows(asset, _moments(table), spec["length_days"], spec["share"])
        return absent_rows(table, spec["modalities"], rows, REASON)
    return _noisy(table, spec, asset)


def perturbation_identity(variant):
    """Campos que identifican una perturbación declarada en recibos e informes."""
    _require(variant in PERTURBATIONS, "La perturbación de entradas no está declarada")
    spec = PERTURBATIONS[variant]
    return dict(
        variant=variant,
        **{key: list(value) if key == "modalities" else value for key, value in spec.items()},
        seed=SEED,
        random=RANDOM,
        missing_reason=REASON if spec["kind"] == BURST else None,
        mask_contract=policy_identity(HISTORICAL_MASKED)["mask_contract"],
    )


# Variantes de lectura: ablaciones de modalidades y perturbaciones declaradas.


def variant_modalities(variant):
    """Modalidades que cambia una variante de lectura declarada."""
    if variant in PERTURBATIONS:
        return PERTURBATIONS[variant]["modalities"]
    return ablated_modalities(variant)


def variant_identity(variant):
    """Identidad de una variante de lectura en recibos e informes."""
    if variant in PERTURBATIONS:
        return perturbation_identity(variant)
    return ablation_identity(variant)


def read_variant(table, variant, asset):
    """Aplicar una variante de lectura a la tabla de muestras de un activo."""
    if variant in PERTURBATIONS:
        return perturb_samples(table, PERTURBATIONS[variant], asset)
    _require(variant in VARIANTS, "La variante de lectura no está declarada")
    return ablate_samples(table, variant)
