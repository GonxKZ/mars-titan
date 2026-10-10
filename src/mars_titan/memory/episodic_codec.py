"""Codec fijo de observaciones preparadas, sin objetivos ni pesos del predictor."""

import functools
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from numbers import Real

import numpy as np

from mars_titan.data.input_policy import MODALITIES
from mars_titan.models.titans.financial_inputs import CPUDecisionBatch, FinancialInputSpec

WIDTH = 64
BLOCKS = (*MODALITIES, "presence")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _positive(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} debe ser un número finito positivo")
    return float(value)


def _immutable(values):
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


def _norm(values):
    return np.sqrt(np.einsum("bi,bi->b", values, values, optimize=False))


def _declared_order(block, transposed):
    """Proyección con el orden de sumas que reproduce `project_codec_block`, en NumPy.

    Dos acumuladores por salida, para las entradas pares e impares. Cada bloque de ocho
    entradas se recorre del par 6-7 al 0-1, el resto por pares en orden creciente, y al final
    se suman los dos acumuladores. Es el orden de `einsum("bi,oi->bo", optimize=False)` en la
    línea base x86-64 de NumPy 2.5. Multiplicar y sumar por separado con NumPy no fusiona las
    operaciones, así que esta función sirve de referencia en cualquier plataforma.
    """
    rows, width = block.shape
    even = np.zeros((rows, transposed.shape[1]))
    odd = np.zeros_like(even)
    index = 0
    while width - index >= 8:
        for offset in (6, 4, 2, 0):
            even = block[:, index + offset, None] * transposed[index + offset] + even
            odd = block[:, index + offset + 1, None] * transposed[index + offset + 1] + odd
        index += 8
    while index < width:
        even = block[:, index, None] * transposed[index] + even
        if index + 1 < width:
            odd = block[:, index + 1, None] * transposed[index + 1] + odd
        index += 2
    return even + odd


@functools.cache
def _einsum_follows_declared_order():
    """Comprobar una vez por proceso que el einsum instalado suma en el orden del núcleo.

    Si NumPy cambiara su núcleo (otra línea base, FMA o más carriles), el enlace nativo y
    einsum darían bits distintos. En ese caso el codec sigue usando einsum.
    """
    generator = np.random.Generator(np.random.PCG64(20261010))
    for width in (1, 2, 3, 7, 8, 9, 15, 16, 17, 33):
        scale = np.ldexp(1.0, generator.integers(-40, 40, size=(5, width)))
        block = generator.standard_normal((5, width)) * scale
        weights = generator.standard_normal((11, width))
        expected = _declared_order(block, np.ascontiguousarray(weights.T))
        actual = np.einsum("bi,oi->bo", block, weights, optimize=False)
        if not np.array_equal(actual.view(np.uint64), expected.view(np.uint64)):
            return False
    return True


def projection_backend(native):
    """Implementación de la proyección que usará `encode` con este enlace nativo.

    El núcleo nativo solo sustituye a einsum si el enlace lo incluye y el einsum de esta
    instalación sigue el orden declarado, es decir, cuando los dos dan los mismos bits.
    """
    if native is None or not hasattr(native, "project_codec_block"):
        return "numpy_einsum"
    return "native" if _einsum_follows_declared_order() else "numpy_einsum"


@dataclass(frozen=True)
class EpisodicEncoding:
    """Claves de entrada y valores FP32 propios, sin grafo ni supervisión."""

    key_inputs: np.ndarray
    values: np.ndarray
    input_l2: np.ndarray
    normalized_l2: np.ndarray
    key_contribution_l2: np.ndarray
    value_contribution_l2: np.ndarray
    block_names: tuple[str, ...]
    flow_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    prediction_at: tuple[int, ...]
    input_available_at: tuple[int, ...]
    input_digest: str
    codec_id: str
    estimated_peak_bytes: int
    _signature: tuple
    _digest: str

    def _arrays(self):
        return (
            self.key_inputs,
            self.values,
            self.input_l2,
            self.normalized_l2,
            self.key_contribution_l2,
            self.value_contribution_l2,
        )

    def _content_digest(self):
        digest = hashlib.sha256(
            _canonical(
                (
                    self.flow_ids,
                    self.sample_ids,
                    self.prediction_at,
                    self.input_available_at,
                    self.input_digest,
                    self.codec_id,
                    self.block_names,
                )
            ).encode()
        )
        for value in self._arrays():
            digest.update(value.tobytes())
        return digest.hexdigest()

    def verify(self):
        """Rechazar cambios en metadatos, geometría o bytes antes de guardar episodios."""
        signature = tuple((value.shape, value.dtype.str, value.strides) for value in self._arrays())
        if signature != self._signature or any(value.flags.writeable for value in self._arrays()):
            raise ValueError("El contrato de la codificación episódica ha cambiado")
        if self._content_digest() != self._digest:
            raise ValueError("La huella de la codificación episódica no coincide")


@dataclass(frozen=True, init=False)
class FrozenEpisodeCodec:
    """Proyección fija 64×64 de bloques normalizados sin ajustar otras observaciones.

    key_inputs incluye una coordenada constante y todavía no tiene norma
    unitaria. El banco aplica su normalización de clave. Las contribuciones
    por bloque son diagnósticos geométricos, no importancia del pronóstico.
    """

    _identity_json: str
    _projection: np.ndarray
    _projection_buffer: np.ndarray
    _transposed: np.ndarray
    _transposed_buffer: np.ndarray
    _input_widths: tuple[int, ...]
    _modality_weights: tuple[float, ...]
    _presence_weight: float
    _key_constant: float
    input_contract_id: str
    max_buffer_bytes: int

    def __init__(
        self,
        specification,
        *,
        seed=1729,
        modality_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
        presence_weight=1.0,
        key_constant=1.0,
        max_buffer_bytes=64 * 1024**2,
    ):
        if not isinstance(specification, FinancialInputSpec):
            raise ValueError("El codec necesita el contrato de entradas financieras")
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("La semilla debe ser un entero uint64")
        if type(max_buffer_bytes) is not int or not 1 <= max_buffer_bytes <= 64 * 1024**2:
            raise ValueError("El presupuesto del codec debe estar entre 1 byte y 64 MiB")
        if not isinstance(modality_weights, (list, tuple)) or len(modality_weights) != len(
            MODALITIES
        ):
            raise ValueError("Se necesita un peso por modalidad")
        weights = tuple(_positive(value, "peso de modalidad") for value in modality_weights)
        presence_weight = _positive(presence_weight, "peso de presencia")
        key_constant = _positive(key_constant, "constante de clave")
        with np.errstate(over="raise", under="ignore"):
            try:
                constant = np.float32(key_constant)
            except FloatingPointError as error:
                raise ValueError("La constante de clave no cabe en FP32") from error
        if not np.isfinite(constant) or constant <= 0:
            raise ValueError("La constante de clave necesita un FP32 positivo")
        widths = tuple(
            specification.dimensions[name] * (specification.context if name == "prices" else 1)
            for name in MODALITIES
        )
        input_contract_id = specification.fingerprint()
        dimensions = sum(widths) + len(MODALITIES)
        matrix_bytes = (2 * WIDTH - 1) * dimensions * 8
        if matrix_bytes * 3 + matrix_bytes // 8 + 65536 > max_buffer_bytes:
            raise ValueError("La proyección supera el presupuesto de memoria del codec")
        generator = np.random.Generator(np.random.PCG64(seed))
        signs = generator.integers(0, 2, size=(2 * WIDTH - 1, dimensions), dtype=np.int8)
        projection = np.empty(signs.shape, dtype=np.float64)
        np.multiply(signs, 2, out=projection)
        projection -= 1
        projection[: WIDTH - 1] /= math.sqrt(WIDTH - 1)
        projection[WIDTH - 1 :] /= math.sqrt(WIDTH)
        projection = _immutable(projection)
        checksum = hashlib.sha256(projection).hexdigest()
        # El núcleo nativo lee cada entrada como una fila contigua de salidas. Liberar los
        # signos antes de copiarla deja el pico en tres matrices, como prevé el presupuesto.
        del signs
        transposed = _immutable(np.ascontiguousarray(projection.T))
        identity = dict(
            schema_version=1,
            recipe="frozen_modal_projection_v1",
            input_specification=specification.identity(),
            input_contract_id=input_contract_id,
            seed=seed,
            modality_weights=list(weights),
            presence_weight=presence_weight,
            presence_scale="one_over_sqrt_five",
            key_constant=key_constant,
            key_width=WIDTH,
            value_width=WIDTH,
            key_layout="63_projected_then_constant",
            key_normalization="delegated_to_native_bank",
            input_normalization="row_l2_per_modality",
            projection="rademacher_einsum_fp64",
            projection_shape=list(projection.shape),
            projection_sha256=checksum,
            compute_dtype="float64",
            stored_dtype="float32",
            numpy_version=np.__version__,
            max_buffer_bytes=max_buffer_bytes,
        )
        for name, value in dict(
            _identity_json=_canonical(identity),
            _projection=projection,
            _projection_buffer=projection.base,
            _transposed=transposed,
            _transposed_buffer=transposed.base,
            _input_widths=widths,
            _modality_weights=weights,
            _presence_weight=presence_weight,
            _key_constant=key_constant,
            input_contract_id=input_contract_id,
            max_buffer_bytes=max_buffer_bytes,
        ).items():
            object.__setattr__(self, name, value)

    def identity(self):
        return json.loads(self._identity_json)

    def fingerprint(self):
        return hashlib.sha256(self._identity_json.encode()).hexdigest()

    @classmethod
    def from_identity(cls, identity, specification):
        """Reconstruir y exigir la misma receta, tipos JSON, versión y proyección."""
        if not isinstance(identity, dict):
            raise ValueError("Falta la identidad del codec")
        try:
            result = cls(
                specification,
                seed=identity["seed"],
                modality_weights=identity["modality_weights"],
                presence_weight=identity["presence_weight"],
                key_constant=identity["key_constant"],
                max_buffer_bytes=identity["max_buffer_bytes"],
            )
            if _canonical(identity) != result._identity_json:
                raise ValueError("La identidad del codec o sus tipos han cambiado")
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError("La identidad del codec está incompleta") from error
        return result

    def _verify_projection(self):
        """Exigir la misma memoria de solo lectura cuya huella se calculó al construir el codec.

        La proyección es una vista de un objeto bytes, que no se puede modificar, y NumPy no
        permite volver a marcarla como escribible. Sustituir el atributo cambia su base. Por eso
        basta con comprobar esa base en cada llamada, sin repetir el SHA-256 de la matriz.
        """
        shape = (2 * WIDTH - 1, sum(self._input_widths) + len(MODALITIES))
        for values, buffer, expected in (
            (self._projection, self._projection_buffer, shape),
            (self._transposed, self._transposed_buffer, shape[::-1]),
        ):
            if (
                values.base is not buffer
                or not isinstance(buffer.base, bytes)
                or values.shape != expected
                or values.dtype != np.float64
                or not values.flags.c_contiguous
                or values.flags.writeable
            ):
                raise ValueError("La proyección congelada ha cambiado")

    def encode(self, batch, *, native=None):
        """Consumir la vista CPU verificada antes de cualquier traslado al predictor.

        Con el enlace episódico nativo, la proyección de cada bloque usa su núcleo con el orden
        de einsum cuando `projection_backend` lo permite. El resultado es el mismo bit a bit.
        """
        if (
            not isinstance(batch, CPUDecisionBatch)
            or batch.input_contract_id != self.input_contract_id
        ):
            raise ValueError("La vista CPU pertenece a otro contrato")
        batch.verify()
        self._verify_projection()
        native_projection = projection_backend(native) == "native"
        size = len(batch.flow_ids)
        temporary = 3 * size * max(self._input_widths) * 8
        peak = (
            self._projection.nbytes
            + self._transposed.nbytes
            + 2 * sys.getsizeof(self._identity_json)
            + temporary
            + size * (4 * WIDTH * 8 + 512)
            + 65536
        )
        if peak > self.max_buffer_bytes:
            raise ValueError("El lote supera el presupuesto de memoria del codec")
        summed = np.zeros((size, 2 * WIDTH - 1), dtype=np.float64)
        input_l2 = np.empty((size, len(MODALITIES)), dtype=np.float64)
        normalized_l2 = np.empty_like(input_l2)
        key_contributions = np.empty((size, len(BLOCKS)), dtype=np.float64)
        value_contributions = np.empty_like(key_contributions)
        start = 0
        for index, name in enumerate(BLOCKS):
            if name == "presence":
                block = batch.presence.astype(np.float64) * (self._presence_weight / math.sqrt(5))
            else:
                block = batch.inputs[name].reshape(size, -1).astype(np.float64)
                norms = _norm(block)
                input_l2[:, index] = norms
                np.divide(block, norms[:, None], out=block, where=norms[:, None] > 0)
                block *= self._modality_weights[index]
                normalized_l2[:, index] = _norm(block)
            end = start + block.shape[1]
            if native_projection:
                contribution = native.project_codec_block(block, self._transposed[start:end])
            else:
                contribution = np.einsum(
                    "bi,oi->bo", block, self._projection[:, start:end], optimize=False
                )
            summed += contribution
            key_contributions[:, index] = _norm(contribution[:, : WIDTH - 1])
            value_contributions[:, index] = _norm(contribution[:, WIDTH - 1 :])
            start = end
            # Evitar que se solapen las salidas de dos proyecciones en bloques estrechos.
            del contribution
        try:
            with np.errstate(over="raise", invalid="raise"):
                keys = np.empty((size, WIDTH), dtype=np.float32)
                keys[:, : WIDTH - 1] = summed[:, : WIDTH - 1]
                keys[:, -1] = self._key_constant
                values = summed[:, WIDTH - 1 :].astype(np.float32)
        except FloatingPointError as error:
            raise ValueError("La codificación no cabe en FP32") from error
        del summed, block, norms
        arrays = tuple(
            _immutable(value)
            for value in (
                keys,
                values,
                input_l2,
                normalized_l2,
                key_contributions,
                value_contributions,
            )
        )
        if any(not np.isfinite(value).all() for value in arrays):
            raise ValueError("La codificación produjo un resultado no finito")
        signature = tuple((value.shape, value.dtype.str, value.strides) for value in arrays)
        result = EpisodicEncoding(
            *arrays,
            BLOCKS,
            batch.flow_ids,
            batch.sample_ids,
            batch.prediction_at,
            batch.input_available_at,
            batch.input_digest,
            self.fingerprint(),
            peak,
            signature,
            "",
        )
        object.__setattr__(result, "_digest", result._content_digest())
        return result
