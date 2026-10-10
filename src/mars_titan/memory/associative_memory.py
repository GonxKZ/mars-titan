"""Memoria asociativa lineal escrita con resultados maduros, por regla delta, proximal o kalman.

Es el comparador B6 de la matriz de experimentos: una matriz A de clave por valor que
solo cambia al aplicar resultados ya maduros. No sustituye la memoria neuronal de
Titans-MAC, cuya sorpresa asociativa usa entradas disponibles, ni el banco episódico,
que conserva episodios recuperables. Las fórmulas, cotas y contraejemplos de las reglas
delta y proximal proceden de `docs/research/memory-mathematics.md`. La regla kalman (PT3)
trata A como el estado de un modelo lineal dinámico con paseo aleatorio y ruido de cohorte
equicorrelado, y su derivación está en `docs/engineering/kalman-associative-memory.md`.

Cada escritura devuelve una memoria nueva y deja intacta la anterior. La lectura no
modifica estado. El orden de aplicación es canónico por disponibilidad de la etiqueta,
decisión e identificador, e independiente del orden de carga.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch

RULES = ("delta", "proximal", "kalman")
# Las claves vienen normalizadas por el codec en FP32. Este margen admite su redondeo.
KEY_NORM_TOLERANCE = 1e-6
WEIGHT_TOLERANCE = 1e-12
MAX_RATE = 1e6
DEFAULT_RATE = 0.1
MAX_SIZE = 512
MAX_COHORT = 8192


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finite(value, name, minimum, maximum):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} debe ser un número finito")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} debe pertenecer a [{minimum}, {maximum}]")
    return float(value)


def _size(value, name):
    if type(value) is not int or not 1 <= value <= MAX_SIZE:
        raise ValueError(f"{name} debe ser un entero entre 1 y {MAX_SIZE}")


def _matrix_digest(matrix):
    payload = matrix.detach().cpu().contiguous().numpy().astype("<f8", copy=False).tobytes()
    digest = hashlib.sha256(str(tuple(matrix.shape)).encode())
    digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True)
class KalmanNoise:
    """Varianzas del modelo lineal dinámico de la regla kalman.

    `process_noise` es q, la varianza que gana cada coordenada de A antes de cada cohorte
    por el paseo aleatorio. `observation_noise` es σ², la varianza de cada resultado.
    `cohort_correlation` es ϱ, la correlación común entre los resultados de una misma
    cohorte, que la deja valer como mucho 1/ϱ observaciones de su componente común.
    `prior_variance` es p0, la varianza inicial de cada coordenada de A. `huber_threshold`
    es c, el residuo estandarizado a partir del cual una fila pesa c/|z|, o None sin recorte.
    """

    process_noise: float
    observation_noise: float
    cohort_correlation: float
    prior_variance: float
    huber_threshold: float | None = None

    def __post_init__(self):
        values = dict(
            process_noise=_finite(self.process_noise, "El ruido de proceso q", 0.0, MAX_RATE),
            observation_noise=_finite(self.observation_noise, "La varianza σ²", 0.0, MAX_RATE),
            cohort_correlation=_finite(self.cohort_correlation, "La correlación ϱ", 0.0, 1.0),
            prior_variance=_finite(self.prior_variance, "La varianza inicial p0", 0.0, MAX_RATE),
        )
        # Con σ² = 0, ϱ = 1 o p0 = 0 la inversa de Σ o la factorización de P no existen.
        if (
            values["observation_noise"] == 0
            or values["cohort_correlation"] == 1
            or values["prior_variance"] == 0
        ):
            raise ValueError("La regla kalman exige σ² > 0, ϱ < 1 y p0 > 0")
        if self.huber_threshold is not None:
            threshold = _finite(self.huber_threshold, "El umbral de Huber", 0.0, MAX_RATE)
            if threshold == 0:
                raise ValueError("El umbral de Huber debe ser positivo")
            values["huber_threshold"] = threshold
        for name, value in values.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class AssociativeMemoryConfig:
    """`rate` es η y `forgetting` es λ. En la regla proximal la retención es ρ = 1 − λ.

    La regla kalman no usa η ni λ y declara sus varianzas en `kalman`. Sin ella el campo
    no existe en la identidad, así que las reglas delta y proximal conservan la suya.
    """

    rule: str
    key_size: int = 64
    value_size: int = 1
    rate: float = DEFAULT_RATE
    forgetting: float = 0.0
    kalman: KalmanNoise | None = None

    def __post_init__(self):
        if self.rule not in RULES:
            raise ValueError("La regla asociativa debe ser delta, proximal o kalman")
        if (self.rule == "kalman") != isinstance(self.kalman, KalmanNoise):
            raise ValueError("Solo la regla kalman declara sus varianzas y siempre las necesita")
        if self.rule == "kalman" and (self.rate != DEFAULT_RATE or self.forgetting != 0.0):
            raise ValueError("La regla kalman no usa rate ni forgetting")
        _size(self.key_size, "La dimensión de clave")
        _size(self.value_size, "La dimensión de valor")
        forgetting = _finite(self.forgetting, "El olvido", 0.0, 1.0)
        rate = _finite(self.rate, "La tasa", 0.0, MAX_RATE)
        # Con ‖k‖ ≤ 1, η ≤ 2 − λ basta para no expandir diferencias en ninguna escritura.
        if self.rule == "delta" and rate > 2.0 - forgetting:
            raise ValueError("La regla delta exige η ≤ 2 − λ para no expandir el estado")
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "forgetting", forgetting)

    @property
    def retention(self):
        return 1.0 - self.forgetting

    def contraction_bound(self):
        """Cota de ‖D⁺‖/‖D‖ para dos estados con las mismas entradas y ‖k‖ ≤ 1.

        Delta: máximo de |1 − λ − η‖k‖²| sobre ‖k‖² ∈ [0, 1], en norma espectral.
        Proximal: ρ/(1 + η λ_min(G)) ≤ ρ en Frobenius. No acota el estado absoluto.
        """
        if self.rule == "kalman":
            raise ValueError("La regla kalman no declara una cota de contracción")
        if self.rule == "delta":
            return max(abs(self.retention), abs(self.retention - self.rate))
        return self.retention

    def identity(self):
        fields = asdict(self)
        kalman = fields.pop("kalman")
        common = dict(
            schema_version=1,
            kind="mature_associative_memory",
            dtype="float64",
            device="cpu",
            key_constraint="l2_norm_at_most_1",
            key_norm_tolerance=KEY_NORM_TOLERANCE,
            order="label_available_at_then_decision_at_then_id",
            write_cursor="strictly_increasing_canonical_key",
        )
        if self.rule != "kalman":
            return dict(
                fields,
                **common,
                empty_cohort="unchanged_without_forgetting",
                update="sequential_rank_one"
                if self.rule == "delta"
                else "cohort_cholesky_lapack_without_inverse",
                weights=None if self.rule == "delta" else "declared_nonnegative_sum_one",
            )
        del fields["rate"], fields["forgetting"]
        return dict(
            fields,
            **common,
            kalman=kalman,
            empty_cohort="unchanged_without_process_noise",
            update="cohort_information_form_cholesky_without_inverse",
            weights=None
            if self.kalman.huber_threshold is None
            else "huber_one_step_on_prior_standardized_residual",
            covariance="shared_by_value_columns",
            cohort_noise="equicorrelated_closed_form_inverse",
            read_variance="k_P_k_plus_sigma2",
            source="post_titans_pt3_v1",
        )

    def fingerprint(self):
        return hashlib.sha256(_canonical(self.identity()).encode()).hexdigest()


@dataclass(frozen=True)
class MatureFeedback:
    """Resultados maduros de un evento: claves estables y valores disponibles.

    `available_at` es el instante desde el que la etiqueta puede usarse. Las claves
    deben proceder del mismo espacio estable con el que se leerá la memoria.
    """

    ids: torch.Tensor
    decision_at: torch.Tensor
    available_at: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    weights: torch.Tensor | None = None


def _check_vector(value, name, size, dtype):
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != dtype
        or value.layout != torch.strided
        or value.shape != (size,)
    ):
        raise ValueError(f"{name} necesita un vector CPU de {size} elementos y tipo {dtype}")


def _check_matrix(value, name, rows, columns):
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float64
        or value.layout != torch.strided
        or value.shape != (rows, columns)
    ):
        raise ValueError(f"{name} necesita una matriz CPU FP64 de forma {(rows, columns)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contiene valores no finitos")


def _check_keys(keys, rows, size):
    _check_matrix(keys, "Las claves", rows, size)
    if rows and torch.linalg.vector_norm(keys, dim=1).max() > 1.0 + KEY_NORM_TOLERANCE:
        raise ValueError("Las claves deben tener norma euclídea no mayor que uno")


def _check_covariance(covariance, size):
    _check_matrix(covariance, "La covarianza", size, size)
    if not torch.equal(covariance, covariance.T):
        raise ValueError("La covarianza debe ser simétrica")
    if int(torch.linalg.cholesky_ex(covariance).info) != 0:
        raise ValueError("La covarianza debe ser definida positiva en FP64")


class AssociativeMemory:
    """Matriz A de clave por valor en FP64 y CPU, con escrituras que devuelven copias.

    Con la regla kalman el estado incluye además la covarianza P de A, compartida por todas
    las columnas de valor y con valor inicial p0 I.
    """

    def __init__(self, config, matrix=None, *, writes=0, cursor=None, covariance=None):
        if not isinstance(config, AssociativeMemoryConfig):
            raise ValueError("La memoria asociativa necesita su configuración identificada")
        shape = (config.key_size, config.value_size)
        if matrix is None:
            matrix = torch.zeros(shape, dtype=torch.float64)
        _check_matrix(matrix, "El estado asociativo", *shape)
        if config.rule != "kalman" and covariance is not None:
            raise ValueError("Solo la regla kalman conserva una covarianza")
        if config.rule == "kalman":
            if covariance is None:
                covariance = config.kalman.prior_variance * torch.eye(
                    config.key_size, dtype=torch.float64
                )
            _check_covariance(covariance, config.key_size)
            covariance = covariance.detach().clone()
        if type(writes) is not int or writes < 0:
            raise ValueError("El contador de escrituras debe ser un entero no negativo")
        if cursor is not None and (
            not isinstance(cursor, tuple)
            or len(cursor) != 3
            or any(type(item) is not int for item in cursor)
        ):
            raise ValueError("El cursor necesita disponibilidad, decisión e identificador")
        if (cursor is None) != (writes == 0):
            raise ValueError("El cursor y el contador de escrituras no son coherentes")
        self.config = config
        self._matrix = matrix.detach().clone()
        self._covariance = covariance
        self.writes = writes
        self.cursor = cursor

    @property
    def matrix(self):
        return self._matrix.clone()

    @property
    def covariance(self):
        """Copia de P con la regla kalman. None con las reglas delta y proximal."""
        return None if self._covariance is None else self._covariance.clone()

    def read(self, keys):
        """Devolver `keys @ A`, es decir Aᵀk por fila, de forma [b, valor] y sin efectos."""
        rows = keys.shape[0] if isinstance(keys, torch.Tensor) and keys.ndim == 2 else -1
        if not 1 <= rows <= MAX_COHORT:
            raise ValueError(f"La lectura necesita entre 1 y {MAX_COHORT} claves")
        _check_keys(keys, rows, self.config.key_size)
        return keys @ self._matrix

    def variance(self, keys):
        """Varianza predictiva kᵀPk + σ² de un resultado nuevo por fila, sin efectos.

        No suma el q del paseo aleatorio que precederá a la siguiente cohorte, así que
        describe la incertidumbre con el estado tal como queda tras la última escritura.
        """
        if self.config.rule != "kalman":
            raise ValueError("Solo la regla kalman tiene varianza predictiva")
        rows = keys.shape[0] if isinstance(keys, torch.Tensor) and keys.ndim == 2 else -1
        if not 1 <= rows <= MAX_COHORT:
            raise ValueError(f"La lectura necesita entre 1 y {MAX_COHORT} claves")
        _check_keys(keys, rows, self.config.key_size)
        quadratic = ((keys @ self._covariance) * keys).sum(dim=1)
        return quadratic + self.config.kalman.observation_noise

    def write(self, feedback, *, cutoff):
        """Aplicar los resultados maduros de un evento y devolver la memoria siguiente."""
        rows, order = self._validate(feedback, cutoff)
        if rows == 0:
            return self
        index = torch.tensor(order, dtype=torch.int64)
        keys = feedback.keys.index_select(0, index)
        values = feedback.values.index_select(0, index)
        covariance = None
        if self.config.rule == "delta":
            matrix = self._delta(keys, values)
        elif self.config.rule == "proximal":
            matrix = self._proximal(keys, values, feedback.weights.index_select(0, index))
        else:
            matrix, covariance = self._kalman(keys, values)
        if not torch.isfinite(matrix).all():
            raise ValueError("La escritura produjo un estado no finito")
        last = order[-1]
        cursor = (
            int(feedback.available_at[last]),
            int(feedback.decision_at[last]),
            int(feedback.ids[last]),
        )
        return AssociativeMemory(
            self.config, matrix, writes=self.writes + rows, cursor=cursor, covariance=covariance
        )

    def _validate(self, feedback, cutoff):
        if not isinstance(feedback, MatureFeedback) or type(cutoff) is not int:
            raise ValueError("La escritura necesita resultados maduros y un corte entero")
        rows = feedback.ids.shape[0] if isinstance(feedback.ids, torch.Tensor) else -1
        if not 0 <= rows <= MAX_COHORT:
            raise ValueError(f"Una escritura admite como máximo {MAX_COHORT} resultados")
        for name in ("ids", "decision_at", "available_at"):
            _check_vector(getattr(feedback, name), name, rows, torch.int64)
        _check_keys(feedback.keys, rows, self.config.key_size)
        _check_matrix(feedback.values, "Los valores", rows, self.config.value_size)
        proximal = self.config.rule == "proximal"
        if (feedback.weights is not None) != (proximal and rows > 0):
            raise ValueError("Solo la regla proximal declara pesos y solo con resultados")
        if rows == 0:
            return 0, []
        ids = feedback.ids.tolist()
        decisions = feedback.decision_at.tolist()
        available = feedback.available_at.tolist()
        if min(ids) < 1 or len(set(ids)) != rows:
            raise ValueError("Los identificadores deben ser positivos y únicos en la escritura")
        if any(not d < a <= cutoff for d, a in zip(decisions, available, strict=True)):
            raise ValueError("Cada etiqueta debe madurar después de su decisión y antes del corte")
        if proximal:
            _check_vector(feedback.weights, "Los pesos", rows, torch.float64)
            weights = feedback.weights
            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("Los pesos de la cohorte deben ser finitos y no negativos")
            if abs(float(weights.sum()) - 1.0) > WEIGHT_TOLERANCE:
                raise ValueError("Los pesos de la cohorte deben sumar uno")
        order = sorted(range(rows), key=lambda i: (available[i], decisions[i], ids[i]))
        first = order[0]
        if self.cursor is not None and (available[first], decisions[first], ids[first]) <= (
            self.cursor
        ):
            raise ValueError("El resultado ya se aplicó o llega fuera del orden canónico")
        return rows, order

    def _delta(self, keys, values):
        retention, rate = self.config.retention, self.config.rate
        if rate * float(torch.linalg.vector_norm(keys, dim=1).max()) ** 2 > 2.0 - (
            self.config.forgetting
        ):
            raise ValueError("Una clave incumple η‖k‖² ≤ 2 − λ")
        matrix = self._matrix.clone()
        for key, value in zip(keys, values, strict=True):
            residual = value - key @ matrix
            matrix = retention * matrix + rate * torch.outer(key, residual)
        return matrix

    def _proximal(self, keys, values, weights):
        weighted = keys * weights.unsqueeze(1)
        gram = weighted.T @ keys
        cross = weighted.T @ values
        system = torch.eye(self.config.key_size, dtype=torch.float64) + self.config.rate * gram
        factor, info = torch.linalg.cholesky_ex(system)
        if int(info) != 0:
            raise ValueError("El sistema proximal no es definido positivo en FP64")
        right = self.config.retention * self._matrix + self.config.rate * cross
        return torch.cholesky_solve(right, factor)

    def _kalman(self, keys, values):
        """Actualización en forma de información de una cohorte con ruido equicorrelado.

        Con P⁻ = P + qI = CCᵀ y M = KᵀΣ⁻¹K, P⁺ = C(I + CᵀMC)⁻¹Cᵀ y A⁺ = A + P⁺KᵀΣ⁻¹(Y − KA).
        Σ⁻¹ = (I − g11ᵀ)/(σ²(1 − ϱ)) con g = ϱ/(1 − ϱ + nϱ) no se forma nunca: basta con
        KᵀK y Kᵀ1. Los pesos de Huber entran como W^{1/2}Σ⁻¹W^{1/2}, que conserva esa forma.
        """
        noise = self.config.kalman
        rows, size = keys.shape
        identity = torch.eye(size, dtype=torch.float64)
        prior = self._covariance + noise.process_noise * identity
        factor, info = torch.linalg.cholesky_ex(prior)
        if int(info) != 0:
            raise ValueError("La covarianza previa no es definida positiva en FP64")
        residual = values - keys @ self._matrix
        weights = torch.ones(rows, dtype=torch.float64)
        if noise.huber_threshold is not None:
            # Un solo paso de pesos de Huber, ψ(z)/z = min(1, c/|z|), con el residuo
            # estandarizado por la varianza predictiva previa de cada fila.
            spread = (((keys @ prior) * keys).sum(dim=1) + noise.observation_noise).sqrt()
            score = torch.linalg.vector_norm(residual, dim=1) / spread
            weights = torch.where(
                score > noise.huber_threshold,
                noise.huber_threshold / score.clamp(min=noise.huber_threshold),
                1.0,
            )
        root = weights.sqrt()
        scale = 1.0 / (noise.observation_noise * (1.0 - noise.cohort_correlation))
        shrink = noise.cohort_correlation / (
            1.0 - noise.cohort_correlation + rows * noise.cohort_correlation
        )
        weighted = keys * weights.unsqueeze(1)
        common = keys.T @ root
        information = scale * (keys.T @ weighted - shrink * torch.outer(common, common))
        evidence = scale * (weighted.T @ residual - shrink * torch.outer(common, root @ residual))
        system = identity + factor.T @ information @ factor
        lower, info = torch.linalg.cholesky_ex(system)
        if int(info) != 0:
            raise ValueError("El sistema de información no es definido positivo en FP64")
        transport = torch.linalg.solve_triangular(lower, factor.T, upper=False)
        covariance = transport.T @ transport
        # TᵀT es simétrica en aritmética exacta. La media con su traspuesta la hace exacta.
        covariance = 0.5 * (covariance + covariance.T)
        matrix = self._matrix + transport.T @ (transport @ evidence)
        if not torch.isfinite(covariance).all():
            raise ValueError("La escritura produjo una covarianza no finita")
        return matrix, covariance

    def export(self):
        payload = dict(
            schema_version=1,
            configuration=self.config.identity(),
            matrix=self.matrix,
            matrix_sha256=_matrix_digest(self._matrix),
            writes=self.writes,
            cursor=list(self.cursor) if self.cursor is not None else None,
        )
        if self._covariance is not None:
            payload.update(
                covariance=self.covariance, covariance_sha256=_matrix_digest(self._covariance)
            )
        return payload

    @classmethod
    def restore(cls, config, payload):
        if not isinstance(config, AssociativeMemoryConfig) or not isinstance(payload, dict):
            raise ValueError("La recuperación necesita configuración y estado exportado")
        expected = {"schema_version", "configuration", "matrix", "matrix_sha256", "writes"}
        if config.rule == "kalman":
            expected |= {"covariance", "covariance_sha256"}
        if set(payload) != expected | {"cursor"} or payload["schema_version"] != 1:
            raise ValueError("El estado exportado no conserva todos sus campos")
        if _canonical(payload["configuration"]) != _canonical(config.identity()):
            raise ValueError("El estado pertenece a otra configuración asociativa")
        matrix = payload["matrix"]
        _check_matrix(matrix, "El estado exportado", config.key_size, config.value_size)
        if _matrix_digest(matrix) != payload["matrix_sha256"]:
            raise ValueError("La matriz no coincide con su huella")
        covariance = payload.get("covariance")
        if config.rule == "kalman":
            _check_covariance(covariance, config.key_size)
            if _matrix_digest(covariance) != payload["covariance_sha256"]:
                raise ValueError("La covarianza no coincide con su huella")
        cursor = payload["cursor"]
        return cls(
            config,
            matrix,
            writes=payload["writes"],
            cursor=tuple(cursor) if isinstance(cursor, list) else cursor,
            covariance=covariance,
        )


CORRECTION_KEYS = ("codec", "constant")


@dataclass(frozen=True)
class MatureCorrection:
    """Corrección escalar de la predicción del núcleo leída de A, el componente B6.

    Con `codec`, la clave es la entrada del codec del banco normalizada en L2 y FP64. Con
    `constant`, la única clave es 1 y A se reduce a un corrector de sesgo, el control que
    permite descartar la dependencia de la clave. El valor escrito es la etiqueta madura menos
    la predicción del núcleo, es decir, la emitida antes de sumar la corrección.
    """

    memory: AssociativeMemoryConfig
    key: str = "codec"

    def __post_init__(self):
        if not isinstance(self.memory, AssociativeMemoryConfig) or self.key not in CORRECTION_KEYS:
            raise ValueError("La corrección necesita su memoria y una clave codec o constant")
        if self.memory.value_size != 1 or self.memory.key_size != (
            64 if self.key == "codec" else 1
        ):
            raise ValueError("La corrección escalar usa claves de 64 o 1 coordenadas y valor 1")

    def identity(self):
        return dict(
            schema_version=1,
            kind="mature_scalar_correction",
            memory=self.memory.identity(),
            key=self.key,
            key_normalization="l2_fp64_of_codec_key_inputs"
            if self.key == "codec"
            else "constant_one",
            value="mature_label_minus_core_prediction",
            read="matrix_of_previous_generation_for_the_whole_event",
            ids="write_count_plus_offset_in_native_canonical_outcome_order",
            proximal_weights="uniform_over_event_cohort",
        )

    def keys(self, key_inputs):
        """Claves FP64 [filas, d] desde las entradas FP32 del codec, en el mismo orden."""
        values = torch.as_tensor(key_inputs)
        if values.ndim != 2 or values.shape[1] != 64 or values.dtype != torch.float32:
            raise ValueError("La corrección necesita las entradas FP32 de 64 coordenadas del codec")
        if self.key == "constant":
            return torch.ones((values.shape[0], 1), dtype=torch.float64)
        values = values.to(dtype=torch.float64, device="cpu")
        norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)
        if not torch.isfinite(values).all() or (norms == 0).any():
            raise ValueError("La clave del codec necesita valores finitos y norma positiva")
        return values / norms

    def feedback(self, *, ids, decision_at, available_at, keys, values):
        """Resultados maduros de un evento con pesos uniformes si la regla es proximal."""
        rows = len(ids)
        weights = (
            torch.full((rows,), 1.0 / rows, dtype=torch.float64)
            if self.memory.rule == "proximal" and rows
            else None
        )
        return MatureFeedback(
            ids=torch.tensor(ids, dtype=torch.int64),
            decision_at=torch.tensor(decision_at, dtype=torch.int64),
            available_at=torch.tensor(available_at, dtype=torch.int64),
            keys=keys,
            values=torch.tensor(values, dtype=torch.float64).reshape(rows, 1),
            weights=weights,
        )
