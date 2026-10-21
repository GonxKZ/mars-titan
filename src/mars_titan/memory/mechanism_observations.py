"""Extraer rasgos antes de MAC desde las filas admitidas y el snapshot anterior."""

import hashlib
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np

from mars_titan.calibration.online_conformal import _instant
from mars_titan.data.input_policy import MODALITIES, masked_inputs, validate_historical_vectors
from mars_titan.data.price_windows import validate_price_window
from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot, _digest_id
from mars_titan.models.titans.financial_inputs import (
    FLOW_ID,
    HISTORICAL_START_US,
    MAX_INPUT_BYTES,
    _buffer,
)
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.models.titans.mechanism_observation import _FIXED, MechanismObservation
from mars_titan.models.titans.session_context import MARKETS

from .episodic_session import DecisionRow


def _sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _row_features(row, spec, cutoff):
    if (
        row.prediction_at != cutoff
        or row.sample_id != f"{row.flow_id}/{cutoff}"
        or _instant(row.input_available_at, "La entrada") > cutoff
    ):
        raise ValueError("La entrada procede del futuro o no pertenece al corte")
    if (
        type(row.inputs) is not MappingProxyType
        or set(row.inputs) != set(MODALITIES)
        or not isinstance(row.presence, np.ndarray)
        or row.presence.dtype != np.bool_
        or row.presence.shape != (5,)
        or not row.presence.flags.c_contiguous
        or row.presence.flags.writeable
        or _buffer(row.presence) is None
    ):
        raise ValueError("La observación necesita inputs y máscaras inmutables admitidos")
    metadata = {k: v for k, v in row.metadata().items() if k != "input_sha256"}
    metadata["input_contract_id"] = spec.fingerprint()
    digest = hashlib.sha256(canonical(metadata).encode())
    values, size = [], row.presence.nbytes
    for index, (name, width) in enumerate(spec.dimensions.items()):
        value = row.inputs[name]
        shape = (spec.context, width) if name == "prices" else (width,)
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.float32
            or value.shape != shape
            or not value.flags.c_contiguous
            or value.flags.writeable
            or _buffer(value) is None
            or not np.isfinite(value).all()
        ):
            raise ValueError("Una modalidad no conserva su forma, precisión, finitud o propiedad")
        digest.update(memoryview(value).cast("B"))
        size += value.nbytes
        if name == "prices":
            validate_price_window(value)
            # El canal de presencia de la ventana no es una magnitud de precio.
            value = value[..., :5]
        if row.presence[index]:
            # Las entradas admitidas son FP32 y cada vector está acotado. Sus cuadrados
            # y su suma caben en FP64, también para el mayor valor finito de FP32.
            rms = float(np.linalg.norm(value.astype(np.float64).ravel())) / math.sqrt(value.size)
            values.append(math.log1p(rms))
        else:
            values.append(0.0)
    digest.update(memoryview(row.presence).cast("B"))
    if digest.hexdigest() != row.input_sha256:
        raise ValueError("La huella de inputs no corresponde a la observación")
    if masked_inputs(spec.input_policy):
        validate_historical_vectors(
            {k: row.inputs[k][None] for k in MODALITIES[1:]},
            row.presence[None],
            spec.representation,
        )
    elif not row.presence.all():
        raise ValueError("La observación estricta requiere todas las modalidades")
    return values, size


def _corrections(values):
    if type(values) is not dict or any(
        not isinstance(key, str)
        or isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        for key, value in values.items()
    ):
        raise ValueError("La calibración necesita correcciones numéricas identificadas")
    try:
        ordered = [float(values[key]) for key in sorted(values)]
    except (OverflowError, ValueError) as error:
        raise ValueError("Una corrección no se puede representar en FP64") from error
    if not ordered or any(not math.isfinite(v) for v in ordered):
        raise ValueError("La calibración necesita correcciones finitas identificadas")
    # Dividir antes de sumar evita desbordar la media de correcciones finitas grandes.
    return [math.fsum(v / len(ordered) for v in ordered), max(abs(v) for v in ordered)]


def extract_observation(rows, *, consumer, state, flow_ids, cutoff, context_id, snapshot=None):
    """No consultar etiquetas, proyecciones episódicas ni el feedback del evento actual.

    La media de cada magnitud incluye todas las filas y asigna cero a las fuentes
    ausentes. La edad resume la disponibilidad agregada por fila, no la de cada
    concepto fundamental o macro. El vector no retiene los arrays de entrada.
    """
    if type(consumer) is not FrozenMechanismConsumer:
        raise ValueError("La extracción necesita el consumidor financiero identificado")
    consumer.verify()
    consumer._check_state(state)
    cutoff = _instant(cutoff, "La observación")
    _digest_id(context_id, "El contexto de la decisión")
    if state.cutoff is not None and state.cutoff >= cutoff:
        raise ValueError("La observación debe seguir al snapshot anterior")
    if type(rows) not in (tuple, list) or not 1 <= len(rows) <= 8192:
        raise ValueError("La observación necesita una cohorte acotada")
    if any(type(row) is not DecisionRow for row in rows) or type(flow_ids) not in (tuple, list):
        raise ValueError("La cohorte necesita filas y flujos tipados")
    if any(
        not isinstance(flow, str) or FLOW_ID.fullmatch(flow) is None
        for flow in (*flow_ids, *(row.flow_id for row in rows))
    ):
        raise ValueError("La cohorte necesita identidades válidas de mercado y activo")
    ordered = sorted(rows, key=lambda row: row.flow_id)
    flows = tuple(row.flow_id for row in ordered)
    consumer.predictor._check_flows(flows)
    if len(set(flow_ids)) != len(flow_ids) or tuple(sorted(flow_ids)) != flows:
        raise ValueError("La observación no contiene exactamente la cohorte declarada")
    entries = 0
    snapshot_id = None
    if snapshot is not None:
        if type(snapshot) is not EpisodeSnapshot or consumer.readout is None:
            raise ValueError("El snapshot necesita un lector y tipo admitidos")
        snapshot.verify()
        if (
            snapshot.cutoff != cutoff
            or snapshot.context_id != context_id
            or snapshot.codec_id != consumer.readout.config.codec_id
            or snapshot.device != consumer.predictor.head.weight.device
            or snapshot.dtype != consumer.predictor.head.weight.dtype
        ):
            raise ValueError("El snapshot no pertenece a la observación")
        entries, snapshot_id = snapshot.count, snapshot._digest
    spec = consumer.predictor.config.inputs
    if masked_inputs(spec.input_policy) and cutoff < HISTORICAL_START_US:
        raise ValueError("La observación histórica no puede preceder al año 2000")
    calculated, total = {}, 0
    for row in ordered:
        values, count = _row_features(row, spec, cutoff)
        total += count
        if total > MAX_INPUT_BYTES:
            raise ValueError("Los inputs de la observación superan 64 MiB")
        calculated[row.flow_id] = values
    groups = []
    for market in sorted({flow.split("/", 1)[0] for flow in flows}):
        selected = [r for r in ordered if r.flow_id.startswith(market + "/")]
        count = len(selected)
        oldest = min(r.input_available_at for r in selected)
        newest = max(r.input_available_at for r in selected)
        committed = state.context.committed_at[MARKETS.index(market)] if state.context else -1
        committed = None if committed < 0 else committed
        calibrated = state.calibration is not None and market in consumer._static["groups"]
        values = [math.log1p(count)]
        values.extend(float(sum(bool(r.presence[j]) for r in selected) / count) for j in range(5))
        values.extend(
            math.fsum(calculated[r.flow_id][j] / count for r in selected) for j in range(5)
        )
        values.extend(
            [
                math.log1p((cutoff - oldest) / 1_000_000),
                math.log1p(entries),
                float(committed is not None),
                0.0 if committed is None else math.log1p((cutoff - committed) / 86_400_000_000),
                float(calibrated),
                math.log1p(state.calibration.pending_rows) if calibrated else 0.0,
            ]
        )
        if calibrated:
            values.extend(
                _corrections(
                    {
                        key: item["correction"]
                        for key, item in consumer._static["groups"][market]["intervals"].items()
                    }
                )
            )
            values.extend(_corrections(state.calibration.corrections(market)))
        else:
            values.extend([0.0] * 4)
        groups.append(
            dict(
                market=market,
                rows=count,
                oldest_input_available_at=oldest,
                newest_input_available_at=newest,
                context_committed_at=committed,
                values=[float(v) for v in values],
            )
        )
    return MechanismObservation(
        dict(
            **_FIXED,
            consumer_id=consumer.model_id,
            parameter_id=consumer.predictor._parameter_id,
            input_contract_id=spec.fingerprint(),
            input_policy=spec.input_policy,
            context_id=context_id,
            producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            flow_ids_sha256=_sha(flows),
            rows_sha256=_sha([r.metadata() for r in ordered]),
            cutoff=cutoff,
            previous_cutoff=state.cutoff,
            snapshot_sha256=snapshot_id,
            bank_entries=entries,
            groups=groups,
        )
    )
