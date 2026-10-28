"""Presupuesto y comprobación durable del diagnóstico E2 con lector regional."""

from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot

from . import regional_episode_trace
from .spaced_utility import _require
from .spaced_utility_state import _fields

KIND = "regional_spaced_session_v1"


def signature(reader):
    return dict(
        schema_version=1,
        kind=KIND,
        reader=reader.config.identity(),
        suppression="same_selected_episode_removed_from_all_views_without_refill",
        work="diagnostic_counts_and_upper_operator_updates_separate_from_ordinary_trace",
    )


def _estimated_bytes(reader, flows, episodes, block_rows):
    support = min(episodes, reader.config.neighbors)
    trace = regional_episode_trace.estimated_bytes(reader, flows, episodes, block_rows)
    transient = reader.estimated_bytes(
        min(len(flows), block_rows), episodes, differentiable=False, refinements=support + 1
    )
    # Escalares y referencias Python de la cohorte completa, además de los tensores.
    # Es una estimación conservadora de trabajo, no una certificación del asignador.
    emissions = len(flows) * (1024 + 128 * support)
    return trace + transient + emissions


def _work(reader, rows, episodes, block_rows, estimate, *, warmup):
    support = 0 if warmup else min(episodes, reader.config.neighbors)
    diagnostic_rows = rows * support
    transformed_rows = diagnostic_rows if support > 1 else 0
    bridge = reader.config.evidence_bridge
    regions = len(bridge.regions)
    if bridge.hierarchy is not None:
        updates = bridge.hierarchy.node_updates(regions, support) * rows if support > 1 else 0
    elif bridge.recurrence is not None:
        updates = bridge.recurrence.node_updates(regions, support) * rows if support > 1 else 0
    else:
        updates = transformed_rows * regions * bridge.cycles
    return dict(
        schema_version=1,
        kind=KIND,
        rows=rows,
        support=support,
        diagnostic_refinements=diagnostic_rows,
        extra_pool_projection_rows=rows if support > 1 else 0,
        extra_pool_projection_calls=(rows + block_rows - 1) // block_rows if support > 1 else 0,
        extra_projected_episode_rows=rows * support if support > 1 else 0,
        diagnostic_initial_region_views=transformed_rows * regions,
        diagnostic_message_updates_upper_bound=updates,
        reader_and_trace_bytes_estimate=estimate,
        update_count_scope="upper_bound_not_observed_adaptive_depth",
    )


def preflight(binding, rows, snapshot, *, context_id, warmup, block_rows):
    reader = binding.consumer.readout
    _require(type(warmup) is bool and 1 <= len(rows) <= 8192, "E2 necesita una cohorte acotada")
    _require(
        type(block_rows) is int and 1 <= block_rows <= reader.config.max_batch,
        "El bloque E2 regional supera el lote admitido por el lector",
    )
    if warmup:
        _require(snapshot is None, "El calentamiento E2 regional no consulta episodios")
        return _work(reader, len(rows), 0, block_rows, 0, warmup=True)
    _require(type(snapshot) is EpisodeSnapshot, "E2 regional necesita un snapshot explícito")
    snapshot.check(
        codec_id=reader.config.codec_id,
        context_id=context_id,
        cutoff=rows[0].prediction_at,
        dtype=binding.dtype,
        device=binding.device,
    )
    support = min(snapshot.count, reader.config.neighbors)
    reader.preflight_node_updates(support + 1, snapshot.count)
    estimate = _estimated_bytes(
        reader, tuple(row.flow_id for row in rows), snapshot.count, block_rows
    )
    _require(
        estimate <= reader.config.max_working_bytes,
        "La cohorte E2 regional supera el presupuesto conjunto antes de MAC",
    )
    return _work(reader, len(rows), snapshot.count, block_rows, estimate, warmup=False)


def verify(session, bundle, event, *, event_kind=None):
    control = session._read(bundle["control"], "control")
    _require(type(control) is dict, "El control E2 regional debe ser un objeto")
    count = control.get("observations")
    _require(type(count) is int and 0 <= count <= 8192, "El control E2 no conserva sus filas")
    flows = tuple(row["flow_id"] for row in event["emissions"])
    # Antes de confirmar, las emisiones distinguen decisión y calentamiento. Después
    # se comprueba también el tipo real del recibo nativo, no un campo autodeclarado.
    kind = (
        ("decision" if flows else "warmup" if count else "settlement")
        if event_kind is None
        else event_kind
    )
    _require(
        type(kind) is str and kind in ("decision", "warmup", "settlement"),
        "El tipo de evento E2 no es válido",
    )
    _require(
        len(flows) == count if kind == "decision" else not flows,
        "Las emisiones E2 regionales no corresponden a las observaciones del evento",
    )
    reader = session.consumer.readout
    session._binding.verify_regional_control(
        control, flows, bundle["cutoff"], session.block_rows, event_kind=kind
    )
    if kind == "settlement":
        _require("spaced_regional" not in control, "La liquidación E2 no inventa lecturas")
        return
    episodes = control["regional_reads"]["episodes"] if kind == "decision" else 0
    estimate = _estimated_bytes(reader, flows, episodes, session.block_rows) if flows else 0
    _require(
        estimate <= reader.config.max_working_bytes,
        "El control E2 regional supera el presupuesto conjunto declarado",
    )
    reader.preflight_node_updates(1 + min(episodes, reader.config.neighbors), episodes)
    expected = _work(reader, count, episodes, session.block_rows, estimate, warmup=kind == "warmup")
    raw = control.get("spaced_regional")
    _fields(raw, expected, "El trabajo E2 regional no conserva sus campos")
    _require(
        all(
            type(raw[name]) is type(value) and raw[name] == value
            for name, value in expected.items()
        ),
        "Los contadores E2 regionales contradicen el soporte, los bloques o su presupuesto",
    )
