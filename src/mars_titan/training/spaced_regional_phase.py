"""Presupuesto y sumas escalares E2 de una fase sin conservar trazas regionales completas."""

from mars_titan.memory.spaced_regional_session import _work
from mars_titan.memory.spaced_utility import _require

SUMS = (
    "diagnostic_refinements",
    "extra_pool_projection_rows",
    "extra_pool_projection_calls",
    "extra_projected_episode_rows",
    "diagnostic_initial_region_views",
    "diagnostic_message_updates_upper_bound",
)


def plan(reader, batches, episodes, *, cutoff):
    _require(1 <= len(batches) <= 256, "La fase E2 necesita bloques acotados")
    flows = tuple(flow for batch in batches for flow in batch.flow_ids)
    _require(
        1 <= len(flows) <= 8192 and len(set(flows)) == len(flows),
        "La cohorte regional E2 necesita flujos distintos y acotados",
    )
    _require(
        all(at == cutoff for batch in batches for at in batch.prediction_at),
        "Los bloques regionales deben compartir el corte antes de preparar el padre",
    )
    block = max(len(batch.flow_ids) for batch in batches)
    _require(block <= reader.config.max_batch, "Un bloque regional supera el lote del lector")
    support = min(episodes, reader.config.neighbors)
    reader.preflight_node_updates(support + 1, episodes)
    transient = reader.estimated_bytes(block, episodes, refinements=support + 1)
    # La fase retiene puntos/IDs, no las trazas tensoriales completas de FinancialSession.
    # La estimación cubre el bloque mayor y las emisiones de la cohorte, no toda la RAM.
    estimate = transient + len(flows) * (1024 + 128 * support)
    _require(
        estimate <= reader.config.max_working_bytes,
        "La fase E2 regional supera el presupuesto conjunto antes de MAC",
    )
    work = _work(reader, len(flows), episodes, block, estimate, warmup=False)
    work["extra_pool_projection_calls"] = len(batches) if support > 1 else 0
    return work


def empty_totals():
    return dict(
        kind="regional_spaced_phase_work_v1",
        decision_events=0,
        decision_rows=0,
        max_support=0,
        max_reader_and_emissions_bytes_estimate=0,
        **dict.fromkeys(SUMS, 0),
    )


def accumulate(totals, work):
    totals["decision_events"] += 1
    totals["decision_rows"] += work["rows"]
    totals["max_support"] = max(totals["max_support"], work["support"])
    totals["max_reader_and_emissions_bytes_estimate"] = max(
        totals["max_reader_and_emissions_bytes_estimate"], work["reader_and_trace_bytes_estimate"]
    )
    for name in SUMS:
        totals[name] += work[name]
