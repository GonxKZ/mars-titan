"""Disco de la retención v2: pico y conservado de la campaña recorrida ventana a ventana.

`training.rolling_retention` ejecuta cada ventana de campaña completa (base, adaptadores,
ablación, políticas, agregados y liberación) antes de la siguiente. Este módulo calcula,
sin leer datos ni modelos, lo que esa ejecución ocupa con los recuentos de cada ventana, la
declaración de almacenamiento y los bytes por fila medidos:

- Durante la ventana se suma lo que deja cada fase sobre lo conservado de las anteriores,
  con el término transitorio de su trabajo más grande (páginas de XGBoost, índices en
  construcción, corpus de los adaptadores y la regeneración de un trabajo al liberar).
- Al liberar, las tablas por fila de la base y de la ablación desaparecen si su
  regeneración es exacta. Las evaluaciones que leerá una política posterior y las tablas de
  los adaptadores quedan compactadas sin pérdida (tabla común y decimales propios).
- Se conservan siempre el estado elegido de cada ajuste, los informes y recibos, los
  agregados por sesión y las cintas de las políticas.

`all_regenerated` supone que todas las regeneraciones salen idénticas y `none_regenerated`
que ninguna lo hace, de modo que todo queda compactado. La realidad está entre los dos. Las
tablas comunes de filas se cuentan hasta el final aunque `_prune_rows` las borre antes, así
que el conservado es una cota superior. Solo se importan módulos estables con rutas
absolutas para poder evaluar el mismo cálculo sobre el plan de otra rama.
"""

import math
from collections import defaultdict

from mars_titan.training.campaign_storage import (
    HELD_OUT,
    INDEXED,
    WRITERS,
    index_rows,
    job_footprint,
)

SCENARIOS = ("all_regenerated", "none_regenerated")
PHASES = ("base", "adapters", "ablation", "rl", "release")
# Registro de retención y huellas que deja cada tabla liberada o compactada.
RECORD_BYTES = 4096


def policy_readers(stage, jobs):
    """Trabajos de las políticas que leen la evaluación de cada predictor elegido de la base.

    Las claves son ámbito, ventana, predictor y la semilla declarada del predictor. Con el
    predictor elegido de la base, un trabajo lee las evaluaciones que da
    `policy_plan.predictor_reads`. Con la cadena, la base solo aporta el predictor de la
    primera ventana de cada ámbito. En las demás la cadena lee las tablas de los adaptadores,
    que la retención compacta y nunca libera.
    """
    from mars_titan.posttraining.staged_chain import scope_windows
    from mars_titan.simulation.policy_plan import CHAIN, predictor_reads

    policies = stage["policies"]
    seed = policies["predictor"]["seed"]
    chain = policies["predictor"]["source"] == CHAIN
    first = {}
    if chain:
        first = {scope: scope_windows(stage["campaign"], scope)[0][0] for scope in stage["scopes"]}
    readers = {}
    for job in jobs:
        for scope, window, predictor in dict.fromkeys(predictor_reads(stage, job)):
            if chain and window != first[scope]:
                continue
            readers.setdefault((scope, window, predictor, seed), []).append(job)
    return readers


def policy_needs(stage, jobs, positions):
    """Última ventana de campaña que lee la evaluación de cada predictor elegido de la base.

    `jobs` es el plan de la etapa de políticas y `positions` da la posición de cada par
    (ámbito, ventana).
    """
    return {
        key: max(positions[job["scope"], job["window"]] for job in readers)
        for key, readers in policy_readers(stage, jobs).items()
    }


def rolling_inputs(jobs, windows, extras, *, ablation=None, adapters=None, rl=None):
    """Trabajos de las etapas, lecturas de las políticas y cintas por ventana de campaña.

    Una política lee la evaluación del predictor con la semilla declarada. Entre los casos
    de una búsqueda cualquiera puede ganar y todos tienen las mismas filas, así que se toma
    el primero del plan, como en `selected_jobs`.
    """
    positions = {
        (scope, window): index
        for index, row in enumerate(windows)
        for scope, window in row["scopes"].items()
    }
    inputs = dict(ablation_jobs=(), adapter_jobs=(), policy_reads={}, tapes={})
    if ablation is not None:
        from mars_titan.training.modality_ablation_stage import load_stage, plan_stage

        inputs["ablation_jobs"] = plan_stage(load_stage(ablation))
    if adapters is not None:
        from mars_titan.posttraining.campaign_stage import load_stage, plan_stage

        inputs["adapter_jobs"] = plan_stage(load_stage(adapters))
    if rl is not None:
        from mars_titan.simulation import policy_plan

        stage = policy_plan.load_stage(rl)
        planned = policy_plan.plan_stage(stage)
        first = {}
        for job in jobs:
            first.setdefault((job["scope"], job["window"], job["arm"], job["seed"]), job["id"])
        reads = {}
        for key, last in policy_needs(stage, planned, positions).items():
            if key in first:
                reads[first[key]] = max(reads.get(first[key], -1), last)
        anchors = {}
        for job in planned:
            anchors[job["scope"], job["market"], job["anchor"], job["predictor"]] = (
                len(job["train"]) + 2
            )
        tapes = defaultdict(int)
        for (scope, _, anchor, _), count in anchors.items():
            tapes[windows[positions[scope, anchor]]["id"]] += count * extras["tape_bytes"]
        inputs.update(policy_reads=reads, tapes=dict(tapes))
    return inputs


def _in(row, job):
    return row["scopes"].get(job["scope"]) == job["window"]


def _partitions(job):
    return HELD_OUT[1:] if job.get("kind") == "carry" else HELD_OUT


def _tables(measured, job, layout, partitions):
    sizes = measured[f"{job['scope']}/{job['window']}"]
    writer = WRITERS[job["model"]]
    return sum(sizes[p]["writers"][writer][layout] for p in partitions)


def regeneration_bytes(job, counts, measured, storage, partitions):
    """Disco de regenerar un trabajo: sus tablas otra vez y, si el modelo indexa, su índice.

    Si no sale idéntica, la tabla compactada se escribe antes de borrar la regeneración.
    """
    written = _tables(measured, job, "current", partitions)
    written += _tables(measured, job, "shared_rows", partitions)
    if job["model"] not in INDEXED:
        return written
    index = storage["index"]
    warmup = None if job["model"] == "episodic_gru" else index["warmup_partition"]
    rows = index_rows(counts, warmup, partitions=partitions, train=False)
    largest = max(2 * counts[p] for p in partitions)
    return written + int(rows * index["bytes_per_row"] + largest * index["build_bytes_per_row"])


def adapter_window(jobs, counts, extras, *, ordered_copy):
    """Adaptadores de una ventana de un ámbito: escrito, compactado y corpus transitorio.

    Sin `ordered_copy` (lectura por bloques desde la vista) el corpus no ocupa disco. Las
    cachés de los padres se conservan como en `storage_budget.adapter_estimate`.
    """
    if not jobs:
        return dict(written=0, compacted=0, transient=0)
    adapter = extras["adapter"]
    held = sum(counts[p] for p in HELD_OUT)
    fixed = adapter["state_bytes"] * adapter["retained_states"] + adapter["job_report_bytes"]
    caches = len({(j["base_arm"], j["seed"]) for j in jobs}) * adapter["parent_cache_bytes"]
    corpus = 0
    if ordered_copy:
        ordered = (counts["train"] + counts["validation"]) * adapter["ordered_row_bytes"]
        corpus = ordered + counts["train"] * adapter["input_row_bytes"]
    return dict(
        written=len(jobs) * (int(held * adapter["row_bytes_current"]) + fixed) + caches,
        compacted=len(jobs) * (int(held * adapter["row_bytes_shared_rows"]) + fixed) + caches,
        transient=corpus,
    )


class _Walk:
    """Recorrido de un escenario: conservado, pico y tablas pendientes de su último lector."""

    def __init__(self, scenario, inputs):
        self.scenario, self.inputs = scenario, inputs
        self.retained, self.peak, self.worst = 0, 0, None
        self.pending, self.shared = {}, set()

    def reach(self, value, label):
        if value > self.peak:
            self.peak, self.worst = value, label

    def base(self, row, index, phases):
        """Ajustes y traslados de la ventana en el orden del plan."""
        counts, measured, storage = (self.inputs[k] for k in ("counts", "measured", "storage"))
        moment, kept, regeneration = self.retained, 0, 0
        aggregate = self.inputs["extras"]["aggregate_bytes_per_row"]
        for job in (j for j in self.inputs["jobs"] if _in(row, j)):
            window = counts[job["scope"]][job["window"]]
            partitions = _partitions(job)
            # Bytes de cada tabla con la disposición actual del escritor.
            written = {p: _tables(measured, job, "current", (p,)) for p in partitions}
            footprint = job_footprint(
                job,
                window,
                storage,
                release=True,
                prediction_bytes=lambda _, p, written=written: written[p],
            )
            phases["base"] = max(
                phases["base"],
                moment + footprint["retained_bytes"] + footprint["transient_bytes"],
            )
            moment += footprint["retained_bytes"]
            kept += footprint["retained_bytes"] - footprint["retained"]["predictions"]
            kept += int(aggregate * sum(window[p] for p in partitions)) + RECORD_BYTES * len(
                partitions
            )
            regeneration = max(
                regeneration, regeneration_bytes(job, window, measured, storage, partitions)
            )
            reader = self.inputs["policy_reads"].get(job["id"], -1)
            if self.scenario == "none_regenerated" or reader > index:
                compacted = _tables(measured, job, "shared_rows", partitions)
                kept += compacted + self._shared(job, partitions)
                if self.scenario == "all_regenerated":
                    self.pending[reader] = self.pending.get(reader, 0) + compacted
        return moment, kept, regeneration

    def _shared(self, job, partitions):
        new = {(job["scope"], job["window"], p) for p in partitions} - self.shared
        self.shared |= new
        measured = self.inputs["measured"]
        return sum(measured[f"{s}/{w}"][p]["shared_rows_bytes"] for s, w, p in new)

    def adapters(self, row, moment, phases):
        written = compacted = transient = 0
        for scope, window in row["scopes"].items():
            part = adapter_window(
                [
                    j
                    for j in self.inputs["adapter_jobs"]
                    if (j["scope"], j["window"]) == (scope, window)
                ],
                self.inputs["counts"][scope][window],
                self.inputs["extras"],
                ordered_copy=self.inputs["ordered_copy"],
            )
            written += part["written"]
            compacted += part["compacted"]
            transient = max(transient, part["transient"])
        phases["adapters"] = moment + written + transient
        return moment + written, compacted

    def ablation(self, row, moment, phases):
        counts, measured, storage = (self.inputs[k] for k in ("counts", "measured", "storage"))
        written_all, kept, regeneration = 0, 0, 0
        for job in (j for j in self.inputs["ablation_jobs"] if _in(row, j)):
            window = counts[job["scope"]][job["window"]]
            written = _tables(measured, job, "current", ("evaluation",))
            build = 0
            if job["model"] in INDEXED:
                build = int(2 * window["evaluation"] * storage["index"]["build_bytes_per_row"])
            phases["ablation"] = max(phases["ablation"], moment + written_all + written + build)
            written_all += written + storage["job_report_bytes"]
            kept += storage["job_report_bytes"] + RECORD_BYTES
            if self.scenario == "none_regenerated":
                kept += _tables(measured, job, "shared_rows", ("evaluation",))
            regeneration = max(
                regeneration,
                regeneration_bytes(job, window, measured, storage, ("evaluation",)),
            )
        phases["ablation"] = max(phases["ablation"], moment + written_all)
        return moment + written_all, kept, regeneration

    def window(self, row, index):
        phases = dict.fromkeys(PHASES, self.retained)
        moment, base_kept, base_regeneration = self.base(row, index, phases)
        moment, adapters_kept = self.adapters(row, moment, phases)
        moment, masked_kept, masked_regeneration = self.ablation(row, moment, phases)
        tapes = self.inputs["tapes"].get(row["id"], 0)
        moment += tapes
        phases["rl"] = moment
        phases["release"] = moment + max(base_regeneration, masked_regeneration)
        for phase, value in phases.items():
            self.reach(value, f"{row['id']}/{phase}")
        freed = sum(self.pending.pop(k) for k in [k for k in self.pending if k <= index])
        self.retained += base_kept + adapters_kept + masked_kept + tapes - freed
        return dict(window=row["id"], retained_bytes=self.retained, **phases)


def rolling_estimate(
    jobs,
    windows,
    counts,
    measured,
    storage,
    extras,
    *,
    ablation_jobs=(),
    adapter_jobs=(),
    policy_reads=None,
    tapes=None,
    ordered_copy=False,
):
    """Pico y conservado de cada escenario, ventana a ventana, en el orden del plan.

    `policy_reads` da, para cada trabajo base cuya evaluación lee una política, la última
    posición de ventana que la lee. `tapes` da los bytes de cintas que conserva cada ventana.
    """
    inputs = dict(
        jobs=jobs,
        counts=counts,
        measured=measured,
        storage=storage,
        extras=extras,
        ablation_jobs=list(ablation_jobs),
        adapter_jobs=list(adapter_jobs),
        policy_reads=policy_reads or {},
        tapes=tapes or {},
        ordered_copy=ordered_copy,
    )
    result = {}
    for scenario in SCENARIOS:
        walk = _Walk(scenario, inputs)
        rows = [walk.window(row, index) for index, row in enumerate(windows)]
        result[scenario] = dict(
            retained_bytes=walk.retained, peak_bytes=walk.peak, peak_at=walk.worst, windows=rows
        )
    return result


def declared_measure(counts, storage):
    """Medida de cada ventana con los bytes por fila declarados, sin tablas sintéticas.

    Sirve a la guardia de cada ventana. Como la declaración solo da la disposición actual,
    la compactada y la tabla común se cuentan con esos mismos bytes, una cota superior.
    """
    rows = storage["prediction_row_bytes"]
    measured = {}
    for scope, windows in counts.items():
        for window, value in windows.items():
            measured[f"{scope}/{window}"] = {
                p: dict(
                    rows=value[p],
                    shared_rows_bytes=math.ceil(value[p] * max(w[p] for w in rows.values())),
                    writers={
                        writer: dict.fromkeys(
                            ("current", "shared_rows"), math.ceil(value[p] * per_row[p])
                        )
                        for writer, per_row in rows.items()
                    },
                )
                for p in HELD_OUT
            }
    return measured


def window_increment(estimate, scenario="all_regenerated"):
    """Mayor crecimiento de una ventana sobre lo conservado antes de empezarla."""
    previous, worst = 0, dict(bytes=0, window=None)
    for row in estimate[scenario]["windows"]:
        top = max(row[phase] for phase in PHASES)
        if top - previous > worst["bytes"]:
            worst = dict(bytes=top - previous, window=row["window"])
        previous = row["retained_bytes"]
    return worst
