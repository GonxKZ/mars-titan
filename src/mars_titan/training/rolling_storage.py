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
  agregados por sesión (también los de la comparación postentrenada de cada ventana) y las
  cintas de las políticas.
- La meseta de un ajuste conjunto ocupa su carpeta hasta que se confirma su continuación,
  que reanuda en ella y la libera. De la meseta solo queda después la copia de su informe.

Con la retención v3 (`v3=True`) cada trabajo de adaptadores ocupa sus tablas compactadas y
un solo estado desde que se confirma, las tablas de la base y de la ablación se compactan al
terminar su fase, las de los adaptadores se liberan como las de la base (salvo la evaluación
del predictor de la cadena, que se conserva hasta su última política) y las cachés de los
padres se borran al cerrar la ventana.

`all_regenerated` supone que todas las regeneraciones salen idénticas y `none_regenerated`
que ninguna lo hace, de modo que todo queda compactado. Con la v3 se añade
`adapters_regenerated`, la proyección de la guardia: los adaptadores se liberan, porque una
regeneración suya que no sale idéntica detiene el recorrido, y la base y la ablación quedan
compactadas. La realidad está entre los dos. Las
tablas comunes de filas se cuentan hasta el final aunque `_prune_rows` las borre antes, así
que el conservado es una cota superior. Solo se importan módulos estables con rutas
absolutas para poder evaluar el mismo cálculo sobre el plan de otra rama.
"""

import math
from collections import Counter, defaultdict

from mars_titan.training.campaign_plan import PLATEAU
from mars_titan.training.campaign_storage import (
    HELD_OUT,
    INDEXED,
    WRITERS,
    index_rows,
    job_footprint,
    shared_index_bytes,
)

SCENARIOS = ("all_regenerated", "none_regenerated")
# Solo con la v3: una regeneración de adaptadores que no sale idéntica detiene el recorrido,
# así que la guardia de cada ventana supone los adaptadores liberados y la base y la
# ablación sin ninguna regeneración idéntica.
ADAPTERS_REGENERATED = "adapters_regenerated"
PHASES = ("base", "adapters", "ablation", "rl", "release")
# Registro de retención y huellas que deja cada tabla liberada o compactada.
RECORD_BYTES = 4096


def policy_tape_bytes(stage, extras):
    """Bytes medidos de una cinta de la etapa de políticas, si la medida cubre su forma.

    La medida de `storage_measurements.tape_bytes` declara los activos con precios y las
    columnas de su diseño. Una cinta de la etapa tiene hasta `max_assets` activos con
    precios y hasta `window_tapes.layout_bound` columnas, las demás vacías. Una medida más
    estrecha subestimaría el disco de todas las cintas, así que se rechaza.
    """
    from mars_titan.simulation.window_tapes import layout_bound, universe_rule

    policies = stage["policies"]
    assets, _ = universe_rule(policies["universe"])
    bound = layout_bound(policies["universe"], policies["train_windows"])
    shape = extras.get("tape_shape") or {}
    if not (shape.get("assets", 0) >= assets and shape.get("layout", 0) >= bound):
        raise ValueError(
            f"La medida de cintas {shape} no cubre {assets} activos con precios en "
            f"{bound} columnas, la forma máxima de la etapa de políticas"
        )
    return extras["tape_bytes"]


def _readers(stage, jobs, *, chained):
    """Trabajos de las políticas que leen cada evaluación, de la base o de la cadena."""
    from mars_titan.simulation.policy_plan import CHAIN, predictor_reads

    from .campaign_chain import scope_windows

    policies = stage["policies"]
    seed = policies["predictor"]["seed"]
    chain = policies["predictor"]["source"] == CHAIN
    first = {}
    if chain:
        first = {scope: scope_windows(stage["campaign"], scope)[0][0] for scope in stage["scopes"]}
    readers = {}
    for job in jobs:
        for scope, window, predictor in dict.fromkeys(predictor_reads(stage, job)):
            if chained != (chain and window != first[scope]):
                continue
            readers.setdefault((scope, window, predictor, seed), []).append(job)
    return readers


def policy_readers(stage, jobs):
    """Trabajos de las políticas que leen la evaluación de cada predictor elegido de la base.

    Las claves son ámbito, ventana, predictor y la semilla declarada del predictor. Con el
    predictor elegido de la base, un trabajo lee las evaluaciones que da
    `policy_plan.predictor_reads`. Con la cadena, la base solo aporta el predictor de la
    primera ventana de cada ámbito. En las demás la cadena lee las tablas de los adaptadores
    (`chain_readers`), que la retención v2 conserva compactadas y la v3 hasta su último lector.
    """
    return _readers(stage, jobs, chained=False)


def chain_readers(stage, jobs):
    """Trabajos de las políticas que leen de la cadena la evaluación del predictor elegido.

    Son las ventanas posteriores a la primera de cada ámbito con `source=chain`, con las
    mismas claves que `policy_readers`. Sin la cadena no hay ninguna.
    """
    return _readers(stage, jobs, chained=True)


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
    el primero del plan, como en `selected_jobs`. La meseta de un ajuste conjunto no emite
    evaluación, así que la lectura recae en su continuación.
    """
    positions = {
        (scope, window): index
        for index, row in enumerate(windows)
        for scope, window in row["scopes"].items()
    }
    inputs = dict(ablation_jobs=(), adapter_jobs=(), policy_reads={}, chain_reads={}, tapes={})
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
            if job.get("phase") != PLATEAU:
                first.setdefault((job["scope"], job["window"], job["arm"], job["seed"]), job["id"])
        reads = {}
        for key, last in policy_needs(stage, planned, positions).items():
            if key in first:
                reads[first[key]] = max(reads.get(first[key], -1), last)
        # Con la cadena, cada predictor y semilla de una ventana posterior lee la evaluación
        # de un trabajo de adaptadores. Se guarda la última ventana que lee cada uno.
        chained = defaultdict(list)
        for (scope, window, _, _), readers in chain_readers(stage, planned).items():
            last = max(positions[job["scope"], job["window"]] for job in readers)
            chained[scope, window].append(last)
        anchors = {}
        for job in planned:
            anchors[job["scope"], job["market"], job["anchor"], job["predictor"]] = (
                len(job["train"]) + 2
            )
        tapes = defaultdict(int)
        tape = policy_tape_bytes(stage, extras)
        for (scope, _, anchor, _), count in anchors.items():
            tapes[windows[positions[scope, anchor]]["id"]] += count * tape
        inputs.update(policy_reads=reads, chain_reads=dict(chained), tapes=dict(tapes))
    return inputs


def _in(row, job):
    return row["scopes"].get(job["scope"]) == job["window"]


def staged_index_bytes(counts, storage):
    """Índice del tramo de ajuste de los adaptadores de una vista: solo las filas nuevas.

    Los ajustes de adaptadores parten de la primera fila nueva sin calentamiento, así que su
    tramo de ajuste es otro índice de la vista. Se acota con el tramo de ajuste completo.
    """
    return int(2 * counts["train"] * storage["index"]["bytes_per_row"])


def _partitions(job):
    """Tramos con tabla del trabajo. La meseta de un ajuste conjunto no escribe ninguno."""
    if job.get("phase") == PLATEAU:
        return ()
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


def comparison_series(jobs):
    """Series (configuración y semilla) de la comparación postentrenada de una ventana de un ámbito.

    Solo se comparan los padres con algún ajuste, porque los de Ridge y XGBoost solo tienen
    el padre congelado. Cada trabajo de un padre comparado es una serie, y cada padre y
    semilla añade el reentreno de la base y la cadena. La cadena se cuenta siempre, aunque
    la declaración no la contraste, así que es una cota superior.
    """
    # "fit" es `campaign_stage.FIT`, sin importar la etapa para poder evaluar otras ramas.
    compared = {(job["base_arm"], job["seed"]) for job in jobs if job.get("kind") == "fit"}
    return sum((job["base_arm"], job["seed"]) in compared for job in jobs) + 2 * len(compared)


def adapter_window(jobs, counts, extras, *, ordered_copy, v3=False):
    """Adaptadores de una ventana de un ámbito: escrito, compactado y corpus transitorio.

    Sin `ordered_copy` (lectura por bloques desde la vista) el corpus no ocupa disco. Las
    cachés de los padres se conservan como en `storage_budget.adapter_estimate`. Los
    agregados por ventana de la comparación postentrenada y sus fuentes se conservan siempre
    con la medida por serie `comparison_aggregate_bytes_per_series`, tomada en la ventana
    con más sesiones y con los cuatro estratos en cada sesión, una cota superior.

    Con `v3`, cada trabajo confirmado conserva un estado y sus tablas compactadas solo hasta
    que se cierra su bloque (el modelo base en la ventana), que guarda sus agregados y las
    regenera y libera. Es transitorio el bloque en curso, con el trabajo que se ajusta (sus
    tablas sin compactar, los demás estados y la caché de su padre) o la regeneración de uno.
    `released` es lo que queda si se liberan todas las tablas, `compacted` si no se libera
    ninguna, `table` las tablas compactadas de un trabajo y `regeneration` el disco de
    regenerar uno.
    """
    if not jobs:
        return dict(written=0, compacted=0, transient=0, released=0, table=0, regeneration=0)
    adapter = extras["adapter"]
    held = sum(counts[p] for p in HELD_OUT)
    current = int(held * adapter["row_bytes_current"])
    table = int(held * adapter["row_bytes_shared_rows"])
    states = 1 if v3 else adapter["retained_states"]
    fixed = adapter["state_bytes"] * states + adapter["job_report_bytes"]
    caches = len({(j["base_arm"], j["seed"]) for j in jobs}) * adapter["parent_cache_bytes"]
    aggregates = math.ceil(
        comparison_series(jobs) * adapter["comparison_aggregate_bytes_per_series"]
    )
    corpus = 0
    if ordered_copy:
        ordered = (counts["train"] + counts["validation"]) * adapter["ordered_row_bytes"]
        corpus = ordered + counts["train"] * adapter["input_row_bytes"]
    if v3:
        block = max(Counter(job["base_arm"] for job in jobs).values()) * table
        running = (
            current
            + adapter["state_bytes"] * (adapter["retained_states"] - 1)
            + adapter["parent_cache_bytes"]
        )
        return dict(
            written=len(jobs) * fixed + aggregates,
            compacted=len(jobs) * (table + fixed) + aggregates,
            transient=corpus + block + max(running, current + table),
            released=len(jobs) * fixed + aggregates,
            table=table,
            regeneration=current + table,
        )
    kept = caches + aggregates
    return dict(
        written=len(jobs) * (current + fixed) + kept,
        compacted=len(jobs) * (table + fixed) + kept,
        transient=corpus,
        released=0,
        table=table,
        regeneration=0,
    )


class _Walk:
    """Recorrido de un escenario: conservado, pico y tablas pendientes de su último lector."""

    def __init__(self, scenario, inputs):
        self.scenario, self.inputs = scenario, inputs
        # Si fallan las regeneraciones de la base y la ablación, y las de los adaptadores.
        self.base_fails = scenario != "all_regenerated"
        self.adapters_fail = scenario == "none_regenerated"
        self.retained, self.peak, self.worst = 0, 0, None
        self.pending, self.shared = {}, set()

    def reach(self, value, label):
        if value > self.peak:
            self.peak, self.worst = value, label

    def base(self, row, index, phases):
        """Ajustes y traslados de la ventana en el orden del plan."""
        counts, measured, storage = (self.inputs[k] for k in ("counts", "measured", "storage"))
        moment, kept, regeneration, compactable = self.retained, 0, 0, 0
        aggregate = self.inputs["extras"]["aggregate_bytes_per_row"]
        waiting, compacted_rows = {}, set()
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
            if job.get("phase") == PLATEAU:
                # La continuación reanuda en la carpeta de la meseta y la libera al confirmarse
                # (`release_confirmed`). De la meseta solo queda la copia de su informe.
                waiting[job["id"]] = footprint["retained_bytes"] - footprint["retained"]["reports"]
                kept += footprint["retained"]["reports"]
                continue
            moment -= waiting.pop(job.get("plateau"), 0)
            kept += footprint["retained_bytes"] - footprint["retained"]["predictions"]
            kept += int(aggregate * sum(window[p] for p in partitions)) + RECORD_BYTES * len(
                partitions
            )
            if partitions:
                regeneration = max(
                    regeneration, regeneration_bytes(job, window, measured, storage, partitions)
                )
                compactable += sum(written.values())
                compactable -= _tables(measured, job, "shared_rows", partitions)
                compacted_rows |= {(job["scope"], job["window"], p) for p in partitions}
            reader = self.inputs["policy_reads"].get(job["id"], -1)
            if self.base_fails or reader > index:
                compacted = _tables(measured, job, "shared_rows", partitions)
                kept += compacted + self._shared(job, partitions)
                if not self.base_fails:
                    self.pending[reader] = self.pending.get(reader, 0) + compacted
        if self.inputs["v3"]:
            # Al terminar la fase, las tablas de la ventana se compactan sobre sus filas comunes.
            moment += self._rows_bytes(compacted_rows) - compactable
        return moment, kept, regeneration

    def _rows_bytes(self, keys):
        measured = self.inputs["measured"]
        return sum(measured[f"{s}/{w}"][p]["shared_rows_bytes"] for s, w, p in keys)

    def _shared(self, job, partitions):
        new = {(job["scope"], job["window"], p) for p in partitions} - self.shared
        self.shared |= new
        return self._rows_bytes(new)

    def adapters(self, row, index, moment, phases):
        """Escrito durante la fase, conservado tras liberar y regeneración más grande.

        Con la v3 y todas las regeneraciones idénticas solo quedan los estados, los informes,
        los agregados y la evaluación del predictor de la cadena que leerá una política
        posterior, hasta esa ventana. Las filas comunes de las tablas compactadas son las de la
        base: durante la ventana ya las cuenta su fase y después se conservan mientras alguna
        tabla compactada las lea.
        """
        v3 = self.inputs["v3"]
        written = kept = transient = regeneration = 0
        for scope, window in row["scopes"].items():
            jobs = [
                j
                for j in self.inputs["adapter_jobs"]
                if (j["scope"], j["window"]) == (scope, window)
            ]
            part = adapter_window(
                jobs,
                self.inputs["counts"][scope][window],
                self.inputs["extras"],
                ordered_copy=self.inputs["ordered_copy"],
                v3=v3,
            )
            transient = max(transient, part["transient"])
            regeneration = max(regeneration, part["regeneration"])
            if not v3 or self.adapters_fail:
                # Sin regeneraciones idénticas, los bloques conservan sus tablas compactadas.
                written += part["compacted"] if v3 else part["written"]
                kept += part["compacted"]
                if v3 and jobs:
                    kept += self._shared(jobs[0], HELD_OUT)
                continue
            # Las evaluaciones de la cadena que leen las políticas siguen en la ventana.
            reads = self.inputs["chain_reads"].get((scope, window), ())
            written += part["written"] + len(reads) * part["table"]
            kept += part["released"]
            later = [
                last for last in self.inputs["chain_reads"].get((scope, window), ()) if last > index
            ]
            if later:
                # Las tablas conservadas siguen leyendo las filas comunes de su ventana.
                kept += self._shared(jobs[0], HELD_OUT)
            for last in later:
                kept += part["table"]
                self.pending[last] = self.pending.get(last, 0) + part["table"]
        phases["adapters"] = moment + written + transient
        return moment + written, kept, regeneration

    def ablation(self, row, moment, phases):
        counts, measured, storage = (self.inputs[k] for k in ("counts", "measured", "storage"))
        written_all, kept, regeneration, compactable = 0, 0, 0, 0
        for job in (j for j in self.inputs["ablation_jobs"] if _in(row, j)):
            window = counts[job["scope"]][job["window"]]
            written = _tables(measured, job, "current", ("evaluation",))
            build = 0
            if job["model"] in INDEXED:
                build = int(2 * window["evaluation"] * storage["index"]["build_bytes_per_row"])
            phases["ablation"] = max(phases["ablation"], moment + written_all + written + build)
            written_all += written + storage["job_report_bytes"]
            kept += storage["job_report_bytes"] + RECORD_BYTES
            if self.base_fails:
                kept += _tables(measured, job, "shared_rows", ("evaluation",))
            regeneration = max(
                regeneration,
                regeneration_bytes(job, window, measured, storage, ("evaluation",)),
            )
            if self.inputs["v3"]:
                compactable += written - _tables(measured, job, "shared_rows", ("evaluation",))
        phases["ablation"] = max(phases["ablation"], moment + written_all)
        # Con la v3 las tablas se compactan al terminar la fase sobre las filas de la base.
        return moment + written_all - compactable, kept, regeneration

    def _indices(self, row):
        """Índices compartidos de las vistas de la ventana que algún trabajo lee.

        La campaña y las fases de la ventana los reutilizan y el recorrido los borra al final
        de su liberación (`window_indices`), así que ocupan durante todas sus fases. Los
        adaptadores leen índices si su familia es cronológica, que son las familias indexadas,
        y los que ajustan añaden el índice de su tramo de ajuste (`staged_index_bytes`).
        """
        counts, storage = self.inputs["counts"], self.inputs["storage"]
        base = (*self.inputs["jobs"], *self.inputs["ablation_jobs"])
        adapters = [
            j for j in self.inputs["adapter_jobs"] if _in(row, j) and j["family"] in INDEXED
        ]
        views = {(j["scope"], j["window"]) for j in base if _in(row, j) and j["model"] in INDEXED}
        views |= {(j["scope"], j["window"]) for j in adapters}
        # `"frozen"` es `posttraining.campaign_stage.FROZEN`: el padre congelado no ajusta.
        staged = {(j["scope"], j["window"]) for j in adapters if j["kind"] != "frozen"}
        return sum(shared_index_bytes(counts[s][w], storage) for s, w in views) + sum(
            staged_index_bytes(counts[s][w], storage) for s, w in staged
        )

    def window(self, row, index):
        indices = self._indices(row)
        self.retained += indices
        phases = dict.fromkeys(PHASES, self.retained)
        moment, base_kept, base_regeneration = self.base(row, index, phases)
        moment, adapters_kept, adapter_regeneration = self.adapters(row, index, moment, phases)
        moment, masked_kept, masked_regeneration = self.ablation(row, moment, phases)
        tapes = self.inputs["tapes"].get(row["id"], 0)
        moment += tapes
        phases["rl"] = moment
        phases["release"] = moment + max(
            base_regeneration, masked_regeneration, adapter_regeneration
        )
        for phase, value in phases.items():
            self.reach(value, f"{row['id']}/{phase}")
        freed = sum(self.pending.pop(k) for k in [k for k in self.pending if k <= index])
        self.retained += base_kept + adapters_kept + masked_kept + tapes - freed - indices
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
    chain_reads=None,
    tapes=None,
    ordered_copy=False,
    v3=False,
):
    """Pico y conservado de cada escenario, ventana a ventana, en el orden del plan.

    `policy_reads` da, para cada trabajo base cuya evaluación lee una política, la última
    posición de ventana que la lee, y `chain_reads`, para cada ámbito y ventana, la última
    posición que lee cada evaluación de la cadena. `tapes` da los bytes de cintas que
    conserva cada ventana. `v3` aplica la retención v3.
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
        chain_reads=chain_reads or {},
        tapes=tapes or {},
        ordered_copy=ordered_copy,
        v3=v3,
    )
    result = {}
    for scenario in SCENARIOS + ((ADAPTERS_REGENERATED,) if v3 else ()):
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
