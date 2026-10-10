"""Retención v2: la campaña ventana a ventana y la liberación de las filas que nadie lee.

La campaña completa no cabe en el disco si conserva todas las tablas por fila. La retención
v2, declarada antes de cualquier resultado, recorre cada ventana de campaña en este orden:

1. `base`: ajustes de todos los brazos, selección, semillas extra del elegido y traslados.
2. `adapters`, `ablation` y `rl`: etapas posteriores de esa ventana. Con el walk-forward
   por etapas, los adaptadores y las políticas reciben un informe de disjunción calculado
   justo antes, y las políticas leen la cadena de la salida de los adaptadores.
3. `aggregates`: agregados por sesión de la comparación de cada ámbito de la ventana
   (`evaluation.window_aggregates`), con sus fuentes limitadas a esa ventana.
4. `release`: cada tabla por fila de la base y de la ablación que ya no lee ninguna fase
   posterior se regenera por inferencia desde el estado elegido y, solo si sale idéntica
   bit a bit, se libera conservando sus huellas. Si no sale idéntica, se compacta sin
   pérdida y se conserva. Las evaluaciones que leerán políticas posteriores y las tablas
   de los adaptadores se compactan sin pérdida y se conservan. Si la declaración nombra
   las políticas como consumidoras, una evaluación que leen solo se libera cuando cada
   trabajo de políticas que la lee ha confirmado su recibo.

Después empieza la ventana siguiente, así que el pico de disco es el de una ventana más lo
conservado. El registro `retention/ledger.json` guarda las fases terminadas de cada ventana
para reanudar tras un corte. Las operaciones de liberación y compactación son idempotentes.
"""

import argparse
import hashlib
import inspect
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data import prediction_files
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation import walk_forward_comparison as comparison
from mars_titan.evaluation import window_aggregates

from . import prediction_regeneration as regeneration
from .campaign_plan import ONLINE

DECLARATION_KIND = "historical_masked_prediction_retention"
LEDGER_KIND = "historical_masked_retention_ledger"
PHASES = ("base", "adapters", "ablation", "rl", "aggregates", "release")
STAGE_PHASES = ("base", "adapters", "ablation", "rl")
AGGREGATES = ("walk_forward", "long_short")
_FIELDS = {
    "schema_version",
    "kind",
    "name",
    "status",
    "declared_on",
    "order",
    "phases",
    "aggregates",
    "release",
    "compact",
    "policy_inputs",
    "consumers",
    "numerics",
    "kept",
}
# Las políticas leen la evaluación de la base al montar sus cintas. Mientras falte el recibo
# de alguna que la lee, esa evaluación no se libera, porque recuperarla exigiría regenerarla
# en la GPU.
POLICY_CONSUMER = dict(reads="evaluation", release_after="confirmed_policy_receipts")
STRICT_FP32 = dict(
    float32_matmul_precision="highest", cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_retention(path):
    """Validar la declaración de la retención v2 y devolverla con su huella."""
    document, digest = read_manifest(Path(path), 1024**2)
    _require(
        isinstance(document, dict)
        and set(document) == _FIELDS
        and document["schema_version"] == 1
        and document["kind"] == DECLARATION_KIND
        and document["status"] == "declared_before_results"
        and document["order"] == "by_window"
        and tuple(document["phases"]) == PHASES,
        "La retención no cumple su contrato de fases por ventana",
    )
    _require(
        tuple(document["aggregates"]) == AGGREGATES,
        "La retención solo libera filas con agregados de la comparación y de la cartera",
    )
    release = document["release"]
    _require(
        release
        == dict(
            stages=["base", "ablation"],
            partitions=["validation", "calibration", "evaluation"],
            requires="bit_exact_regeneration",
            otherwise="compact_lossless",
        ),
        "Solo se liberan filas regeneradas bit a bit de la base y de la ablación",
    )
    _require(
        document["compact"] == ["adapters"]
        and document["policy_inputs"] == "compact_until_the_last_reading_policy_window"
        and document["numerics"] == STRICT_FP32,
        "La retención compacta los adaptadores, guarda las entradas de las políticas y exige "
        "FP32 estricto",
    )
    _require(
        document["consumers"] in ({}, dict(rl=POLICY_CONSUMER)),
        "Las políticas solo pueden declararse como consumidoras de la evaluación hasta "
        "confirmar sus recibos",
    )
    return dict(document, path=str(Path(path).resolve()), sha256=digest)


def load_schedule(path, campaign):
    """Ventanas de campaña en orden, con la ventana de cada ámbito que las forma."""
    document, _ = read_manifest(Path(path), 64 * 1024**2)
    _require(
        isinstance(document, dict)
        and document.get("campaign") == campaign["sha256"]
        and isinstance(document.get("windows"), list)
        and document["windows"],
        "El orden de ventanas no pertenece a esta campaña",
    )
    resolved = campaign["comparison_config"]["resolved_scopes"]
    rows, seen = [], set()
    for row in document["windows"]:
        scopes = row.get("scopes")
        _require(
            isinstance(row.get("window"), str)
            and isinstance(scopes, dict)
            and scopes
            and all(
                scope in campaign["scopes"] and window in resolved[scope]["windows"]
                for scope, window in scopes.items()
            ),
            f"La ventana {row.get('window')} no corresponde a los ámbitos de la campaña",
        )
        pairs = set(scopes.items())
        _require(not pairs & seen, f"La ventana {row['window']} repite un par de la campaña")
        seen |= pairs
        rows.append(dict(id=row["window"], scopes=dict(scopes)))
    expected = {(s, w) for s in campaign["scopes"] for w in resolved[s]["windows"]}
    _require(seen == expected, "El orden no recorre exactamente las ventanas de la campaña")
    return rows


def _name(job_id):
    """Nombre de carpeta estable para un trabajo cuyo identificador contiene barras."""
    readable = re.sub(r"[^A-Za-z0-9.+-]+", "_", job_id)[:80]
    return f"{readable}-{hashlib.sha256(job_id.encode()).hexdigest()[:12]}"


class Rolling:
    """Estado de un recorrido: campaña confirmada, etapas, inventario y registro."""

    def __init__(
        self,
        retention,
        campaign_path,
        views,
        output,
        windows,
        *,
        ablation=None,
        adapters=None,
        rl=None,
        regenerators=None,
        ablation_executors=None,
        disk=None,
        edition=None,
    ):
        """`disk` activa la guardia por ventana: `storage` (declaración), `extras` (medidas
        de agregados, adaptadores y cintas), `adapter_blocks` (lectura sin copia ordenada) y,
        en las pruebas, `usage` en lugar de `shutil.disk_usage`. `edition` es la edición de
        precios sin ajustar que necesitan la cartera larga y corta y los estratos de liquidez si
        la comparación los declara.
        """
        from . import masked_campaign as engine

        self.retention, self.campaign_path = retention, Path(campaign_path)
        self.views, self.output, self.windows = views, Path(output), windows
        self.ablation, self.adapters, self.rl = ablation, adapters, rl
        self._policies = None
        self.regenerators = engine.regenerators() if regenerators is None else regenerators
        self.ablation_executors = ablation_executors
        self.disk = disk
        self.edition = None if edition is None else Path(edition)
        # Asignador de liquidez compartido por todas las ventanas y ámbitos del recorrido: lee
        # cada activo de la edición una vez por proceso y no una por ventana.
        self._liquidity = None
        self.folder = self.output / "retention"
        # Con las políticas declaradas como consumidoras, sin su etapa no se sabría qué
        # evaluaciones leen y se podrían liberar antes de montar sus cintas.
        _require(
            (rl is not None) == ("rl" in retention["consumers"]),
            "La etapa de políticas debe acompañar a la retención que la declara consumidora, "
            "y solo a esa",
        )
        self.campaign, _ = engine._confirmed_state(campaign_path, views, output)
        # Sin la edición no hay agregados de la cartera ni estratos de liquidez, y sus filas
        # no se podrían liberar.
        declared = self.comparison_config()
        _require(
            comparison.LONG_SHORT_FIELD not in declared or self.edition is not None,
            "La comparación declara la cartera larga y corta: falta la edición de precios",
        )
        _require(
            comparison.LIQUIDITY_FIELD not in declared or self.edition is not None,
            "La comparación declara estratos de liquidez: falta la edición de precios",
        )
        self.positions = {
            (scope, window): index
            for index, row in enumerate(windows)
            for scope, window in row["scopes"].items()
        }

    # Registro de fases por ventana.

    @property
    def ledger_path(self):
        return self.folder / "ledger.json"

    def ledger(self):
        if not self.ledger_path.is_file():
            return dict(
                schema_version=1,
                kind=LEDGER_KIND,
                retention_sha256=self.retention["sha256"],
                campaign_sha256=self.campaign["sha256"],
                windows={},
            )
        document, _ = read_manifest(self.ledger_path, 64 * 1024**2)
        _require(
            document.get("kind") == LEDGER_KIND
            and document.get("retention_sha256") == self.retention["sha256"]
            and document.get("campaign_sha256") == self.campaign["sha256"],
            "El registro pertenece a otra retención o a otra campaña",
        )
        return document

    def done(self, window, phase):
        return phase in self.ledger()["windows"].get(window, {}).get("phases", {})

    def mark(self, window, phase, **record):
        document = self.ledger()
        entry = document["windows"].setdefault(window, dict(phases={}))
        entry["phases"][phase] = dict(record, finished_at_utc=datetime.now(UTC).isoformat())
        atomic_json(self.ledger_path, document)

    # Estado confirmado de la base y de las etapas.

    def base_state(self, upto):
        """Recibos confirmados de la base en las ventanas hasta `upto`, en orden del plan."""
        from . import masked_campaign as engine
        from .campaign_plan import plan_campaign

        _, state = engine._confirmed_state(self.campaign_path, self.views, self.output)
        jobs = [
            job
            for job in plan_campaign(self.campaign)
            if self.positions[job["scope"], job["window"]] <= upto
        ]
        for job in engine.with_dependencies(plan_campaign(self.campaign), [j["id"] for j in jobs]):
            case, _, sources = state.resolve(job)
            receipt = state.confirmed(job, state.job_identity(job, case, sources))
            _require(receipt is not None, f"Falta confirmar {job['id']} antes de liberar")
            state.receipts[job["id"]] = receipt
        return state, jobs

    def ablation_state(self, upto):
        from . import modality_ablation_stage as stage_module

        pairs = {pair for pair, index in self.positions.items() if index <= upto}
        stage, base, output = stage_module._opened(
            self.ablation["stage"],
            self.views,
            self.output,
            self.ablation["output"],
            pairs,
        )
        identity = stage_module._identity(stage, base.views)
        executors = self.ablation_executors or stage_module.EXECUTORS
        state = stage_module._Stage(stage, base, output, identity, executors, None)
        jobs = [
            job
            for job in stage_module.plan_stage(stage)
            if self.positions[job["scope"], job["window"]] <= upto
        ]
        for job in jobs:
            receipt = state.confirmed(job, state.job_identity(job))
            _require(receipt is not None, f"Falta confirmar {job['id']} antes de liberar")
            state.receipts[job["id"]] = receipt
        return state, jobs

    def policy_stage(self):
        """Etapa de políticas cargada y su plan, que no cambian durante el recorrido."""
        if self._policies is None:
            from mars_titan.simulation import policy_plan

            stage = policy_plan.load_stage(self.rl["stage"])
            self._policies = stage, policy_plan.plan_stage(stage)
        return self._policies

    def policy_reads(self, state, index):
        """Trabajos base cuya evaluación lee una política y los trabajos de políticas que la leen.

        Solo cuentan las ventanas hasta `index`, las únicas con recibos confirmados.
        """
        if self.rl is None:
            return {}
        from .rolling_storage import policy_readers

        reads = {}
        for (scope, window, arm, chosen), readers in policy_readers(*self.policy_stage()).items():
            if self.positions.get((scope, window), index + 1) <= index:
                key, _ = state.selected(scope, window, arm, chosen)
                reads.setdefault(key, []).extend(readers)
        return reads

    def policy_keep(self, state, index):
        """Trabajos base cuya evaluación leerá una política, con la última ventana que la lee."""
        return {
            key: max(self.positions[job["scope"], job["window"]] for job in readers)
            for key, readers in self.policy_reads(state, index).items()
        }

    def require_policy_receipts(self, readers):
        """Rechazar la liberación mientras una política que lee esas evaluaciones no confirme.

        Cada trabajo de políticas debe tener su recibo completado en la salida de la etapa,
        con la identidad registrada en su `stage.json` y esa etapa. El recibo solo se escribe
        después de montar las cintas, así que con él la tabla ya no hace falta. Liberarla
        antes obligaría a regenerarla en la GPU para la política atrasada.
        """
        from mars_titan.simulation.campaign_stage import RECEIPT_KIND, _digest

        stage, _ = self.policy_stage()
        output = Path(self.rl["output"])
        marker = output / "stage.json"
        identity = read_manifest(marker, 8 * 1024**2)[0] if marker.is_file() else None
        current = isinstance(identity, dict) and identity.get("stage_sha256") == stage["sha256"]
        missing = []
        for job_id in sorted(readers):
            path = output / "jobs" / job_id / "receipt.json"
            receipt = read_manifest(path, 8 * 1024**2)[0] if current and path.is_file() else {}
            if not (
                receipt.get("kind") == RECEIPT_KIND
                and receipt.get("status") == "completed"
                and receipt.get("identity", {}).get("id") == job_id
                and receipt["identity"].get("stage_identity_sha256") == _digest(identity)
            ):
                missing.append(job_id)
        _require(
            not missing,
            f"Faltan {len(missing)} recibos de las políticas que leen las evaluaciones que se "
            "iban a liberar: " + ", ".join(missing[:8]) + (" y otros" if len(missing) > 8 else ""),
        )

    # Guardia de disco de cada ventana.

    def disk_projection(self, row):
        """Crecimiento previsto de una ventana sobre lo ya conservado, con la declaración.

        Usa el escenario sin regeneraciones exactas y la medida declarada, que cuentan las
        tablas compactadas y comunes por arriba. Incluye las páginas de XGBoost y los índices
        de cada ajuste, el corpus de los adaptadores sin lectura por bloques, la ablación,
        las cintas y la regeneración de un trabajo al liberar.
        """
        from . import masked_campaign as engine
        from .campaign_plan import plan_campaign
        from .campaign_storage import load_storage, view_counts
        from .rolling_storage import declared_measure, rolling_estimate, window_increment

        storage = load_storage(self.disk["storage"])
        extras = self.disk["extras"]
        _, state = engine._confirmed_state(self.campaign_path, self.views, self.output)
        counts = view_counts({s: state.views[s] for s in row["scopes"]})
        stages = self._stage_jobs(extras)
        estimate = rolling_estimate(
            [j for j in plan_campaign(self.campaign) if j["scope"] in row["scopes"]],
            [row],
            counts,
            declared_measure(counts, storage),
            storage,
            extras,
            ordered_copy=not self.disk.get("adapter_blocks", False),
            **stages,
        )
        return window_increment(estimate, "none_regenerated"), storage

    def _stage_jobs(self, extras):
        from .rolling_storage import rolling_inputs

        return rolling_inputs(
            [],
            self.windows,
            extras,
            ablation=self.ablation and self.ablation["stage"],
            adapters=self.adapters and self.adapters["stage"],
            rl=self.rl and self.rl["stage"],
        )

    def require_disk(self, row):
        """Rechazar una ventana nueva cuyo crecimiento no cabe en el disco sobre el margen."""
        from .campaign_storage import DiskGuard

        increment, storage = self.disk_projection(row)
        options = {"usage": self.disk["usage"]} if "usage" in self.disk else {}
        guard = DiskGuard(self.output, storage["margin_bytes"], **options)
        return guard.require_launch(
            dict(peak_bytes=increment["bytes"], peak_job=f"la ventana {row['id']}")
        )

    # Fases.

    def comparison_config(self):
        return comparison.load_config(Path(self.campaign["comparison_path"]))

    def aggregates(self, row):
        """Guardar los agregados de la comparación de cada ámbito de una ventana."""
        from . import masked_campaign as engine
        from . import modality_ablation_stage as stage_module

        config = self.comparison_config()
        folder = self.folder / "aggregates"
        written = {}
        for scope, window in row["scopes"].items():
            restricted = comparison.restrict_windows(config, scope, [window])
            path = engine.write_sources(
                self.campaign_path, self.views, self.output, scope, window=window
            )
            sources = comparison.load_sources(path, restricted, scope)
            # Los agregados se puntúan con los brazos y familias del ámbito, como la
            # comparación final que los leerá (en US y CN, los del diseño conjunto).
            restricted = comparison.scope_config(restricted, scope)
            masked = None
            if self.ablation is not None and comparison.ABLATION_FIELD in config:
                masked_path = stage_module.write_sources(
                    self.ablation["stage"],
                    self.views,
                    self.output,
                    self.ablation["output"],
                    scope,
                    window=window,
                )
                masked = comparison._ablation_sources(masked_path, restricted, sources)
            if self._liquidity is None:
                self._liquidity = comparison.liquidity_source(config, self.edition)
            records = dict(
                walk_forward=window_aggregates.write(
                    folder, restricted, sources, window, masked, liquidity=self._liquidity
                )
            )
            if comparison.LONG_SHORT_FIELD in config:
                records["long_short"] = window_aggregates.write_long_short(
                    folder, restricted, sources, window, self.edition
                )
            written[scope] = {
                name: dict(
                    path=str(Path(record["path"]).relative_to(self.folder)),
                    sha256=record["sha256"],
                    bytes=record["bytes"],
                )
                for name, record in records.items()
            }
        return written

    def release(self, index):
        """Liberar o compactar las tablas de las ventanas hasta `index` que nadie leerá."""
        base, jobs = self.base_state(index)
        keep = self.policy_keep(base, index)
        if self.rl is not None:
            # Antes de liberar nada: cada evaluación que deja de conservarse ya la ha leído
            # cada política que la necesita.
            reads = self.policy_reads(base, index)
            self.require_policy_receipts(
                {
                    reader["id"]
                    for job in jobs
                    if keep.get(job["id"], -1) <= index
                    for reader in reads.get(job["id"], ())
                }
            )
        totals = dict(
            released=0, kept_for_policies=0, not_regenerable=0, declared_not_regenerable=0
        )
        freed = 0
        for job in jobs:
            receipt = base.receipts[job["id"]]
            report_path = self.output / receipt["report"]["path"]
            report, _ = read_manifest(report_path, 16 * 1024**2)
            tables = regeneration.originals(report_path.parent, report)
            if keep.get(job["id"], -1) > index:
                # Una política posterior leerá su evaluación: se compacta y se conserva.
                totals["kept_for_policies"] += 1
                freed += self._compact(job, tables)
                continue
            if job.get("regenerable", True) is False or job.get("kind") == ONLINE:
                # Sus predicciones dependen de algo más que la inferencia (un control en línea
                # actualiza sus pesos mientras predice): se compacta sin regenerar.
                totals["declared_not_regenerable"] += 1
                freed += self._compact(job, tables)
                continue

            def regenerate(destination, job=job):
                return regeneration._regenerate_base(
                    self.campaign, base, job, destination, self.regenerators
                )

            freed += self._release_job("base", job, tables, regenerate, totals)
        if self.ablation is not None:
            state, ablation_jobs = self.ablation_state(index)
            # Sin su sección en la comparación, la ablación no tiene agregados: se compacta.
            aggregated = comparison.ABLATION_FIELD in self.comparison_config()
            for job in ablation_jobs:
                receipt = state.receipts[job["id"]]
                record = receipt["prediction"]
                tables = {"evaluation": (state.output / record["path"], record["sha256"])}
                if not aggregated:
                    freed += self._compact(job, tables)
                    continue

                def regenerate(destination, job=job):
                    return regeneration._regenerate_ablation(state, job, destination)

                freed += self._release_job("ablation", job, tables, regenerate, totals)
        if self.adapters is not None:
            freed += self._compact_adapters(index)
        pruned = self._prune_rows()
        return dict(totals, freed_bytes=freed, pruned_rows_bytes=pruned)

    def _rows(self, job, partition):
        folder = self.folder / "rows" / job["scope"] / job["window"] / partition
        safe_destination(folder)
        return folder

    def _compact(self, job, tables):
        freed = 0
        for partition, (path, digest) in tables.items():
            if prediction_files.verify(path, digest) == prediction_files.PRESENT:
                before = path.stat().st_size
                record = prediction_files.compact(path, digest, self._rows(job, partition))
                freed += before - record["compact"]["bytes"]
        return freed

    def _release_job(self, stage, job, tables, regenerate, totals):
        """Regenerar, comparar y liberar solo si todas las tablas salen idénticas.

        Una comparación escrita en `regeneration-reports` es definitiva: al reanudar o en
        ventanas posteriores no se vuelve a regenerar un trabajo ya comparado. Un error
        antes de comparar (por ejemplo, falta de memoria) conserva las tablas y se reintenta
        en la liberación siguiente. Los errores permanentes, como un ajuste sin FP32
        estricto, fallan antes de cualquier inferencia.
        """
        states = {p: prediction_files.verify(path, d) for p, (path, d) in tables.items()}
        if all(state == prediction_files.RELEASED for state in states.values()):
            return 0
        name = _name(f"{stage}:{job['id']}")
        decision = self.folder / "regeneration-reports" / f"{name}.json"
        destination = self.folder / "regeneration" / name
        report = read_manifest(decision, 16 * 1024**2)[0] if decision.is_file() else None
        decided = report is not None and "error" not in report
        if not decided:
            if destination.exists():
                # Un corte anterior dejó una regeneración sin decidir: se repite entera.
                shutil.rmtree(destination)
            try:
                report = regenerate(destination)
            except Exception as error:  # noqa: BLE001 - se registra y se conserva la tabla
                report = dict(identical=False, error=f"{type(error).__name__}: {error}")
            decision.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(decision, dict(report, job=job["id"], stage=stage))
        freed = 0
        if report.get("identical") is True:
            for partition, (path, digest) in tables.items():
                if states[partition] == prediction_files.RELEASED:
                    continue
                size = (
                    path.stat().st_size
                    if states[partition] == prediction_files.PRESENT
                    else prediction_files.entry(path)["compact"]["bytes"]
                )
                prediction_files.release(
                    path,
                    digest,
                    stage=stage,
                    job=job["id"],
                    partition=partition,
                    regeneration_sha256=sha256(decision),
                )
                freed += size
            totals["released"] += 1
        else:
            totals["not_regenerable"] += 0 if decided else 1
            freed += self._compact(job, tables)
        if destination.exists():
            shutil.rmtree(destination)
        return freed

    def _compact_adapters(self, index):
        """Compactar sin pérdida las tablas confirmadas de los adaptadores de la ventana."""
        freed = 0
        output = Path(self.adapters["output"])
        for receipt_path in sorted((output / "jobs").rglob("receipt.json")):
            receipt, _ = read_manifest(receipt_path, 8 * 1024**2)
            identity = receipt.get("identity", {})
            pair = (identity.get("scope"), identity.get("window"))
            if self.positions.get(pair, len(self.windows)) > index:
                continue
            job = dict(id=identity.get("id"), scope=pair[0], window=pair[1])
            tables = {
                partition: (output / record["path"], record["sha256"])
                for partition, record in receipt.get("predictions", {}).items()
            }
            freed += self._compact(job, tables)
        return freed

    def _prune_rows(self):
        """Borrar las tablas comunes a las que ya no apunta ningún archivo compactado."""
        root = self.folder / "rows"
        if not root.is_dir():
            return 0
        referenced = set()
        for base in (
            self.output,
            *(Path(s["output"]) for s in (self.ablation, self.adapters) if s),
        ):
            for record in base.rglob(prediction_files.RETENTION_FILE):
                referenced |= prediction_files.rows_references(record.parent)
        pruned = 0
        for path in root.rglob("rows-*.parquet"):
            if path.resolve() not in referenced:
                pruned += path.stat().st_size
                path.unlink()
        return pruned


def _supports_window(runner):
    return "window" in inspect.signature(runner).parameters


def disjunction_report(campaign, views, output, posttraining, window, phase):
    """Calcular y guardar el informe de disjunción que exige una etapa por etapas.

    Se calcula de nuevo antes de cada etapa que lo necesita, porque la RL lee selecciones de
    la cadena que la etapa de adaptadores acaba de confirmar. Recorre las vistas, los recibos
    de la campaña base y, si existe, la salida del posentrenamiento. El informe se guarda en
    `retention/disjunction/<ventana>-<fase>.json` también cuando registra fallos, para
    revisarlos, y en ese caso el recorrido se detiene antes de la etapa.
    """
    from .chain_disjunction import verify

    report = verify(campaign, views, campaign_output=output, posttraining=posttraining)
    path = Path(output) / "retention" / "disjunction" / f"{window}-{phase}.json"
    atomic_json(path, report)
    _require(
        not report["failures"],
        f"La disjunción registra {len(report['failures'])} fallos antes de {phase} en "
        f"{window}: revisa {path}",
    )
    return path


def default_runners(campaign_path, views, output, *, storage=None, stages=None, stop=None):
    """Órdenes de cada fase limitadas a una ventana de campaña.

    Necesitan el filtro `window` de las etapas, que define el orden por ventanas de la
    campaña A v2. Sin él se rechaza el recorrido antes de ejecutar nada. Si la campaña
    declara `walk_forward_stages`, los adaptadores y las políticas reciben un informe de
    disjunción recién calculado y las políticas leen la cadena de la salida de los adaptadores.
    """
    from mars_titan.posttraining import campaign_stage as adapter_stage
    from mars_titan.simulation import campaign_stage as rl_stage

    from . import masked_campaign as engine
    from . import modality_ablation_stage as ablation_stage
    from .campaign_plan import load_campaign

    stages = stages or {}
    campaign = load_campaign(campaign_path)
    chain_output = stages.get("adapters", {}).get("output")

    def checked(window, phase):
        if not campaign.get("walk_forward_stages"):
            return None
        return disjunction_report(campaign, views, output, chain_output, window, phase)

    for runner in (engine.run_campaign, ablation_stage.run_stage, adapter_stage.run_stage):
        _require(
            _supports_window(runner),
            f"{runner.__module__}.{runner.__name__} no admite una ventana de campaña",
        )
    runners = dict(
        base=lambda window: engine.run_campaign(
            campaign_path, views, output, storage=storage, stop=stop, window=window
        )
    )
    if "adapters" in stages:
        entry = stages["adapters"]
        runners["adapters"] = lambda window: adapter_stage.run_stage(
            entry["stage"],
            views,
            output,
            entry["output"],
            stop=stop,
            window=window,
            disjunction=checked(window, "adapters"),
        )
    if "ablation" in stages:
        entry = stages["ablation"]
        runners["ablation"] = lambda window: ablation_stage.run_stage(
            entry["stage"], views, output, entry["output"], stop=stop, window=window
        )
    if "rl" in stages:
        entry = stages["rl"]
        _require(_supports_window(rl_stage.run_stage), "La etapa de políticas no admite ventana")
        runners["rl"] = lambda window: rl_stage.run_stage(
            entry["stage"],
            views,
            output,
            entry["edition"],
            entry["output"],
            stop=stop,
            window=window,
            chain_output=chain_output,
            disjunction=checked(window, "rl"),
        )
    return runners


def run_rolling(rolling, runners, *, stop=None):
    """Recorrer las ventanas en orden, con cada fase una sola vez y reanudable."""
    summary = {}
    for index, row in enumerate(rolling.windows):
        started = any(rolling.done(row["id"], phase) for phase in PHASES)
        if rolling.disk is not None and not started:
            rolling.require_disk(row)
        for phase in STAGE_PHASES:
            if stop is not None and stop.requested:
                return dict(status="paused", windows=summary)
            if phase in runners and not rolling.done(row["id"], phase):
                status = runners[phase](row["id"])["status"]
                if status != "completed":
                    return dict(status=status, window=row["id"], windows=summary)
                rolling.mark(row["id"], phase)
        if not rolling.done(row["id"], "aggregates"):
            rolling.mark(row["id"], "aggregates", written=rolling.aggregates(row))
        if not rolling.done(row["id"], "release"):
            rolling.mark(row["id"], "release", **rolling.release(index))
        summary[row["id"]] = rolling.ledger()["windows"][row["id"]]
    return dict(status="completed", windows=summary)


def main(argv=None):
    from . import masked_campaign as engine
    from .campaign_plan import load_campaign

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retention", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--views", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument(
        "--extras", type=Path, required=True, help="Medidas de agregados, adaptadores y cintas"
    )
    parser.add_argument(
        "--adapter-blocks",
        action="store_true",
        help="Los adaptadores leen la vista por bloques, sin copia ordenada",
    )
    for name in ("adapter", "ablation", "rl"):
        parser.add_argument(f"--{name}-stage", type=Path)
        parser.add_argument(f"--{name}-output", type=Path)
    parser.add_argument(
        "--edition", type=Path, help="Edición de precios sin ajustar (políticas y cartera)"
    )
    args = parser.parse_args(argv)
    views = engine._views_argument(args.views)
    retention = load_retention(args.retention)
    windows = load_schedule(args.schedule, load_campaign(args.campaign))
    stages = {}
    for name, key in (("adapter", "adapters"), ("ablation", "ablation"), ("rl", "rl")):
        stage, output = getattr(args, f"{name}_stage"), getattr(args, f"{name}_output")
        _require((stage is None) == (output is None), f"--{name}-stage necesita su salida")
        if stage is not None:
            stages[key] = dict(stage=stage, output=output)
    if "rl" in stages:
        _require(args.edition is not None, "Las políticas necesitan --edition")
        stages["rl"]["edition"] = args.edition
    runners = default_runners(
        args.campaign, views, args.output, storage=args.storage, stages=stages
    )
    rolling = Rolling(
        retention,
        args.campaign,
        views,
        args.output,
        windows,
        ablation=stages.get("ablation"),
        adapters=stages.get("adapters"),
        rl=stages.get("rl"),
        edition=args.edition,
        disk=dict(
            storage=args.storage,
            extras=json.loads(args.extras.read_text()),
            adapter_blocks=args.adapter_blocks,
        ),
    )
    result = run_rolling(rolling, runners)
    print(json.dumps(dict(status=result["status"], windows=len(result["windows"]))))
    return 0 if result["status"] == "completed" else 1
