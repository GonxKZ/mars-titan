"""Retención v2: la campaña ventana a ventana y la liberación de las filas que nadie lee.

La campaña completa no cabe en el disco si conserva todas las tablas por fila. La retención
v2, declarada antes de cualquier resultado, recorre cada ventana de campaña en este orden:

1. `base`: ajustes de todos los modelos, selección, semillas extra del elegido y traslados.
2. `adapters`, `ablation` y `rl`: etapas posteriores de esa ventana. Con el walk-forward
   por etapas, los adaptadores y las políticas reciben un informe de disjunción calculado
   justo antes, y las políticas leen la cadena de la salida de los adaptadores.
3. `aggregates`: agregados por sesión de la comparación de cada ámbito de la ventana
   (`evaluation.window_aggregates`), con sus fuentes limitadas a esa ventana, y el efecto
   de elegir entre los casos de búsqueda (`evaluation.selection_effect`). Con la
   publicación declarada (`training.campaign_publication`), también los de la comparación
   postentrenada de cada padre con trabajos de la etapa en la ventana, porque esa
   comparación lee la calibración y la evaluación del modelo base, que se liberan después.
   Cada agregado se guarda con su registro de integridad
   (`integrity.comparison_integrity`): mismas filas en todos los modelos y métricas por
   sesión recalculadas desde las filas. Si falla, el recorrido se detiene antes de liberar.
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

La retención v3 (#611) mantiene la etapa de adaptadores completa y cambia qué se guarda y
cuándo. Cada trabajo de adaptadores compacta sus tablas al confirmarse y deja solo su estado
elegido. Las tablas de la base y de la ablación se compactan al terminar su fase. Al liberar,
las tablas de los adaptadores se regeneran como las de la base y solo se liberan si salen
idénticas bit a bit, por bloques de modelo base y después de guardar sus agregados. Las
evaluaciones del predictor que elige la cadena se conservan compactadas hasta la última
política que las lee, y la caché de cada padre se borra al cerrarlo. Con la declaración v2
todo se comporta como antes.
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
from mars_titan.evaluation import selection_effect, window_aggregates
from mars_titan.evaluation import walk_forward_comparison as comparison
from mars_titan.integrity import comparison_integrity

from . import prediction_regeneration as regeneration
from . import window_indices
from .campaign_plan import ONLINE, PLATEAU

DECLARATION_KIND = "historical_masked_prediction_retention"
LEDGER_KIND = "historical_masked_retention_ledger"
PHASES = ("base", "adapters", "ablation", "rl", "aggregates", "release")
STAGE_PHASES = ("base", "adapters", "ablation", "rl")
AGGREGATES = ("walk_forward", "long_short", "selection_effect")
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
RELEASE = dict(
    stages=["base", "ablation"],
    partitions=["validation", "calibration", "evaluation"],
    requires="bit_exact_regeneration",
    otherwise="compact_lossless",
)
# Retención v3: campos añadidos, liberación de los adaptadores y compactación por fase.
V3_FIELDS = {"adapter_states", "adapter_release", "adapter_mismatch", "parent_caches"}
V3_RELEASE = dict(RELEASE, stages=["base", "ablation", "adapters"])
V3_COMPACT = dict(base="after_phase", adapters="on_confirmation", ablation="after_phase")
V3_STATES = "selected_only_after_confirmation"
V3_ADAPTER_RELEASE = "by_base_arm_block_after_its_aggregates"
V3_MISMATCH = "compact_lossless_and_stop"
V3_CACHES = "released_when_each_parent_closes"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_retention(path):
    """Validar la declaración de la retención v2 o v3 y devolverla con su huella."""
    document, digest = read_manifest(Path(path), 1024**2)
    _require(
        isinstance(document, dict)
        and set(document) in (_FIELDS, _FIELDS | V3_FIELDS)
        and document["schema_version"] == 1
        and document["kind"] == DECLARATION_KIND
        and document["status"] == "declared_before_results"
        and document["order"] == "by_window"
        and tuple(document["phases"]) == PHASES,
        "La retención no cumple su contrato de fases por ventana",
    )
    _require(
        tuple(document["aggregates"]) == AGGREGATES,
        "La retención solo libera filas con agregados de la comparación, de la cartera y del "
        "efecto de la selección",
    )
    v3 = V3_FIELDS <= set(document)
    _require(
        document["release"] == (V3_RELEASE if v3 else RELEASE),
        "Solo se liberan filas regeneradas bit a bit de la base y de la ablación y, con la "
        "v3, también de los adaptadores",
    )
    _require(
        document["compact"] == (V3_COMPACT if v3 else ["adapters"])
        and document["policy_inputs"] == "compact_until_the_last_reading_policy_window"
        and document["numerics"] == STRICT_FP32,
        "La retención compacta los adaptadores, guarda las entradas de las políticas y exige "
        "FP32 estricto",
    )
    _require(
        not v3
        or (
            document["adapter_states"] == V3_STATES
            and document["adapter_release"] == V3_ADAPTER_RELEASE
            and document["adapter_mismatch"] == V3_MISMATCH
            and document["parent_caches"] == V3_CACHES
        ),
        "La v3 deja solo el estado elegido de cada adaptador, libera sus tablas por bloques de "
        "modelo base, se detiene si una no se repite y borra la caché de cada padre al cerrarlo",
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
        publication=None,
        unrecorded_platform=None,
        adapter_device="cuda:0",
        adapter_lease=None,
    ):
        """`disk` activa la guardia por ventana: `storage` (declaración), `extras` (medidas
        de agregados, adaptadores y cintas), `adapter_blocks` (lectura sin copia ordenada) y,
        en las pruebas, `usage` en lugar de `shutil.disk_usage`. `edition` es la edición de
        precios sin ajustar que necesitan la cartera larga y corta y los estratos de liquidez si
        la comparación los declara. `publication` es la declaración del paso final. Con la
        etapa de adaptadores es obligatoria y su comparación postentrenada debe ser la de esa
        etapa, porque sus agregados por ventana se guardan antes de liberar.
        `unrecorded_platform` nombra el perfil al que se atribuyen los recibos anteriores al
        registro de la plataforma al publicar los agregados. `adapter_device` y
        `adapter_lease` son el dispositivo y la reserva de la GPU con los que la v3 regenera
        los adaptadores (CPU y un sustituto de la reserva en las pruebas).
        """
        from . import masked_campaign as engine
        from .campaign_publication import load_publication

        self.retention, self.campaign_path = retention, Path(campaign_path)
        self.views, self.output, self.windows = views, Path(output), windows
        self.ablation, self.adapters, self.rl = ablation, adapters, rl
        self._policies = None
        self.regenerators = engine.regenerators() if regenerators is None else regenerators
        self.ablation_executors = ablation_executors
        self.disk = disk
        self.edition = None if edition is None else Path(edition)
        self.unrecorded_platform = unrecorded_platform
        self.adapter_device, self.adapter_lease = adapter_device, adapter_lease
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
        self.publication = None if publication is None else load_publication(publication)
        stage = None if self.publication is None else self.publication["posttraining"]
        _require(
            self.publication is None
            or self.publication["campaign_config"]["sha256"] == self.campaign["sha256"],
            "La publicación declara otra campaña",
        )
        # Sin la comparación postentrenada declarada, la liberación borraría las filas del
        # modelo base que esa comparación lee antes de guardar sus agregados.
        _require(
            (adapters is None and stage is None)
            or (
                adapters is not None
                and stage is not None
                and Path(adapters["stage"]).resolve() == Path(stage["stage"]["path"])
            ),
            "La etapa de adaptadores del recorrido y la comparación postentrenada de la "
            "publicación van juntas y deben ser la misma etapa",
        )
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

    @property
    def v3(self):
        """Si la declaración es la retención v3, que también libera los adaptadores."""
        return "adapters" in self.retention["release"]["stages"]

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

    def chain_keep(self, index, only=None):
        """Trabajos de adaptadores que elige la cadena y cuya evaluación leerá una política.

        Devuelve, para cada trabajo elegido en una ventana hasta `index`, la última posición
        de ventana que lo lee y los trabajos de políticas que lo leen. La primera ventana de
        cada ámbito elige el estado de la base y la cubre `policy_keep`. `only` limita la
        búsqueda a un ámbito, una ventana y un modelo base, cuyas selecciones ya existen.
        """
        if self.rl is None or self.adapters is None:
            return {}
        from .campaign_chain import read_selection
        from .rolling_storage import chain_readers

        keep = {}
        for (scope, window, arm, seed), readers in chain_readers(*self.policy_stage()).items():
            if self.positions.get((scope, window), index + 1) > index:
                continue
            if only is not None and (scope, window, arm) != only:
                continue
            selection = read_selection(Path(self.adapters["output"]), scope, window, arm, seed)
            _require(
                selection is not None,
                f"Falta la selección de la cadena de {scope}/{window}/{arm} con la semilla {seed}",
            )
            last, found = keep.get(selection["selected"]["job"], (-1, []))
            last = max([last, *(self.positions[job["scope"], job["window"]] for job in readers)])
            keep[selection["selected"]["job"]] = (last, found + list(readers))
        return keep

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
        las cintas y la regeneración de un trabajo al liberar. Con la v3 los adaptadores se
        proyectan liberados, porque una regeneración suya que no sale idéntica detiene el
        recorrido (`_require_identical`).
        """
        from . import masked_campaign as engine
        from .campaign_plan import plan_campaign
        from .campaign_storage import load_storage, view_counts
        from .rolling_storage import (
            ADAPTERS_REGENERATED,
            declared_measure,
            rolling_estimate,
            window_increment,
        )

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
            v3=self.v3,
            **stages,
        )
        scenario = ADAPTERS_REGENERATED if self.v3 else "none_regenerated"
        return window_increment(estimate, scenario), storage

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
                self.campaign_path,
                self.views,
                self.output,
                scope,
                window=window,
                unrecorded_platform=self.unrecorded_platform,
            )
            sources = comparison.load_sources(path, restricted, scope)
            # Los agregados se puntúan con los modelos y familias del ámbito, como la
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
                    unrecorded_platform=self.unrecorded_platform,
                )
                masked = comparison._ablation_sources(masked_path, restricted, sources)
            # La integridad de la ventana se comprueba antes de liberar ninguna de sus filas.
            walked, checked = comparison_integrity.write_checked(
                folder, restricted, sources, window, masked, liquidity=self.liquidity()
            )
            records = dict(walk_forward=walked, integrity=checked)
            if comparison.LONG_SHORT_FIELD in config:
                records["long_short"] = window_aggregates.write_long_short(
                    folder, restricted, sources, window, self.edition
                )
            # Medir la elección entre búsquedas exige la evaluación de todos los casos, también
            # la de los no elegidos, y la liberación la borra después.
            _, searches = engine.search_records(
                self.campaign_path, self.views, self.output, scope, window=window
            )
            records["selection_effect"] = selection_effect.write_window(
                folder, config, scope, window, searches.get(window, {})
            )
            written[scope] = {name: self._record(record) for name, record in records.items()}
            parents = self.posttraining_aggregates(scope, window, path, folder)
            if parents:
                written[scope]["posttraining"] = parents
        return written

    def _record(self, record):
        path = Path(record["path"])
        return dict(
            path=str(path.relative_to(self.folder)),
            sha256=record["sha256"],
            bytes=record.get("bytes", path.stat().st_size),
        )

    def posttraining_aggregates(self, scope, window, base_sources, folder):
        """Agregados de la comparación postentrenada de cada padre con trabajos en la ventana.

        `base_sources` es el manifiesto de la campaña limitado a la ventana. Se escriben antes
        de la liberación, que borra la calibración y la evaluación del modelo base que leen.
        Cada padre hereda los estratos de liquidez de la comparación de la campaña, así que
        usa el mismo asignador que sus agregados.
        """
        from mars_titan.posttraining import stage_comparison

        stage = self.publication and self.publication["posttraining"]
        if stage is None or scope not in stage["stage"]["scopes"]:
            return {}
        written = {}
        for base_arm in stage["configs"]:
            if window not in stage_comparison.compared_windows(stage, scope, base_arm):
                continue
            block = self._block_record(scope, window, base_arm)
            if block is not None and block["aggregates"] is not None:
                # La retención v3 ya los guardó al cerrar el bloque, antes de liberar sus filas.
                written[base_arm] = block["aggregates"]
                continue
            written[base_arm] = self._posttraining_aggregate(
                scope, window, base_arm, base_sources, folder
            )
        return written

    def _posttraining_aggregate(self, scope, window, base_arm, base_sources, folder):
        """Agregados de la comparación postentrenada de un padre en una ventana."""
        from mars_titan.posttraining import stage_comparison

        record = stage_comparison.write_window_aggregates(
            self.publication["posttraining_path"],
            scope,
            base_arm,
            window,
            base_sources=base_sources,
            stage_output=self.adapters["output"],
            root=folder,
            loaded=self.publication["posttraining"],
            liquidity=self.liquidity(),
            unrecorded_platform=self.unrecorded_platform,
        )
        return dict(self._record(record), integrity=self._record(record["integrity"]))

    def liquidity(self):
        """Asignador de liquidez compartido por todas las ventanas y ámbitos del recorrido."""
        if self._liquidity is None:
            self._liquidity = comparison.liquidity_source(self.comparison_config(), self.edition)
        return self._liquidity

    # Bloques de la retención v3.

    def _block_path(self, scope, window, base_arm):
        return self.folder / "blocks" / scope / window / f"{base_arm}.json"

    def _block_record(self, scope, window, base_arm):
        """Registro del bloque cerrado, con sus agregados comprobados, o None."""
        path = self._block_path(scope, window, base_arm)
        if not path.is_file():
            return None
        record, _ = read_manifest(path, 1024**2)
        stored = record["aggregates"]
        for item in () if stored is None else (stored, stored["integrity"]):
            _require(
                sha256(self.folder / item["path"]) == item["sha256"],
                f"Los agregados del bloque {scope}/{window}/{base_arm} han cambiado",
            )
        return record

    def block(self, state, scope, window, base_arm):
        """Retención v3: cerrar en la etapa el bloque de un modelo base en una ventana.

        La etapa lo llama al confirmar la última selección de la cadena del modelo en la
        ventana, con su reserva de la GPU. Primero se guardan los agregados de su comparación
        postentrenada, que leen todas sus tablas. Después se regeneran y se liberan sus tablas,
        salvo la evaluación del predictor elegido que leerá una política de esta ventana o de
        una posterior: esa se conserva compactada y la libera la ventana de su último lector.
        Así la fase no acumula las tablas de toda la ventana. Es idempotente.
        """
        from mars_titan.posttraining import stage_comparison
        from mars_titan.posttraining.campaign_stage import plan_stage

        from . import masked_campaign as engine

        jobs = [
            job
            for job in plan_stage(state.stage)
            if (job["scope"], job["window"], job["base_arm"]) == (scope, window, base_arm)
        ]
        if not jobs:
            return None
        path = self._block_path(scope, window, base_arm)
        record = self._block_record(scope, window, base_arm)
        if record is None:
            aggregates = None
            stage = self.publication and self.publication["posttraining"]
            if (
                stage is not None
                and scope in stage["stage"]["scopes"]
                and base_arm in stage["configs"]
                and window in stage_comparison.compared_windows(stage, scope, base_arm)
            ):
                sources = engine.write_sources(
                    self.campaign_path,
                    self.views,
                    self.output,
                    scope,
                    window=window,
                    unrecorded_platform=self.unrecorded_platform,
                )
                aggregates = self._posttraining_aggregate(
                    scope, window, base_arm, sources, self.folder / "aggregates"
                )
            record = dict(aggregates=aggregates)
            atomic_json(path, record)
        index = self.positions[scope, window]
        keep = self.chain_keep(index, only=(scope, window, base_arm))
        totals = dict(released=0, kept_for_policies=0, not_regenerable=0)
        freed = 0
        for job in jobs:
            receipt = state.receipts[job["id"]]
            tables = {
                partition: (state.output / item["path"], item["sha256"])
                for partition, item in receipt["predictions"].items()
            }
            if job["id"] in keep:
                totals["kept_for_policies"] += 1
                continue

            def regenerate(destination, job=job):
                return regeneration._regenerate_adapter(state, job, destination)

            freed += self._release_job("adapters", job, tables, regenerate, totals)
        record = dict(record, **totals, freed_bytes=freed)
        atomic_json(path, record)
        self._require_identical("adapters", jobs)
        return record

    def _require_identical(self, stage, jobs):
        """Retención v3: detener el recorrido si alguna regeneración de adaptadores no salió
        idéntica bit a bit o falló.

        Sus tablas ya están compactadas sin pérdida, pero la comparación supone que cada tabla
        se repite desde su estado, así que hay que revisarlo antes de seguir. Una decisión
        con error se repite al reanudar. Una diferencia comparada es definitiva y la detención
        se repite hasta que se revise.
        """
        failed = []
        for job in jobs:
            path = self.folder / "regeneration-reports" / f"{_name(f'{stage}:{job["id"]}')}.json"
            if path.is_file() and read_manifest(path, 16 * 1024**2)[0].get("identical") is not True:
                failed.append(job["id"])
        _require(
            not failed,
            f"{len(failed)} regeneraciones de {stage} no salen idénticas o fallaron, con las "
            "tablas compactadas: "
            + ", ".join(failed[:8])
            + (" y otras" if len(failed) > 8 else ""),
        )

    def release(self, index):
        """Liberar o compactar las tablas de las ventanas hasta `index` que nadie leerá."""
        base, jobs = self.base_state(index)
        keep = self.policy_keep(base, index)
        chain = self.chain_keep(index) if self.v3 else {}
        if self.rl is not None:
            # Antes de liberar nada: cada evaluación que deja de conservarse ya la ha leído
            # cada política que la necesita, también las de la cadena con la v3.
            reads = self.policy_reads(base, index)
            readers = {
                reader["id"]
                for job in jobs
                if keep.get(job["id"], -1) <= index
                for reader in reads.get(job["id"], ())
            }
            readers |= {
                reader["id"] for last, found in chain.values() if last <= index for reader in found
            }
            self.require_policy_receipts(readers)
        totals = dict(
            released=0, kept_for_policies=0, not_regenerable=0, declared_not_regenerable=0
        )
        freed = 0
        for job in jobs:
            if job.get("phase") == PLATEAU:
                # La meseta de un ajuste conjunto no escribe tablas. Las del ajuste son las de
                # su continuación, que se regenera desde el estado elegido.
                continue
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
        extra = {}
        if self.adapters is not None and self.v3:
            adapters = dict(released=0, kept_for_policies=0, not_regenerable=0)
            last = {job_id: position for job_id, (position, _) in chain.items()}
            freed += self._release_adapters(index, last, adapters)
            extra = dict(adapters=adapters, parent_cache_bytes=self._release_parent_caches(index))
        elif self.adapters is not None:
            freed += self._compact_adapters(index)
        pruned = self._prune_rows()
        indices = self._release_indices(base, index)
        return dict(
            totals, freed_bytes=freed, pruned_rows_bytes=pruned, index_bytes=indices, **extra
        )

    def after_phase(self, index, row, phase):
        """Retención v3: compactar sin pérdida las tablas de la fase que acaba de terminar.

        Las fases siguientes y los agregados las leen compactadas con `prediction_files.read`.
        Los adaptadores ya se compactan al confirmar cada trabajo y se liberan por bloques, así
        que aquí solo se recogen los que se confirmaron antes sin compactar y se resumen sus
        bloques. Devuelve lo que se añade al registro de la fase. Con la v2 no hace nada.
        """
        if not self.v3:
            return {}
        freed, record = 0, {}
        if phase == "base":
            base, jobs = self.base_state(index)
            for job in jobs:
                if (
                    job.get("phase") == PLATEAU
                    or self.positions[job["scope"], job["window"]] != index
                ):
                    continue
                report_path = self.output / base.receipts[job["id"]]["report"]["path"]
                report, _ = read_manifest(report_path, 16 * 1024**2)
                freed += self._compact(job, regeneration.originals(report_path.parent, report))
        elif phase == "adapters" and self.adapters is not None:
            freed = self._compact_adapters(index, window_only=True)
            blocks = [
                read_manifest(path, 1024**2)[0]
                for scope, window in row["scopes"].items()
                for path in sorted((self.folder / "blocks" / scope / window).glob("*.json"))
            ]
            record["blocks"] = {
                key: sum(block.get(key, 0) for block in blocks)
                for key in ("released", "kept_for_policies", "not_regenerable", "freed_bytes")
            }
            record["blocks"]["count"] = len(blocks)
        elif phase == "ablation" and self.ablation is not None:
            state, jobs = self.ablation_state(index)
            for job in jobs:
                if self.positions[job["scope"], job["window"]] == index:
                    item = state.receipts[job["id"]]["prediction"]
                    tables = {"evaluation": (state.output / item["path"], item["sha256"])}
                    freed += self._compact(job, tables)
        return dict(record, compacted_bytes=freed)

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

    def _compact_adapters(self, index, window_only=False):
        """Compactar sin pérdida las tablas confirmadas de los adaptadores hasta la ventana
        `index`, o solo las de esa ventana con `window_only`.

        Con `window_only` (retención v3, al terminar la fase) deja además solo el estado
        elegido de cada ajuste, como hace la etapa al confirmar cada trabajo.
        """
        from .checkpoints import release_recovery_states

        freed = 0
        output = Path(self.adapters["output"])
        for receipt_path in sorted((output / "jobs").rglob("receipt.json")):
            receipt, _ = read_manifest(receipt_path, 8 * 1024**2)
            identity = receipt.get("identity", {})
            pair = (identity.get("scope"), identity.get("window"))
            position = self.positions.get(pair, len(self.windows))
            if position > index or (window_only and position != index):
                continue
            job = dict(id=identity.get("id"), scope=pair[0], window=pair[1])
            tables = {
                partition: (output / record["path"], record["sha256"])
                for partition, record in receipt.get("predictions", {}).items()
            }
            freed += self._compact(job, tables)
            if window_only:
                for marker in sorted(receipt_path.parent.rglob("checkpoints/identity.json")):
                    freed += release_recovery_states(marker.parent)
        return freed

    def _release_adapters(self, index, keep, totals):
        """Retención v3: regenerar y liberar las tablas de los adaptadores hasta `index`.

        Cada trabajo se regenera desde su estado elegido con `_release_job`, como la base, y
        sus tablas solo se liberan si salen idénticas bit a bit. La evaluación que leerá una
        política posterior (`keep` da su última ventana) se conserva compactada. Los trabajos
        ya liberados no se vuelven a abrir.
        """
        from mars_titan.posttraining import campaign_stage as adapters

        output = Path(self.adapters["output"])
        pairs = {pair for pair, position in self.positions.items() if position <= index}
        state, jobs = adapters.opened_stage(
            self.adapters["stage"],
            self.views,
            self.output,
            output,
            pairs,
            device=self.adapter_device,
            stop=regeneration._Running(),
        )
        pending = [job for job in jobs if not self._released(output, state.folder(job))]
        freed = 0
        if not pending:
            return freed
        with (self.adapter_lease or adapters._gpu_lease)() as state.lease:
            try:
                for job in pending:
                    receipt = state.confirmed(job, state.job_identity(job))
                    _require(receipt is not None, f"Falta confirmar {job['id']} antes de liberar")
                    state.receipts[job["id"]] = receipt
                    tables = {
                        partition: (output / record["path"], record["sha256"])
                        for partition, record in receipt["predictions"].items()
                    }
                    if keep.get(job["id"], -1) > index:
                        totals["kept_for_policies"] += 1
                        freed += self._compact(job, tables)
                        continue

                    def regenerate(destination, job=job):
                        return regeneration._regenerate_adapter(state, job, destination)

                    freed += self._release_job("adapters", job, tables, regenerate, totals)
            finally:
                state.close()
        self._require_identical("adapters", pending)
        return freed

    @staticmethod
    def _released(output, folder):
        """Si todas las tablas de un trabajo confirmado ya están liberadas."""
        path = folder / "receipt.json"
        if not path.is_file():
            return False
        receipt, _ = read_manifest(path, 8 * 1024**2)
        return all(
            prediction_files.verify(output / record["path"], record["sha256"])
            == prediction_files.RELEASED
            for record in receipt["predictions"].values()
        )

    def _release_parent_caches(self, index):
        """Retención v3: borrar las cachés de predicciones de los padres hasta `index`.

        Solo aceleran los ajustes de su ventana, que ya están confirmados, y se reconstruyen
        desde el padre si hiciera falta. La regeneración predice con el padre, no con ellas.
        """
        freed = 0
        root = Path(self.adapters["output"]) / "windows-data"
        for (scope, window), position in self.positions.items():
            if position > index:
                continue
            parents = root / scope / window / "parents"
            for path in sorted(parents.glob("*/seed-*/predictions.sqlite*")):
                if path.is_file() and not path.is_symlink():
                    freed += path.stat().st_size
                    path.unlink()
        return freed

    def _release_indices(self, base, index):
        """Borrar los índices compartidos de las vistas de las ventanas hasta `index`.

        Va después de todas las regeneraciones de la liberación, que los reutilizan. Ningún
        trabajo posterior lee la vista de una ventana cerrada: los adaptadores y las políticas
        de la ventana siguiente leen la suya.
        """
        return sum(
            window_indices.release_view_indices(base.views[scope]["windows"][window]["sha256"])
            for (scope, window), position in self.positions.items()
            if position <= index
        )

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


def default_runners(
    campaign_path,
    views,
    output,
    *,
    storage=None,
    stages=None,
    stop=None,
    unrecorded_platform=None,
    execution=None,
    retention=None,
    on_block=None,
):
    """Órdenes de cada fase limitadas a una ventana de campaña.

    Necesitan el filtro `window` de las etapas, que define el orden por ventanas de la
    campaña A v2. Sin él se rechaza el recorrido antes de ejecutar nada. Si la campaña
    declara `walk_forward_stages`, los adaptadores y las políticas reciben un informe de
    disjunción recién calculado y las políticas leen la cadena de la salida de los adaptadores.
    `execution` es la declaración de ejecución de la campaña base. Sus trazas de #448 llegan
    también a los adaptadores y a las políticas.
    Las políticas contrastan su plataforma con los recibos de la base, y `unrecorded_platform`
    nombra el perfil de los anteriores a ese registro. Con la retención v3 (`retention`), la
    etapa de adaptadores compacta cada trabajo al confirmarlo en las filas comunes de la
    retención y llama a `on_block` (`Rolling.block`) al cerrar cada modelo base en una ventana.
    """
    from mars_titan.posttraining import campaign_stage as adapter_stage
    from mars_titan.simulation import campaign_stage as rl_stage

    from . import masked_campaign as engine
    from . import modality_ablation_stage as ablation_stage
    from .campaign_plan import load_campaign
    from .campaign_resources import Execution, load_execution

    stages = stages or {}
    campaign = load_campaign(campaign_path)
    chain_output = stages.get("adapters", {}).get("output")
    if execution is not None and not isinstance(execution, Execution):
        execution = load_execution(execution, scopes=tuple(campaign["scopes"]))
    traces = None if execution is None else execution.traces

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
            campaign_path,
            views,
            output,
            storage=storage,
            stop=stop,
            window=window,
            execution=execution,
        )
    )
    if "adapters" in stages:
        entry = stages["adapters"]
        v3 = retention is not None and "adapters" in retention["release"]["stages"]
        rows = Path(output) / "retention" / "rows" if v3 else None
        runners["adapters"] = lambda window: adapter_stage.run_stage(
            entry["stage"],
            views,
            output,
            entry["output"],
            stop=stop,
            window=window,
            disjunction=checked(window, "adapters"),
            traces=traces,
            compact_rows=rows,
            on_block=on_block if v3 else None,
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
            listing_status=entry["listing_status"],
            stop=stop,
            window=window,
            chain_output=chain_output,
            disjunction=checked(window, "rl"),
            traces=traces,
            unrecorded_platform=unrecorded_platform,
        )
    return runners


def run_rolling(rolling, runners, *, stop=None):
    """Recorrer las ventanas en orden, con cada fase una sola vez y reanudable.

    Todas las fases y la liberación comparten las huellas y los índices de la campaña.
    """
    from . import masked_campaign as engine

    with engine.scoped_environment(engine.shared_files(rolling.output)):
        return _run_windows(rolling, runners, stop)


def _run_windows(rolling, runners, stop):
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
                rolling.mark(row["id"], phase, **rolling.after_phase(index, row, phase))
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
    parser.add_argument(
        "--listing-status",
        type=Path,
        help="Tabla del estado de cotización que declaran las políticas",
    )
    parser.add_argument("--publication", type=Path, help="Declaración del paso final de la campaña")
    parser.add_argument(
        "--execution", type=Path, help="Declaración de ejecución (recursos y trazas de #448)"
    )
    parser.add_argument(
        "--unrecorded-platform",
        help="Perfil de hardware al que se atribuyen los recibos sin plataforma",
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
        _require(args.listing_status is not None, "Las políticas necesitan --listing-status")
        stages["rl"].update(edition=args.edition, listing_status=args.listing_status)
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
        publication=args.publication,
        unrecorded_platform=args.unrecorded_platform,
        disk=dict(
            storage=args.storage,
            extras=json.loads(args.extras.read_text()),
            adapter_blocks=args.adapter_blocks,
        ),
    )
    runners = default_runners(
        args.campaign,
        views,
        args.output,
        storage=args.storage,
        stages=stages,
        execution=args.execution,
        unrecorded_platform=args.unrecorded_platform,
        retention=retention,
        on_block=rolling.block,
    )
    result = run_rolling(rolling, runners)
    print(json.dumps(dict(status=result["status"], windows=len(result["windows"]))))
    return 0 if result["status"] == "completed" else 1
