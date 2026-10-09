"""Postentrenamiento de un brazo con lector episódico: núcleo, lectura o ambos.

El brazo de MARS-TITAN o CM-v1 es un núcleo Titans-MAC `mac_online` congelado y un lector
episódico ajustado sobre él. `ReadoutAdapterTrainer` recorre los mismos instantes, bancos y
etiquetas maduras que `ReadoutTrainer` y solo cambia lo que se ajusta:

- `episodic_readout`: adaptadores de `query_projection` y `value_projection` del lector. El
  núcleo sigue congelado y se reutiliza la repetición del lector del ajuste base.
- `core`: adaptadores del núcleo (cabeza, lectura MAC y fusión). La predicción depende ahora
  de parámetros del núcleo, así que cada tramo se emite con el mismo cálculo diferenciable,
  se corta el grafo y al actualizar se repite el núcleo por bloques de flujos desde el
  estado de cada flujo al empezar el tramo, con BPTT dentro del tramo, como la acumulación
  de `training.financial_run`. El lector se aplica sobre cada bloque repetido con la misma
  instantánea del banco que vio al emitir.
- `full_continuation`: todos los parámetros del lector desde su estado elegido, como el
  ajuste base pero con el presupuesto de la matriz.

Los pesos rápidos siguen la regla asociativa de Titans en cada observación. El banco
conserva su política, sus claves del codec y sus errores maduros de la predicción emitida.
Con el control C en modo penalty (núcleo de B+C y B+C+M), el objetivo del núcleo suma sus
términos sobre el operador MAC que se ejecuta, que ahora incluye los adaptadores. Sin
adaptadores en el núcleo, C no tiene nada que ajustar y el padre llega en modo disabled.
"""

import hashlib
import importlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import torch

from mars_titan.data.storage import sha256
from mars_titan.models.predictive_adaptation import adapter_names, base_digest
from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.episodic_readout import apply_episodic_readout
from mars_titan.models.titans.frozen_financial import _numerics
from mars_titan.training.financial_run import ChronologicalInference, _split, _stack
from mars_titan.training.mars_titan_run import MarsTitanInference, ReadoutTrainer

_OWN = (
    "mars_titan.posttraining.readout_adapters",
    "mars_titan.models.predictive_adaptation",
)


class ReadoutAdapterTrainer(ReadoutTrainer):
    """Ajustar adaptadores del núcleo o del lector, o continuar el lector, sin otros cambios.

    `posttraining` es la declaración del caso que entra en la identidad. `core_rows` es el
    tamaño de los bloques de flujos de la repetición del núcleo (`accumulation_rows` de la
    receta del padre). Con None, el tramo se repite con todos sus flujos.
    """

    def __init__(self, predictor, readout, recipe, *, posttraining, core_rows=None, **options):
        if not isinstance(posttraining, dict) or not posttraining:
            raise ValueError("El postentrenamiento del lector necesita su declaración")
        if core_rows is not None and (type(core_rows) is not int or not 1 <= core_rows <= 256):
            raise ValueError("Los bloques de flujos del núcleo deben ser un entero de 1 a 256")
        self.core = bool(adapter_names(predictor))
        self.core_rows, self._source, self._measured = core_rows, None, None
        # Por etapas, las escalas M3 son las congeladas del brazo padre en su ventana.
        self._parent_scalers = (posttraining.get("placement") or {}).get("scalers")
        control = predictor.local_control
        self.penalized = self.core and control is not None and control.config.mode == "penalty"
        super().__init__(predictor, readout, recipe, **options)
        if not self.core and not adapter_names(readout) and posttraining.get("control") is None:
            raise ValueError("Sin adaptadores el caso debe ser la continuación del lector")
        self.base_sha256 = base_digest(predictor) if self.core else None
        if self.core:
            self.identity["parent"]["state"] = "selected_checkpoint_frozen_with_core_adapters"
        self.identity["posttraining"] = dict(
            posttraining,
            core_rows=core_rows,
            core_base_parameters_sha256=self.base_sha256,
            core_gradient=(
                "segment_replayed_by_flow_blocks_from_segment_start_bptt_v1" if self.core else None
            ),
            control_penalty_on_adapted_operator=self.penalized,
        )
        self.identity = json.loads(canonical(self.identity))
        self.run_id = hashlib.sha256(canonical(self.identity).encode()).hexdigest()

    def _check_scalers(self, scalers, train):
        if self._parent_scalers is None:
            return ReadoutTrainer._check_scalers(scalers, train)
        if (
            asdict(scalers) != self._parent_scalers
            or scalers.decision_end > train.phase.decision_start
        ):
            raise ValueError("Las escalas M3 no son las del brazo padre anteriores al ajuste")
        return None

    # Con el núcleo adaptado, el padre debe ser mac_online sin diagnóstico C y solo pueden
    # requerir gradiente sus correcciones. Sin núcleo rige la regla del ajuste base.
    def _check_parent(self, predictor):
        if not self.core:
            return MarsTitanInference._check_parent(predictor)
        control = predictor.local_control
        adapters = set(adapter_names(predictor))
        if predictor.config.variant != "mac_online" or (
            control is not None and control.config.mode == "diagnostic"
        ):
            raise ValueError("El núcleo adaptado es Titans-MAC mac_online, sin diagnóstico C")
        if predictor.training or any(
            value.requires_grad != (name in adapters)
            for name, value in predictor.named_parameters()
        ):
            raise ValueError("El núcleo del postentrenamiento solo ajusta sus adaptadores")
        return None

    def _parameter_groups(self, readout, admission):
        readout_adapters = adapter_names(readout)
        if not self.core and not readout_adapters:
            return super()._parameter_groups(readout, admission)
        if any(
            value.requires_grad
            for name, value in readout.named_parameters()
            if name not in readout_adapters
        ):
            raise ValueError("Con adaptadores, el lector solo ajusta sus correcciones")
        roles, named = {}, {}
        if readout_adapters:
            values = dict(readout.named_parameters())
            roles["episodic_read_adapters"] = list(readout_adapters)
            named.update({name: values[name] for name in readout_adapters})
        if self.core:
            values = dict(self.predictor.named_parameters())
            names = adapter_names(self.predictor)
            roles["core_adapters"] = [f"core.{name}" for name in names]
            named.update({f"core.{name}": values[name] for name in names})
        return roles, [], named

    @staticmethod
    def _code():
        own = {name: sha256(Path(importlib.import_module(name).__file__)) for name in _OWN}
        return {**ReadoutTrainer._code(), **own}

    def _check_runtime(self):
        if not self.core:
            return super()._check_runtime()
        if (
            self._code() != self.identity["implementation"]
            or _numerics() != self.identity["numerics"]
        ):
            raise ValueError("El código o la configuración numérica cambiaron durante el recorrido")
        self.predictor.verify_parameter_identity()
        if base_digest(self.predictor) != self.base_sha256:
            raise ValueError("El ajuste cambió parámetros del núcleo fuera de sus adaptadores")
        return None

    # Los checkpoints guardan también las correcciones del núcleo, que no forman parte del
    # lector. Al reanudar se recargan y se comprueba la huella sellada del núcleo.
    def _extra_state(self):
        if not self.core:
            return {}
        values = dict(self.predictor.named_parameters())
        return dict(
            core_adapters={
                name: values[name].detach().cpu().clone() for name in adapter_names(self.predictor)
            },
            core_sha256=self.predictor._parameter_id,
        )

    def _restore_extra(self, state):
        if not self.core:
            return
        values = dict(self.predictor.named_parameters())
        saved = state.get("core_adapters")
        if not isinstance(saved, dict) or list(saved) != adapter_names(self.predictor):
            raise ValueError("El checkpoint no contiene los adaptadores del núcleo")
        with torch.no_grad():
            for name, value in saved.items():
                if value.shape != values[name].shape or value.dtype != values[name].dtype:
                    raise ValueError("Los adaptadores del núcleo no conservan forma o precisión")
                values[name].copy_(value)
        self.predictor._seal_parameters()
        if self.predictor._parameter_id != state["core_sha256"]:
            raise ValueError("Los adaptadores recuperados del núcleo no conservan su huella")

    # Se recuerda la fuente que se evalúa porque el plan de C del evento depende de ella.
    def evaluate(self, source, *, stop=None, rows=None):
        self._source = source
        return super().evaluate(source, stop=stop, rows=rows)

    def _event_plan(self, run, phase, event, batches, *, train):
        """Con C penalty, la selección del evento se fija antes de dividirlo, como en Titans."""
        if not self.penalized:
            return {}
        source = self.train if phase == self.train.phase else self._source
        return ChronologicalInference._control_plan(self, run, source, event, batches)

    def _observe(self, run, phase, event, *, train):
        self._measured = []
        try:
            super()._observe(run, phase, event, train=train)
            if self._measured:
                keep = train and self.core and event.at >= phase.decision_start
                ChronologicalInference._penalty(run, self._measured, keep=keep, replay=True)
        finally:
            self._measured = None

    def _prepare_block(self, run, batch, plan, *, train):
        """Emitir con el cálculo diferenciable de la repetición y cortar el grafo.

        En ajuste con el núcleo adaptado se guarda el estado de cada flujo al empezar el tramo.
        La predicción emitida y la repetición recorren así los mismos núcleos.
        """
        replay = self.core and train
        if replay:
            for flow in batch.flow_ids:
                if flow not in run.starts:
                    ((_, run.starts[flow]),) = _split(run.flows[flow], detach=True)
        state = _stack([run.flows[flow] for flow in batch.flow_ids])
        with torch.set_grad_enabled(replay):
            prepared = self.predictor.prepare(batch, state, differentiable=replay, **plan)
        run.flows.update(_split(prepared.next_state, detach=replay))
        control = prepared.local_control
        if control is not None and self._measured is not None:
            self._measured.append((control.penalty.detach(), control.flow_ids))
        if not replay:
            return prepared
        return replace(
            prepared,
            point_predictions=prepared.point_predictions.detach(),
            working_state=prepared.working_state.detach(),
            quantiles=None if prepared.quantiles is None else prepared.quantiles.detach(),
            local_control=None,
        )

    def _block_extra(self, batch, plan):
        return dict(batch=batch, plan=plan) if self.core else {}

    # El paso del lector puede no recibir gradiente cuando la instantánea no tiene episodios.
    def _backward_block(self, loss, record):
        """Sin episodios en la instantánea, la lectura episódica no interviene en la pérdida.

        Es el único caso en que el bloque no aporta gradiente a los adaptadores del lector, y
        solo se admite cuando no se ajusta nada más.
        """
        if loss.requires_grad:
            return loss.backward()
        snapshot = record["snapshot"]
        if (
            self.core
            or not adapter_names(self.readout)
            or (snapshot is not None and snapshot.count)
        ):
            raise ValueError("La pérdida del bloque no depende de los parámetros ajustables")
        return None

    def _update(self, run, at):
        if not self.core:
            return super()._update(run, at)
        total = len(run.used)
        if total:
            loss_sum = self._replay(run, total)
            torch.nn.utils.clip_grad_norm_(
                self.trainable, self.recipe.max_grad_norm or math.inf, error_if_nonfinite=True
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            # El paso es la única modificación admitida del núcleo. Se sella de nuevo.
            self.predictor._seal_parameters()
            self.global_step += 1
            run.counters["updates"] += 1
            run.counters["labels_in_loss"] += total
            run.counters["loss_sum"] += loss_sum * total
            if run.penalties:
                run.counters["control_groups_in_objective"] = run.counters.get(
                    "control_groups_in_objective", 0
                ) + len(run.penalties)
            if self.audit is not None:
                used = tuple((flow, decision_at) for flow, decision_at, _, _ in run.used)
                self.audit.append(("update", at, self.global_step, used))
        elif run.penalties:
            # Sin etiquetas maduras no hay paso: C no añade actualizaciones al presupuesto.
            run.counters["control_groups_discarded"] = run.counters.get(
                "control_groups_discarded", 0
            ) + len(run.penalties)
        parameter_id = self.predictor._parameter_id
        for state in list(run.flows.values()):
            run.flows.update(_split(state, detach=True, parameter_id=parameter_id))
        run.blocks.clear()
        run.used.clear()
        run.starts.clear()
        run.penalties.clear()
        run.instants = 0
        run.counters["segments"] += 1
        return None

    def _replay(self, run, total):
        """Repetir el tramo por bloques de flujos y acumular el gradiente de la pérdida media.

        Los flujos son independientes dados los parámetros, que no cambian dentro del tramo.
        Cada bloque de flujos parte del estado de sus flujos al empezar el tramo y recorre
        los bloques emitidos en orden. Un bloque repetido con todas sus filas debe reproducir
        exactamente la predicción emitida. Con la penalización C, cada evento se repite con su
        plan y el bloque suma los términos de sus flujos medidos divididos por el número de
        grupos del tramo, de modo que la suma de los bloques es la media de los grupos.
        """
        predictor, readout = self.predictor, self.readout
        matured = {}
        for flow, decision_at, block, value in run.used:
            matured.setdefault(block, {})[flow, decision_at] = value
        controlled = [flow for flows in run.penalties for flow in flows]
        order = list(dict.fromkeys([*(flow for flow, _, _, _ in run.used), *controlled]))
        size = self.core_rows or len(order)
        loss_sum, replayed, used = 0.0, 0, 0
        for start in range(0, len(order), size):
            members = set(order[start : start + size])
            states = {flow: run.starts[flow] for flow in members if flow in run.starts}
            terms, penalties = [], []
            for block in sorted(run.blocks):
                record = run.blocks[block]
                batch = record["batch"]
                rows = [i for i, flow in enumerate(batch.flow_ids) if flow in members]
                if not rows:
                    continue
                whole = len(rows) == len(batch.flow_ids)
                selected = batch if whole else batch.select(rows)
                if any(flow not in states for flow in selected.flow_ids):
                    raise ValueError("Un flujo repetido no tiene su estado al empezar el tramo")
                state = _stack([states[flow] for flow in selected.flow_ids])
                with torch.enable_grad():
                    prepared = predictor.prepare(
                        selected, state, differentiable=True, **record["plan"]
                    )
                states.update(_split(prepared.next_state))
                if prepared.local_control is not None:
                    penalties.append(prepared.local_control.penalty)
                    replayed += len(prepared.local_control.flow_ids)
                labels = matured.get(block, {})
                keys = list(zip(selected.flow_ids, selected.prediction_at, strict=True))
                positions = [i for i, key in enumerate(keys) if key in labels]
                if not positions:
                    continue
                result = apply_episodic_readout(
                    replace(prepared, next_state=None, detached_tokens=None),
                    predictor.head,
                    readout,
                    record["snapshot"],
                    context_id=record["context"],
                    cutoff=record["cutoff"],
                    differentiable=True,
                )
                if whole and not torch.equal(result.point_predictions.detach(), record["issued"]):
                    raise ValueError("La repetición del bloque no reproduce su predicción emitida")
                index = torch.tensor(positions, dtype=torch.int64, device=self.device)
                target = torch.tensor(
                    [labels[keys[i]] for i in positions], dtype=self.dtype, device=self.device
                )
                chosen = replace(
                    result,
                    point_predictions=result.point_predictions.index_select(0, index),
                    quantiles=None
                    if result.quantiles is None
                    else result.quantiles.index_select(0, index),
                )
                loss = self._loss(chosen, target) * (len(positions) / total)
                if not torch.isfinite(loss).item():
                    raise ValueError("La pérdida del tramo no es finita")
                loss_sum += float(loss.detach())
                used += len(positions)
                terms.append(loss)
            if penalties:
                terms.append(sum(penalties) / len(run.penalties))
            if terms:
                objective = sum(terms[1:], terms[0])
                if not torch.isfinite(objective).item():
                    raise ValueError("El objetivo del tramo no es finito")
                objective.backward()
        if used != total or replayed != len(controlled):
            raise ValueError("La repetición del tramo no reproduce sus etiquetas ni flujos de C")
        return loss_sum
