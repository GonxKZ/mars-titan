"""Registro de los ciclos de autoedición acotada `posthoc_self_edit_v1` (G1, #748).

Sigue el bucle exterior de SEAL (Zweiger y colaboradores, versión 2,
https://arxiv.org/html/2506.10943v2): se generan ediciones, se aplican y se evalúan en datos
posteriores. Aquí no hay modelo de lenguaje. Las ediciones son configuraciones de una
gramática cerrada y las propone un generador que solo ve la partición G. Cada ciclo fija sus
reglas antes de abrir S:

1. **Declaración.** El generador y un placebo aleatorio proponen el mismo número N de
   configuraciones distintas de la misma gramática, ninguna igual a la vigente. N lo fija el
   plan y vale 8. El placebo muestrea sin reemplazo con `SeedSequence([semilla, ciclo])`.
2. **Selección** en un primer bloque de S. Se elige la mejor de cada lista, con empates por el
   orden declarado. La del generador se adopta solo si mejora a la vigente. Las N de cada
   lista quedan cobradas aunque se rechacen.
3. **Contraste** en un bloque posterior de S: la elegida del generador menos la elegida del
   placebo, con la comparación emparejada por bloques de días (`paired_comparisons`). El
   ciclo bate al placebo si el intervalo declarado queda entero del lado favorable.

Un ciclo solo puede declararse si el anterior está contrastado y batió a su placebo. Así un
segundo ciclo, de salida o de arquitectura, no existe si el primero no supera a un generador
aleatorio con la misma gramática y el mismo N. El estado se exporta con su huella y se
restaura para continuar un ciclo a medias.

Los registros de `proposals.ledger` (#711) siguen el mismo contrato de cuota fija, cargos sin
devolución y recibos sellados. No se reutilizan sus clases porque exigen una `MatureReview`
de una `FinancialSession` y recetas `FinancialConfig`, que no existen sobre predicciones
guardadas.
"""

import copy
import hashlib
import json
import math

import numpy as np

from mars_titan.environments.cohorts import FINAL_TEST_START_US

KIND = "posthoc_self_edit_cycles_v1"
SCHEMA_VERSION = 1
CANDIDATES = 8
CONTRAST = "generated_minus_placebo"
STATUSES = ("declared", "selected", "assessed")
_PLAN = {"candidates", "placebo_seed", "decision"}
_DECISION = {"block_length", "replicates", "seed", "confidence"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _instant(value, name):
    _require(
        type(value) is int and 0 <= value < FINAL_TEST_START_US,
        f"{name} debe ser un instante UTC anterior al test reservado",
    )
    return value


def checked_plan(plan):
    """Validar el plan declarado antes del primer ciclo."""
    _require(isinstance(plan, dict) and set(plan) == _PLAN, f"El plan declara {sorted(_PLAN)}")
    decision = plan["decision"]
    _require(
        type(plan["candidates"]) is int and 1 <= plan["candidates"] <= 64,
        "N debe ser un entero entre 1 y 64",
    )
    _require(
        type(plan["placebo_seed"]) is int and 0 <= plan["placebo_seed"] < 2**32,
        "La semilla del placebo debe ser un entero de 32 bits",
    )
    _require(
        isinstance(decision, dict)
        and set(decision) == _DECISION
        and all(type(decision[name]) is int for name in ("block_length", "replicates", "seed"))
        and type(decision["confidence"]) is float
        and 0.5 <= decision["confidence"] < 1,
        "La regla de contraste necesita bloque, réplicas, semilla y confianza",
    )
    return copy.deepcopy(plan)


def placebo(domain, current, *, count, seed, cycle):
    """`count` configuraciones distintas de la gramática, al azar y sin la vigente."""
    choices = [item for item in domain if item != current]
    _require(count <= len(choices), "La gramática no tiene tantas configuraciones")
    generator = np.random.default_rng(np.random.SeedSequence([seed, cycle]))
    return [choices[index] for index in generator.choice(len(choices), count, replace=False)]


def _best(scores, loss):
    values = np.asarray(scores, dtype=np.float64)
    return int(np.argmin(values) if loss else np.argmax(values))


class SelfEditLedger:
    """Ciclos declarados, elegidos y contrastados, con la puerta entre ciclos."""

    def __init__(self, plan, grammars):
        self._plan = checked_plan(plan)
        self._grammars = dict(grammars)
        self._cycles = []

    def _validated(self, grammar, items, current):
        check = self._grammars[grammar]
        _require(
            isinstance(items, list) and len(items) == self._plan["candidates"],
            f"Cada lista declara exactamente N = {self._plan['candidates']} configuraciones",
        )
        items = [check(item) for item in items]
        encoded = [_canonical(item) for item in items]
        _require(len(set(encoded)) == len(encoded), "Una lista repite configuraciones")
        _require(_canonical(current) not in encoded, "Una candidata repite la vigente")
        return items

    def current(self, grammar, reference):
        """Configuración vigente de una gramática: la última adoptada o la de referencia."""
        for cycle in reversed(self._cycles):
            if cycle["grammar"] == grammar and cycle["status"] != "declared":
                adopted = cycle["selection"]["adopted"]
                return copy.deepcopy(cycle["current"] if adopted is None else adopted)
        return copy.deepcopy(reference)

    @property
    def plan(self):
        return copy.deepcopy(self._plan)

    def cycle(self, index):
        """Copia de un ciclo registrado."""
        _require(
            type(index) is int and 1 <= index <= len(self._cycles),
            "El ciclo no existe en este registro",
        )
        return copy.deepcopy(self._cycles[index - 1])

    def declare(self, grammar, *, current, generated, generator, domain, fit):
        """Fijar N, candidatas y placebo de un ciclo antes de abrir S.

        `domain` enumera la gramática completa y de él sale el placebo. `fit` describe la
        partición G que vio el generador: su huella y la mayor madurez de sus etiquetas.
        Falla si el ciclo anterior no está contrastado o no batió al placebo.
        """
        if self._cycles:
            last = self._cycles[-1]
            _require(last["status"] == "assessed", "El ciclo anterior no está contrastado")
            _require(
                last["verdict"]["beats_placebo"],
                "El ciclo anterior no batió al placebo y no se abre otro",
            )
        _require(grammar in self._grammars, f"La gramática {grammar!r} no está registrada")
        current = self._grammars[grammar](current)
        index = len(self._cycles) + 1
        generated = self._validated(grammar, generated, current)
        placebo_items = self._validated(
            grammar,
            placebo(
                domain,
                current,
                count=self._plan["candidates"],
                seed=self._plan["placebo_seed"],
                cycle=index,
            ),
            current,
        )
        _require(isinstance(generator, str) and generator, "El generador necesita un nombre")
        _require(
            isinstance(fit, dict)
            and set(fit) == {"sha256", "matured_until"}
            and isinstance(fit["sha256"], str)
            and len(fit["sha256"]) == 64,
            "La partición G se describe con su huella y su madurez",
        )
        _instant(fit["matured_until"], "La madurez de G")
        self._cycles.append(
            dict(
                index=index,
                grammar=grammar,
                status="declared",
                current=current,
                generator=generator,
                generated=generated,
                placebo=dict(seed=[self._plan["placebo_seed"], index], items=placebo_items),
                partitions=dict(fit=dict(fit)),
                selection=None,
                verdict=None,
            )
        )
        return index

    def _cycle(self, index, status):
        self.cycle(index)
        cycle = self._cycles[index - 1]
        _require(cycle["status"] == status, f"El ciclo {index} no está en el estado {status}")
        return cycle

    def select(self, index, *, partition, scores, loss):
        """Registrar la selección en el primer bloque de S y cobrar las 2N candidatas.

        `partition` describe el bloque: huella, primera emisión y mayor madurez. `scores`
        contiene la puntuación media por sesión de la vigente y de cada lista en su orden.
        """
        cycle = self._cycle(index, "declared")
        _require(type(loss) is bool, "Se declara si la puntuación es una pérdida")
        partition = self._partition(partition, cycle["partitions"]["fit"]["matured_until"])
        count = self._plan["candidates"]
        _require(
            isinstance(scores, dict)
            and set(scores) == {"current", "generated", "placebo"}
            and len(scores["generated"]) == len(scores["placebo"]) == count,
            "La selección puntúa la vigente y las N candidatas de cada lista",
        )
        values = [scores["current"], *scores["generated"], *scores["placebo"]]
        _require(
            all(type(value) is float and math.isfinite(value) for value in values),
            "Las puntuaciones de selección deben ser decimales finitos",
        )
        chosen = _best(scores["generated"], loss)
        rival = _best(scores["placebo"], loss)
        best = scores["generated"][chosen]
        improves = best < scores["current"] if loss else best > scores["current"]
        rejected = [
            dict(
                ordinal=ordinal,
                reason="no_es_la_mejor_en_seleccion"
                if ordinal != chosen
                else "no_mejora_la_vigente",
            )
            for ordinal in range(count)
            if ordinal != chosen or not improves
        ]
        cycle["partitions"]["select"] = partition
        cycle["selection"] = dict(
            loss=loss,
            scores=copy.deepcopy(scores),
            generated=chosen,
            placebo=rival,
            adopted=copy.deepcopy(cycle["generated"][chosen]) if improves else None,
            charged=2 * count,
            rejected=rejected,
        )
        cycle["status"] = "selected"
        return copy.deepcopy(cycle["selection"])

    @staticmethod
    def _partition(partition, after):
        _require(
            isinstance(partition, dict)
            and set(partition) == {"sha256", "first_emission", "matured_until"}
            and isinstance(partition["sha256"], str)
            and len(partition["sha256"]) == 64,
            "El bloque se describe con su huella, su primera emisión y su madurez",
        )
        first = _instant(partition["first_emission"], "La primera emisión del bloque")
        _instant(partition["matured_until"], "La madurez del bloque")
        _require(first >= after, "El bloque empieza antes de que maduren las etiquetas anteriores")
        return dict(partition)

    def selected(self, index):
        """Las dos configuraciones elegidas de un ciclo seleccionado o contrastado."""
        cycle = self._cycles[index - 1]
        _require(cycle["selection"] is not None, f"El ciclo {index} no tiene selección")
        selection = cycle["selection"]
        return (
            copy.deepcopy(cycle["generated"][selection["generated"]]),
            copy.deepcopy(cycle["placebo"]["items"][selection["placebo"]]),
        )

    def assess(self, index, *, partition, comparison):
        """Registrar el contraste en un bloque posterior y decidir si el ciclo bate al placebo.

        `comparison` es el resultado de `compare_series` con el único contraste
        `generated_minus_placebo`, o None si las dos listas eligieron la misma configuración.
        El bloque empieza después de que maduren el de selección y el contraste del ciclo
        anterior, así que cada veredicto usa filas que ningún veredicto previo vio.
        """
        cycle = self._cycle(index, "selected")
        after = cycle["partitions"]["select"]["matured_until"]
        if index > 1:
            after = max(after, self._cycles[index - 2]["partitions"]["assess"]["matured_until"])
        partition = self._partition(partition, after)
        generated, rival = self.selected(index)
        loss = cycle["selection"]["loss"]
        if _canonical(generated) == _canonical(rival):
            _require(comparison is None, "Con la misma configuración no hay contraste")
            verdict = dict(beats_placebo=False, reason="misma_configuracion", contrast=None)
        else:
            row = self._contrast(comparison, loss)
            interval = row["interval"]
            if interval is None:
                verdict = dict(beats_placebo=False, reason="sin_intervalo", contrast=row)
            else:
                beats = interval[1] < 0 if loss else interval[0] > 0
                reason = "intervalo_favorable" if beats else "intervalo_no_favorable"
                verdict = dict(beats_placebo=beats, reason=reason, contrast=row)
        cycle["partitions"]["assess"] = partition
        cycle["verdict"] = verdict
        cycle["status"] = "assessed"
        return copy.deepcopy(verdict)

    def _contrast(self, comparison, loss):
        decision = self._plan["decision"]
        _require(
            isinstance(comparison, dict)
            and comparison.get("kind") == "paired_session_contrasts"
            and comparison.get("loss") is loss
            and comparison.get("confidence") == decision["confidence"]
            and comparison["resampling"]["block_length"] == decision["block_length"]
            and comparison["resampling"]["replicates"] == decision["replicates"]
            and comparison["resampling"]["seed"] == decision["seed"]
            and comparison["multiplicity"]["family_size"] == 1,
            "El contraste no sigue la regla declarada en el plan",
        )
        (row,) = comparison["contrasts"]
        _require(
            row["name"] == CONTRAST and row["coefficients"] == {"generated": 1.0, "placebo": -1.0},
            "El contraste debe ser la elegida del generador menos la del placebo",
        )
        return dict(
            estimate=row["estimate"],
            interval=row["interval"],
            sessions=row["sessions"],
            metric=comparison["metric"],
            cohort_sha256=comparison["cohort_sha256"],
        )

    def charged(self):
        """Configuraciones cobradas en todos los ciclos con selección, rechazadas incluidas."""
        return sum(cycle["selection"]["charged"] for cycle in self._cycles if cycle["selection"])

    def export(self):
        """Estado completo con su huella, en JSON."""
        payload = dict(
            schema_version=SCHEMA_VERSION,
            kind=KIND,
            plan=copy.deepcopy(self._plan),
            grammars=sorted(self._grammars),
            cycles=copy.deepcopy(self._cycles),
        )
        payload["state_sha256"] = _sha256(payload)
        return payload

    @classmethod
    def restore(cls, payload, grammars):
        """Reconstruir el registro tras comprobar su huella, sus gramáticas y su secuencia."""
        _require(
            isinstance(payload, dict)
            and set(payload)
            == {"schema_version", "kind", "plan", "grammars", "cycles", "state_sha256"}
            and payload["schema_version"] == SCHEMA_VERSION
            and payload["kind"] == KIND,
            "El estado no conserva el contrato del registro",
        )
        body = {key: value for key, value in payload.items() if key != "state_sha256"}
        _require(_sha256(body) == payload["state_sha256"], "El estado no coincide con su huella")
        _require(payload["grammars"] == sorted(grammars), "El estado usa otras gramáticas")
        ledger = cls(payload["plan"], grammars)
        cycles = payload["cycles"]
        for position, cycle in enumerate(cycles, 1):
            _require(
                cycle["index"] == position
                and cycle["status"] in STATUSES
                and (position == len(cycles) or cycle["verdict"]["beats_placebo"] is True),
                "La secuencia de ciclos no es coherente",
            )
        ledger._cycles = copy.deepcopy(payload["cycles"])
        return ledger
