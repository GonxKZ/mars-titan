"""Cohortes de ajuste con las visitas de repaso de un control de `replay_continuation`.

`ReplayInputs` conserva las filas nuevas, su orden en cada época y la validación de
`PairedInputs`, y añade en cada época las mismas visitas de repaso, fijadas antes del primer
ajuste. El repaso de sesiones anteriores lee otra fuente de la misma vista, abierta solo con
las sesiones planificadas, que el padre ajustó completas y maduraron antes de la primera
fila nueva. Si caben en el presupuesto de bloque, la fuente las lee una vez y sirven a todas
las épocas. La repetición lee las filas nuevas. La normalización y la rejilla siguen siendo
las de las filas nuevas, que son las reales del ajuste, y el repaso no las cambia.
"""

from mars_titan.environments.cohort_order import Visit

from . import replay_continuation as replay
from .inputs import PairedInputs, fingerprint

REPLAY, REPEAT = "replay", "repeat"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class ReplayInputs(PairedInputs):
    """Filas nuevas y repaso de un caso.

    `train` es la fuente de las filas nuevas, limitada con `since`. Para el repaso de
    sesiones anteriores, `candidates` resume las sesiones admitidas (`moments`, `rows` y,
    con el reparto por régimen, `routes`) y `open_earlier(moments)` abre la fuente de las
    sesiones elegidas. La repetición no necesita ninguna de las dos. `close` cierra la fuente
    abierta aquí.
    """

    read_arms = ("real", REPLAY, REPEAT)

    def __init__(
        self,
        train,
        validation,
        parent,
        *,
        spec,
        batch_size,
        seed,
        candidates=None,
        open_earlier=None,
    ):
        super().__init__(train, validation, parent)
        self.spec = replay.check_spec(spec)
        policy = spec["policy"]
        source = replay.SOURCES[policy]
        _require(
            (candidates is None) == (open_earlier is None) == (source == replay.NEW),
            "Solo el repaso de sesiones anteriores lee otra fuente",
        )
        since = getattr(train, "since", None)
        _require(type(since) is int and since > 0, "El repaso exige ajustar desde las filas nuevas")
        _require(
            type(batch_size) is int and 1 <= batch_size <= 4096,
            "El lote debe contener entre una y 4096 filas",
        )
        _require(type(seed) is int and 0 <= seed < 2**32, "La semilla del repaso no es válida")
        self.since, self.batch_size, self.seed, self.earlier = since, batch_size, seed, None
        # Actualizaciones de una época de la continuación completa, que recorre una vez
        # cada sesión nueva en lotes del mismo tamaño.
        self.new_updates = replay.epoch_updates([count for _, count in train.index], batch_size)
        extra = replay.extra_updates(self.new_updates, spec["ratio"])
        if candidates is None:
            moments = [at for at, _ in train.index]
            rows = [count for _, count in train.index]
            routes = None
        else:
            moments, rows = list(candidates["moments"]), list(candidates["rows"])
            routes = candidates.get("routes") if policy == replay.REGIME else None
            _require(
                len(moments) == len(rows)
                and all(type(at) is int and at < since for at in moments)
                and all(type(count) is int and count >= 1 for count in rows),
                "Las candidatas del repaso son sesiones anteriores con filas",
            )
        ordered = replay.order(range(len(rows)), policy, seed=seed, routes=routes)
        chosen = replay.plan(ordered, rows, batch_size=batch_size, updates=extra)
        if candidates is None:
            origin, self.plan = train, chosen
        else:
            expected = {moments[i]: rows[i] for i, _ in chosen}
            planned = sorted(expected)
            self.earlier = origin = open_earlier(planned)
            try:
                _require(
                    origin.partition == "train"
                    and origin.source_sha256 == train.source_sha256
                    and [list(row) for row in origin.index]
                    == [[at, expected[at]] for at in planned],
                    "La fuente del repaso no contiene exactamente las sesiones planificadas",
                )
            except BaseException:
                origin.close()
                raise
            position = {at: i for i, (at, _) in enumerate(origin.index)}
            self.plan = [(position[moments[i]], batches) for i, batches in chosen]
        self.arm = REPEAT if candidates is None else REPLAY
        self.limits = {
            session: min(origin.index[session][1], batches * batch_size)
            for session, batches in self.plan
        }
        self.extra = dict(
            updates=extra,
            rows=sum(self.limits.values()),
            sessions=[[origin.index[session][0], self.limits[session]] for session, _ in self.plan],
        )
        self.identity = dict(
            self.identity,
            replay=dict(
                spec,
                source=source,
                since=since,
                batch_size=batch_size,
                seed=seed,
                streams=[replay.SELECTION_STREAM, replay.ORDER_STREAM],
                candidates=None
                if candidates is None
                else dict(
                    sessions_sha256=replay.digest([moments, rows]),
                    count=len(moments),
                    not_fitted=candidates.get("not_fitted"),
                    late=candidates.get("late"),
                    parent_view=candidates.get("parent_view"),
                ),
                routes=None
                if routes is None
                else dict(sha256=replay.digest(list(routes)), rule=replay.RULE.identity()),
                plan_sha256=replay.digest(self.extra["sessions"]),
                updates=extra,
                rows=self.extra["rows"],
            ),
        )
        self.sha256 = fingerprint(self.identity)

    def _visits(self, partition, condition, epoch, seed):
        visits = super()._visits(partition, condition, epoch, seed)
        if partition != "train":
            return visits
        _require(seed == self.seed, "El repaso se planificó con otra semilla")
        slots, permutation = replay.positions(len(visits), len(self.plan), seed=seed, epoch=epoch)
        extra = [Visit(self.arm, -1, self.plan[i][0], True) for i in permutation]
        result, new, added = [], iter(visits), iter(extra)
        taken = set(slots)
        for slot in range(len(visits) + len(extra)):
            result.append(next(added) if slot in taken else next(new))
        return result

    def _source(self, partition, visit):
        if visit.arm == REPLAY:
            return self.earlier
        return super()._source(partition, visit)

    def _rows(self, partition, visit):
        if visit.arm in (REPLAY, REPEAT):
            return self.limits[visit.cohort]
        return super()._rows(partition, visit)

    def _admit(self, visit, raw):
        if visit.arm == REPLAY:
            _require(
                raw["prediction_at"] < self.since
                and bool((raw["target_available_at"] < self.since).all()),
                "Una sesión de repaso madura después de la primera fila nueva",
            )

    def _origin(self, visit, condition):
        return visit.arm

    def budget(self, condition, batch_size):
        """Filas y actualizaciones de cada época: las nuevas más el repaso, ya recortado."""
        _require(
            condition == "real" and batch_size == self.batch_size,
            "El repaso se planificó con otro lote o condición",
        )
        real = self.counts["train"]
        return dict(
            real_rows=real,
            extra_rows=self.extra["rows"],
            rows=real + self.extra["rows"],
            updates=self.new_updates + self.extra["updates"],
        )

    def close(self):
        if self.earlier is not None:
            self.earlier.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
