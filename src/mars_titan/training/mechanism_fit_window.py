"""Auditar prefijos nativos elegibles sin normalizar señales ni ajustar parámetros."""

from dataclasses import asdict, dataclass

from mars_titan.memory.financial_session import FinancialSession, _check_callbacks
from mars_titan.memory.mechanism_collection_record import CollectedMechanisms
from mars_titan.memory.mechanism_supervision import MechanismSupervision
from mars_titan.models.titans.config import bounded_integer
from mars_titan.models.titans.mechanism_catalog import MechanismCatalog

from .mechanism_fit_contract import MechanismFitConfig


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class MechanismFitWindow:
    """Una sesión y un prefijo explícito, sin leer el archivo al construir la petición."""

    session: FinancialSession
    generations: tuple
    through_generation: int

    def __post_init__(self):
        _require(type(self.session) is FinancialSession, "La ventana necesita una sesión nativa")
        _require(
            type(self.generations) is tuple and 1 <= len(self.generations) <= 256,
            "Faltan las generaciones explícitas de la ventana",
        )
        for generation in self.generations:
            bounded_integer(generation, "generación de ventana", 1, 2**63 - 1)
        _require(
            list(self.generations) == sorted(set(self.generations)),
            "Las generaciones de ventana se repiten o desordenan",
        )
        bounded_integer(self.through_generation, "fin de prefijo", self.generations[-1], 2**63 - 1)
        _require(
            self.through_generation - self.generations[0] + 1 <= 256,
            "El prefijo supera sus 256 registros",
        )


def _read(request, *, before, maximum):
    return request.session.mechanism_collections_at(
        request.generations,
        through_generation=request.through_generation,
        before=before,
        max_bytes=maximum,
    )


def iter_fit_windows(sources, *, config, max_bytes=None):
    """Validar una ventana a la vez, con límites globales y sin construir objetivos.

    Es lectura de datos para auditoría. El ajuste y su lector de señales mantienen sus
    propias guardias antes de invocar este recorrido. `max_bytes` permite una auditoría
    con un límite menor que el protocolo y se registra en su salida.
    """
    _require(type(config) is MechanismFitConfig, "Falta el protocolo de ventanas")
    config.__post_init__()
    _require(config.source_mode == "native_windows", "El protocolo no declara ventanas nativas")
    _require(
        type(sources) in (list, tuple) and 1 <= len(sources) <= config.max_sources,
        "Las ventanas fuente deben ser explícitas y acotadas",
    )
    maximum = config.max_source_bytes if max_bytes is None else max_bytes
    bounded_integer(maximum, "bytes de auditoría", 1, config.max_source_bytes)
    total, observed, cohorts = 0, 0, set()
    for request in sources:
        _require(type(request) is MechanismFitWindow, "Falta una petición de ventana identificada")
        request.__post_init__()
        session = request.session
        _check_callbacks(session)
        session.consumer.verify()
        _require(
            session.consumer.model_id == config.parent_consumer_id,
            "La ventana pertenece a otro consumidor padre",
        )
        config.check_catalog(session.consumer.catalog)
        snapshot = session.snapshot()
        _require(total < maximum, "Las ventanas agotaron sus bytes acumulados")
        view = _read(request, before=config.fit_end, maximum=maximum - total)
        stamp = view["as_of"]
        total += stamp["source_bytes"]
        _require(total <= maximum, "Las ventanas superan sus bytes acumulados")
        collections = []
        for item in view["collections"]:
            packet = CollectedMechanisms.restore(item["collection"])
            body = packet.payload()
            config.check_catalog(MechanismCatalog.restore(body["catalog"]))
            _require(
                body["catalog"] == session.consumer.catalog.identity(),
                "La colección cambia el catálogo del consumidor padre",
            )
            observation = body["observation"]["observation"]
            _require(
                observation["consumer_id"] == config.parent_consumer_id
                and config.fit_start <= observation["cutoff"] < config.fit_end,
                "La decisión pertenece a otro padre o tramo de ajuste",
            )
            supervision = MechanismSupervision.restore(packet, item["supervision"])
            _require(
                all(
                    target is None or target["available_at"] < config.fit_end
                    for target in supervision.record()["outcomes"]
                ),
                "La ventana contiene disponibilidad posterior al ajuste",
            )
            for group in observation["groups"]:
                observed += 1
                _require(
                    observed <= config.max_cohorts, "Las ventanas superan sus cohortes acumuladas"
                )
                key = group["market"], observation["cutoff"]
                _require(
                    group["market"] in dict(config.market_weights) and key not in cohorts,
                    "El mercado no está declarado o la ventana repite una cohorte",
                )
                cohorts.add(key)
            collections.append(
                (
                    packet,
                    supervision,
                    dict(
                        generation=item["generation"],
                        packet_reference=item["packet_reference"],
                        collection_sha256=packet.sha256,
                        observation_sha256=body["observation"]["sha256"],
                        censored=item["censored"],
                        resolutions=item["resolutions"],
                    ),
                )
            )
        _require(session.snapshot() == snapshot, "La sesión fuente cambió durante la auditoría")
        yield (
            collections,
            dict(
                contract_id=stamp["contract_id"],
                model_id=stamp["model_id"],
                phase=asdict(session.phase),
                generation=request.through_generation,
                before=stamp["before"],
                collection_generations=stamp["collection_generations"],
                window_sha256=stamp["window_sha256"],
                native_records=stamp["native_records"],
                source_bytes=stamp["source_bytes"],
                collections=[item[2] for item in collections],
                budget_bytes=maximum,
            ),
        )


def audit_fit_windows(sources, *, config, max_bytes=None):
    """Resumir procedencia y estados de filas, sin emitir ganancias ni tensores de ajuste."""
    records = []
    counts = dict(
        observed_cohorts=0, mature_rows=0, pending_rows=0, excluded_rows=0, censored_rows=0
    )
    for collections, record in iter_fit_windows(sources, config=config, max_bytes=max_bytes):
        for packet, supervision, metadata in collections:
            counts["observed_cohorts"] += len(
                packet.payload()["observation"]["observation"]["groups"]
            )
            outcomes = supervision.record()["outcomes"]
            censored = len(metadata["censored"])
            counts["mature_rows"] += sum(t is not None and t["status"] == "label" for t in outcomes)
            counts["excluded_rows"] += sum(
                t is not None and t["status"] != "label" for t in outcomes
            )
            counts["censored_rows"] += censored
            counts["pending_rows"] += sum(t is None for t in outcomes) - censored
        records.append(record)
    return dict(sources=records, source_bytes=sum(r["source_bytes"] for r in records), **counts)
