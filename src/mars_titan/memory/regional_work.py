"""Reservar trabajo bajo el OutputLock del coordinador, sin publicar predicciones.

El llamante posee el bloqueo y valida la cabeza nativa que entrega a reconcile.
Los SHA detectan cambios incoherentes, pero no autentican un rollback completo del archivo.
"""

import hashlib
import json
import os
import re
import tempfile
import zlib
from copy import deepcopy
from pathlib import Path

from .session_artifacts import SessionArtifacts

RESOURCES = frozenset(
    (
        "context_queries",
        "selector_decisions",
        "local_node_evaluations",
        "summary_evaluations",
        "core_observations",
        "mac_observations",
        "reader_refinements",
        "reader_searches",
        "context_commits",
    )
)
EVENT_KINDS = frozenset(("warmup", "decision", "settlement"))
KIND = "regional_work_ledger_v2"
RECEIPT = "regional_work_reservation"
MAX_INTEGER = 2**63 - 1


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, minimum=0, maximum=MAX_INTEGER):
    _require(
        type(value) is int and minimum <= value <= maximum,
        "El contador regional necesita un entero dentro de sus límites",
    )
    return value


def _digest(value):
    _require(
        type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        "La reserva regional necesita una identidad SHA-256",
    )
    return value


def _event_kind(value):
    _require(
        type(value) is str and value in EVENT_KINDS,
        "El evento regional debe ser warmup, decision o settlement",
    )
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "El índice regional repite un campo")
        result[key] = value
    return result


def _prefix(value):
    _require(
        type(value) is dict and set(value) == {"generation", "cursor", "checkpoint_sha256"},
        "El prefijo regional no conserva su contrato",
    )
    _integer(value["generation"])
    _integer(value["cursor"])
    _require(value["generation"] == value["cursor"], "El prefijo contradice el cursor nativo")
    _digest(value["checkpoint_sha256"])
    return dict(value)


def _latest(value):
    _require(
        type(value) is dict
        and set(value) == {"generation", "cursor", "checkpoint_sha256", "reservation_id"},
        "La cabeza regional no conserva su contrato",
    )
    _prefix({key: part for key, part in value.items() if key != "reservation_id"})
    if value["reservation_id"] is not None:
        _digest(value["reservation_id"])
    return dict(value)


class RegionalWorkLedger:
    """Cargos conservadores por cohorte, con un solo intento pendiente.

    No toma otro OutputLock. El coordinador debe incluir references() antes de podar.
    confirm es contabilidad y solo reconcile admite una cabeza enlazada como confirmación.
    Las unidades observadas pertenecen a la traza de ejecución, no a esta reserva.
    """

    def __init__(
        self,
        directory,
        *,
        artifacts,
        contract_id,
        model_id,
        program_id,
        limits,
        max_attempts=2,
        max_index_bytes=65536,
    ):
        _require(type(artifacts) is SessionArtifacts, "La reserva necesita artefactos de sesión")
        self.artifacts, self.native = artifacts, artifacts.native
        directory = Path(directory)
        self.native.require_safe_path(str(directory))
        self.directory = directory.resolve(strict=True)
        self.native.require_safe_path(str(artifacts.directory))
        artifact_path = artifacts.directory.resolve()
        _require(
            self.directory.is_dir()
            and artifact_path != self.directory
            and artifact_path.is_relative_to(self.directory),
            "Los artefactos deben pertenecer al directorio protegido de la reserva",
        )
        _require(
            type(limits) is dict and bool(limits) and set(limits) <= RESOURCES,
            "Los recursos regionales deben ser un subconjunto fijo del registro",
        )
        self.limits = {name: _integer(value) for name, value in sorted(limits.items())}
        self.max_attempts = _integer(max_attempts, 1, 32)
        self.max_index_bytes = _integer(max_index_bytes, 256, 65536)
        self.configuration = dict(
            contract_id=_digest(contract_id),
            model_id=_digest(model_id),
            program_id=_digest(program_id),
            limits=self.limits,
            max_attempts=self.max_attempts,
            max_index_bytes=self.max_index_bytes,
        )
        self.configuration_id = hashlib.sha256(_json(self.configuration)).hexdigest()
        self.path = self.directory / "regional-work.json"
        self.native.require_safe_path(str(self.path))

    def _units(self, value):
        _require(
            type(value) is dict and set(value) == set(self.limits),
            "Las unidades no conservan los recursos declarados",
        )
        return {name: _integer(value[name]) for name in self.limits}

    def _payload(self, cohort, ordinal, units, previous):
        return dict(
            schema_version=2,
            configuration_sha256=self.configuration_id,
            history_before=cohort["history_before"],
            cutoff=cohort["cutoff"],
            event_kind=cohort["event_kind"],
            prefix=cohort["prefix"],
            rows_digest=cohort["rows_digest"],
            ordinal=ordinal,
            units=units,
            previous_reservation_id=previous,
        )

    def _empty_history(self, initial):
        summary = dict(
            schema_version=1,
            configuration_sha256=self.configuration_id,
            initial_prefix=_prefix(initial),
            cohorts=0,
            attempts=0,
            charged_units=dict.fromkeys(self.limits, 0),
            last_reservation_id=None,
            frontier=None,
        )
        return dict(summary, sha256=hashlib.sha256(_json(summary)).hexdigest())

    def _history(self, summary):
        _require(
            type(summary) is dict
            and set(summary)
            == {
                "schema_version",
                "configuration_sha256",
                "initial_prefix",
                "cohorts",
                "attempts",
                "charged_units",
                "last_reservation_id",
                "frontier",
                "sha256",
            }
            and type(summary["schema_version"]) is int
            and summary["schema_version"] == 1
            and summary["configuration_sha256"] == self.configuration_id,
            "El resumen regional no conserva su contrato",
        )
        initial = _prefix(summary["initial_prefix"])
        cohorts = _integer(summary["cohorts"])
        attempts = _integer(summary["attempts"])
        self._units(summary["charged_units"])
        _digest(summary["sha256"])
        if not cohorts:
            _require(summary == self._empty_history(initial), "El historial vacío cambia su saldo")
            return summary
        _require(
            cohorts <= attempts <= cohorts * self.max_attempts,
            "El resumen regional no conserva los intentos",
        )
        frontier = summary["frontier"]
        _require(
            type(frontier) is dict
            and set(frontier)
            == {"prefix", "generation", "cursor", "cutoff", "event_kind", "reservation_id"},
            "La frontera regional no conserva su esquema",
        )
        base = _prefix(frontier["prefix"])
        generation = _integer(frontier["generation"], 1)
        _require(
            generation == initial["generation"] + cohorts == base["generation"] + 1
            and _integer(frontier["cursor"], 1) == generation
            and _digest(frontier["reservation_id"]) == _digest(summary["last_reservation_id"]),
            "La frontera regional contradice el prefijo acumulado",
        )
        _integer(frontier["cutoff"], 1)
        _event_kind(frontier["event_kind"])
        return summary

    @staticmethod
    def _terminal_cohort(cohort):
        terminal = deepcopy(cohort)
        terminal.pop("confirmed_prefix", None)
        terminal["attempts"][-1].update(
            status="confirmed", bundle_generation=terminal["prefix"]["generation"] + 1
        )
        return terminal

    def _fold(self, summary, cohort):
        terminal = self._terminal_cohort(cohort)
        last = terminal["attempts"][-1]["reservation_id"]
        return dict(
            summary,
            cohorts=_integer(summary["cohorts"] + 1),
            attempts=_integer(summary["attempts"] + len(terminal["attempts"])),
            charged_units={
                name: _integer(summary["charged_units"][name] + terminal["charged_units"][name])
                for name in self.limits
            },
            last_reservation_id=last,
            frontier=dict(
                prefix=terminal["prefix"],
                generation=terminal["prefix"]["generation"] + 1,
                cursor=terminal["prefix"]["cursor"] + 1,
                cutoff=terminal["cutoff"],
                event_kind=terminal["event_kind"],
                reservation_id=last,
            ),
            sha256=hashlib.sha256(bytes.fromhex(summary["sha256"]) + _json(terminal)).hexdigest(),
        )

    def _check_cohort(self, cohort, history, *, receipts=None, terminal=False):
        fields = {
            "cutoff",
            "event_kind",
            "prefix",
            "rows_digest",
            "history_before",
            "charged_units",
            "attempts",
        }
        if not terminal:
            fields.add("confirmed_prefix")
        _require(
            type(cohort) is dict and set(cohort) == fields,
            "La cohorte regional no conserva su esquema",
        )
        _integer(cohort["cutoff"], 1)
        _event_kind(cohort["event_kind"])
        base = _prefix(cohort["prefix"])
        _digest(cohort["rows_digest"])
        _require(
            cohort["history_before"] == history,
            "La cohorte regional contradice su historia previa sellada",
        )
        attempts = cohort["attempts"]
        _require(
            type(attempts) is list and 1 <= len(attempts) <= self.max_attempts,
            "La cohorte regional no conserva sus intentos acotados",
        )
        if receipts is not None:
            _require(
                type(receipts) is list and len(receipts) == len(attempts),
                "El archivo regional perdió recibos completos",
            )
        charged, previous = dict.fromkeys(self.limits, 0), history["last_reservation_id"]
        for ordinal, attempt in enumerate(attempts, 1):
            _require(
                type(attempt) is dict
                and set(attempt)
                == {
                    "reservation_id",
                    "ordinal",
                    "units",
                    "reference",
                    "status",
                    "observed_units",
                    "bundle_generation",
                },
                "El intento regional no conserva su esquema",
            )
            _require(
                _integer(attempt["ordinal"], 1) == ordinal
                and attempt["status"] in ("reserved", "confirmed", "abandoned")
                and attempt["observed_units"] is None,
                "El intento regional cambia su orden, estado o trabajo desconocido",
            )
            units = self._units(attempt["units"])
            charged = {name: _integer(charged[name] + units[name]) for name in self.limits}
            _require(
                all(charged[name] <= limit for name, limit in self.limits.items()),
                "La cohorte regional excede un recurso reservado",
            )
            payload = self._payload(cohort, ordinal, units, previous)
            identity = hashlib.sha256(_json(payload)).hexdigest()
            _require(
                attempt["reservation_id"] == identity, "El intento regional cambió de identidad"
            )
            self._reference(attempt["reference"], RECEIPT)
            if receipts is None:
                stored = self.artifacts.read(
                    attempt["reference"], identity=self.configuration["contract_id"], kind=RECEIPT
                )
            else:
                stored = receipts[ordinal - 1]
            _require(
                _json(stored) == _json(dict(payload, reservation_id=identity)),
                "El recibo regional contradice sus unidades o prefijo",
            )
            if attempt["status"] == "confirmed":
                _require(
                    _integer(attempt["bundle_generation"], 1) == base["generation"] + 1,
                    "La confirmación regional pertenece a otra generación",
                )
            else:
                _require(
                    attempt["bundle_generation"] is None,
                    "Un intento sin confirmar declara generación",
                )
            if ordinal != len(attempts):
                _require(attempt["status"] == "abandoned", "Un intento pendiente fue sustituido")
            previous = identity
        _require(
            self._units(cohort["charged_units"]) == charged,
            "La cohorte regional perdió cargos reservados",
        )
        if terminal:
            _require(
                attempts[-1]["status"] == "confirmed",
                "El archivo regional no contiene una cohorte terminal",
            )

    def _reference(self, reference, kind):
        header = self.artifacts._header(self.configuration["contract_id"], kind)
        _require(
            type(reference) is dict
            and set(reference) == set(header) | {"name", "sha256", "bytes"}
            and _json({name: reference[name] for name in header}) == _json(header),
            "La referencia regional pertenece a otro contrato",
        )
        digest = _digest(reference["sha256"])
        _require(reference["name"] == digest + ".bin", "La referencia regional cambió de nombre")
        _integer(reference["bytes"], 1, self.artifacts.max_bytes)

    def _validate(self, state):
        _require(
            type(state) is dict
            and set(state)
            == {
                "schema_version",
                "kind",
                "configuration",
                "initial_prefix",
                "latest",
                "cohorts",
                "archived",
                "archive",
            }
            and type(state["schema_version"]) is int
            and state["schema_version"] == 2
            and state["kind"] == KIND
            and _json(state["configuration"]) == _json(self.configuration),
            "El índice regional no conserva su identidad y límites",
        )
        initial = _prefix(state["initial_prefix"])
        summary = self._history(state["archived"])
        _require(summary["initial_prefix"] == initial, "El resumen regional cambia su origen")
        cohorts = state["cohorts"]
        _require(
            type(cohorts) is list and len(cohorts) <= min(256, self.artifacts.max_files),
            "El índice regional supera el número de cohortes",
        )
        anchor = dict(initial, reservation_id=None)
        cutoff, previous_kind, references = 0, None, 0
        if summary["cohorts"]:
            _require(
                bool(cohorts) and type(cohorts[0]) is dict and "prefix" in cohorts[0],
                "El prefijo archivado perdió el tramo retenido",
            )
            frontier = summary["frontier"]
            base = _prefix(cohorts[0]["prefix"])
            _require(
                base["generation"] == frontier["generation"],
                "El tramo regional no continúa la generación archivada",
            )
            anchor = dict(base, reservation_id=summary["last_reservation_id"])
            cutoff, previous_kind = frontier["cutoff"], frontier["event_kind"]
        _latest(state["latest"])
        for position, cohort in enumerate(cohorts):
            self._check_cohort(cohort, summary)
            base = _prefix(cohort["prefix"])
            kind = cohort["event_kind"]
            _require(
                cohort["cutoff"] > cutoff
                or (
                    kind == "settlement"
                    and previous_kind != "settlement"
                    and cohort["cutoff"] == cutoff
                ),
                "El corte regional no avanza o no corresponde a settlement",
            )
            _require(
                base == {key: value for key, value in anchor.items() if key != "reservation_id"},
                "La cohorte regional no continúa el prefijo confirmado",
            )
            cutoff, previous_kind = cohort["cutoff"], kind
            attempts = cohort["attempts"]
            references += len(attempts)
            confirmed = cohort["confirmed_prefix"]
            if confirmed is not None:
                confirmed = _prefix(confirmed)
                _require(
                    attempts[-1]["status"] == "confirmed"
                    and confirmed["generation"] == base["generation"] + 1
                    and confirmed["checkpoint_sha256"] != base["checkpoint_sha256"],
                    "El prefijo regional no enlaza la confirmación",
                )
                anchor = dict(confirmed, reservation_id=attempts[-1]["reservation_id"])
                summary = self._fold(summary, cohort)
            else:
                _require(
                    position == len(cohorts) - 1, "Una cohorte posterior elude la confirmación"
                )
        _require(references <= self.artifacts.max_files, "La reserva supera los archivos admitidos")
        _require(state["latest"] == anchor, "La cabeza regional no concilia con sus cohortes")
        archive = state["archive"]
        if archive is None:
            _require(not state["archived"]["cohorts"], "El resumen regional perdió su archivo")
        else:
            self._archive_stamp(archive)
            linked = next(
                (
                    cohort
                    for cohort in cohorts
                    if cohort["confirmed_prefix"] is not None
                    and cohort["confirmed_prefix"]["generation"] == archive["generation"]
                    and cohort["attempts"][-1]["reservation_id"] == archive["reservation_id"]
                ),
                None,
            )
            _require(
                linked is not None,
                "El archivo regional no enlaza el tramo confirmado",
            )
            _require(
                _json(self._archive_event(archive)) == _json(self._archive_payload(linked)),
                "El ancla regional contradice su cohorte confirmada",
            )
        return state

    def _read(self):
        self.native.require_safe_path(str(self.path))
        try:
            content = self.native.read_blob(str(self.path), self.max_index_bytes)
            state = json.loads(content, object_pairs_hook=_unique)
        except (OSError, RuntimeError, UnicodeError, RecursionError) as error:
            raise ValueError("No se puede recuperar el índice regional acotado") from error
        return self._validate(state)

    def _write(self, state):
        self._validate(state)
        content = _json(state)
        _require(
            len(content) <= self.max_index_bytes, "El índice regional supera su límite de bytes"
        )
        self.native.require_safe_path(str(self.path))
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.directory, prefix=".regional-work-", suffix=".pending", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def initialize(self, prefix):
        """Crear el índice una vez. La recuperación nunca entra por esta operación."""
        prefix = _prefix(prefix)
        _require(not self.path.exists(), "La reserva regional ya tiene un índice")
        self._write(
            dict(
                schema_version=2,
                kind=KIND,
                configuration=self.configuration,
                initial_prefix=prefix,
                latest=dict(prefix, reservation_id=None),
                cohorts=[],
                archived=self._empty_history(prefix),
                archive=None,
            )
        )

    def resume(self, latest):
        """Exigir el índice previo y conciliarlo antes de podar o ejecutar otra propuesta."""
        self.reconcile(latest)

    def reserve(self, cutoff, prefix, rows_digest, units, *, event_kind="decision"):
        event_kind = _event_kind(event_kind)
        cutoff, prefix, rows_digest = _integer(cutoff, 1), _prefix(prefix), _digest(rows_digest)
        units = self._units(units)
        state = self._read()
        expected = {key: value for key, value in state["latest"].items() if key != "reservation_id"}
        _require(prefix == expected, "La reserva no parte de la cabeza conciliada")
        cohorts = state["cohorts"]
        same = bool(cohorts) and (
            cutoff == cohorts[-1]["cutoff"]
            and prefix == cohorts[-1]["prefix"]
            and event_kind == cohorts[-1]["event_kind"]
        )
        if not same:
            _require(
                not cohorts or cohorts[-1]["confirmed_prefix"] is not None,
                "Una cohorte posterior necesita confirmación nativa conciliada",
            )
            _require(
                not cohorts
                or cutoff > cohorts[-1]["cutoff"]
                or (
                    event_kind == "settlement"
                    and cohorts[-1]["event_kind"] != "settlement"
                    and cutoff == cohorts[-1]["cutoff"]
                ),
                "El corte regional no avanza o no corresponde a settlement",
            )
            history = state["archived"]
            for previous_cohort in cohorts:
                history = self._fold(history, previous_cohort)
            cohorts.append(
                dict(
                    cutoff=cutoff,
                    event_kind=event_kind,
                    prefix=prefix,
                    rows_digest=rows_digest,
                    history_before=history,
                    charged_units=dict.fromkeys(self.limits, 0),
                    attempts=[],
                    confirmed_prefix=None,
                )
            )
        cohort = cohorts[-1]
        _require(
            cutoff == cohort["cutoff"]
            and event_kind == cohort["event_kind"]
            and prefix == cohort["prefix"]
            and rows_digest == cohort["rows_digest"]
            and cohort["confirmed_prefix"] is None,
            "La reserva cambia o reabre una cohorte ya identificada",
        )
        attempts = cohort["attempts"]
        _require(
            len(attempts) < self.max_attempts
            and (not attempts or attempts[-1]["status"] == "abandoned"),
            "La cohorte agota sus intentos o aún tiene uno pendiente",
        )
        charged = {name: cohort["charged_units"][name] + units[name] for name in self.limits}
        _require(
            all(charged[name] <= limit for name, limit in self.limits.items()),
            "La reserva excede un recurso de la cohorte",
        )
        previous = next(
            (
                item["attempts"][-1]["reservation_id"]
                for item in reversed(cohorts)
                if item["attempts"]
            ),
            state["archived"]["last_reservation_id"],
        )
        payload = self._payload(cohort, len(attempts) + 1, units, previous)
        identity = hashlib.sha256(_json(payload)).hexdigest()
        reference = self.artifacts.stage(
            dict(payload, reservation_id=identity),
            identity=self.configuration["contract_id"],
            kind=RECEIPT,
        )
        receipt = dict(
            reservation_id=identity,
            ordinal=len(attempts) + 1,
            units=units,
            reference=reference,
            status="reserved",
            observed_units=None,
            bundle_generation=None,
        )
        attempts.append(receipt)
        cohort["charged_units"] = charged
        self._write(state)
        return deepcopy(receipt)

    def confirm(self, reservation_id, bundle_generation):
        """Registrar contablemente una confirmación, sin acreditar el commit nativo."""
        _digest(reservation_id)
        _integer(bundle_generation, 1)
        state = self._read()
        _require(bool(state["cohorts"]), "No existe una reserva regional que confirmar")
        cohort = state["cohorts"][-1]
        attempt = cohort["attempts"][-1]
        _require(
            attempt["reservation_id"] == reservation_id
            and attempt["status"] in ("reserved", "confirmed")
            and bundle_generation == cohort["prefix"]["generation"] + 1,
            "La confirmación regional no corresponde al intento y generación",
        )
        attempt.update(status="confirmed", bundle_generation=bundle_generation)
        self._write(state)

    def reconcile(self, latest):
        """Conservar cargos ante abandono o enlazar una única generación confirmada."""
        latest = _latest(latest)
        state = self._read()
        if latest == state["latest"]:
            if state["cohorts"]:
                cohort = state["cohorts"][-1]
                attempt = cohort["attempts"][-1]
                if cohort["confirmed_prefix"] is None and attempt["status"] != "abandoned":
                    attempt.update(status="abandoned", bundle_generation=None)
                    self._write(state)
            return
        _require(bool(state["cohorts"]), "La cabeza avanzó sin una reserva regional")
        cohort = state["cohorts"][-1]
        attempt = cohort["attempts"][-1]
        _require(
            cohort["confirmed_prefix"] is None
            and attempt["status"] in ("reserved", "confirmed")
            and latest["generation"] == cohort["prefix"]["generation"] + 1
            and latest["checkpoint_sha256"] != cohort["prefix"]["checkpoint_sha256"]
            and latest["reservation_id"] == attempt["reservation_id"],
            "La cabeza nativa no enlaza el intento regional pendiente",
        )
        cohort["confirmed_prefix"] = {
            key: value for key, value in latest.items() if key != "reservation_id"
        }
        attempt.update(status="confirmed", bundle_generation=latest["generation"])
        state["latest"] = latest
        self._write(state)

    def references(self):
        """Referencias vivas de cargos reservados, confirmados y abandonados, antes de poda."""
        state = self._read()
        references = [
            attempt["reference"] for cohort in state["cohorts"] for attempt in cohort["attempts"]
        ]
        if state["archive"] is not None:
            references.append(state["archive"]["reference"])
        return deepcopy(list({reference["name"]: reference for reference in references}.values()))

    def snapshot(self):
        return deepcopy(self._read())

    def _archive_payload(self, cohort):
        terminal = self._terminal_cohort(cohort)
        attempt = terminal["attempts"][-1]
        return dict(
            reservation_id=attempt["reservation_id"],
            reference=attempt["reference"],
            cohort=terminal,
            receipts=[
                self.artifacts.read(
                    item["reference"], identity=self.configuration["contract_id"], kind=RECEIPT
                )
                for item in terminal["attempts"]
            ],
        )

    def archive_payload(self, reservation_id):
        """Proponer evidencia completa, sin acreditar ni publicar una confirmación."""
        _digest(reservation_id)
        state = self._read()
        _require(bool(state["cohorts"]), "No existe una reserva regional para archivar")
        cohort = state["cohorts"][-1]
        _require(
            cohort["attempts"][-1]["reservation_id"] == reservation_id
            and cohort["attempts"][-1]["status"] in ("reserved", "confirmed"),
            "El archivo no corresponde al intento regional pendiente",
        )
        return deepcopy(self._archive_payload(cohort))

    def verify_archive_payload(self, payload, *, generation, cutoff, rows_digest, event_kind=None):
        """Comprobar evidencia inline, aunque ya no existan sus blobs individuales."""
        _integer(generation, 1)
        _integer(cutoff, 1)
        _digest(rows_digest)
        _require(
            type(payload) is dict
            and set(payload) == {"reservation_id", "reference", "cohort", "receipts"}
            and type(payload["cohort"]) is dict
            and "history_before" in payload["cohort"],
            "El archivo regional no conserva su cuerpo canónico",
        )
        cohort = payload["cohort"]
        history = self._history(cohort["history_before"])
        self._check_cohort(cohort, history, receipts=payload["receipts"], terminal=True)
        expected_generation = history["initial_prefix"]["generation"] + history["cohorts"] + 1
        _require(
            generation == cohort["prefix"]["generation"] + 1 == expected_generation
            and cutoff == cohort["cutoff"]
            and rows_digest == cohort["rows_digest"]
            and payload["reservation_id"] == cohort["attempts"][-1]["reservation_id"]
            and payload["reference"] == cohort["attempts"][-1]["reference"],
            "El archivo regional contradice el evento confirmado",
        )
        if event_kind is not None:
            _require(
                _event_kind(event_kind) == cohort["event_kind"],
                "El archivo regional pertenece a otro tipo de evento",
            )
        if history["cohorts"]:
            frontier = history["frontier"]
            _require(
                cutoff > frontier["cutoff"]
                or (
                    cohort["event_kind"] == "settlement"
                    and frontier["event_kind"] != "settlement"
                    and cutoff == frontier["cutoff"]
                ),
                "El archivo regional retrocede o repite settlement",
            )
        else:
            _require(
                cohort["prefix"] == history["initial_prefix"],
                "El archivo regional cambia su prefijo inicial",
            )
        return deepcopy(payload)

    def _archive_stamp(self, archive):
        _require(
            type(archive) is dict and set(archive) == {"reference", "generation", "reservation_id"},
            "El archivo regional necesita una referencia confirmada",
        )
        _integer(archive["generation"], 1)
        _digest(archive["reservation_id"])
        self._reference(archive["reference"], "mechanism_trace_chunk")

    def _archive_event(self, archive):
        self._archive_stamp(archive)
        block = self.artifacts.read(
            archive["reference"],
            identity=self.configuration["contract_id"],
            kind="mechanism_trace_chunk",
        )
        _require(
            type(block) is dict
            and set(block) == {"schema_version", "model_id", "events", "encoding", "compression"}
            and type(block["schema_version"]) is int
            and block["schema_version"] == 1
            and block["model_id"] == self.configuration["model_id"]
            and block["encoding"] == "canonical_json_utf8_v1"
            and block["compression"] in ("none", "zlib1")
            and type(block["events"]) is list
            and 1 <= len(block["events"]) <= 64,
            "El bloque regional no conserva el historial de mecanismos",
        )
        encoded = block["events"][-1]
        _require(
            type(encoded) is bytes and 0 < len(encoded) <= self.artifacts.max_bytes,
            "El evento regional excede sus bytes admitidos",
        )
        if block["compression"] == "zlib1":
            decoder = zlib.decompressobj()
            try:
                encoded = decoder.decompress(encoded, self.artifacts.max_bytes + 1)
            except zlib.error as error:
                raise ValueError("El evento regional no conserva su compresión") from error
            _require(
                len(encoded) <= self.artifacts.max_bytes
                and decoder.eof
                and not decoder.unused_data
                and not decoder.unconsumed_tail,
                "El evento regional comprimido es incompleto o excede su límite",
            )
        try:
            event = json.loads(encoded, object_pairs_hook=_unique)
        except (ValueError, UnicodeError, RecursionError) as error:
            raise ValueError("El evento regional no conserva su JSON") from error
        _require(
            type(event) is dict
            and set(event)
            == {
                "schema_version",
                "model_id",
                "generation",
                "cutoff",
                "kind",
                "native_record_sha256",
                "payload",
            }
            and type(event["schema_version"]) is int
            and event["schema_version"] == 1
            and event["model_id"] == self.configuration["model_id"]
            and type(event["generation"]) is int
            and event["generation"] == archive["generation"]
            and type(event["payload"]) is dict
            and "regional_work" in event["payload"],
            "El archivo regional no contiene la generación enlazada",
        )
        _digest(event["native_record_sha256"])
        _event_kind(event["kind"])
        work = event["payload"]["regional_work"]
        _require(
            type(work) is dict and type(work.get("cohort")) is dict,
            "El evento regional perdió su cohorte inline",
        )
        self.verify_archive_payload(
            work,
            generation=event["generation"],
            cutoff=event["cutoff"],
            rows_digest=work["cohort"].get("rows_digest"),
            event_kind=event["kind"],
        )
        _require(
            work["reservation_id"] == archive["reservation_id"],
            "El archivo regional pertenece a otra reserva",
        )
        return work

    def compact(self, latest, *, archive):
        """Retirar cohortes solo tras verificar su archivo nativo en el coordinador.

        El llamante acredita que reference pertenece al historial del bundle de latest.
        Aquí se comprueban el bloque sellado, el cuerpo inline y su continuidad local.
        Se conservan dos cohortes conciliadas y cualquier cohorte activa.
        """
        latest = _latest(latest)
        state = self._read()
        self._archive_stamp(archive)
        _require(
            latest == state["latest"]
            and archive["generation"] == latest["generation"]
            and archive["reservation_id"] == latest["reservation_id"],
            "La compactación necesita la cabeza nativa conciliada y su archivo",
        )
        confirmed = [
            cohort for cohort in state["cohorts"] if cohort["confirmed_prefix"] is not None
        ]
        _require(bool(confirmed), "La compactación no tiene cohortes conciliadas")
        work = self._archive_event(archive)
        _require(
            _json(work) == _json(self._archive_payload(confirmed[-1])),
            "El archivo regional no cubre los recibos de la cabeza confirmada",
        )
        count = max(0, len(confirmed) - 2)
        if not count:
            return
        summary = state["archived"]
        for cohort in state["cohorts"][:count]:
            _require(
                cohort["confirmed_prefix"] is not None,
                "La compactación no puede retirar una cohorte pendiente",
            )
            summary = self._fold(summary, cohort)
        state.update(archived=summary, archive=deepcopy(archive), cohorts=state["cohorts"][count:])
        self._write(state)
