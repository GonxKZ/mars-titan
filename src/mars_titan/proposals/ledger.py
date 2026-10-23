"""Cuota finita y recibos recuperables, sin seleccionar ni desplegar consumidores.

Los sellos detectan cambios incoherentes. No autentican un archivo local completo
ni impiden restaurar coherentemente una versión anterior de todos sus archivos.
"""

import fcntl
import json
import os
import re
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from mars_titan.data.storage import atomic_json

from .contracts import ProposalRegistry
from .mature_review import (
    MAX_REVIEW_BYTES,
    MAX_REVIEW_ROWS,
    MatureReview,
    bounded_integer,
    encoded,
    fingerprint,
    identifier,
    require,
)

KIND = "finite_registered_proposals_v1"
RECEIPT_KIND = "registered_configuration_proposal_v1"
LIMIT_MAXIMA = {
    "max_candidates": 8,
    "max_attempts": 32,
    "max_review_rows": MAX_REVIEW_ROWS,
    "max_review_bytes": MAX_REVIEW_BYTES,
    "max_state_bytes": 1024**2,
}
DISPOSITIONS = frozenset(("proposed", "rejected", "abandoned"))
RECEIPT_FIELDS = {
    "schema_version",
    "kind",
    "proposal_id",
    "ordinal",
    "registry_id",
    "review_id",
    "parent_consumer_id",
    "window_sha256",
    "events_sha256",
    "configuration_id",
    "request_id",
    "status",
    "reason",
    "previous_receipt_sha256",
    "selection_performed",
    "deployment_performed",
}


def _request_id(value):
    require(
        type(value) is str
        and 1 <= len(value.encode()) <= 128
        and all(character.isprintable() for character in value),
        "La petición necesita un identificador acotado y legible",
    )


def _reason(value):
    require(
        type(value) is str
        and 1 <= len(value.encode()) <= 256
        and all(character.isprintable() for character in value),
        "El resultado necesita un motivo legible y acotado",
    )


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "El estado repite un campo JSON")
        result[key] = value
    return result


@dataclass(frozen=True, init=False)
class ProposalReceipt:
    """Resultado sellado de un intento. proposed no significa selected o deployed."""

    _json: str
    _sha256: str

    def __init__(self, body):
        content = encoded(body, MAX_REVIEW_BYTES)
        object.__setattr__(self, "_json", content)
        object.__setattr__(self, "_sha256", fingerprint(body))

    def record(self):
        body = json.loads(self._json)
        require(fingerprint(body) == self._sha256, "El recibo no conserva su sello")
        return dict(body, receipt_sha256=self._sha256)


class ProposalLedger:
    """Una ejecución finita con un solo intento pendiente y cargos sin reembolso."""

    def __init__(
        self, directory, *, registry, review, limits, resume=False, generation_plan_id=None
    ):
        require(type(registry) is ProposalRegistry, "Falta el registro tipado de configuraciones")
        require(type(review) is MatureReview, "Falta la revisión madura tipada")
        require(type(resume) is bool, "resume debe ser booleano")
        require(
            type(limits) is dict and set(limits) == set(LIMIT_MAXIMA), "La cuota cambió su esquema"
        )
        for name, maximum in LIMIT_MAXIMA.items():
            bounded_integer(limits[name], name, 1, maximum)
        self.registry, self.review, self.limits = registry, review, dict(limits)
        self.generation_plan_id = generation_plan_id
        registry.identity()
        require(
            len(registry.entries) <= limits["max_candidates"], "El registro supera sus candidatas"
        )
        value = review.record()
        require(len(value["rows"]) <= limits["max_review_rows"], "La revisión supera sus filas")
        encoded(value, limits["max_review_bytes"])
        self._identity = self._bindings()
        self._identity_json = encoded(self._identity, MAX_REVIEW_BYTES)
        self.directory = Path(directory)
        self.path = self.directory / "state.json"
        initial = self._sealed(
            dict(
                schema_version=self._version(),
                kind=KIND,
                identity=self._identity,
                review=value,
                charged_attempts=0,
                attempts=[],
            )
        )
        # El límite se comprueba antes de crear la salida, también con una cuota pequeña.
        encoded(initial, limits["max_state_bytes"], pretty=True)
        with self._locked(create=not resume):
            if resume:
                self._read()
                self._clean_temporaries()
            else:
                require(
                    not self.path.exists() and not self.path.is_symlink(), "La ejecución ya existe"
                )
                require(
                    all(path.name == ".lock" for path in self.directory.iterdir()),
                    "La salida contiene archivos ajenos a la nueva ejecución",
                )
                atomic_json(self.path, initial)

    def _bindings(self):
        review = self.review.record()
        result = dict(
            registry_id=self.registry.registry_id,
            baseline_id=self.registry.baseline_id,
            review_id=review["review_id"],
            parent_consumer_id=review["parent_consumer_id"],
            window_sha256=review["window_sha256"],
            events_sha256=review["events_sha256"],
            limits=self.limits,
        )
        if self.generation_plan_id is not None:
            result["generation_plan_id"] = identifier(
                self.generation_plan_id, "El plan de generación"
            )
        return result

    def _version(self):
        return 2 if "generation_plan_id" in self._identity else 1

    def _check_bindings(self):
        require(
            encoded(self._bindings(), MAX_REVIEW_BYTES) == self._identity_json,
            "La ejecución cambió de registro, revisión, cuota o plan",
        )

    @contextmanager
    def _locked(self, *, create=False):
        for path in (self.directory, *self.directory.parents):
            require(not path.is_symlink(), "La salida no puede atravesar enlaces simbólicos")
        require(create or self.directory.is_dir(), "No existe una ejecución que reabrir")
        if create:
            self.directory.mkdir(parents=True, exist_ok=True)
        require(self.directory.is_dir(), "La salida no es un directorio")
        descriptor = os.open(
            self.directory / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600
        )
        try:
            require(stat.S_ISREG(os.fstat(descriptor).st_mode), "El bloqueo no es regular")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            self._check_bindings()
            yield
        finally:
            os.close(descriptor)

    @staticmethod
    def _sealed(body):
        value = {key: part for key, part in body.items() if key != "state_sha256"}
        return dict(value, state_sha256=fingerprint(value))

    def _proposal_id(self, ordinal, configuration_id, request_id):
        return fingerprint(
            dict(
                identity=self._identity,
                ordinal=ordinal,
                configuration_id=configuration_id,
                request_id=request_id,
            )
        )

    def _receipt(self, ordinal, configuration_id, request_id, previous):
        result = dict(
            schema_version=self._version(),
            kind=RECEIPT_KIND,
            proposal_id=self._proposal_id(ordinal, configuration_id, request_id),
            ordinal=ordinal,
            **{
                name: self._identity[name]
                for name in (
                    "registry_id",
                    "review_id",
                    "parent_consumer_id",
                    "window_sha256",
                    "events_sha256",
                )
            },
            configuration_id=configuration_id,
            request_id=request_id,
            status="reserved",
            reason=None,
            previous_receipt_sha256=previous,
            selection_performed=False,
            deployment_performed=False,
        )
        if "generation_plan_id" in self._identity:
            result["generation_plan_id"] = self._identity["generation_plan_id"]
        return result

    def _check_attempts(self, state):
        attempts = state["attempts"]
        require(type(attempts) is list, "Faltan los intentos de la ejecución")
        bounded_integer(len(attempts), "intentos conservados", 0, self.limits["max_attempts"])
        bounded_integer(
            state["charged_attempts"], "intentos cargados", len(attempts), len(attempts)
        )
        previous, requests = None, set()
        eligible = self.review.record()["mature_rows"] > 0
        fields = RECEIPT_FIELDS | {"receipt_sha256"}
        if "generation_plan_id" in self._identity:
            fields.add("generation_plan_id")
        for ordinal, receipt in enumerate(attempts, 1):
            require(
                type(receipt) is dict and set(receipt) == fields,
                "Un recibo cambió su esquema",
            )
            request = receipt["request_id"]
            _request_id(request)
            require(request not in requests, "El estado repite una petición")
            requests.add(request)
            configuration_id = receipt["configuration_id"]
            self.registry.resolve(configuration_id)
            expected = self._receipt(ordinal, configuration_id, request, previous)
            status, reason = receipt["status"], receipt["reason"]
            if status == "reserved":
                require(
                    ordinal == len(attempts) and reason is None, "El intento pendiente no es único"
                )
            else:
                require(
                    type(status) is str and status in DISPOSITIONS,
                    "Estado de propuesta desconocido",
                )
                _reason(reason)
                if status == "proposed":
                    require(eligible, "Falta evidencia madura elegible")
                expected.update(status=status, reason=reason)
            require(
                encoded(receipt, MAX_REVIEW_BYTES)
                == encoded(ProposalReceipt(expected).record(), MAX_REVIEW_BYTES),
                "El recibo contradice su cargo, identidad o enlace anterior",
            )
            previous = receipt["receipt_sha256"]

    def _read(self):
        try:
            descriptor = os.open(
                self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
            )
            with os.fdopen(descriptor, "rb") as stream:
                metadata = os.fstat(stream.fileno())
                require(
                    stat.S_ISREG(metadata.st_mode)
                    and 0 < metadata.st_size <= self.limits["max_state_bytes"],
                    "El estado no es regular o supera sus bytes",
                )
                data = stream.read(self.limits["max_state_bytes"] + 1)
            require(len(data) <= self.limits["max_state_bytes"], "El estado supera sus bytes")
            state = json.loads(data, object_pairs_hook=_unique)
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
            raise ValueError("No se puede leer el estado confirmado de propuestas") from error
        require(
            type(state) is dict
            and set(state)
            == {
                "schema_version",
                "kind",
                "identity",
                "review",
                "charged_attempts",
                "attempts",
                "state_sha256",
            },
            "El estado cambió su esquema",
        )
        require(
            type(state["schema_version"]) is int
            and state["schema_version"] == self._version()
            and state["kind"] == KIND
            and state["identity"] == self._identity
            and state["review"] == self.review.record()
            and state == self._sealed(state),
            "El estado no conserva su identidad, revisión o sello",
        )
        self._check_attempts(state)
        return state

    def _clean_temporaries(self):
        temporary = []
        for path in self.directory.iterdir():
            if path.name in {".lock", "state.json"}:
                continue
            require(
                re.fullmatch(r"\.state\.json\.[A-Za-z0-9_-]{1,64}", path.name) is not None
                and not path.is_symlink()
                and path.is_file()
                and path.stat().st_size <= self.limits["max_state_bytes"],
                "La ejecución contiene una entrada ajena o no acotada",
            )
            temporary.append(path)
            require(len(temporary) <= 1, "La ejecución acumula temporales sin reconciliar")
        for path in temporary:
            path.unlink()

    def _publish(self, state):
        sealed = self._sealed(state)
        self._check_attempts(sealed)
        encoded(sealed, self.limits["max_state_bytes"], pretty=True)
        atomic_json(self.path, sealed)

    def snapshot(self):
        """Releer el estado confirmado bajo bloqueo, sin entregar alias mutables."""
        with self._locked():
            return self._read()

    def reserve(self, configuration_id, *, request_id):
        """Cargar una petición admitida antes de producir su recibo terminal."""
        self.registry.resolve(configuration_id)
        _request_id(request_id)
        with self._locked():
            state = self._read()
            self._clean_temporaries()
            for previous in state["attempts"]:
                if previous["request_id"] == request_id:
                    require(
                        previous["configuration_id"] == configuration_id,
                        "La petición ya identifica otra configuración",
                    )
                    return ProposalReceipt(
                        {key: value for key, value in previous.items() if key != "receipt_sha256"}
                    )
            attempts = state["attempts"]
            require(
                not attempts or attempts[-1]["status"] != "reserved", "Hay un intento pendiente"
            )
            require(
                state["charged_attempts"] < self.limits["max_attempts"], "La cuota está agotada"
            )
            previous = None if not attempts else attempts[-1]["receipt_sha256"]
            receipt = ProposalReceipt(
                self._receipt(len(attempts) + 1, configuration_id, request_id, previous)
            )
            attempts.append(receipt.record())
            state["charged_attempts"] += 1
            self._publish(state)
            return receipt

    def finish(self, proposal_id, *, disposition, reason):
        """Finalizar metadatos. proposed exige evidencia, no ganancia positiva."""
        identifier(proposal_id, "La propuesta")
        require(type(disposition) is str and disposition in DISPOSITIONS, "Resultado desconocido")
        _reason(reason)
        with self._locked():
            state = self._read()
            self._clean_temporaries()
            found = next(
                (
                    (index, receipt)
                    for index, receipt in enumerate(state["attempts"])
                    if receipt["proposal_id"] == proposal_id
                ),
                None,
            )
            if found is None:
                raise ValueError("La propuesta no pertenece a esta ejecución")
            index, previous = found
            if previous["status"] != "reserved":
                require(
                    previous["status"] == disposition and previous["reason"] == reason,
                    "El intento ya tiene otro resultado terminal",
                )
                return ProposalReceipt(
                    {key: value for key, value in previous.items() if key != "receipt_sha256"}
                )
            if disposition == "proposed":
                require(self.review.record()["mature_rows"] > 0, "Falta evidencia madura elegible")
            body = {key: value for key, value in previous.items() if key != "receipt_sha256"}
            body.update(status=disposition, reason=reason)
            receipt = ProposalReceipt(body)
            state["attempts"][index] = receipt.record()
            self._publish(state)
            return receipt
