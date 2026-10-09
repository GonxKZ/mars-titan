"""Campaña recuperable de adaptación con presupuesto explícito por fases."""

import copy
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training.learning_hold import require_learning_allowed

from .adaptation_scenarios import FAMILIES

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulation/adaptive-campaign.json"
LAUNCHER = PROJECT_ROOT / "scripts/run_native_ppo.py"
ENTRYPOINT = PROJECT_ROOT / "scripts/run_adaptive_campaign.py"
VARIANTS = (
    "ppo",
    "double_dqn",
    "ppo_window",
    "ppo_gru",
    "ppo_episodic",
    "ppo_hmm",
    "ppo_episodic_hmm",
)
AUXILIARY = ("ppo_recent_aux", "ppo_replay_aux")
SEEDS = (42, 43, 44)
SPLIT_COUNTS = {"train": 32, "validation": 16, "audit": 64}
JSON_LIMIT = 32 * 1024**2
FILE_LIMIT = 256 * 1024**2
STOP_TIMEOUT = 80


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(document):
    encoded = json.dumps(
        document, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _number(value, name, *, minimum=0):
    _require(
        type(value) in (int, float) and math.isfinite(value) and value >= minimum,
        f"{name} debe ser un número finito dentro de su intervalo",
    )
    return float(value)


def _integer(value, name, *, minimum=0, maximum=2**63 - 1):
    _require(
        type(value) is int and minimum <= value <= maximum, f"{name} debe ser un entero acotado"
    )
    return value


def _hash(value):
    _require(
        isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value),
        "Falta una huella SHA256 válida",
    )
    return value


def _json(path, maximum=JSON_LIMIT):
    safe_destination(path)
    return read_manifest(path, maximum)


def _file_hash(path, maximum=FILE_LIMIT):
    safe_destination(path)
    metadata = path.stat()
    _require(
        stat.S_ISREG(metadata.st_mode) and 0 < metadata.st_size <= maximum,
        "Un archivo de campaña no es regular o excede su presupuesto",
    )
    return sha256(path)


def _relative(root, value):
    _require(
        isinstance(value, str)
        and value
        and not Path(value).is_absolute()
        and all(part not in {".", ".."} for part in Path(value).parts),
        "Las rutas de campaña deben ser relativas y no salir de su directorio",
    )
    return root / value


def _confirm_bytes(path, payload):
    safe_destination(path)
    if path.exists():
        _require(
            path.stat().st_size == len(payload) and path.read_bytes() == payload,
            "Un archivo congelado de campaña ha cambiado",
        )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".campaign-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _settings(path):
    settings, digest = _json(path, 1024**2)
    expected = {
        "schema_version",
        "base_config",
        "variants",
        "seeds",
        "budgets_hours",
        "pilot_transitions",
        "transition_grid",
        "prediction_margin",
        "evaluation_transitions",
        "cooldown_seconds",
        "max_pause_retries",
        "auxiliary_gate",
        "audit_cost_bps",
        "final_test_opened",
    }
    _require(
        set(settings) in (expected, expected | {"initial_reserve_seconds"})
        and type(settings["schema_version"]) is int
        and settings["schema_version"] in (1, 2)
        and settings["variants"] == list(VARIANTS)
        and settings["seeds"] == list(SEEDS)
        and settings["budgets_hours"] == dict(pilot=12, main=108, auxiliary=36, reserve=12)
        and settings["pilot_transitions"] == 8192
        and settings["transition_grid"]
        == ([65536, 262144, 524288] if settings["schema_version"] == 1 else [1048576])
        and settings["prediction_margin"] == 1.25
        and settings["evaluation_transitions"] == 16384
        and settings["cooldown_seconds"] == 30
        and settings["max_pause_retries"] == 3
        and settings["auxiliary_gate"]
        == dict(minimum_log_growth_gain=0.001, maximum_drawdown_increase=0.02)
        and settings["audit_cost_bps"] == [0, 10, 25]
        and settings["final_test_opened"] is False,
        "La configuración no conserva el protocolo de 168 horas y sus comparaciones",
    )
    initial_reserve = settings.get("initial_reserve_seconds", 0)
    _require(
        _number(initial_reserve, "reserva inicial") <= 12 * 3600,
        "La reserva inicial debe caber en las 12 horas de reserva",
    )
    settings = dict(settings, initial_reserve_seconds=initial_reserve)
    base_path = _relative(path.parent, settings["base_config"])
    base, base_digest = _json(base_path, 1024**2)
    _require(
        base.get("schema_version") == (2 if settings["schema_version"] == 1 else 3)
        and base.get("environments") == 16
        and base["training"]["rollout_transitions"] == 1024
        and base["hyperparameters"]["minibatch_size"] == 64
        and (
            base["selection"]["early_stopping"] is False
            if settings["schema_version"] == 1
            else base["selection"]
            == dict(
                early_stopping=True,
                min_transitions=131072,
                patience=8,
                min_delta=0.0001,
                metric="ruin_count_then_mean_log_growth",
            )
            and base["training"]["total_transitions"] == 1048576
        )
        and base["agent"]["hmm_file"] is None
        and base.get("final_test_opened") is False,
        "La base de campaña no conserva la edición de selección, N16, rollout1024 y minibatch64",
    )
    return settings, base, {"campaign": digest, "base": base_digest}


def select_grid(costs, settings):
    """Elegir presupuesto con tiempos, sin recibir puntuaciones de validación."""
    expected = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    _require(
        len(costs) == len(expected)
        and {(row["variant"], row["seed"]) for row in costs} == expected
        and all(
            row["status"] == "completed" and row["transitions"] == settings["pilot_transitions"]
            for row in costs
        ),
        "El piloto necesita tres ejecuciones completas por variante",
    )
    maxima = {
        variant: max(
            _number(row["budget_seconds"], "duración del piloto", minimum=1e-9)
            for row in costs
            if row["variant"] == variant
        )
        for variant in VARIANTS
    }
    interval = settings["evaluation_transitions"]
    pilot_evaluations = 1 + math.ceil(settings["pilot_transitions"] / interval)
    predictions = []
    for transitions in settings["transition_grid"]:
        evaluations = 1 + math.ceil(transitions / interval)
        factor = 1 + transitions / settings["pilot_transitions"] + evaluations / pilot_evaluations
        seconds = math.fsum(maxima.values()) * factor * len(SEEDS) * settings["prediction_margin"]
        predictions.append(
            dict(transitions=transitions, predicted_seconds=seconds, evaluations=evaluations)
        )
    admitted = [
        row
        for row in predictions
        if row["predicted_seconds"] <= settings["budgets_hours"]["main"] * 3600
    ]
    _require(
        admitted,
        f"El piloto no permite completar ni {settings['transition_grid'][0]} transiciones "
        "en las 108 horas principales",
    )
    return dict(
        **admitted[-1],
        model="conservative_fixed_training_evaluation_envelope",
        predictions=predictions,
        maximum_pilot_seconds=maxima,
        margin=settings["prediction_margin"],
        score_used=False,
    )


def auxiliary_gate(metrics, settings):
    """Comparar los tres pares de semillas sin consultar la reserva de auditoría."""
    selected = [row for row in metrics if row["variant"] in {"ppo_hmm", "ppo_episodic_hmm"}]
    expected = {(variant, seed) for variant in ("ppo_hmm", "ppo_episodic_hmm") for seed in SEEDS}
    _require(
        len(selected) == len(expected)
        and {(row["variant"], row["seed"]) for row in selected} == expected,
        "La consolidación necesita los tres pares completos de HMM y memoria con HMM",
    )

    def aggregate(variant):
        rows = [row["best"] for row in selected if row["variant"] == variant]
        episodes = sum(_integer(row["episodes"], "episodios", minimum=1) for row in rows)
        ruins = sum(_integer(row["ruin_count"], "ruinas") for row in rows)
        means = {}
        for key in ("mean_log_growth", "mean_max_drawdown"):
            for row in rows:
                _number(row[key], key, minimum=-1e6 if key == "mean_log_growth" else 0)
            means[key] = sum(Decimal(str(row[key])) * row["episodes"] for row in rows) / episodes
        return episodes, ruins, means

    control_episodes, control_ruins, control = aggregate("ppo_hmm")
    candidate_episodes, candidate_ruins, candidate = aggregate("ppo_episodic_hmm")
    _require(
        control_episodes == candidate_episodes,
        "Los controles de consolidación no evalúan la misma población",
    )
    gain = candidate["mean_log_growth"] - control["mean_log_growth"]
    drawdown = candidate["mean_max_drawdown"] - control["mean_max_drawdown"]
    thresholds = settings["auxiliary_gate"]
    enabled = (
        gain >= Decimal(str(thresholds["minimum_log_growth_gain"]))
        and candidate_ruins <= control_ruins
        and (drawdown <= Decimal(str(thresholds["maximum_drawdown_increase"])))
    )
    return dict(
        enabled=enabled,
        source="validation_only",
        control="ppo_hmm",
        candidate="ppo_episodic_hmm",
        log_growth_gain=float(gain),
        drawdown_increase=float(drawdown),
        control_ruins=control_ruins,
        candidate_ruins=candidate_ruins,
        episodes=control_episodes,
        thresholds=thresholds,
    )


class FrozenInputs:
    def __init__(self, scenarios, binary, config, build_identity=None):
        self.scenarios = scenarios
        self.binary = binary
        self.config_path = config
        self.settings, self.base, config_hashes = _settings(config)
        self.index, index_hash = _json(scenarios)
        _require(
            type(self.index.get("schema_version")) is int
            and self.index["schema_version"] == 1
            and self.index.get("kind") == "adaptation_scenarios"
            and self.index.get("status") == "completed"
            and self.index.get("domain") == "synthetic"
            and self.index.get("final_test_opened") is False,
            "El índice no es un catálogo sintético completo con el test cerrado",
        )
        self.records = self.index.get("records")
        _require(
            isinstance(self.records, list)
            and len(self.records) == len(FAMILIES) * sum(SPLIT_COUNTS.values()),
            "El catálogo necesita 256 fuentes train, 128 de validación y 512 de auditoría",
        )
        names, seeds, manifests = set(), set(), set()
        for row in self.records:
            _require(
                row.get("family") in FAMILIES
                and row.get("split") in SPLIT_COUNTS
                and row.get("partition") == ("train" if row["split"] == "train" else "validation")
                and row.get("evaluator_only") is (row["split"] == "audit")
                and isinstance(row.get("name"), str)
                and re.fullmatch(r"[a-z0-9_-]{1,160}", row["name"])
                and row.get("path") == row["name"]
                and row["name"] not in names,
                "Una fuente no conserva su familia, partición o ruta relativa",
            )
            _integer(row["seed"], "semilla del escenario", maximum=2**32 - 1)
            _require(
                row["seed"] not in seeds and _hash(row["manifest_sha256"]) not in manifests,
                "El catálogo reutiliza semillas o manifiestos entre escenarios",
            )
            _hash(row["context_sha256"])
            names.add(row["name"])
            seeds.add(row["seed"])
            manifests.add(row["manifest_sha256"])
        for family in FAMILIES:
            for split, count in SPLIT_COUNTS.items():
                _require(
                    sum(row["family"] == family and row["split"] == split for row in self.records)
                    == count,
                    "El catálogo no está equilibrado por familia y partición",
                )
        self.partitions = {}
        for split, count in SPLIT_COUNTS.items():
            families = {
                family: sorted(
                    (
                        row
                        for row in self.records
                        if row["split"] == split and row["family"] == family
                    ),
                    key=lambda row: row["seed"],
                )
                for family in FAMILIES
            }
            self.partitions[split] = [
                families[family][ordinal] for ordinal in range(count) for family in FAMILIES
            ]
        self.source_files = []
        for row in [*self.partitions["train"], *self.partitions["validation"]]:
            self.check_source(row, remember=True)
        hmm_record = self.index.get("hmm", {})
        self.hmm_path = _relative(scenarios.parent, hmm_record.get("path"))
        hmm, hmm_hash = _json(self.hmm_path, 4 * 1024**2)
        _require(
            hmm_hash == hmm_record.get("sha256")
            and hmm.get("fit_split") == "train"
            and hmm.get("inference") == "forward_filter_only"
            and hmm.get("feature_indices") == [4, 5, 6]
            and isinstance(hmm.get("train_manifest_sha256"), list)
            and len(hmm["train_manifest_sha256"]) == len(self.partitions["train"])
            and set(hmm["train_manifest_sha256"])
            == {row["manifest_sha256"] for row in self.partitions["train"]},
            "El HMM no está ligado exclusivamente al conjunto train congelado",
        )
        self.hmm_bytes = self.hmm_path.read_bytes()
        self.build_path = build_identity or binary.parent / "native-build-identity-Release.txt"
        _file_hash(self.build_path, 1024**2)
        build_bytes = self.build_path.read_bytes()
        header, _, body = build_bytes.partition(b"\n")
        self.build_hash = hashlib.sha256(body).hexdigest()
        _require(
            header == f"sha256={self.build_hash}".encode() and b"build_type[7]=Release\n" in body,
            "La identidad de compilación no conserva un Release verificable",
        )
        source_match = re.search(rb"(?:^|\n)source_sha256\[64\]=([a-f0-9]{64})(?:\n|$)", body)
        _require(source_match is not None, "Falta la identidad de las fuentes nativas compiladas")
        self.source_hash = source_match[1].decode()
        _require(os.access(binary, os.X_OK), "El binario de campaña no es ejecutable")
        self.identity = dict(
            schema_version=1,
            kind="adaptive_campaign",
            config_hashes=config_hashes,
            settings=self.settings,
            base_configuration=self.base,
            index_sha256=index_hash,
            binary_sha256=_file_hash(binary),
            build_identity_sha256=_file_hash(self.build_path),
            native_build_sha256=self.build_hash,
            native_source_sha256=self.source_hash,
            hmm_sha256=hmm_hash,
            source_manifests=[
                dict(
                    name=row["name"],
                    split=row["split"],
                    manifest_sha256=row["manifest_sha256"],
                    context_sha256=row["context_sha256"],
                )
                for row in self.records
            ],
            source_order={
                split: [row["manifest_sha256"] for row in rows]
                for split, rows in self.partitions.items()
            },
            code=dict(
                module=_file_hash(Path(__file__)),
                entrypoint=_file_hash(ENTRYPOINT),
                launcher=_file_hash(LAUNCHER),
                uv_lock=_file_hash(PROJECT_ROOT / "uv.lock"),
            ),
            final_test_opened=False,
        )

    def check_source(self, row, *, remember=False):
        directory = _relative(self.scenarios.parent, row["path"])
        manifest, digest = _json(directory / "manifest.json", 4 * 1024**2)
        context, context_digest = _json(directory / "context.json", 4 * 1024**2)
        _require(
            digest == row["manifest_sha256"]
            and context_digest == row["context_sha256"]
            and manifest.get("final_test_opened") is False
            and manifest["identity"]["domain"] == "synthetic"
            and manifest["identity"]["partition"] == row["partition"]
            and manifest["identity"]["parent_id"] == self.index["parent_id"]
            and manifest["identity"]["source"]["generator"]["seed"] == row["seed"]
            and manifest["identity"]["source"]["generator"]["split"] == row["split"]
            and manifest["identity"]["source"]["generator"]["family"] == row["family"]
            and context.get("market_manifest_sha256") == digest,
            "Una fuente no conserva el manifiesto, contexto o procedencia congelados",
        )
        for path, description in (
            (
                directory / "market.parquet",
                dict(sha256=manifest["file_sha256"], bytes=manifest["file_bytes"]),
            ),
            (_relative(directory, context["file"]["path"]), context["file"]),
        ):
            _require(
                _file_hash(path) == description["sha256"]
                and path.stat().st_size == description["bytes"],
                "Los datos de un escenario no conservan tamaño y SHA256",
            )
        if remember:
            self.source_files.extend(
                (
                    (directory / "manifest.json", digest),
                    (directory / "context.json", context_digest),
                )
            )

    def unchanged(self):
        checks = [
            (self.scenarios, self.identity["index_sha256"]),
            (self.binary, self.identity["binary_sha256"]),
            (self.build_path, self.identity["build_identity_sha256"]),
            (self.hmm_path, self.identity["hmm_sha256"]),
            (self.config_path, self.identity["config_hashes"]["campaign"]),
            (
                _relative(self.config_path.parent, self.settings["base_config"]),
                self.identity["config_hashes"]["base"],
            ),
            (Path(__file__), self.identity["code"]["module"]),
            (ENTRYPOINT, self.identity["code"]["entrypoint"]),
            (LAUNCHER, self.identity["code"]["launcher"]),
            (PROJECT_ROOT / "uv.lock", self.identity["code"]["uv_lock"]),
            *self.source_files,
        ]
        _require(
            all(_file_hash(path) == digest for path, digest in checks),
            "La identidad de código, configuración o fuentes cambió durante la campaña",
        )


@contextmanager
def _campaign_lock(output):
    descriptor = os.open(
        output / ".campaign.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600
    )
    try:
        metadata = os.fstat(descriptor)
        _require(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == os.getuid()
            and metadata.st_nlink == 1,
            "El bloqueo de campaña no es un archivo privado regular",
        )
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def _execute(command, stop):
    child = subprocess.Popen(command, start_new_session=True)
    requested = None
    try:
        while child.poll() is None:
            if stop() and requested is None:
                child.send_signal(signal.SIGTERM)
                requested = time.monotonic()
            if requested is not None and time.monotonic() - requested > STOP_TIMEOUT:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
                return 75
            time.sleep(0.1)
        return child.returncode
    finally:
        if child.poll() is None:
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)


class Campaign:
    def __init__(self, inputs, output, resume, executor, stop, sleep):
        self.inputs, self.output = inputs, output
        self.execute, self.stop, self.sleep = executor, stop, sleep
        self.identity_hash = _digest(inputs.identity)
        self.journal_path = output / "campaign.json"
        if resume and not self.journal_path.exists():
            self.validate_setup()
            resume = False
        if resume:
            envelope, _ = _json(self.journal_path)
            self.state = envelope["payload"]
            _require(
                envelope["sha256"] == _digest(self.state)
                and self.state["identity_sha256"] == self.identity_hash,
                "La identidad o el diario de campaña no coincide con esta recuperación",
            )
            _require(
                isinstance(self.state.get("cases"), list) and len(self.state["cases"]) <= 75,
                "La cola persistida de campaña excede su presupuesto",
            )
            self.validate_queue()
        else:
            self.state = dict(
                schema_version=1,
                identity_sha256=self.identity_hash,
                phase="pilot",
                status="running",
                cases=[],
                choice=None,
                gate=None,
                freeze_sha256=None,
                audit_opened=False,
                budget_complete=True,
                reason=None,
            )
            atomic_json(output / "identity.json", inputs.identity)
        self._ensure_file(output / "configs/hmm.json", inputs.hmm_bytes)
        if not self.state["cases"]:
            self.add_cases("pilot", VARIANTS, inputs.settings["pilot_transitions"])
        self.publish()

    def _ensure_file(self, path, payload):
        _confirm_bytes(path, payload)

    def validate_setup(self):
        _require(
            all(
                path.name in {".campaign.lock", "identity.json", "configs"}
                for path in self.output.iterdir()
            ),
            "No se puede reconstruir un diario ausente después de iniciar la campaña",
        )
        identity_path = self.output / "identity.json"
        if identity_path.exists():
            _require(
                _json(identity_path)[0] == self.inputs.identity,
                "La identidad inicial no coincide con esta recuperación",
            )
        configs = self.output / "configs"
        if configs.exists():
            expected = {"hmm.json"} | {
                f"pilot-{variant}-{seed}.json" for variant in VARIANTS for seed in SEEDS
            }
            _require(
                all(path.name in expected for path in configs.iterdir()),
                "La preparación incompleta contiene configuraciones ajenas",
            )

    def validate_queue(self):
        identifiers = set()
        for case in self.state["cases"]:
            stage, variant, seed = case.get("stage"), case.get("variant"), case.get("seed")
            allowed = (
                VARIANTS
                if stage in {"pilot", "main"}
                else AUXILIARY
                if stage == "auxiliary"
                else (*VARIANTS, *AUXILIARY)
            )
            _require(
                stage in {"pilot", "main", "auxiliary", "audit"}
                and variant in allowed
                and seed in SEEDS,
                "El diario contiene un caso fuera de la comparación declarada",
            )
            identifier = f"{stage}-{variant}-{seed}"
            config_stage = (
                ("auxiliary" if variant in AUXILIARY else "main") if stage == "audit" else stage
            )
            _require(
                case.get("id") == identifier
                and identifier not in identifiers
                and case.get("output") == f"{stage}/{variant}-{seed}"
                and case.get("watch") == f"watch/{identifier}.json"
                and case.get("config") == f"configs/{config_stage}-{variant}-{seed}.json",
                "El diario contiene rutas ajenas o casos duplicados",
            )
            if stage == "audit":
                _require(
                    case.get("training_output") == f"{config_stage}/{variant}-{seed}",
                    "La auditoría apunta a otra selección",
                )
            identifiers.add(identifier)

    def add_cases(self, stage, variants, transitions):
        for variant in variants:
            for seed in SEEDS:
                identifier = f"{stage}-{variant}-{seed}"
                if any(row["id"] == identifier for row in self.state["cases"]):
                    continue
                config = copy.deepcopy(self.inputs.base)
                config["training"].update(seed=seed, total_transitions=transitions)
                if stage == "pilot" and self.inputs.settings["schema_version"] == 2:
                    config["selection"].update(early_stopping=False, min_transitions=0)
                config["evaluation_transitions"] = self.inputs.settings["evaluation_transitions"]
                config["agent"].update(variant=variant, markov_fields=[], hmm_file=None)
                if variant in {"ppo_hmm", "ppo_episodic_hmm", *AUXILIARY}:
                    config["agent"].update(
                        markov_fields=[4, 5, 6],
                        hmm_file=dict(path="hmm.json", sha256=self.inputs.identity["hmm_sha256"]),
                    )
                relative = f"configs/{identifier}.json"
                path = self.output / relative
                if path.exists():
                    _require(
                        _json(path)[0] == config,
                        "Una configuración de caso ya existe con otro contenido",
                    )
                else:
                    atomic_json(path, config)
                self.state["cases"].append(
                    dict(
                        id=identifier,
                        stage=stage,
                        variant=variant,
                        seed=seed,
                        transitions=transitions,
                        config=relative,
                        config_sha256=_file_hash(path),
                        output=f"{stage}/{variant}-{seed}",
                        watch=f"watch/{identifier}.json",
                        status="pending",
                        attempts=0,
                        pauses=0,
                        active_seconds=0.0,
                        budget_seconds=0.0,
                        watch_identity=None,
                        accounted_starts=0,
                        watch_budget_seconds=0.0,
                        receipt_sha256=None,
                        best=None,
                        reason=None,
                    )
                )
        _require(
            sum(row["stage"] in {"main", "auxiliary"} for row in self.state["cases"]) <= 27,
            "La campaña supera 27 ajustes principales y auxiliares",
        )

    def publish(self):
        self.state["updated_at"] = datetime.now(UTC).isoformat()
        atomic_json(self.journal_path, dict(payload=self.state, sha256=_digest(self.state)))
        report = dict(
            schema_version=1,
            kind="adaptive_campaign",
            activity="rl",
            model="adaptive_comparison",
            status=self.state["status"],
            phase=self.state["phase"],
            updated_at=self.state["updated_at"],
            identity_sha256=self.identity_hash,
            budget_seconds=sum(row["budget_seconds"] for row in self.state["cases"]),
            active_seconds=sum(row["active_seconds"] for row in self.state["cases"]),
            budget_limit_seconds=168 * 3600,
            initial_reserved_seconds=self.inputs.settings["initial_reserve_seconds"],
            unobserved_reserved_seconds=math.fsum(
                row["budget_seconds"] - row["active_seconds"] for row in self.state["cases"]
            ),
            budget_complete=self.state.get("budget_complete", True),
            budgets_hours=self.inputs.settings["budgets_hours"],
            chosen_transitions=self.state["choice"]["transitions"]
            if self.state["choice"]
            else None,
            completed_cases=sum(row["status"] == "completed" for row in self.state["cases"]),
            active_case=self.state.get("active_case"),
            auxiliary_gate=self.state["gate"],
            audit_opened=self.state["audit_opened"],
            selection_frozen=self.state["freeze_sha256"] is not None,
            reason=self.state["reason"],
            final_test_opened=False,
            parent_frozen=True,
            analysis_domain="technical",
            domain="synthetic",
        )
        report["budget_seconds"] += self.inputs.settings["initial_reserve_seconds"]
        atomic_json(self.output / "run.json", report)
        public_fields = {
            "path",
            "stage",
            "variant",
            "seed",
            "status",
            "config_sha256",
            "planned_transitions",
            "transitions",
        }
        atomic_json(
            self.output / "registry.json",
            dict(
                schema_version=1,
                kind="adaptive_campaign",
                status=self.state["status"],
                planned_runs=len(self.state["cases"]),
                runs=[
                    {key: value for key, value in row.items() if key in public_fields}
                    for row in self.run_rows()
                ],
            ),
        )
        return report

    def run_rows(self):
        return [
            dict(
                path=case["output"],
                planned_transitions=case["transitions"],
                transitions=case.get("confirmed_transitions", 0),
                **{
                    field: case[field]
                    for field in (
                        "stage",
                        "variant",
                        "seed",
                        "status",
                        "config_sha256",
                        "active_seconds",
                        "budget_seconds",
                        "optimizer_steps",
                        "auxiliary_steps",
                        "auxiliary_samples",
                        "stopping_reason",
                    )
                    if field in case
                },
            )
            for case in self.state["cases"]
        ]

    def remaining(self, stage):
        total = (
            math.fsum(row["budget_seconds"] for row in self.state["cases"])
            + self.inputs.settings["initial_reserve_seconds"]
        )
        stage_total = math.fsum(
            row["budget_seconds"] for row in self.state["cases"] if row["stage"] == stage
        )
        allocation = "reserve" if stage == "audit" else stage
        if allocation == "reserve":
            stage_total += self.inputs.settings["initial_reserve_seconds"]
        return max(
            0.0,
            min(
                168 * 3600 - total,
                self.inputs.settings["budgets_hours"][allocation] * 3600 - stage_total,
            ),
        )

    def watch(self, case, *, missing=False):
        path = self.output / case["watch"]
        if not path.exists():
            _require(
                missing and case["attempts"] == 0,
                "Falta la contabilidad persistida del intento nativo",
            )
            return None
        record, _ = _json(path, 64 * 1024)
        if record.get("schema_version") != 2 or record.get("budget_complete") is not True:
            self.state["budget_complete"] = False
            raise ValueError("La vigilancia no acredita una contabilidad completa del presupuesto")
        _hash(record.get("identity_sha256"))
        _require(
            case["watch_identity"] in (None, record["identity_sha256"]),
            "El watch pertenece a otra identidad de caso",
        )
        case["watch_identity"] = record["identity_sha256"]
        active = _number(record.get("active_seconds"), "tiempo activo")
        budget = _number(record.get("budget_seconds"), "tiempo presupuestado")
        _require(
            active >= case["active_seconds"]
            and budget >= case.get("watch_budget_seconds", 0)
            and budget >= active,
            "La contabilidad del caso ha retrocedido",
        )
        case.update(
            active_seconds=active,
            budget_seconds=budget,
            watch_budget_seconds=budget,
            attempts=_integer(record["starts"], "intentos", maximum=1_000_000),
            pauses=_integer(record["pauses"], "pausas", maximum=1_000_000),
        )
        if record.get("child_active"):
            if record.get("parent_death_signal") == "SIGKILL":
                case["budget_seconds"] += _number(
                    record["unobserved_reserve_seconds"], "reserva no observada"
                )
            else:
                self.state["budget_complete"] = False
                raise ValueError("Hay un intento activo anterior sin cierre confirmado")
        return record

    def command(self, case):
        command = [
            sys.executable,
            str(LAUNCHER),
            "--config",
            str(self.output / case["config"]),
            "--output",
            str(self.output / case["output"]),
            "--binary",
            str(self.inputs.binary),
            "--watch-state",
            str(self.output / case["watch"]),
            "--active-seconds-limit",
            str(self.remaining(case["stage"])),
        ]
        if case["stage"] == "audit":
            command.extend(("--audit-run", str(self.output / case["training_output"])))
            for row in self.inputs.partitions["audit"]:
                command.extend(
                    ("--audit-tape", str(_relative(self.inputs.scenarios.parent, row["path"])))
                )
        else:
            for split, option in (("train", "--train-tape"), ("validation", "--validation-tape")):
                for row in self.inputs.partitions[split]:
                    command.extend(
                        (option, str(_relative(self.inputs.scenarios.parent, row["path"])))
                    )
        if (self.output / case["output"]).exists():
            command.append("--resume")
        return command

    def validate_convergence_receipt(self, directory, receipt, config, identity_hash):
        """Contrastar el recibo con el último estado confirmado de selección."""
        envelope, _ = _json(directory / "ppo-index.json")
        index = envelope["payload"]
        _require(
            envelope.get("sha256") == _digest(index)
            and index.get("identity_sha256") == identity_hash,
            "El índice de selección no conserva su identidad y sello",
        )
        record = index["recent"][0]
        bundle = _relative(directory, "ppo-" + _hash(record["sha256"]))
        _require(record.get("bundle") == bundle.name, "El checkpoint reciente no es canónico")
        manifest, digest = _json(bundle / "manifest.json")
        metadata, metadata_digest = _json(bundle / "metadata.json")
        description = manifest["files"]["metadata.json"]
        _require(
            digest == record["sha256"]
            and manifest.get("identity_sha256") == identity_hash
            and metadata_digest == description["sha256"]
            and (bundle / "metadata.json").stat().st_size == description["bytes"],
            "El estado confirmado no conserva sus hashes",
        )
        progress = metadata["progress"]
        early = receipt.get("stopping_reason") == "early_stop"
        transitions = _integer(receipt.get("transitions"), "transiciones")
        selection = config["selection"]
        minimum = selection["min_transitions"]
        interval = config["evaluation_transitions"]
        history = progress.get("evaluation_cursors")
        _require(
            isinstance(history, list)
            and 1 <= len(history) <= 4096
            and history == receipt.get("evaluation_cursors"),
            "El selector no conserva una historia de validaciones completa y acotada",
        )
        history = [
            _integer(cursor, "cursor de validación", maximum=transitions) for cursor in history
        ]
        _require(
            history[0] == 0
            and history[-1] == transitions
            and all(
                0 < current - previous < interval + config["environments"]
                and (
                    current - previous >= interval
                    or current == config["training"]["total_transitions"]
                )
                for previous, current in pairwise(history)
            ),
            "Los cursores no respetan la programación de validaciones completas",
        )
        evaluations = len(history)
        best = receipt["best"]
        best_transition = _integer(
            best.get("transitions"), "transiciones seleccionadas", maximum=transitions
        )
        _require(best_transition in history, "El mejor estado no tiene una validación completa")
        eligible = sum(cursor > max(minimum, best_transition) for cursor in history)
        stale = _integer(receipt.get("stale_evaluations"), "paciencia", maximum=eligible)
        _require(
            receipt.get("schema_version") == 3
            and receipt.get("selection") == dict(selection, policy="greedy_argmax")
            and metadata.get("schema_version") == 3
            and metadata.get("configuration") == config
            and metadata.get("transitions") == transitions
            and metadata.get("optimizer_steps") == receipt["optimizer_steps"]
            and progress.get("evaluated_transitions") == transitions
            and progress.get("evaluated_optimizer_steps") == receipt["optimizer_steps"]
            and progress.get("evaluations") == receipt.get("evaluations") == evaluations
            and progress.get("stale_evaluations") == stale == eligible
            and progress.get("best") == best
            and progress.get("status") == ("early_stopped" if early else "completed")
            and (
                not early
                or (
                    selection["early_stopping"] is True
                    and receipt["optimizer_steps"] > 0
                    and minimum < transitions < config["training"]["total_transitions"]
                    and stale >= selection["patience"]
                )
            ),
            "El recibo no acredita el mínimo, la paciencia y la selección terminal confirmada",
        )

    def validate_training_receipt(self, case, expected_status):
        directory = self.output / case["output"]
        receipt, digest = _json(directory / "run.json")
        identity_record, _ = _json(directory / "identity.json")
        identity = identity_record["identity"]
        config = _json(self.output / case["config"])[0]
        _require(
            receipt.get("status") == expected_status
            and receipt.get("kind") == "native_ppo"
            and receipt.get("seed") == case["seed"]
            and receipt.get("agent_variant") == case["variant"]
            and receipt.get("device") == "cuda:0"
            and receipt.get("diagnostic") is False
            and receipt.get("final_test_opened") is False
            and receipt.get("parent_frozen") is True
            and receipt.get("identity_sha256") == identity_record.get("sha256")
            and identity_record.get("sha256") == _digest(identity)
            and identity.get("configuration") == config
            and identity.get("native_build_sha256") == self.inputs.build_hash
            and identity.get("native_source_sha256") == self.inputs.source_hash,
            "El recibo nativo no conserva estado, identidad, algoritmo o configuración",
        )
        for split in ("train", "validation"):
            expected = [
                (row["manifest_sha256"], row["context_sha256"])
                for row in self.inputs.partitions[split]
            ]
            actual = identity["sources"][split]
            _require(
                len(actual) == len(expected)
                and [(row["manifest_sha256"], row["context_sha256"]) for row in actual] == expected,
                "El recibo nativo no usa todas las fuentes congeladas de su partición",
            )
        counters = {
            key: _integer(receipt.get(key), key)
            for key in ("optimizer_steps", "auxiliary_steps", "auxiliary_samples")
        }
        _require(
            counters["auxiliary_steps"]
            <= counters["auxiliary_samples"]
            <= 64 * counters["auxiliary_steps"]
            and (case["variant"] in AUXILIARY or counters["auxiliary_steps"] == 0),
            "Los contadores auxiliares no corresponden a las muestras y variante del caso",
        )
        case.update(
            confirmed_transitions=_integer(
                receipt.get("transitions"), "transiciones confirmadas", maximum=case["transitions"]
            ),
            **counters,
        )
        if expected_status == "completed":
            convergent = self.inputs.settings["schema_version"] == 2 and case["stage"] in {
                "main",
                "auxiliary",
            }
            budget_complete = (
                receipt.get("transitions") == case["transitions"]
                and receipt.get("stopping_reason") == "budget_exhausted"
            )
            _require(
                receipt.get("total_steps") == case["transitions"]
                and (
                    budget_complete
                    or (convergent and receipt.get("stopping_reason") == "early_stop")
                ),
                "El caso terminó sin completar su presupuesto o una parada validada",
            )
            if self.inputs.settings["schema_version"] == 2:
                self.validate_convergence_receipt(
                    directory, receipt, config, identity_record["sha256"]
                )
                case["stopping_reason"] = receipt["stopping_reason"]
            best = receipt["best"]
            _require(
                best.get("episodes") == len(self.inputs.partitions["validation"]),
                "La mejor validación no está completa",
            )
            _integer(best.get("ruin_count"), "ruinas", maximum=best["episodes"])
            _number(best.get("mean_log_growth"), "crecimiento logarítmico", minimum=-1e6)
            _number(best.get("mean_max_drawdown"), "drawdown medio")
            self.validate_metrics(
                best.get("validation_metrics"),
                "validation",
                [config["environment"]["cost_bps"]],
                with_cost=False,
            )
            _require(
                case["best"] in (None, best), "La mejor validación cambió tras confirmar el caso"
            )
            case.update(
                best=best,
                receipt_sha256=digest,
                training_identity_sha256=receipt["identity_sha256"],
            )
        return receipt

    def validate_metrics(self, rows, split, costs, *, with_cost=True):
        expected = {
            (row["manifest_sha256"], cost)
            for row in self.inputs.partitions[split]
            for cost in costs
        }
        _require(
            isinstance(rows, list) and len(rows) == len(expected),
            "La evaluación no contiene todos los escenarios y costes",
        )
        observed = set()
        for row in rows:
            key = row.get("manifest_sha256"), row.get("cost_bps") if with_cost else costs[0]
            _require(
                key in expected and key not in observed and row.get("completed") is True,
                "La evaluación mezcla fuentes, duplica episodios o contiene resultados incompletos",
            )
            observed.add(key)
            for field in ("net_return", "max_drawdown", "costs", "turnover"):
                _number(row.get(field), field, minimum=-1 if field == "net_return" else 0)
            _integer(row.get("steps"), "pasos evaluados", minimum=1, maximum=8192)

    def validate_audit_receipt(self, case, expected_status):
        report, digest = _json(self.output / case["output"] / "audit.json")
        frozen, freeze_digest = _json(self.output / "freeze.json")
        _require(
            freeze_digest == self.state["freeze_sha256"],
            "La selección congelada perdió su identidad",
        )
        selected = next(
            row for row in frozen["selections"] if row["output"] == case["training_output"]
        )
        identity = report.get("identity", {})
        checkpoint = identity.get("selected_checkpoint", {})
        _require(
            report.get("kind") == "native_ppo_audit"
            and report.get("status") == expected_status
            and report.get("seed") == case["seed"]
            and report.get("identity_sha256") == _digest(identity)
            and identity.get("selected_identity_sha256") == selected["training_identity_sha256"]
            and checkpoint.get("bundle") == selected["bundle"]["bundle"]
            and checkpoint.get("sha256") == selected["bundle"]["sha256"]
            and identity.get("policy_sha256") == selected["policy_sha256"]
            and identity.get("configuration") == _json(self.output / case["config"])[0]
            and identity.get("native_build_sha256") == self.inputs.build_hash
            and identity.get("native_source_sha256") == self.inputs.source_hash
            and identity.get("device") == "cuda:0"
            and identity.get("diagnostic") is False
            and identity.get("cost_bps") == self.inputs.settings["audit_cost_bps"]
            and identity.get("final_test_opened") is False,
            "El recibo de auditoría no corresponde al caso congelado",
        )
        _require(
            [(row["manifest_sha256"], row["context_sha256"]) for row in identity.get("sources", [])]
            == [
                (row["manifest_sha256"], row["context_sha256"])
                for row in self.inputs.partitions["audit"]
            ],
            "La auditoría no conserva el orden de sus fuentes reservadas",
        )
        if expected_status == "completed":
            count = len(self.inputs.partitions["audit"]) * len(
                self.inputs.settings["audit_cost_bps"]
            )
            _require(
                report.get("confirmed_episodes") == count and report.get("total_episodes") == count,
                "La auditoría no ha confirmado todos los episodios",
            )
            self.validate_metrics(
                report.get("metrics"), "audit", self.inputs.settings["audit_cost_bps"]
            )
            case.update(
                receipt_sha256=digest, audit_summary=self.group_metrics(report["metrics"], "audit")
            )
        return report

    def group_metrics(self, metrics, split, fixed_cost=None):
        families = {row["manifest_sha256"]: row["family"] for row in self.inputs.partitions[split]}
        groups = {}
        for row in metrics:
            key = families[row["manifest_sha256"]], row.get("cost_bps", fixed_cost)
            groups.setdefault(key, []).append(row)
        return [
            dict(
                family=family,
                cost_bps=cost,
                period="episode_complete",
                episodes=len(rows),
                **{
                    f"mean_{field}": math.fsum(row[field] for row in rows) / len(rows)
                    for field in ("net_return", "max_drawdown", "costs", "turnover")
                },
            )
            for (family, cost), rows in sorted(groups.items())
        ]

    def stop_case(self, case, status, reason):
        case.update(status=status, reason=reason)
        self.state.update(status="blocked" if status == "blocked" else "paused", reason=reason)
        self.publish()
        return False

    def run_case(self, case):
        receipt_name = "audit.json" if case["stage"] == "audit" else "run.json"
        if case["status"] == "completed":
            _require(
                _file_hash(self.output / case["output"] / receipt_name) == case["receipt_sha256"],
                "Un recibo completado cambió después de confirmarlo",
            )
            validator = (
                self.validate_audit_receipt
                if case["stage"] == "audit"
                else self.validate_training_receipt
            )
            validator(case, "completed")
            return True
        if case["status"] == "blocked":
            return self.stop_case(case, "blocked", case["reason"])
        while True:
            if self.stop():
                return self.stop_case(case, "paused", "requested_pause")
            record = self.watch(case, missing=True)
            if (
                record
                and not record.get("child_active")
                and case["status"] == "running"
                and record["starts"] > case.get("accounted_starts", 0)
            ):
                outcome = self.process_result(case, record.get("returncode"), record)
                if outcome is not None:
                    return outcome
                continue
            if case["pauses"] > self.inputs.settings["max_pause_retries"]:
                return self.stop_case(case, "blocked", "pause_retry_limit")
            if self.remaining(case["stage"]) <= 75:
                return self.stop_case(case, "paused", "active_budget_exhausted")
            self.inputs.unchanged()
            if case["stage"] == "audit":
                self.verify_freeze()
            _require(
                _file_hash(self.output / case["config"]) == case["config_sha256"],
                "La configuración congelada de un caso ha cambiado",
            )
            case["status"] = "running"
            self.state["active_case"] = case["id"]
            self.state.update(status="running", reason=None)
            self.publish()
            limit = self.remaining(case["stage"])
            before = case["budget_seconds"]
            code = self.execute(self.command(case), self.stop)
            record = self.watch(case, missing=code == 1)
            if case["budget_seconds"] - before > limit + 1e-6:
                return self.stop_case(case, "blocked", "active_budget_overrun")
            if code not in (0, 2, 3, 4):
                return self.process_result(case, code, record)
            if record is not None:
                _require(
                    (record.get("returncode") == code and not record.get("child_active"))
                    or code == 3,
                    "El lanzador no confirmó el cierre y código de su hijo",
                )
            self.publish()
            outcome = self.process_result(case, code, record)
            if outcome is not None:
                return outcome

    def process_result(self, case, code, record):
        if code != 3:
            case["accounted_starts"] = case["attempts"]
        if code in (0, 2):
            case.update(status="blocked", reason="receipt_not_confirmed")
        if code == 0:
            validator = (
                self.validate_audit_receipt
                if case["stage"] == "audit"
                else self.validate_training_receipt
            )
            validator(case, "completed")
            case.update(status="completed", reason=None)
            self.state["active_case"] = None
            self.publish()
            return True
        if code == 3:
            pending = record and (
                record.get("child_active") or record["starts"] > case.get("accounted_starts", 0)
            )
            case["status"] = "running" if pending else "waiting"
            self.state.update(status="paused", reason="waiting_for_gpu")
            self.publish()
            self.sleep(self.inputs.settings["cooldown_seconds"])
            return None
        if code == 4:
            return self.stop_case(case, "paused", "active_budget_exhausted")
        if code == 2:
            _require(
                record is not None and record.get("recoverable_checkpoint") is True,
                "La pausa no contiene confirmación de checkpoint recuperable",
            )
            validator = (
                self.validate_audit_receipt
                if case["stage"] == "audit"
                else self.validate_training_receipt
            )
            validator(case, "paused")
            case.update(status="paused", reason=None)
            self.publish()
            if record.get("reason") == "active_budget_exhausted" or self.stop():
                return self.stop_case(case, "paused", record.get("reason") or "requested_pause")
            if case["pauses"] > self.inputs.settings["max_pause_retries"]:
                return self.stop_case(case, "blocked", "pause_retry_limit")
            self.sleep(self.inputs.settings["cooldown_seconds"])
            return None
        return self.stop_case(case, "blocked", f"child_exit_{code}")

    def freeze(self):
        selections = []
        for case in self.state["cases"]:
            if case["stage"] not in {"main", "auxiliary"}:
                continue
            _require(
                case["status"] == "completed", "No se congela una selección con ajustes incompletos"
            )
            directory = self.output / case["output"]
            envelope, _ = _json(directory / "ppo-index.json")
            index = envelope["payload"]
            _require(
                envelope["sha256"] == _digest(index),
                "El índice del checkpoint no conserva su sello",
            )
            selected = index["best"]
            _require(
                isinstance(selected, dict)
                and selected.get("bundle") == "ppo-" + _hash(selected.get("sha256")),
                "Falta el mejor checkpoint confirmado de un caso",
            )
            bundle = _relative(directory, selected["bundle"])
            manifest, digest = _json(bundle / "manifest.json")
            _require(
                digest == selected["sha256"],
                "El manifiesto del mejor checkpoint no conserva su hash",
            )
            _require(
                manifest.get("identity_sha256") == case["training_identity_sha256"]
                and index.get("identity_sha256") == case["training_identity_sha256"],
                "El mejor checkpoint no pertenece a la identidad del entrenamiento",
            )
            for name in ("metadata.json", "policy.pt", "rollout.pt"):
                description = manifest["files"][name]
                _require(
                    _file_hash(bundle / name) == description["sha256"]
                    and (bundle / name).stat().st_size == description["bytes"],
                    "El mejor checkpoint perdió un archivo confirmado",
                )
            metadata, _ = _json(bundle / "metadata.json")
            _require(
                metadata.get("progress", {}).get("best") == case["best"]
                and metadata.get("transitions") == case["best"]["transitions"]
                and metadata.get("optimizer_steps") == case["best"]["optimizer_steps"],
                "El checkpoint no corresponde a la selección de mejor validación del recibo",
            )
            selections.append(
                dict(
                    case_id=case["id"],
                    output=case["output"],
                    config=case["config"],
                    config_sha256=case["config_sha256"],
                    bundle=selected,
                    policy_sha256=manifest["files"]["policy.pt"]["sha256"],
                    receipt_sha256=case["receipt_sha256"],
                    training_identity_sha256=case["training_identity_sha256"],
                    variant=case["variant"],
                    seed=case["seed"],
                )
            )
        document = dict(
            schema_version=1,
            identity_sha256=self.identity_hash,
            selections=selections,
            auxiliary_gate=self.state["gate"],
            audit_sources=[
                row for row in self.inputs.identity["source_manifests"] if row["split"] == "audit"
            ],
            selection_uses="validation_only",
            final_test_opened=False,
        )
        path = self.output / "freeze.json"
        if path.exists():
            previous = _json(path)[0]
            _require(
                {key: value for key, value in previous.items() if key != "frozen_at"} == document,
                "La selección congelada cambió antes de la auditoría",
            )
        else:
            document["frozen_at"] = datetime.now(UTC).isoformat()
            atomic_json(path, document)
        self.state["freeze_sha256"] = _file_hash(path)
        self.publish()

        for row in self.inputs.partitions["audit"]:
            self.inputs.check_source(row)
        self.state["audit_opened"] = True
        for selected in selections:
            identifier = f"audit-{selected['variant']}-{selected['seed']}"
            if not any(row["id"] == identifier for row in self.state["cases"]):
                self.state["cases"].append(
                    dict(
                        id=identifier,
                        stage="audit",
                        variant=selected["variant"],
                        seed=selected["seed"],
                        transitions=0,
                        config=selected["config"],
                        config_sha256=selected["config_sha256"],
                        training_output=selected["output"],
                        output=f"audit/{selected['variant']}-{selected['seed']}",
                        watch=f"watch/{identifier}.json",
                        status="pending",
                        attempts=0,
                        pauses=0,
                        active_seconds=0.0,
                        budget_seconds=0.0,
                        watch_identity=None,
                        accounted_starts=0,
                        watch_budget_seconds=0.0,
                        receipt_sha256=None,
                        best=None,
                        reason=None,
                    )
                )
        self.publish()

    def verify_freeze(self):
        frozen, digest = _json(self.output / "freeze.json")
        _require(digest == self.state["freeze_sha256"], "La selección congelada ha cambiado")
        for selected in frozen["selections"]:
            directory = self.output / selected["output"]
            index = _json(directory / "ppo-index.json")[0]["payload"]
            _require(
                index["best"] == selected["bundle"]
                and _file_hash(self.output / selected["config"]) == selected["config_sha256"]
                and _file_hash(directory / "run.json") == selected["receipt_sha256"]
                and _file_hash(directory / selected["bundle"]["bundle"] / "policy.pt")
                == selected["policy_sha256"],
                "Un checkpoint, recibo o configuración cambió después de congelar la selección",
            )

    def summary(self):
        rows = []
        for case in self.state["cases"]:
            if case["status"] != "completed" or case["stage"] == "pilot":
                continue
            groups = (
                case.get("audit_summary")
                if case["stage"] == "audit"
                else self.group_metrics(
                    case["best"]["validation_metrics"],
                    "validation",
                    self.inputs.base["environment"]["cost_bps"],
                )
            )
            rows.extend(
                dict(
                    variant=case["variant"],
                    seed=case["seed"],
                    split="audit" if case["stage"] == "audit" else "validation",
                    **group,
                )
                for group in groups
            )
        paired_work = []
        for seed in SEEDS:
            pair = {
                case["variant"]: case
                for case in self.state["cases"]
                if case["stage"] == "auxiliary"
                and case["seed"] == seed
                and case["status"] == "completed"
            }
            if len(pair) == len(AUXILIARY):
                first, second = (pair[variant] for variant in AUXILIARY)
                paired_work.append(
                    dict(
                        seed=seed,
                        **{
                            f"same_{field}": (
                                first.get("confirmed_transitions", 0)
                                == second.get("confirmed_transitions", 0)
                                if field == "transitions"
                                and self.inputs.settings["schema_version"] == 2
                                else first[field] == second[field]
                            )
                            for field in (
                                "transitions",
                                "optimizer_steps",
                                "auxiliary_steps",
                                "auxiliary_samples",
                            )
                        },
                    )
                )
        atomic_json(
            self.output / "summary.json",
            dict(
                schema_version=1,
                analysis_domain="technical",
                runs=self.run_rows(),
                auxiliary_work=paired_work,
                work_counters_scope="last_confirmed_native_receipt",
                metrics=rows,
                period_scope="episode_complete",
                audit_used_for_selection=False,
                final_test_opened=False,
            ),
        )

    def run(self, pilot_only):
        for case in self.state["cases"]:
            self.watch(case, missing=True)
        for stage in ("pilot", "main", "auxiliary", "audit"):
            self.state["phase"] = stage
            if stage == "main":
                self.state["choice"] = select_grid(
                    [
                        {
                            key: row[key]
                            for key in (
                                "variant",
                                "seed",
                                "status",
                                "transitions",
                                "budget_seconds",
                            )
                        }
                        for row in self.state["cases"]
                        if row["stage"] == "pilot"
                    ],
                    self.inputs.settings,
                )
                self.add_cases("main", VARIANTS, self.state["choice"]["transitions"])
                if pilot_only:
                    self.state.update(status="paused", phase="pilot_complete", reason="pilot_only")
                    self.summary()
                    return self.publish()
            elif stage == "auxiliary":
                self.state["gate"] = auxiliary_gate(
                    [row for row in self.state["cases"] if row["stage"] == "main"],
                    self.inputs.settings,
                )
                if self.state["gate"]["enabled"]:
                    self.add_cases("auxiliary", AUXILIARY, self.state["choice"]["transitions"])
            elif stage == "audit":
                self.freeze()
            self.publish()
            for case in [row for row in self.state["cases"] if row["stage"] == stage]:
                if not self.run_case(case):
                    self.summary()
                    return self.publish()
            self.summary()
        self.state.update(status="completed", phase="completed", reason=None, active_case=None)
        return self.publish()


def run_adaptive_campaign(
    scenarios,
    binary,
    output,
    *,
    config=DEFAULT_CONFIG,
    build_identity=None,
    resume=False,
    pilot_only=False,
    executor=None,
    stop=None,
    sleep=time.sleep,
):
    """Ejecutar una cola secuencial sin adquirir el bloqueo GPU del lanzador."""
    require_learning_allowed("la campaña de adaptación con PPO nativo")
    paths = [Path(path).absolute() for path in (scenarios, binary, output, config)]
    for path in paths:
        safe_destination(path)
    scenarios, binary, output, config = (path.resolve() for path in paths)
    _require(
        not output.is_relative_to(scenarios.parent)
        and not scenarios.parent.is_relative_to(output)
        and not binary.is_relative_to(output)
        and not config.is_relative_to(output),
        "La salida de campaña debe permanecer separada de escenarios, binario y configuración",
    )
    inputs = FrozenInputs(
        scenarios, binary, config, Path(build_identity) if build_identity else None
    )
    _require(
        (resume and output.is_dir()) or (not resume and not output.exists()),
        "Usa una salida nueva o recuperación explícita de campaña",
    )
    output.mkdir(parents=True, exist_ok=resume)
    with _campaign_lock(output):
        campaign = Campaign(
            inputs, output, resume, executor or _execute, stop or (lambda: False), sleep
        )
        try:
            return campaign.run(pilot_only)
        except Exception as error:
            campaign.state.update(status="blocked", reason=type(error).__name__)
            campaign.publish()
            raise
