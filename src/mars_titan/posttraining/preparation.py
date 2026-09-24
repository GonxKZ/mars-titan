"""Preparar aumentos compartidos y comprobar la procedencia de los codificadores."""

import fcntl
import os
from dataclasses import asdict, replace
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.episodes.augmentation import augmentation_windows, paired_world
from mars_titan.episodes.encoding import EncodedWorld
from mars_titan.episodes.storage import EpisodeSource, write_world
from mars_titan.episodes.windows import EpisodeView, EpisodeWindow
from mars_titan.episodes.worlds import DAY
from mars_titan.training.checkpoints import StopRequest
from mars_titan.training.run_receipts import initialize_receipt

from .inputs import fingerprint


def encoder_contract(supervision, encoded):
    """Seguir la huella de codificación declarada por la supervisión, sin abrir datos."""
    source, source_hash = read_manifest(Path(supervision), 8 * 1024**2)
    encoding, encoding_hash = read_manifest(Path(encoded), 8 * 1024**2)
    spec = encoding.get("configuration", {}).get("encoders")
    if (
        source.get("configuration", {}).get("source_manifest_sha256") != encoding_hash
        or source.get("final_test_opened") is not False
        or encoding.get("final_test_opened") is not False
        or not isinstance(spec, dict)
        or not spec
        or source.get("context_sessions") != encoding.get("context_sessions")
    ):
        raise ValueError("La codificación no está ligada a la supervisión o ha abierto el test")
    return dict(
        supervision_sha256=source_hash,
        encoded_sha256=encoding_hash,
        encoders=spec,
        context=source["context_sessions"],
    )


def _validate_receipt(report, identity, windows, source, output):
    rows = sum(source.index[i][1] for w in windows for i in range(w.decision_start, w.stop))
    records = report.get("episodes")
    if (
        report.get("identity") != identity
        or report.get("final_test_opened") is not False
        or type(report.get("extra_rows")) is not int
        or report["extra_rows"] != rows
        or report.get("status") not in {"running", "paused", "completed"}
        or not isinstance(records, list)
        or len(records) > len(windows)
        or (report["status"] == "completed" and len(records) != len(windows))
    ):
        raise ValueError("El recibo de aumento no conserva la identidad y el presupuesto")
    for i, record in enumerate(records):
        window = windows[i]
        if record.get("path") != f"episode-{i:05d}/manifest.json" or record.get(
            "source_window"
        ) != asdict(window):
            raise ValueError("El recibo de aumento no conserva el prefijo de ventanas")
        metadata, digest = read_manifest(output / record["path"], 8 * 1024**2)
        expected_counts = [source.index[j][1] for j in range(window.start, window.stop)]
        if (
            digest != record.get("sha256")
            or metadata.get("status") != "completed"
            or metadata.get("partition") != "train"
            or metadata.get("final_test_opened") is not False
            or [row[1] for row in metadata.get("index", [])] != expected_counts
            or metadata.get("identity", {}).get("world", {}).get("encoder_spec")
            != identity["encoders"]
        ):
            raise ValueError("El recibo apunta a un episodio incompatible o alterado")
        counterpart = EpisodeWindow(
            digest,
            0,
            window.decision_start - window.start,
            window.stop - window.start,
            metadata["index"][-1][0] + DAY,
            "train",
            "synthetic",
        )
        if record.get("synthetic_window") != asdict(counterpart):
            raise ValueError("Las ventanas sintéticas del recibo no conservan el emparejamiento")


def prepare_augmentation(
    source,
    output,
    encoders,
    *,
    expected_spec,
    seed,
    decisions,
    warmup,
    volatility,
    stop=None,
    check_resources=None,
):
    """Confirmar episodios sintéticos una vez y compartirlos entre padres y objetivos."""
    if encoders.spec != expected_spec:
        raise ValueError("El codificador no coincide con el usado por los datos reales")
    windows = augmentation_windows(source, seed=seed, decisions=decisions, warmup=warmup)
    identity = dict(
        train_sha256=source.manifest_sha256,
        windows=[asdict(w) for w in windows],
        seed=seed,
        volatility=volatility,
        encoders=expected_spec,
        fraction=0.25,
        code_sha256=sha256(Path(__file__)),
    )
    output, stop = Path(output), stop or StopRequest()
    safe_destination(output)
    output.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        output / ".augmentation.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        confirmed = initialize_receipt(
            output, identity, record="augmentation.json", lock=".augmentation.lock"
        )
        report = (
            read_manifest(output / "augmentation.json", 8 * 1024**2)[0]
            if confirmed
            else dict(
                identity=identity,
                status="running",
                episodes=[],
                extra_rows=sum(
                    source.index[i][1] for w in windows for i in range(w.decision_start, w.stop)
                ),
                final_test_opened=False,
            )
        )
        _validate_receipt(report, identity, windows, source, output)
        atomic_json(output / "augmentation.json", report)
        for i, window in enumerate(windows):
            if stop.requested:
                raise InterruptedError("Preparación de episodios interrumpida")
            if check_resources is not None:
                check_resources()
            if i < len(report["episodes"]):
                continue
            world_seed = int(fingerprint([seed, i])[:8], 16)
            world, counterpart = paired_world(
                source, window, seed=world_seed, volatility=volatility
            )
            encoded = EncodedWorld(world, encoders, expected_spec=expected_spec)
            if encoded.shapes != source.shapes:
                raise ValueError("Los datos reales y sintéticos no comparten las dimensiones")
            destination = output / f"episode-{i:05d}"
            result = write_world(
                encoded,
                destination,
                resume=destination.exists(),
                stop=lambda: stop.requested,
                check_resources=check_resources,
            )
            if result["status"] != "completed":
                raise InterruptedError("Hay un episodio pendiente de confirmar")
            path = destination / "manifest.json"
            digest = sha256(path)
            counterpart = replace(counterpart, source_sha256=digest)
            report["episodes"].append(
                dict(
                    path=str(path.relative_to(output)),
                    sha256=digest,
                    source_window=asdict(window),
                    synthetic_window=asdict(counterpart),
                )
            )
            atomic_json(output / "augmentation.json", report)
        report["status"] = "completed"
        atomic_json(output / "augmentation.json", report)
        return report
    except InterruptedError:
        report["status"] = "paused"
        atomic_json(output / "augmentation.json", report)
        raise
    finally:
        os.close(descriptor)


class EpisodeFactory:
    """Abrir como máximo una fuente sintética y verificar su huella al cambiar de episodio."""

    def __init__(self, root, report):
        if report.get("status") != "completed" or report.get("final_test_opened") is not False:
            raise ValueError("El aumento necesita episodios completos y test cerrado")
        self.root, self.records = Path(root), report["episodes"]
        self.current, self.view = None, None
        self.windows = [EpisodeWindow(**r["source_window"]) for r in self.records]
        self.identity = {str(i): record["sha256"] for i, record in enumerate(self.records)}

    def __call__(self, index):
        if type(index) is not int or not 0 <= index < len(self.records):
            raise ValueError("El episodio no pertenece al aumento confirmado")
        if self.current != index:
            self.close()
            record = self.records[index]
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("El episodio debe permanecer en su carpeta")
            path = self.root / relative
            safe_destination(path)
            if sha256(path) != record["sha256"]:
                raise ValueError("El manifiesto del episodio ha cambiado")
            source = EpisodeSource(path)
            try:
                self.view = EpisodeView(source, EpisodeWindow(**record["synthetic_window"]))
            except BaseException:
                source.close()
                raise
            self.current = index
        return self.view

    def close(self):
        if self.view is not None:
            self.view.source.close()
        self.current, self.view = None, None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
