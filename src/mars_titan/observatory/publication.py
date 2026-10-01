"""Publicación acotada desde un checkout dedicado a datos públicos."""

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from mars_titan.data.storage import atomic_json

from .collector import digest, safe_path


@dataclass
class PublicationSchedule:
    interval: float = 300
    last_digest: str | None = None
    last_terminal: str | None = None
    last_success: float = -300
    retry_at: float = 0
    failures: int = 0

    def due(self, content, terminal, now):
        return (
            content != self.last_digest
            and now >= self.retry_at
            and (terminal != self.last_terminal or now - self.last_success >= self.interval)
        )

    def succeeded(self, content, terminal, now):
        self.last_digest, self.last_terminal = content, terminal
        self.last_success, self.failures, self.retry_at = now, 0, now

    def failed(self, now):
        self.failures += 1
        self.retry_at = now + min(300, 15 * 2 ** min(self.failures - 1, 5))


def publication_identity(snapshot):
    body = {k: v for k, v in snapshot.items() if k != "generated_at"}
    terminal = [
        (r["run_id"], r["attempt_id"], r["status"])
        for r in snapshot["runs"]
        if r["status"] in {"completed", "failed", "cancelled"}
    ]
    return digest(body), digest(terminal)


class GitPublisher:
    """Enviar solo el índice y sus páginas, sin tocar el checkout científico."""

    def __init__(self, checkout, *, repository="GonxKZ/mars-titan", dispatch=True):
        self.checkout = Path(checkout)
        self.repository, self.dispatch = repository, dispatch
        if self.git("branch", "--show-current").strip() != "observatory-data":
            raise ValueError("La publicación requiere la rama exclusiva observatory-data")
        tracked = self.git("ls-files").splitlines()
        if any(not self.allowed(path) for path in tracked):
            raise ValueError("La rama de datos contiene archivos ajenos al contrato público")

    @staticmethod
    def allowed(path):
        return path == "observatory.json" or bool(re.fullmatch(r"pages/[a-f0-9]{64}\.json", path))

    def git(self, *args):
        result = subprocess.run(
            ["git", *args],
            cwd=self.checkout,
            text=True,
            capture_output=True,
            timeout=45,
            check=True,
        )
        return result.stdout

    def publish(self, output, index=None):
        output = Path(output)
        index = index or json.loads((output / "observatory.json").read_text())
        if index.get("schema_version") != 2 or index.get("project") != "MARS-TITAN":
            raise ValueError("La publicación no tiene el contrato del observatorio")
        names = ["observatory.json", *index["pagination"]["pages"]]
        if len(names) > 4097:
            raise ValueError("La publicación supera el máximo de páginas")
        documents = []
        for name in names:
            if not self.allowed(name):
                raise ValueError("Ruta pública no admitida")
            path = safe_path(output, name)
            if path.stat().st_size > 8 * 1024**2:
                raise ValueError("Página pública demasiado grande")
            document = index if name == "observatory.json" else json.loads(path.read_text())
            if name.startswith("pages/") and name != f"pages/{digest(document)}.json":
                raise ValueError("La huella de la página no coincide")
            documents.append((name, document))
        for name, document in documents:
            atomic_json(safe_path(self.checkout, name), document)
        self.git("add", "--", *names)
        if self.git("diff", "--cached", "--name-only").strip():
            self.git("commit", "-m", "chore(observatory): update public campaign records")
        revision = self.git("rev-parse", "HEAD").strip()
        self.git("push", "origin", "HEAD:refs/heads/observatory-data")
        if self.dispatch:
            subprocess.run(
                [
                    "gh",
                    "workflow",
                    "run",
                    "pages.yml",
                    "--repo",
                    self.repository,
                    "--ref",
                    "main",
                    "-f",
                    f"data_sha={revision}",
                ],
                capture_output=True,
                text=True,
                timeout=45,
                check=True,
            )
        return revision


class PublicationWorker:
    """Mantener como máximo un envío pendiente fuera del ciclo de recolección."""

    def __init__(self, publisher):
        self.publisher = publisher
        self.schedule = PublicationSchedule()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="observatory-publish")
        self.future = None
        self.pending = None

    def tick(self, snapshot, index, output, now):
        result = None
        if self.future is not None and self.future.done():
            try:
                revision = self.future.result()
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                self.schedule.failed(now)
                result = ("failed", type(error).__name__)
            else:
                self.schedule.succeeded(*self.pending, now)
                result = ("published", revision)
            self.future = None
        content, terminal = publication_identity(snapshot)
        if self.future is None and self.schedule.due(content, terminal, now):
            self.pending = (content, terminal)
            self.future = self.executor.submit(self.publisher.publish, output, index)
        return result

    def close(self):
        self.executor.shutdown(wait=True)
        if self.future is not None:
            return self.future.result()
        return None
