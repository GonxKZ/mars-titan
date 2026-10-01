"""Recoger informes pequeños cada 15 segundos y publicar cambios cada cinco minutos."""

import argparse
import fcntl
import json
import logging
import signal
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

from mars_titan.observatory.collector import Collector, write_pages
from mars_titan.observatory.publication import (
    GitPublisher,
    PublicationWorker,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/observatory/campaigns.json"))
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish-checkout", type=Path)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    for destination in (args.state_dir, args.output):
        if destination.resolve().is_relative_to((args.root / "data").resolve()):
            parser.error("El estado y la salida deben quedar fuera de los datos científicos")
    config = json.loads(args.config.read_text())
    if config.get("schema_version") != 1 or not 1 <= len(config["sources"]) <= 64:
        parser.error("Configuración de campañas inválida")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    stop_requested = False

    def request_stop(*_args):
        nonlocal stop_requested
        # El manejador puede interrumpir un bloqueo del hilo principal.
        stop_requested = True

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    publisher = GitPublisher(args.publish_checkout) if args.publish_checkout else None
    worker = PublicationWorker(publisher) if publisher else None
    try:
        with ExitStack() as cleanup:
            if worker:
                cleanup.callback(worker.close)
            with (args.state_dir / "collector.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with Collector(args.root, args.state_dir / "sources.sqlite") as collector:
                    while not stop_requested:
                        started = time.monotonic()
                        try:
                            snapshot = collector.collect(config["sources"])
                            index = write_pages(snapshot, args.output)
                            if worker:
                                result = worker.tick(snapshot, index, args.output, started)
                                if result:
                                    logging.info("Estado de publicación: %s, %s", *result)
                            logging.info(
                                "Registros: %d. Bytes leídos: %d. Tiempo: %.3f s.",
                                len(snapshot["runs"]),
                                collector.bytes_read,
                                time.monotonic() - started,
                            )
                        except (OSError, ValueError) as error:
                            logging.error(
                                "Se conserva el registro anterior: %s", type(error).__name__
                            )
                            if not args.watch:
                                raise
                        if not args.watch:
                            break
                        if not stop_requested:
                            time.sleep(max(0, 15 - (time.monotonic() - started)))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        logging.error("Recolección o publicación sin confirmar: %s", type(error).__name__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
