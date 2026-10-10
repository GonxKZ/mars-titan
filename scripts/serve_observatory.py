"""Servir el observatorio en local y empujar sus cambios en directo.

El servidor solo lee: la salida del recolector, los resúmenes de las campañas por
ventanas, las trazas de aprendizaje y la telemetría del equipo. Se detiene con Ctrl+C.
"""

import argparse
import json
import signal
import threading
from pathlib import Path

from mars_titan.observatory.collector import WINDOW_SOURCE, identifier, safe_path
from mars_titan.observatory.live_server import Limits, LiveObservatory

ROOT = Path(__file__).resolve().parents[1]


def campaign(value):
    label, separator, path = value.partition("=")
    if not separator:
        path, label = value, Path(value).name
    return label, Path(path)


def declared_campaigns(config, root):
    """Campañas por ventanas que declara la configuración del recolector."""
    sources = json.loads(Path(config).read_text())["sources"]
    return {
        identifier(source["id"]): safe_path(Path(root).absolute(), source["path"])
        for source in sources
        if source["kind"] == WINDOW_SOURCE
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=ROOT / "site")
    parser.add_argument(
        "--public-dir", type=Path, help="Salida del recolector con observatory.json"
    )
    parser.add_argument(
        "--campaign",
        type=campaign,
        action="append",
        default=[],
        help="Carpeta de una campaña por ventanas, con la forma ETIQUETA=RUTA o RUTA",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Configuración del recolector. Se siguen sus campañas por ventanas",
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="Raíz de las rutas de --config")
    parser.add_argument(
        "--traces-dir", type=Path, help="Carpeta con index.json y paquetes de trazas"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--allow-remote", action="store_true", help="Permitir una interfaz no local"
    )
    parser.add_argument("--max-clients", type=int, default=Limits.max_clients)
    parser.add_argument("--events-per-second", type=float, default=Limits.events_per_second)
    parser.add_argument("--telemetry-seconds", type=float, default=Limits.telemetry_seconds)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("Puerto fuera de rango")
    limits = Limits(
        max_clients=args.max_clients,
        max_connections=max(Limits.max_connections, args.max_clients + 8),
        events_per_second=args.events_per_second,
        telemetry_seconds=args.telemetry_seconds,
        idle_telemetry_seconds=max(Limits.idle_telemetry_seconds, args.telemetry_seconds),
    )
    try:
        campaigns = declared_campaigns(args.config, args.root) if args.config else {}
        for label, path in args.campaign:
            if label in campaigns:
                parser.error(f"La campaña {label} ya figura en la configuración")
            campaigns[label] = path
        server = LiveObservatory(
            (args.host, args.port),
            site_dir=args.site,
            public_dir=args.public_dir,
            campaigns=campaigns,
            traces_dir=args.traces_dir,
            limits=limits,
            allow_remote=args.allow_remote,
        )
    except ValueError as error:
        parser.error(str(error))

    def stop(*_args):
        # shutdown() espera al bucle del servidor, así que se llama desde otro hilo.
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    host, port = server.server_address[:2]
    print(f"Observatorio en directo: http://{host}:{port}/", flush=True)
    try:
        server.serve()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
