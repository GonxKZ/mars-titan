"""La señal de parada no puede adquirir un bloqueo interrumpido en el mismo hilo."""

import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_stop_signal_does_not_deadlock_an_interrupted_wait(tmp_path, signum):
    root = tmp_path / "root"
    root.mkdir()
    configuration = tmp_path / "campaigns.json"
    configuration.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": [
                    {"id": "fixture", "path": "runs", "kind": "archive", "domain": "technical"}
                ],
            }
        )
    )
    program = """
import os
import signal
import sys
import threading
import time
from scripts.collect_observatory import main

signum = int(sys.argv.pop())
def interrupted_wait(event, timeout=None):
    with event._cond:
        os.kill(os.getpid(), signum)
    return False

probe = threading.Event()
threading.Event.wait = interrupted_wait
time.sleep = lambda duration: interrupted_wait(probe, duration)
raise SystemExit(main())
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            "--root",
            str(root),
            "--config",
            str(configuration),
            "--state-dir",
            str(tmp_path / "state"),
            "--output",
            str(tmp_path / "public"),
            "--watch",
            str(int(signum)),
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        timeout=3,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "public/observatory.json").is_file()
