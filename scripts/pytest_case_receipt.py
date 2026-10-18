"""Recibo de colección y fases para el lanzador local de pruebas aisladas."""

import json
import os
from pathlib import Path

_receipt = {}


def _write():
    path = Path(os.environ["MARS_TITAN_PYTEST_RECEIPT"])
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(_receipt, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def pytest_sessionstart(session):
    _receipt.clear()
    _receipt.update(schema_version=1, pid=os.getpid(), selected=[], phases=[], exit_code=None)
    _write()


def pytest_collection_finish(session):
    _receipt["selected"] = [item.nodeid for item in session.items]
    _write()


def pytest_runtest_logreport(report):
    _receipt["phases"].append(
        dict(
            nodeid=report.nodeid,
            when=report.when,
            outcome=report.outcome,
            seconds=report.duration,
            detail=str(report.longrepr)[:8192] if report.longrepr is not None else None,
        )
    )
    _write()


def pytest_sessionfinish(session, exitstatus):
    _receipt["exit_code"] = int(exitstatus)
    _write()
