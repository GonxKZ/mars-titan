"""Resumen de rangos NVTX de una exportación de Nsight Systems con y sin grafos ni copias."""

import importlib.util
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "nsys_summary", ROOT / "benchmarks/nsys_range_summary.py"
)
nsys_summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nsys_summary)


def export(path, *, graphs, copies):
    """Exportación mínima: un rango con dos llamadas y una llamada fuera del rango."""
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE StringIds (id INTEGER, value TEXT);
        CREATE TABLE NVTX_EVENTS (text TEXT, textId INTEGER, start INTEGER, end INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME
            (start INTEGER, end INTEGER, nameId INTEGER, correlationId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (correlationId INTEGER, start INTEGER, end INTEGER);
        INSERT INTO StringIds VALUES (1, 'cudaLaunchKernel_v7000'), (2, 'cudaMemcpyAsync_v3020');
        INSERT INTO NVTX_EVENTS VALUES ('readout.emit', NULL, 0, 10000000);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME
            VALUES (1000, 2000, 1, 7), (3000, 4000, 2, 8), (20000000, 20001000, 1, 9);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (7, 5000, 2005000), (9, 20002000, 20003000);
        """
    )
    if graphs:
        db.executescript(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
            " (correlationId INTEGER, start INTEGER, end INTEGER);"
            "INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (7, 2005000, 3005000);"
        )
    if copies:
        db.executescript(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY"
            " (correlationId INTEGER, copyKind INTEGER, bytes INTEGER);"
            "CREATE TABLE ENUM_CUDA_MEMCPY_OPER (id INTEGER, label TEXT);"
            "INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (1, 'Host-to-Device');"
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (8, 1, 4096);"
        )
    db.commit()
    db.close()
    return path


@pytest.mark.parametrize("graphs", [True, False], ids=["con-grafos", "sin-grafos"])
@pytest.mark.parametrize("copies", [True, False], ids=["con-copias", "sin-copias"])
def test_missing_activity_tables_count_as_zero(tmp_path, graphs, copies):
    path = export(tmp_path / "perfil.sqlite", graphs=graphs, copies=copies)
    emit = nsys_summary.summary(path, ["readout.emit"])["readout.emit"]
    assert emit["kernels_per_repeat"] == 1 and emit["kernel_ms_per_repeat"] == 2.0
    assert emit["graphs_per_repeat"] == (1 if graphs else 0)
    assert emit["graph_ms_per_repeat"] == (1.0 if graphs else 0)
    assert emit["api_calls_per_repeat"]["cudaLaunchKernel"]["count"] == 1
    assert emit["copies_per_repeat"] == (
        {"Host-to-Device": dict(count=1, bytes=4096)} if copies else {}
    )


def test_other_database_errors_still_stop_the_summary(tmp_path):
    path = export(tmp_path / "perfil.sqlite", graphs=False, copies=False)
    db = sqlite3.connect(path)
    db.executescript(
        "DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL; CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (x);"
    )
    db.close()
    with pytest.raises(sqlite3.OperationalError, match="no such column"):
        nsys_summary.summary(path, ["readout.emit"])
