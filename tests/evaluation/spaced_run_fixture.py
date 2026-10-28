"""Informes E2 manuales para comprobar un comparador, sin ajustar ni ejecutar modelos."""

import copy
import hashlib
import json
from dataclasses import asdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.memory.spaced_study import load_study
from mars_titan.models.quantile_head import LEVELS, QUANTILE_COLUMNS
from mars_titan.training.spaced_window import _code, _digest, _session_table
from tests.memory.spaced_analysis_fixture import analysis_document
from tests.memory.test_spaced_analysis_study import profile

DAY = 86_400_000_000


def _stamp(text):
    return hashlib.sha256(text.encode()).hexdigest()


def update_prediction(folder, report, partition, table):
    """Escribir las tablas reales del fixture y conciliar sus resúmenes declarados."""
    count = len(table)
    sessions = {}
    for row in table.to_pylist():
        key = row["market"], row["prediction_at"]
        sessions.setdefault(key, []).append(row["prediction"] - row["target"])
    metrics = dict(
        labels=count,
        predictions=count,
        observations=count,
        session_count=len(sessions),
        session_mae=float(np.mean([np.mean(np.abs(v)) for v in sessions.values()])),
        session_mse=float(np.mean([np.mean(np.square(v)) for v in sessions.values()])),
    )
    path = folder / f"{partition}-predictions.parquet"
    pq.write_table(table, path, row_group_size=7, compression="zstd")
    session_path = folder / f"{partition}-session-metrics.parquet"
    pq.write_table(_session_table([table], metrics), session_path, compression="zstd")
    evidence = None
    if report["request"]["arm"] != "baseline":
        evidence_path = folder / f"{partition}-evidence.json"
        atomic_json(evidence_path, dict(kind="manual_evidence_fixture", no_model_execution=True))
        evidence = dict(path=evidence_path.name, sha256=sha256(evidence_path))
    report["predictions"][partition] = dict(
        path=path.name,
        sha256=sha256(path),
        rows=count,
        bytes=path.stat().st_size,
        metrics=metrics,
        sessions=dict(path=session_path.name, sha256=sha256(session_path), rows=len(sessions)),
        evidence=evidence,
    )
    return report


def manual_runs(
    root, *, contract="standard", quantiles=False, days=21, sparse_cn=False, weighting="session"
):
    root.mkdir()
    config = analysis_document(json.loads(profile(contract).read_text()), contract)
    config["analysis"].update(
        replicates=30, block_length=2, sensitivity_block_lengths=[1, 3], market_weighting=weighting
    )
    study_path = root / "study.json"
    atomic_json(study_path, config)
    study, study_sha = load_study(study_path)
    kind = (
        "spaced_regional_utility_frozen_window_v1"
        if contract == "regional"
        else "spaced_utility_frozen_window_v1"
    )
    phases, tables = {}, {}
    for index, partition in enumerate(("validation", "calibration", "evaluation")):
        start = 1_600_000_000_000_000 + index * 60 * DAY
        phases[partition] = dict(
            partition=partition,
            warmup_start=start,
            decision_start=start,
            decision_end=start + days * DAY,
            close_at=start + days * DAY,
        )
        rows = []
        for day in range(days):
            for market, assets in (("US", 4), ("CN", 1 if day % 2 else 3)):
                if sparse_cn and market == "CN" and day % 2:
                    continue
                for asset in range(assets):
                    at = start + day * DAY
                    flow = f"{market}/FIXTURE_{asset}"
                    rows.append(
                        dict(
                            sample_id=f"{flow}/{at}",
                            asset_id=flow,
                            market=market,
                            prediction_at=at,
                            target=asset * 0.02 - 0.01 + day * 0.0001,
                        )
                    )
        table = pa.Table.from_pylist(rows)
        position = table.column_names.index("prediction_at")
        table = table.set_column(
            position, "prediction_at", table["prediction_at"].cast(pa.timestamp("us", "UTC"))
        )
        tables[partition] = table.take(
            np.random.Generator(np.random.PCG64(73)).permutation(len(table))
        )
    selected = dict(
        schema_version=1,
        kind="spaced_selected_regional_reader_v1"
        if contract == "regional"
        else "spaced_selected_reader_v1",
        state_transfer="shared_parameters_only",
        anchor=dict(
            run_id=_stamp("manual_anchor"),
            run_sha256=_stamp("anchor_report"),
            selected_sha256=_stamp("manual_selection"),
        ),
        view_sha256=_stamp("manual_view"),
        seed=42,
        variant_sha256=_stamp("manual_variant"),
        parent=dict(
            run_id=_stamp("manual_parent"),
            run_sha256=_stamp("parent_report"),
            parameters_sha256=_stamp("manual_parent_parameters"),
        ),
        readout_sha256=_stamp("manual_reader_parameters"),
        consumer_id=_stamp("manual_consumer"),
    )
    if contract == "regional":
        selected["reader_contract"] = contract
    reports = {}
    offsets = dict(
        baseline=(0.04, 0.01),
        reservoir=(0.04, 0.01),
        utility=(0.02, 0.005),
        permuted_utility=(0.04, 0.01),
        uniform_frontier=(0.03, 0.01),
        recent=(0.035, 0.012),
    )
    code = _code(regional=contract == "regional")
    for arm in study.arms:
        folder = root / arm
        folder.mkdir()
        request = dict(
            schema_version=1,
            kind=kind,
            arm=arm,
            device="cpu",
            view_sha256=selected["view_sha256"],
            anchor_run_sha256=selected["anchor"]["run_sha256"],
            anchor_selected_sha256=selected["anchor"]["selected_sha256"],
            source_manifest_sha256=_stamp("manual_source"),
            study_sha256=study_sha,
            study=study.identity(),
            protocol_sha256=_stamp("manual_protocol"),
            code=code,
            native_sha256=_stamp("fixture_without_native_execution"),
        )
        retention = study.retention_for(arm)
        if retention is None:
            retention = RetentionConfig(
                capacity=study.retention.capacity, seed=study.retention.seed, frontier=8
            )
        identity = dict(
            schema_version=1,
            kind=kind,
            request=request,
            selected=copy.deepcopy(selected),
            phases=copy.deepcopy(phases),
            indices={name: _stamp("manual_index_" + name) for name in phases},
            expected_rows={name: len(table) for name, table in tables.items()},
            retention=asdict(retention),
            evidence_enabled=arm != "baseline",
            state_transfer="selected_shared_parameters_only",
            state_reset="each_phase_empty_fast_memory_bank_and_evidence",
            interrupted_phase="repeat_from_start",
        )
        if contract == "regional":
            identity["reader_contract"] = contract
        record = dict(
            schema_version=1,
            kind=kind,
            status="completed",
            request=request,
            identity=identity,
            run_id=_digest(identity),
            predictions={},
            attempts=[],
            final_test_opened=False,
            technical_fixture="manual_reports_without_model_execution",
        )
        for partition, table in tables.items():
            offset = np.where(table["market"].to_numpy() == "US", *offsets[arm])
            prediction = table["target"].to_numpy() + offset
            current = table.append_column("prediction", pa.array(prediction)).append_column(
                "zero", pa.array(np.zeros(len(table)))
            )
            if quantiles:
                for name, value in zip(
                    QUANTILE_COLUMNS, (-0.1, -0.04, 0.0, 0.04, 0.1), strict=True
                ):
                    current = current.append_column(name, pa.array(prediction + value))
                assert len(LEVELS) == 5
            update_prediction(folder, record, partition, current)
        atomic_json(folder / "run.json", record)
        reports[arm] = folder / "run.json"
    return dict(reports=reports, study=study_path, root=root)


def rewrite(source, arm, *, record=None, table=None, partition="evaluation"):
    path = source["reports"][arm]
    value = copy.deepcopy(json.loads(path.read_text()) if record is None else record)
    if table is not None:
        update_prediction(path.parent, value, partition, table)
    atomic_json(path, value)
