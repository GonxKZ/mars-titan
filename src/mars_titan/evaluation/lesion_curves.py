"""Curvas B4 de una ventana y semilla, con sesiones completas y remuestreo temporal.

Las columnas lesion_* contienen pronósticos completos emitidos antes de madurar sus
resultados. zero conserva el núcleo sin corrector y residual_zero es el predictor cero.
Este análisis no selecciona máscaras, no reentrena y no acredita generalización entre
semillas ni verifica por sí solo la disponibilidad temporal de las entradas originales.
"""

import platform
import tempfile
import time
from dataclasses import fields
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.memory.area_memory import AreaLayout
from mars_titan.memory.associative_memory import AssociativeMemoryConfig, MatureCorrection
from mars_titan.memory.lesion_study import load_study
from mars_titan.training.learning_hold import require_learning_allowed

from .forecast_files import check_prediction_schema, score_prediction_file
from .paired_comparisons import compare_series, delta

MAX_SESSIONS = 50_000
_BASE = ("sample_id", "market", "prediction_at", "target", "prediction")
POINT_ONLY = "El diagnóstico puntúa solo puntos y no altera ni evalúa los cuantiles originales"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _geometry(configuration, protocol):
    """Regenerar las máscaras sin cargar pesos, puente, etiquetas ni datos del padre."""
    _require(isinstance(configuration, dict), "Falta la geometría B4")
    try:
        correction = configuration["correction"]
        layout = AreaLayout(**{f.name: correction["areas"][f.name] for f in fields(AreaLayout)})
        # Solo el número de áreas y 2dν intervienen en las máscaras. El corrector
        # auxiliar no se lee ni se escribe y no sustituye la identidad guardada.
        reference = MatureCorrection(
            AssociativeMemoryConfig("proximal", key_size=correction["memory"]["key_size"]),
            areas=layout,
        )
        expected = protocol.bind(reference).identity()
    except (KeyError, TypeError) as error:
        raise ValueError("La geometría B4 está incompleta") from error
    _require(set(configuration) == set(expected), "La geometría B4 tiene campos desconocidos")
    _require(
        all(configuration[k] == expected[k] for k in expected if k != "correction"),
        "Las máscaras B4 no corresponden al protocolo y dimensiones declarados",
    )
    return tuple(case["name"] for case in expected["cases"])


def _source(report_path, study, partition):
    document, digest = read_manifest(report_path, maximum=16 * 1024**2)
    _require(
        isinstance(document, dict)
        and document.get("kind")
        in (
            "mars_titan_correction_window",
            "mars_titan_correction_carried_predictions",
        )
        and document.get("status") == "completed"
        and document.get("final_test_opened") is False,
        "El análisis B4 necesita una ventana completada de desarrollo",
    )
    protocol = document.get("request", {}).get("lesions", document.get("lesions"))
    _require(protocol == study.protocol.identity(), "El protocolo B4 ha cambiado")
    info = document.get("predictions", {}).get(partition)
    _require(isinstance(info, dict), "El tramo no tiene predicciones B4")
    configuration = info.get("metrics", {}).get("lesions", {}).get("configuration")
    names = _geometry(configuration, study.protocol)
    filename = info.get("path")
    _require(
        isinstance(filename, str)
        and Path(filename).name == filename
        and filename.endswith(".parquet"),
        "La ruta de predicciones debe ser un archivo local",
    )
    path = report_path.parent / filename
    safe_destination(path)
    _require(path.is_file(), "Falta el archivo de predicciones")
    _require(
        path.stat().st_size == info.get("bytes") and sha256(path) == info.get("sha256"),
        "La huella o tamaño de las predicciones B4 ha cambiado",
    )
    return (
        path,
        info,
        names,
        dict(
            report_sha256=digest,
            seed=document.get("request", document.get("anchor", {})).get("seed"),
            configuration=configuration,
            prediction_sha256=info["sha256"],
            partition=partition,
        ),
    )


def _schema(path, info, names):
    columns = (*_BASE, *(f"lesion_{name}" for name in names))
    schema = check_prediction_schema(path, info.get("rows"), columns)
    _require(
        {n for n in schema.names if n.startswith("lesion_")} == set(columns[5:]),
        "Las columnas B4 no corresponden a sus condiciones",
    )
    return columns


def _scores(path, columns, names, analysis, temporary, batch_size):
    points = {name: f"lesion_{name}" for name in names}
    points["residual_zero"] = None
    return score_prediction_file(
        path,
        points,
        rows=pq.ParquetFile(path).metadata.num_rows,
        minimum=analysis["rank_ic_min_assets"],
        temporary=temporary,
        batch_size=batch_size,
        max_sessions=MAX_SESSIONS,
        equal_points=(("prediction", "lesion_intact"),),
    )


def analyze_lesion_run(report, study, *, partition, output, batch_size=4096):
    """Publicar curvas de una ventana completada, solo si el aprendizaje está permitido."""
    require_learning_allowed("el análisis científico de lesiones B4")
    _require(partition in ("validation", "calibration", "evaluation"), "Tramo B4 desconocido")
    _require(type(batch_size) is int and 1 <= batch_size <= 8192, "Lote Arrow B4 inválido")
    report, study_path, output = Path(report), Path(study), Path(output)
    safe_destination(report)
    safe_destination(output)
    outside_source(report.parent, output)
    _require(not output.exists(), "La salida B4 ya existe")
    study, study_sha = load_study(study_path)
    path, info, names, provenance = _source(report, study, partition)
    columns = _schema(path, info, names)
    declaration_path = report.parent / "lesion-study.json"
    provenance["analysis_declaration_sha256"] = None
    if declaration_path.exists():
        declaration, declaration_sha = read_manifest(declaration_path, maximum=64 * 1024)
        _require(
            isinstance(declaration, dict)
            and declaration.get("study_sha256") == study_sha
            and declaration.get("analysis") == study.analysis
            and declaration.get("protocol") == study.protocol.identity()
            and declaration.get("seed") == provenance["seed"],
            "La declaración B4 anterior a la ventana ha cambiado",
        )
        provenance["analysis_declaration_sha256"] = declaration_sha
    started = time.perf_counter()
    # El spill se aloja junto al destino, nunca en el tmpfs de /tmp por defecto.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".b4-", dir=output.parent) as directory:
        scores, resources = _scores(
            path, columns, names, study.analysis, Path(directory), batch_size
        )
        contrasts = {f"{name}_minus_intact": delta("intact", name) for name in names[1:]}
        contrasts.update(
            intact_minus_residual_zero=delta("residual_zero", "intact"),
            core_minus_residual_zero=delta("residual_zero", "zero"),
        )
        options = {k: v for k, v in study.analysis.items() if k != "rank_ic_min_assets"}
        comparisons = {
            metric: compare_series(
                {n: s.series(metric) for n, s in scores.items()}, contrasts, **options
            )
            for metric in ("mae", "rank_ic")
        }
        tables = [
            s.to_table().append_column("condition", pa.array([n] * len(s.mae)))
            for n, s in scores.items()
        ]
        _require(
            sha256(path) == info["sha256"]
            and sha256(report) == provenance["report_sha256"]
            and sha256(study_path) == study_sha,
            "Las fuentes B4 han cambiado durante el análisis",
        )
        if provenance["analysis_declaration_sha256"] is not None:
            _require(
                sha256(declaration_path) == provenance["analysis_declaration_sha256"],
                "La declaración B4 ha cambiado durante el análisis",
            )
        # Reservar un destino nuevo evita sobrescribir resultados anteriores.
        output.mkdir()
        sessions = output / "sessions.parquet"
        pq.write_table(pa.concat_tables(tables), sessions, compression="zstd")
        result = dict(
            schema_version=1,
            kind="lesion_session_curves_v1",
            status="completed",
            final_test_opened=False,
            source=provenance,
            study_sha256=study_sha,
            analysis=study.analysis,
            rows=info["rows"],
            resources=resources,
            scores={
                n: dict(
                    s.summary(market_weighting=study.analysis["market_weighting"]),
                    quantiles_reason=POINT_ONLY,
                )
                for n, s in scores.items()
            },
            comparisons=comparisons,
            sessions=dict(path=sessions.name, sha256=sha256(sessions), rows=sum(map(len, tables))),
            software=dict(
                python=platform.python_version(),
                numpy=np.__version__,
                pyarrow=pa.__version__,
                duckdb=duckdb.__version__,
            ),
            code={
                name: sha256(Path(__file__).with_name(name + ".py"))
                for name in (
                    "lesion_curves",
                    "forecast_files",
                    "forecast_panel",
                    "forecast_scores",
                    "paired_comparisons",
                )
            },
            seconds=time.perf_counter() - started,
            limits=[
                "Una ventana y semilla no acreditan generalización entre semillas.",
                "Las máscaras evalúan lectura sobre una trayectoria sin lesiones de escritura.",
                "zero es el núcleo sin corrector. residual_zero es el predictor cero.",
                "Los intervalos simultáneos pertenecen a cada familia métrica por separado.",
                "No se eligen máscaras ni arquitecturas con estas curvas.",
                "No se valida aquí la disponibilidad temporal de las entradas originales.",
            ],
        )
        atomic_json(output / "curves.json", result)
    return result
