"""Destilación temporalmente válida de un profesor a Titans-MAC (#57, MT-055).

La destilación desplaza cómputo de la inferencia al ajuste: un profesor más caro, el lector
de MARS-TITAN sobre su padre, deja sus salidas en las filas de ajuste y un alumno más barato,
el núcleo Titans-MAC, aprende a la vez de las etiquetas y de esas salidas (Hinton et al.,
2015). Solo compensa si el profesor predice mejor y su inferencia cuesta claramente más.

Las salidas del profesor en su propio tramo de ajuste son objetivos derivados dentro de la
muestra, no predicciones fuera de muestra. Se generan con una sola pasada cronológica del
profesor ya elegido sobre ese tramo (banco vacío y memoria rápida inicial al empezar, como
cualquier validación), así que su memoria en cada fila solo contiene etiquetas maduras antes
de la decisión. Sus pesos, en cambio, han visto todo el tramo. Por eso el archivo declara
`in_sample` y su único uso: la pérdida del alumno. Nunca sustituyen a un error emitido, ni
alimentan un banco o una puerta, ni se presentan como predicciones históricas.

El alumno minimiza, en cada tramo con etiquetas maduras,

    (1 − w) · pérdida de la tarea + w · media de |salida del alumno − salida del profesor|,

con la pérdida de la tarea de su receta (pinball con la cabeza de cuantiles) y la distancia
L1 sobre cada salida numérica: los cinco cuantiles o el punto. No se trata como destilación
de una densidad. El peso `w` se elige con la validación de la ventana y el alumno conserva
su propia calibración.
"""

import argparse
import json
import math
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.models.quantile_head import LEVELS

KIND = "teacher_targets"
USE = "student_distillation_loss_only"
DISTANCE = "mean_absolute_difference_per_output_v1"
MEMORY = "single_chronological_pass_of_the_train_phase_from_initial_state_v1"
POINT = "point"
DECLARATION_KIND = "temporal_distillation_declaration"
_MAX_ROWS = 50_000_000


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_weight(value):
    """Peso de la distancia al profesor: None sin destilación o una fracción entre 0 y 1."""
    _require(
        value is None or (type(value) is float and math.isfinite(value) and 0 < value < 1),
        "El peso de la destilación es una fracción entre 0 y 1",
    )
    return value


def bound_source(source):
    """Identidad del tramo de ajuste a la que quedan ligadas las salidas del profesor."""
    _require(source.phase.partition == "train", "Las salidas del profesor son del tramo de ajuste")
    manifest = source.dataset.manifest
    return dict(
        source_sha256=source.dataset.identity,
        view_sha256=source.identity,
        representation=manifest["representation"],
        input_policy=manifest["input_policy"],
        phase=asdict(source.phase),
    )


def _outputs(levels):
    return dict(kind="quantiles", levels=list(LEVELS)) if levels else dict(kind=POINT)


def _check_values(values, phase, decisions, width):
    _require(
        values.ndim == 2 and values.shape[1] == width and np.isfinite(values).all(),
        "Las salidas del profesor deben ser finitas y del ancho declarado",
    )
    _require(
        bool(np.all((decisions >= phase["decision_start"]) & (decisions < phase["decision_end"]))),
        "Una salida del profesor queda fuera de las decisiones del tramo de ajuste",
    )
    _require(
        bool(np.all(np.diff(values, axis=1) >= 0)), "Los cuantiles del profesor no están ordenados"
    )


def teacher_identity(arm, report_path, source):
    """Identidad de un profesor ya elegido, ajustado con el mismo tramo que el alumno.

    El informe debe ser el de una ejecución completada, sin la reserva final abierta, con su
    estado elegido y con el mismo corpus, índice y fase de ajuste que `source`.
    """
    report_path = Path(report_path)
    report, digest = read_manifest(report_path, 64 * 1024**2)
    identity = report.get("identity", {})
    _require(
        report.get("status") == "completed"
        and report.get("final_test_opened") is False
        and isinstance(report.get("best_checkpoint"), dict),
        "El profesor es una ejecución completada con su estado elegido",
    )
    _require(
        identity.get("dataset_sha256") == source.dataset.identity
        and identity.get("sources", {}).get("train")
        == dict(index_sha256=source.identity, phase=asdict(source.phase)),
        "El profesor no se ajustó con el tramo de ajuste del alumno",
    )
    return dict(
        arm=arm,
        run_id=report["run_id"],
        report_sha256=digest,
        best_checkpoint_sha256=report["best_checkpoint"]["sha256"],
        best_epoch=report["best_epoch"],
    )


def generate_teacher_targets(folder, inference, source, *, arm, report):
    """Recorrer una vez el tramo de ajuste con el profesor elegido y guardar sus salidas.

    `inference` es un entrenador del profesor con la identidad de la ejecución `report`. Antes
    de recorrer el tramo carga el estado elegido de esa ejecución, comprobado con su huella.
    Su `evaluate` empieza con banco vacío y memoria rápida inicial y emite cada salida antes
    de conocer la etiqueta. La etiqueta no se guarda.
    """
    teacher = teacher_identity(arm, report, source)
    _require(
        getattr(inference, "run_id", None) == teacher["run_id"]
        and Path(report).resolve() == (Path(inference.output) / "run.json").resolve(),
        "Las salidas no las emite la ejecución declarada del profesor",
    )
    inference._load_best(json.loads(Path(report).read_text()))
    rows = []
    inference.evaluate(source, rows=rows)
    quantiles = inference.quantiles
    return _write(Path(folder), rows, source=source, teacher=teacher, quantiles=quantiles)


def _write(folder, rows, *, source, teacher, quantiles):
    _require(not folder.exists(), "Las salidas del profesor se escriben en una carpeta nueva")
    bound = bound_source(source)
    width = len(LEVELS) if quantiles else 1
    keys = [(flow, int(decision)) for flow, decision, *_ in rows]
    _require(rows and len(set(keys)) == len(keys), "Cada fila del profesor aparece una sola vez")
    if quantiles:
        _require(
            all(row[4] is not None and len(row[4]) == width for row in rows),
            "Cada fila del profesor lleva sus cinco cuantiles",
        )
        values = np.asarray([row[4] for row in rows], dtype=np.float64)
    else:
        values = np.asarray([[row[2]] for row in rows], dtype=np.float64)
    decisions = np.asarray([decision for _, decision in keys], dtype=np.int64)
    _check_values(values, bound["phase"], decisions, width)
    order = sorted(range(len(keys)), key=lambda i: (keys[i][1], keys[i][0]))
    table = pa.table(
        dict(
            flow=pa.array([keys[i][0] for i in order], pa.string()),
            decision_at=pa.array(decisions[order], pa.int64()),
            outputs=pa.array(values[order].tolist(), pa.list_(pa.float64(), width)),
        )
    )
    folder.mkdir(parents=True)
    path = folder / "teacher-targets.parquet"
    pq.write_table(table, path)
    manifest = dict(
        schema_version=1,
        kind=KIND,
        use=USE,
        in_sample=True,
        out_of_sample_predictions=False,
        labels_stored=False,
        memory=MEMORY,
        teacher=teacher,
        source=bound,
        outputs=_outputs(quantiles),
        rows=len(keys),
        cut_us=int(decisions.max()),
        targets_path=path.name,
        targets_sha256=sha256(path),
        final_test_opened=False,
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    atomic_json(folder / "manifest.json", manifest)
    return folder / "manifest.json"


class TeacherTargets:
    """Salidas del profesor ligadas al tramo de ajuste del alumno y a un profesor declarado.

    Rechaza un archivo de otra vista, representación, política de entrada, fase, cabeza,
    memoria o profesor aunque coincidan las dimensiones, y uno cuyo contenido no conserve su
    huella. `values` da las salidas de las filas pedidas y falla si falta alguna.
    """

    def __init__(self, manifest, *, source, teacher, quantiles):
        meta, digest = read_manifest(Path(manifest), 1024**2)
        _require(
            meta.get("kind") == KIND
            and meta.get("use") == USE
            and meta.get("in_sample") is True
            and meta.get("out_of_sample_predictions") is False
            and meta.get("labels_stored") is False
            and meta.get("final_test_opened") is False,
            "El archivo no declara salidas del profesor solo para la pérdida del alumno",
        )
        _require(meta.get("memory") == MEMORY, "Las salidas del profesor usan otra memoria")
        _require(
            meta.get("source") == bound_source(source), "Las salidas del profesor son de otro tramo"
        )
        _require(meta.get("teacher") == teacher, "Las salidas son de otro profesor")
        _require(meta.get("outputs") == _outputs(quantiles), "Las salidas son de otra cabeza")
        path = Path(manifest).parent / meta["targets_path"]
        _require(
            meta["targets_path"] == "teacher-targets.parquet"
            and not path.is_symlink()
            and sha256(path) == meta["targets_sha256"],
            "El archivo de salidas del profesor ha cambiado",
        )
        table = pq.read_table(path)
        width = len(LEVELS) if quantiles else 1
        _require(
            table.schema.names == ["flow", "decision_at", "outputs"]
            and 1 <= table.num_rows == meta["rows"] <= _MAX_ROWS,
            "Las salidas del profesor no conservan su esquema y sus filas",
        )
        flows = table["flow"].to_pylist()
        decisions = table["decision_at"].to_numpy()
        values = np.asarray(table["outputs"].to_pylist(), dtype=np.float64).reshape(-1, width)
        _check_values(values, meta["source"]["phase"], decisions, width)
        _require(int(decisions.max()) == meta["cut_us"], "El corte declarado no es el de las filas")
        self.index = {key: i for i, key in enumerate(zip(flows, decisions.tolist(), strict=True))}
        _require(len(self.index) == len(flows), "Cada fila del profesor aparece una sola vez")
        self.values_cpu, self.quantiles = torch.from_numpy(values), quantiles
        self.source = meta["source"]
        self.identity = dict(
            manifest_sha256=digest,
            targets_sha256=meta["targets_sha256"],
            teacher=teacher,
            rows=meta["rows"],
            cut_us=meta["cut_us"],
            use=USE,
        )

    def values(self, keys, *, device, dtype):
        """Salidas del profesor de las filas (flujo, decisión), en ese orden."""
        positions = [self.index.get((flow, int(at))) for flow, at in keys]
        _require(None not in positions, "Falta la salida del profesor de una fila del ajuste")
        chosen = self.values_cpu[positions]
        if not self.quantiles:
            chosen = chosen[:, 0]
        return chosen.to(device=device, dtype=dtype)


def objective(task, prediction, teacher, weight):
    """Pérdida del alumno: (1 − w) · tarea + w · distancia L1 media al profesor."""
    _require(
        prediction.shape == teacher.shape, "La salida del profesor no tiene la forma del alumno"
    )
    distance = (prediction - teacher).abs().mean()
    return (1 - weight) * task + weight * distance


def load_declaration(path):
    """Comprobar la declaración del contraste antes de generar ningún objetivo."""
    meta, digest = read_manifest(Path(path), 256 * 1024)
    weights = meta.get("student", {}).get("distillation_weights")
    _require(
        meta.get("kind") == DECLARATION_KIND
        and meta.get("schema_version") == 1
        and meta.get("status") == "declared_not_executed"
        and meta.get("final_test_opened") is False,
        "La declaración de la destilación no conserva su tipo y su estado",
    )
    _require(
        isinstance(weights, list)
        and weights
        and len(set(weights)) == len(weights)
        and all(checked_weight(value) is not None for value in weights),
        "La declaración fija los pesos candidatos de la destilación",
    )
    _require(
        meta["student"]["direct_control"] == meta["student"]["arm"]
        and meta["teacher"]["arm"] != meta["student"]["arm"],
        "El alumno directo es el mismo modelo sin destilación y el profesor es otro",
    )
    _require(
        meta["targets"] == dict(use=USE, memory=MEMORY, distance=DISTANCE, in_sample=True),
        "La declaración no fija el uso, la memoria y la distancia de las salidas",
    )
    return meta, digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["check"])
    parser.add_argument("--declaration", type=Path, required=True)
    args = parser.parse_args(argv)
    meta, digest = load_declaration(args.declaration)
    print(
        json.dumps(
            dict(
                sha256=digest,
                status=meta["status"],
                teacher=meta["teacher"]["arm"],
                student=meta["student"]["arm"],
                distillation_weights=meta["student"]["distillation_weights"],
                final_test_opened=False,
            ),
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
