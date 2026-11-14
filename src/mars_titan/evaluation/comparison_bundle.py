"""ZIP verificable de resultados de comparación, sin pesos ni datos de entrada.

Solo se copian registros, informes JSON y tablas de predicciones que sus productores
declaran. La lectura del ZIP comprueba cada huella y no extrae rutas arbitrarias.
"""

import hashlib
import io
import json
import os
import re
import shutil
import stat
import struct
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from mars_titan.data.storage import outside_source

KIND = "comparison_campaign_results_v1"
PARTITIONS = ("validation", "calibration", "evaluation")
MAX_FILES = 16_384
MAX_JSON_BYTES = 16 * 1024**2
MAX_FILE_BYTES = 512 * 1024**2
MAX_TOTAL_BYTES = 128 * 1024**3
MAX_TABLE_ROWS = 1_100_000
_CHUNK = 1024**2
_HASH = re.compile(r"[a-f0-9]{64}")
_ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True).encode("utf-8")


def _json(data, *, limit=MAX_JSON_BYTES):
    _require(len(data) <= limit, "El JSON supera el límite de lectura")

    def invalid(value):
        raise ValueError(f"El JSON contiene {value}")

    return json.loads(data, parse_constant=invalid)


def _relative(value):
    _require(
        isinstance(value, str) and value and not any(c in value for c in ("\\", ":", "\x00")),
        "ruta no válida",
    )
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and str(path) == value
        and all(part not in (".", "..") for part in path.parts),
        "La ruta debe ser relativa y no salir del paquete",
    )
    return path


def _source(root, relative):
    path = Path(root).joinpath(*_relative(relative).parts)
    _require(
        path.is_file() and not path.is_symlink(), f"Falta un archivo regular en la ruta {relative}"
    )
    _require(path.resolve().is_relative_to(Path(root).resolve()), "La ruta sale del trabajo")
    return path


def _read(path, expected=None, *, limit=MAX_JSON_BYTES):
    _require(path.is_file() and not path.is_symlink(), f"Falta el archivo regular {path.name}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    _require(len(data) <= limit, f"El archivo {path.name} supera el límite")
    digest = hashlib.sha256(data).hexdigest()
    _require(expected in (None, digest), f"El archivo {path.name} no conserva su huella")
    return _json(data, limit=limit), digest


class _Payloads:
    def __init__(self):
        self.items = {}

    def data(self, name, value):
        self._add(name, _json_bytes(value), None)

    def file(self, name, path, expected=None):
        self._add(name, Path(path), expected)

    def _add(self, name, source, expected):
        _relative(name)
        _require(name not in self.items, "Un archivo del paquete está repetido")
        size = len(source) if isinstance(source, bytes) else source.stat().st_size
        _require(size <= MAX_FILE_BYTES, f"El archivo {name} supera el límite")
        self.items[name] = source, expected, size
        _require(len(self.items) < MAX_FILES, "El paquete supera el límite de archivos")

    def write(self, archive):
        records = {}
        for name, (source, expected, size) in self.items.items():
            digest, written = hashlib.sha256(), 0
            before = None
            if isinstance(source, bytes):
                stream = io.BytesIO(source)
            else:
                _require(not source.is_symlink(), "Ha cambiado la ruta del archivo")
                status = source.stat()
                before = status.st_ino, status.st_size, status.st_mtime_ns
                stream = source.open("rb")
            info = zipfile.ZipInfo(name)
            info.compress_type = (
                zipfile.ZIP_DEFLATED if name.endswith((".json", ".log")) else zipfile.ZIP_STORED
            )
            with stream, archive.open(info, "w", force_zip64=True) as target:
                while block := stream.read(_CHUNK):
                    written += len(block)
                    _require(written <= size, f"El archivo {name} cambió durante la exportación")
                    digest.update(block)
                    target.write(block)
            actual = digest.hexdigest()
            _require(written == size, f"El archivo {name} cambió durante la exportación")
            _require(expected in (None, actual), f"El archivo {name} no conserva su huella")
            if before is not None:
                status = source.stat()
                _require(
                    before == (status.st_ino, status.st_size, status.st_mtime_ns),
                    f"El archivo {name} cambió durante la exportación",
                )
            records[name] = dict(bytes=written, sha256=actual)
        return records


def _cohorts(declaration, layout, payloads, destination):
    result = {}
    for origin in declaration["origins"]:
        year = origin["year"]
        receipt, _ = _read(layout.cohort(year) / "cohort.json")
        view, digest = _read(
            layout.cohort(year) / "manifest.json",
            receipt["manifest_sha256"],
            limit=64 * 1024**2,
        )
        selection = receipt["selection"]
        for root in view["roots"].values():
            outside_source(Path(root), destination)
        _require(selection["size"] == declaration["cohort"]["size"], "La cohorte tiene otro tamaño")
        entry = dict(
            manifest_sha256=digest,
            selection=selection,
            fold=view["temporal_view"]["fold"],
            input_policy=view["input_policy"],
            mask_contract=view["mask_contract"],
        )
        name = f"cohorts/{year}.json"
        payloads.data(name, entry)
        result[str(year)] = dict(file=name, **entry)
    return result


def _report(payloads, root, relative, name, expected=None):
    path = _source(root, relative)
    document, digest = _read(path, expected)
    _require(isinstance(document, dict), "El informe no es un objeto JSON")
    _require(document.get("final_test_opened") is not True, "El informe ha abierto el test final")
    payloads.file(name, path, digest)
    return document


def _tables(payloads, job, root, report, seed):
    predictions = report.get("predictions")
    _require(isinstance(predictions, dict), "Faltan las predicciones del trabajo terminado")
    result = {}
    for partition in PARTITIONS:
        record = predictions.get(partition, {})
        relative = record.get("path")
        _relative(relative)
        _require(relative.endswith(".parquet"), "La ruta de predicciones no es Parquet")
        digest = record.get("sha256")
        _require(
            isinstance(digest, str) and _HASH.fullmatch(digest), "Falta la huella de predicciones"
        )
        name = f"predictions/{job.key}/{seed}/{partition}.parquet"
        payloads.file(name, _source(root, relative), digest)
        result[partition] = name
    return result


def _job(declaration, layout, job, cohort, payloads):
    from mars_titan.training.comparison_campaign import DONE, read_record

    _require(_ID.fullmatch(job.model) is not None, "El modelo no tiene una ruta válida")
    record = read_record(layout, job.key)
    record_name = f"records/{job.key}.json"
    path = layout.record(job.key)
    if path.exists():
        record, digest = _read(path)
        payloads.file(record_name, path, digest)
    else:
        payloads.data(record_name, record)
    entry = dict(
        model=job.model,
        year=job.year,
        seed=job.seed,
        runner=job.runner,
        status=record["status"],
        record=record_name,
        report=None,
        fit_report=None,
        tables={},
        seed_reports={},
        trial_reports={},
        needs=job.needs,
        log=None,
    )
    log = layout.log(job.key)
    if log.is_file():
        _require(not log.is_symlink(), "El registro de salida no es un archivo regular")
        entry["log"] = f"logs/{job.key}.log"
        payloads.file(entry["log"], log)
    root = layout.output(job.key)
    if not (root / "run.json").exists():
        _require(record["status"] != DONE, "El trabajo terminado no tiene informe")
        return entry
    entry["report"] = f"reports/{job.key}/run.json"
    report = _report(payloads, root, "run.json", entry["report"])
    fit = report.get("fit")
    if isinstance(fit, dict):
        entry["fit_report"] = f"reports/{job.key}/fit.json"
        _report(payloads, root, fit["path"], entry["fit_report"], fit["sha256"])
    elif (root / "fit/run.json").is_file():
        entry["fit_report"] = f"reports/{job.key}/fit.json"
        _report(payloads, root, "fit/run.json", entry["fit_report"])
    if record["status"] != DONE:
        return entry
    _require(report.get("status") == DONE, "El informe del trabajo no está terminado")
    if job.runner in ("fit", "reader"):
        _require(entry["fit_report"] is not None, "Falta el historial del ajuste terminado")
    request = report.get("request", {})
    _require(
        request.get("view_sha256") == cohort["manifest_sha256"],
        "El trabajo pertenece a otra cohorte",
    )
    _require(request.get("window", job.fold) == job.fold, "El trabajo pertenece a otra ventana")
    if job.seed is not None:
        _require(request.get("seed") == job.seed, "El trabajo pertenece a otra semilla")
    expected_id = record.get("result", {}).get("run_id")
    _require(
        expected_id is None or expected_id == report.get("run_id"), "El registro nombra otro ajuste"
    )
    if job.runner == "tabular":
        seeds = report.get("seeds", {})
        _require(
            set(seeds) == {str(s) for s in declaration["seeds"]},
            "Faltan semillas del control tabular",
        )
        for seed, item in seeds.items():
            path = _source(root, item["path"])
            name = f"reports/{job.key}/seeds/{seed}.json"
            child = _report(payloads, root, item["path"], name, item["sha256"])
            _require(
                child.get("seed") == int(seed) and child.get("status") == DONE,
                "La semilla tabular no está terminada",
            )
            entry["seed_reports"][seed] = name
            entry["tables"][seed] = _tables(payloads, job, path.parent, child, seed)
    else:
        seed = "shared" if job.seed is None else str(job.seed)
        entry["tables"][seed] = _tables(payloads, job, root, report, seed)
    for name, item in (report.get("fits") or {}).items():
        _require(_ID.fullmatch(name) is not None, "La ruta del ajuste tabular no es válida")
        destination = f"reports/{job.key}/fits/{name}.json"
        _report(payloads, root, item["path"] + "/run.json", destination, item["run_sha256"])
        entry["trial_reports"][name] = destination
    return entry


def export_results(declaration, declaration_sha256, layout, machine, destination):
    """Guardar un ZIP nuevo con resultados confirmados y estados de los trabajos incompletos.

    La autorización se consulta para identificar la ejecución, pero no se transporta un
    permiso de aprendizaje. Una cola que esté escribiendo puede hacer fallar la comprobación
    de huellas. Para obtener un corte estable se debe pausar antes de exportar.
    """
    from mars_titan.training.comparison_campaign import DONE, checked_declaration, plan

    checked_declaration(declaration)
    destination = Path(destination)
    _require(destination.suffix.lower() == ".zip", "El destino debe ser un archivo ZIP")
    _require(not destination.exists(), "El archivo de destino ya existe")
    outside_source(layout.state, destination)
    outside_source(layout.views, destination)
    authorization, _ = _read(layout.authorization)
    _require(
        authorization.get("declaration_sha256") == declaration_sha256,
        "La declaración no es la autorizada en este estado",
    )
    commit = authorization.get("code_commit")
    _require(
        isinstance(commit, str) and re.fullmatch(r"[a-f0-9]{40}", commit),
        "Falta el commit del entrenamiento",
    )
    payloads = _Payloads()
    payloads.data("declaration.json", declaration)
    cohorts = _cohorts(declaration, layout, payloads, destination)
    jobs = {
        job.key: _job(declaration, layout, job, cohorts[str(job.year)], payloads)
        for job in plan(declaration, machine)
    }
    complete = sum(job["status"] == DONE for job in jobs.values())
    manifest = dict(
        schema_version=1,
        kind=KIND,
        created_at_utc=datetime.now(UTC).isoformat(),
        machine=machine,
        models_included=False,
        dataset_included=False,
        final_test_opened=False,
        complete=complete == len(jobs),
        completed_jobs=complete,
        provenance=dict(
            training_commit=commit,
            code_changes_sha256=authorization.get("code_changes_sha256"),
            authorized_declaration_sha256=declaration_sha256,
            declaration_sha256=hashlib.sha256(_json_bytes(declaration)).hexdigest(),
        ),
        cohorts=cohorts,
        jobs=jobs,
    )
    total = sum(item[2] for item in payloads.items.values())
    _require(total + MAX_JSON_BYTES <= MAX_TOTAL_BYTES, "El paquete supera el límite de bytes")
    manifest["payload_bytes"] = total
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require(
        shutil.disk_usage(destination.parent).free
        >= total + len(payloads.items) * 4096 + MAX_JSON_BYTES,
        "No hay espacio para escribir el ZIP",
    )
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            with zipfile.ZipFile(stream, "w", allowZip64=True) as archive:
                manifest["files"] = payloads.write(archive)
                metadata = _json_bytes(manifest)
                _require(len(metadata) <= MAX_JSON_BYTES, "El manifiesto supera el límite")
                archive.writestr("manifest.json", metadata, compress_type=zipfile.ZIP_DEFLATED)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ValueError("El archivo de destino ya existe") from error
    finally:
        Path(temporary).unlink(missing_ok=True)
    return manifest


def _check_directory(path):
    """Acotar el directorio central antes de que ZipFile lo cargue en memoria.

    Los paquetes se escriben en un volumen y sin comentario ZIP. Aunque usen ZIP64 por el
    tamaño de las predicciones, su directorio y su número de entradas caben en estos campos.
    """
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), "El ZIP debe ser un archivo regular")
    _require(
        path.stat().st_size <= MAX_TOTAL_BYTES + MAX_JSON_BYTES, "El ZIP supera el límite de bytes"
    )
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        _require(stream.tell() >= 22, "El ZIP no tiene una cabecera válida")
        stream.seek(-22, os.SEEK_END)
        signature, disk, directory_disk, disk_entries, entries, size, _, comment = struct.unpack(
            "<4s4H2LH", stream.read(22)
        )
    _require(
        signature == b"PK\x05\x06"
        and disk == directory_disk == comment == 0
        and disk_entries == entries,
        "El ZIP no tiene la cabecera de un paquete de resultados",
    )
    _require(
        entries <= MAX_FILES and size <= MAX_FILES * 1024, "El directorio del ZIP supera el límite"
    )


class ResultsBundle:
    """Lector de un paquete autocontenido. Verifica los bytes antes de exponer tablas."""

    def __init__(self, path):
        self.path = Path(path)
        self.archive = None

    def __enter__(self):
        try:
            _check_directory(self.path)
            self.archive = zipfile.ZipFile(self.path)
            self._validate()
        except (KeyError, TypeError, AttributeError) as error:
            if self.archive is not None:
                self.archive.close()
            raise ValueError(
                "El paquete tiene metadatos incompletos o de tipo incorrecto"
            ) from error
        except BaseException:
            if self.archive is not None:
                self.archive.close()
            raise
        return self

    def __exit__(self, *_):
        self.archive.close()

    def _validate(self):
        from mars_titan.training.comparison_campaign import DONE, checked_declaration, plan

        entries = self.archive.infolist()
        names = [info.filename for info in entries]
        _require(len(names) == len(set(names)), "El ZIP tiene miembros repetidos")
        _require(0 < len(names) <= MAX_FILES, "El ZIP supera el límite de archivos")
        _require(
            sum(info.file_size for info in entries) <= MAX_TOTAL_BYTES,
            "El ZIP supera el límite de bytes",
        )
        for info in entries:
            _relative(info.filename)
            _require(
                not info.is_dir() and not stat.S_ISLNK(info.external_attr >> 16),
                "El ZIP contiene una ruta que no es un archivo regular",
            )
            _require(
                not info.flag_bits & 1
                and info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                "El ZIP utiliza un formato no admitido",
            )
            _require(
                info.file_size
                <= (MAX_JSON_BYTES if info.filename.endswith(".json") else MAX_FILE_BYTES),
                "Un archivo del ZIP supera el límite",
            )
        _require("manifest.json" in names, "Falta el manifiesto del ZIP")
        self.manifest = _json(self.archive.read("manifest.json"))
        m = self.manifest
        _require(
            isinstance(m, dict)
            and m.get("kind") == KIND
            and type(m.get("schema_version")) is int
            and m["schema_version"] == 1
            and m.get("final_test_opened") is False,
            "El manifiesto no es un paquete de resultados admitido",
        )
        _require(
            m.get("models_included") is False and m.get("dataset_included") is False,
            "El paquete no debe contener modelos ni dataset",
        )
        files = m.get("files")
        _require(
            isinstance(files, dict) and set(files) == set(names) - {"manifest.json"},
            "El ZIP contiene archivos no declarados o faltan archivos",
        )
        for name, record in files.items():
            _require(
                isinstance(record, dict)
                and type(record.get("bytes")) is int
                and isinstance(record.get("sha256"), str)
                and _HASH.fullmatch(record["sha256"]),
                "La huella declarada no es válida",
            )
            _require(
                record["bytes"] == self.archive.getinfo(name).file_size,
                "El tamaño no coincide con el declarado",
            )
            digest = hashlib.sha256()
            with self.archive.open(name) as stream:
                while block := stream.read(_CHUNK):
                    digest.update(block)
            _require(
                digest.hexdigest() == record["sha256"], f"El archivo {name} no conserva su huella"
            )
        _require(
            m["payload_bytes"] == sum(record["bytes"] for record in files.values()),
            "El tamaño total del contenido no coincide",
        )
        self.declaration = self.read_json("declaration.json")
        checked_declaration(self.declaration, check_recipe_files=False)
        provenance = m["provenance"]
        _require(
            isinstance(provenance.get("training_commit"), str)
            and re.fullmatch(r"[a-f0-9]{40}", provenance["training_commit"])
            and all(
                isinstance(provenance.get(k), str) and _HASH.fullmatch(provenance[k])
                for k in ("declaration_sha256", "authorized_declaration_sha256")
            ),
            "La procedencia no identifica el código y la declaración",
        )
        _require(
            m["provenance"]["declaration_sha256"] == files["declaration.json"]["sha256"],
            "La declaración tiene otra huella",
        )
        expected = {job.key: job for job in plan(self.declaration, m["machine"])}
        _require(set(m["jobs"]) == set(expected), "Los trabajos no coinciden con la declaración")
        _require(
            set(m["cohorts"]) == {str(o["year"]) for o in self.declaration["origins"]},
            "Faltan cohortes de la declaración",
        )
        used = {"declaration.json"}
        for year, cohort in m["cohorts"].items():
            name = f"cohorts/{year}.json"
            _require(cohort["file"] == name, "La ruta de la cohorte no es válida")
            _require(
                self.read_json(name) == {k: v for k, v in cohort.items() if k != "file"},
                "La cohorte no coincide con su manifiesto",
            )
            used.add(name)
        for key, job in expected.items():
            row = m["jobs"][key]
            _require(
                row["model"] == job.model
                and row["year"] == job.year
                and row["seed"] == job.seed
                and row["runner"] == job.runner
                and row["needs"] == job.needs,
                "La identidad del trabajo ha cambiado",
            )
            _require(row["record"] == f"records/{key}.json", "La ruta del registro no es válida")
            record = self.read_json(row["record"])
            _require(record.get("status") == row["status"], "El estado del trabajo no coincide")
            _require(
                row["status"] in ("pending", "running", "paused", "failed", "completed", "blocked"),
                "El estado del trabajo no es válido",
            )
            if row["status"] == DONE:
                _require(row["report"] is not None, "Falta el informe del trabajo terminado")
                if job.runner in ("fit", "reader"):
                    _require(
                        row["fit_report"] is not None, "Falta el historial del ajuste terminado"
                    )
            used.add(row["record"])
            if row["log"] is not None:
                _require(
                    row["log"] == f"logs/{key}.log", "La ruta del registro de salida no es válida"
                )
                used.add(row["log"])
            for field, filename in (("report", "run.json"), ("fit_report", "fit.json")):
                if row[field] is not None:
                    _require(
                        row[field] == f"reports/{key}/{filename}",
                        "La ruta del informe no es válida",
                    )
                    used.add(row[field])
            for field, folder in (("seed_reports", "seeds"), ("trial_reports", "fits")):
                for label, name in row[field].items():
                    _require(
                        _ID.fullmatch(label) and name == f"reports/{key}/{folder}/{label}.json",
                        "La ruta del informe auxiliar no es válida",
                    )
                    used.add(name)
            expected_seeds = (
                {str(s) for s in self.declaration["seeds"]}
                if job.runner == "tabular"
                else {"shared" if job.seed is None else str(job.seed)}
            )
            _require(
                set(row["tables"]) == (expected_seeds if row["status"] == DONE else set()),
                "Faltan semillas o hay tablas de un trabajo incompleto",
            )
            for seed, tables in row["tables"].items():
                _require(set(tables) == set(PARTITIONS), "Faltan particiones de predicciones")
                for partition, name in tables.items():
                    _require(
                        name == f"predictions/{key}/{seed}/{partition}.parquet",
                        "La ruta de predicciones no es válida",
                    )
                    used.add(name)
        _require(used == set(files), "Hay archivos no declarados por los trabajos")
        for key, job in expected.items():
            self._validate_reports(job, m["jobs"][key])
        complete = sum(row["status"] == DONE for row in m["jobs"].values())
        _require(
            m["completed_jobs"] == complete and m["complete"] is (complete == len(expected)),
            "El recuento de trabajos terminados no coincide",
        )

    def _validate_reports(self, job, row):
        from mars_titan.training.comparison_campaign import DONE

        if row["report"] is None:
            return
        report = self.read_json(row["report"])
        _require(report.get("final_test_opened") is not True, "El informe ha abierto el test final")
        if row["status"] != DONE:
            return
        request = report.get("request", {})
        _require(report.get("status") == DONE, "El informe no está terminado")
        _require(
            request.get("view_sha256")
            == self.manifest["cohorts"][str(job.year)]["manifest_sha256"],
            "El informe corresponde a otra cohorte",
        )
        _require(request.get("window", job.fold) == job.fold, "El informe tiene otra ventana")
        if job.seed is not None:
            _require(request.get("seed") == job.seed, "El informe tiene otra semilla")
        if row["fit_report"] is not None:
            _require(
                report["fit"]["sha256"] == self.manifest["files"][row["fit_report"]]["sha256"],
                "El historial no pertenece al ajuste declarado",
            )
        if job.needs:
            parent = self.manifest["jobs"].get(job.needs)
            _require(parent is not None and parent["status"] == DONE, "Falta el padre terminado")
            parent_report = self.read_json(parent["report"])
            _require(
                request.get("parent", {}).get("run_sha256")
                == self.manifest["files"][parent["report"]]["sha256"]
                and request["parent"].get("checkpoint_sha256")
                == parent_report["checkpoint"]["sha256"],
                "El lector no pertenece al checkpoint de su padre",
            )

    def read_json(self, name):
        _require(
            name in self.manifest["files"] and name.endswith(".json"), "El JSON no está declarado"
        )
        return _json(self.archive.read(name))

    def read_table(self, name):
        """Leer un Parquet acotado conservando sus columnas para las métricas posteriores."""
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        _require(
            name in self.manifest["files"] and name.endswith(".parquet"),
            "La tabla no está declarada",
        )
        data = self.archive.read(name)
        parquet = pq.ParquetFile(
            pa.BufferReader(data), read_dictionary=["sample_id", "asset_id", "market"]
        )
        _require(
            parquet.metadata.num_rows <= MAX_TABLE_ROWS and parquet.metadata.num_columns <= 64,
            "La tabla supera el límite de filas o columnas",
        )
        _require(
            sum(
                parquet.metadata.row_group(i).total_byte_size for i in range(parquet.num_row_groups)
            )
            <= MAX_FILE_BYTES,
            "La tabla supera el límite descomprimido",
        )
        table = parquet.read(use_threads=False)
        for name, maximum in (("sample_id", 256), ("asset_id", 64), ("market", 8)):
            if name not in table.column_names:
                continue
            for chunk in table[name].chunks:
                values = chunk.dictionary if pa.types.is_dictionary(chunk.type) else chunk
                if pa.types.is_string(values.type) or pa.types.is_large_string(values.type):
                    _require(
                        (pc.max(pc.binary_length(values)).as_py() or 0) <= maximum,
                        "Una identidad supera el límite de longitud",
                    )
            if pa.types.is_dictionary(table[name].type):
                table = table.set_column(
                    table.schema.get_field_index(name), name, table[name].cast(pa.large_string())
                )
        return table
