"""Validar y resumir los cuatro estratos técnicos sin importar bibliotecas numéricas."""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

STRATA = (("float32", 1), ("float32", 256), ("float64", 1), ("float64", 256))
CONTEXTS = (None, "market_memory_v1")
LEGACY = "legacy_self_uniform_interleaved"
ARMS = (
    LEGACY,
    "hierarchy_tree_local",
    "hierarchy_star_local",
    "hierarchy_permuted_tree_local",
    "hierarchy_tree_no_feedback_local",
    "hierarchy_flat_local",
    "hierarchy_tree_common_skip",
    "hierarchy_flat_common_skip",
)
SAMPLE_FIELDS = {
    "seconds",
    "allocated_bytes",
    "reserved_bytes",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "free_global_vram_bytes",
    "total_global_vram_bytes",
    "rss_bytes",
    "process_lifetime_peak_rss_bytes",
}
MAX_JSON_BYTES = 16 * 1024**2
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class ValidationError(ValueError):
    """Un artefacto no permite sostener la agregación completa."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def number(value, name, *, positive=False):
    finite = (
        -sys.float_info.max <= value <= sys.float_info.max
        if type(value) is int
        else math.isfinite(value)
        if type(value) is float
        else False
    )
    require(finite, f"{name} debe ser finito")
    require(value > 0 if positive else value >= 0, f"{name} no tiene un valor admisible")
    return value


def integer(value, name):
    require(type(value) is int and value >= 0, f"{name} debe ser un entero no negativo")
    return value


def digest(value):
    require(
        type(value) is str and DIGEST.fullmatch(value),
        "Falta una huella SHA-256 válida",
    )
    return value


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def relative_cost(value, baseline):
    result = number(value, "coste") / number(baseline, "denominador legacy", positive=True)
    require(math.isfinite(result), "El cociente de coste no es finito")
    return result


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Clave JSON duplicada: {key}")
        result[key] = value
    return result


def read_json(path):
    require(path.is_file(), f"Falta el estrato o su manifiesto: {path.name}")
    require(path.stat().st_size <= MAX_JSON_BYTES, f"El JSON supera 16 MiB: {path.name}")
    raw = path.read_bytes()
    require(len(raw) <= MAX_JSON_BYTES, f"El JSON supera 16 MiB: {path.name}")

    def invalid_constant(value):
        raise ValidationError(f"Constante JSON no finita: {value}")

    try:
        data = json.loads(raw, object_pairs_hook=unique_object, parse_constant=invalid_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValidationError(f"JSON inválido: {path.name}") from error
    return data, {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def parameter_shapes(record, element_size):
    supplied = record.get("parameter_shapes")
    require(type(supplied) is dict and 0 < len(supplied) <= 8192, "Faltan nombres y formas")
    shapes = {}
    for name, shape in supplied.items():
        require(
            type(name) is str and type(shape) is list,
            "Nombre o forma de tensor inválido",
        )
        require(
            len(shape) <= 8 and all(type(n) is int and 0 < n <= 2**31 for n in shape),
            "Forma de tensor inválida",
        )
        if not name.startswith("buffer/"):
            shapes[name] = shape
    require(bool(shapes), "No hay parámetros para comparar")
    elements = sum(math.prod(shape) for shape in shapes.values())
    require(
        integer(record.get("parameter_elements"), "elementos") == elements,
        "El recuento de parámetros contradice las formas",
    )
    require(
        integer(record.get("parameter_bytes"), "bytes de parámetros") == elements * element_size,
        "Los bytes de parámetros no corresponden a la precisión",
    )
    return shapes


def financial_configuration(record, context, arm, plan):
    configuration = record.get("financial_configuration")
    require(type(configuration) is dict, "Falta la configuración financiera completa")
    profile = plan["profile_overrides"]
    for name in ("hidden_size", "seed", "max_batch"):
        require(
            type(configuration.get(name)) is int and configuration[name] == profile[name],
            f"La configuración financiera cambia el perfil: {name}",
        )
    head = configuration.get("output_head")
    require(
        configuration.get("output") == profile["head"]
        and type(head) is dict
        and head.get("name") == profile["head"],
        "La configuración no conserva la cabeza de cuantiles en output_head",
    )
    require(
        configuration.get("variant") == "mac_online",
        "La configuración cambia el modo MAC",
    )
    declared_context = configuration.get("session_context")
    require(
        (context is None and "session_context" not in configuration)
        or (
            context is not None
            and type(declared_context) is dict
            and declared_context.get("kind") == context
        ),
        "La configuración financiera no conserva el contexto de la casilla",
    )
    execution = configuration.get("regional_execution")
    require(type(execution) is dict, "Falta la configuración de ejecución regional")
    raw = plan["arm_configurations"][arm]
    require(
        set(raw) <= set(execution)
        and canonical({name: execution[name] for name in raw}) == canonical(raw),
        "La etiqueta de configuración no corresponde a sus opciones regionales reales",
    )
    kind, program = (
        ("pre_mac_regional_program_v1", "summary_interleaved_v1")
        if arm == LEGACY
        else ("pre_mac_regional_hierarchy_v1", "hierarchical_group_root_v1")
    )
    require(
        type(execution.get("schema_version")) is int
        and execution["schema_version"] == 1
        and execution.get("kind") == kind
        and execution.get("program") == program,
        "La configuración regional cambia su familia o su programa",
    )
    common = {name: value for name, value in configuration.items() if name != "regional_execution"}
    return identity(configuration), identity(common), identity(execution)


def record_summary(record, context, arm):
    require(record.get("status") == "passed", "Hay una condición incompleta")
    samples, warmups = record.get("samples"), record.get("warmup_seconds")
    require(
        type(samples) is list and len(samples) == 5,
        "Cada condición necesita cinco muestras",
    )
    require(
        type(warmups) is list and len(warmups) == 2,
        "Cada condición necesita dos calentamientos",
    )
    seconds = []
    for sample in samples:
        require(
            type(sample) is dict and set(sample) == SAMPLE_FIELDS,
            "La muestra no conserva su esquema",
        )
        seconds.append(number(sample["seconds"], "duración de prepare", positive=True))
        for key in SAMPLE_FIELDS - {"seconds"}:
            integer(sample[key], key)
        require(
            sample["reserved_bytes"] >= sample["allocated_bytes"],
            "Reserva Torch menor que asignación",
        )
        require(
            sample["peak_allocated_bytes"] >= sample["allocated_bytes"],
            "Pico Torch asignado contradictorio",
        )
        require(
            sample["peak_reserved_bytes"]
            >= max(sample["reserved_bytes"], sample["peak_allocated_bytes"]),
            "Pico Torch reservado contradictorio",
        )
        require(
            sample["free_global_vram_bytes"] <= sample["total_global_vram_bytes"],
            "VRAM global contradictoria",
        )
    for value in warmups:
        number(value, "calentamiento", positive=True)
    mean, median = statistics.mean(seconds), statistics.median(seconds)
    batch = record["batch"]
    computed = {
        "min_ms": 1000 * min(seconds),
        "median_ms": 1000 * median,
        "max_ms": 1000 * max(seconds),
        "rows_per_second": batch / mean,
    }
    for key, expected in computed.items():
        supplied = number(record.get(key), key, positive=True)
        require(
            math.isclose(supplied, expected, rel_tol=1e-10, abs_tol=1e-12),
            f"El valor {key} contradice sus muestras",
        )
    work = {
        "local_node_evaluations": 20 * batch,
        "summary_evaluations": (2 if arm == LEGACY else 10) * batch,
    }
    require(
        canonical(record.get("observed_regional_work")) == canonical(work),
        "Trabajo regional incompatible con el calendario",
    )
    program = "summary_interleaved_v1" if arm == LEGACY else "hierarchical_group_root_v1"
    require(
        canonical(record.get("selected_program_rows")) == canonical({program: batch}),
        "El programa no cubre todas las filas",
    )
    require(
        record.get("observed_prefix_calls") == {"core": 2, "mac": 2},
        "El prefijo no tiene dos preparaciones y dos MAC",
    )
    require(
        record.get("mac_module_present") is True
        and record.get("associative_update_enabled") is True,
        "El modo de memoria no es mac_online",
    )
    require(
        record.get("context_present") is (context is not None),
        "El contexto no corresponde a la casilla",
    )
    dtype = record["dtype"]
    shapes = parameter_shapes(record, 4 if dtype == "float32" else 8)
    return {
        "cell_id": record["cell_id"],
        "arm": arm,
        "context": context,
        "dtype": dtype,
        "batch": batch,
        "pass_index": record["pass_index"],
        "position": record["position"],
        "sample_seconds": seconds,
        "warmup_seconds": list(warmups),
        "statistics": {
            "min_seconds": min(seconds),
            "median_seconds": median,
            "mean_seconds": mean,
            "max_seconds": max(seconds),
            "rows_per_second": batch / mean,
        },
        "memory": {
            "allocated_sampled_max_bytes": max(s["allocated_bytes"] for s in samples),
            "reserved_sampled_max_bytes": max(s["reserved_bytes"] for s in samples),
            "peak_allocated_max_reported_bytes": max(s["peak_allocated_bytes"] for s in samples),
            "peak_reserved_max_reported_bytes": max(s["peak_reserved_bytes"] for s in samples),
            "rss_sampled_max_bytes": max(s["rss_bytes"] for s in samples),
            "process_lifetime_peak_rss_max_reported_bytes": max(
                s["process_lifetime_peak_rss_bytes"] for s in samples
            ),
            "free_global_vram_min_reported_bytes": min(
                s["free_global_vram_bytes"] for s in samples
            ),
        },
        "parameter_elements": record["parameter_elements"],
        "parameter_bytes": record["parameter_bytes"],
        "parameter_shapes_sha256": identity(shapes),
        "full_tensor_sha256": digest(record.get("parameter_sha256")),
        "input_sha256": digest(record.get("input_sha256")),
        "initial_state_sha256": digest(record.get("initial_state_sha256")),
        "model_state_sha256": digest(record.get("model_state_sha256")),
        "observed_regional_work": work,
        "initialization_seconds": number(record.get("initialization_seconds"), "inicialización"),
        "prefix_seconds": number(record.get("prefix_seconds"), "preparación del prefijo"),
    }, shapes


def aggregate_jobs(jobs, manifests):
    require(
        set(jobs) == set(STRATA) and set(manifests) == set(STRATA),
        "Se necesitan los cuatro estratos completos",
    )
    first_manifest, protocol, hold = None, None, None
    rows, job_summaries, shape_groups, hashes, states, input_groups = (
        [],
        [],
        {},
        {},
        {},
        {},
    )
    configurations, common_configurations = {}, {}
    cells = [(context, arm) for context in CONTEXTS for arm in ARMS]
    ids = [f"{context or 'none'}/{arm}" for context, arm in cells]
    for dtype, batch in STRATA:
        report, manifest = jobs[dtype, batch], manifests[dtype, batch]
        require(type(report) is dict, "El estrato necesita un objeto JSON")
        require(type(manifest) is dict and bool(manifest), "Falta el manifiesto de fuentes")
        for name, value in manifest.items():
            require(type(name) is str, "Nombre de fuente inválido")
            digest(value)
        require(
            all(
                name in manifest
                for name in (
                    "benchmarks/regional_hierarchy_matrix_cuda.py",
                    "benchmarks/regional_program_matrix_cuda.py",
                    "src/mars_titan/models/titans/regional_hierarchy.py",
                )
            ),
            "El manifiesto no fija el productor y el componente",
        )
        if first_manifest is None:
            first_manifest = manifest
        require(
            manifest == first_manifest,
            "Los manifiestos de fuentes difieren entre estratos",
        )
        require(report.get("status") == "passed", f"Estrato incompleto: {dtype}/{batch}")
        require(
            report.get("source_manifest_unchanged") is True
            and report.get("hold_unchanged") is True,
            "Fuentes o guarda sin cierre verificado",
        )
        current_hold = digest(report.get("hold_before"))
        require(
            current_hold == digest(report.get("hold_sha256_final")),
            "Cambió la guarda durante el estrato",
        )
        hold = current_hold if hold is None else hold
        require(current_hold == hold, "Los estratos no comparten la misma guarda")
        plan = report.get("protocol")
        require(type(plan) is dict, "Falta el protocolo")
        protocol = plan if protocol is None else protocol
        require(
            canonical(plan) == canonical(protocol),
            "Los protocolos de los estratos difieren",
        )
        for key, value in {
            "condition_records": 128,
            "prefix_prepare_calls": 256,
            "timed_prepare_calls": 896,
            "total_prepare_calls": 1152,
            "warmup_per_record": 2,
            "repetitions_per_record": 5,
            "optimizer_steps": 0,
        }.items():
            require(
                type(plan.get(key)) is int and plan[key] == value,
                f"Cantidad de protocolo incompatible: {key}",
            )
        require(
            plan.get("scientific_evaluation") is False and plan.get("final_test_access") is False,
            "El protocolo no es una comprobación técnica cerrada",
        )
        require(
            plan.get("parameter_comparison")
            == "exact_values_names_shapes_within_context_and_dtype",
            "El protocolo no declara la comparación de pesos",
        )
        require(
            canonical(plan.get("profile_overrides"))
            == canonical(
                {"hidden_size": 32, "seed": 42, "max_batch": 256, "head": "quantile_head_v1"}
            ),
            "El protocolo cambia el perfil fijo H32, semilla42, lote físico256 o cuantiles",
        )
        configurations_by_arm = plan.get("arm_configurations")
        require(
            type(configurations_by_arm) is dict and set(configurations_by_arm) == set(ARMS),
            "El protocolo no contiene las ocho configuraciones de referencia",
        )
        for arm_name, raw in configurations_by_arm.items():
            expected_fields = (
                {"selection", "program", "context_threshold"}
                if arm_name == LEGACY
                else {"topology", "groups", "feedback", "readout"}
            )
            require(
                type(raw) is dict and set(raw) == expected_fields,
                "La configuración de referencia no conserva todas sus opciones raw",
            )
        require(
            plan.get("dtypes") == ["float32", "float64"] and plan.get("batches") == [1, 256],
            "Precisiones o lotes distintos",
        )
        declared = plan.get("cells")
        require(
            type(declared) is list
            and all(type(cell) is dict for cell in declared)
            and [cell.get("cell_id") for cell in declared] == ids,
            "Las casillas no conservan los ocho controles y dos contextos",
        )
        require(
            all(
                cell.get("variant") == "mac_online"
                and cell.get("context") == context
                and cell.get("arm") == arm
                for cell, (context, arm) in zip(declared, cells, strict=True)
            ),
            "Una casilla cambia modo, contexto o configuración",
        )
        stratum = report.get("execution_stratum")
        require(
            canonical(stratum)
            == canonical({"dtype": dtype, "batch": batch, "expected_records": 32}),
            "El archivo declara otro estrato",
        )
        records = report.get("records")
        require(
            type(records) is list and len(records) == 32,
            "Cada estrato necesita 32 registros",
        )
        for key, expected in (
            ("prefix_calls_completed", 64),
            ("timed_calls_completed", 224),
        ):
            require(
                type(report.get(key)) is int and report[key] == expected,
                f"Cantidad de llamadas incorrecta: {key}",
            )
        seen = set()
        for ordinal, record in enumerate(records):
            require(type(record) is dict, "Registro inválido")
            cell_id, pass_index = record.get("cell_id"), record.get("pass_index")
            key = (cell_id, pass_index)
            require(
                type(cell_id) is str and type(pass_index) is int and key not in seen,
                "Registro de condición duplicado o inválido",
            )
            seen.add(key)
            expected_pass, position = divmod(ordinal, 16)
            expected_id = ids[position] if expected_pass == 0 else ids[15 - position]
            require(
                pass_index == expected_pass
                and type(record.get("position")) is int
                and record["position"] == position
                and cell_id == expected_id,
                "El registro no conserva el orden global y su inverso",
            )
            require(
                record.get("dtype") == dtype
                and type(record.get("batch")) is int
                and record["batch"] == batch,
                "La condición pertenece a otro estrato",
            )
            context, arm = cells[ids.index(cell_id)]
            configuration_sha, common_sha, execution_sha = financial_configuration(
                record, context, arm, plan
            )
            row, shapes = record_summary(record, context, arm)
            row["financial_configuration_sha256"] = configuration_sha
            row["common_financial_configuration_sha256"] = common_sha
            row["regional_execution_identity_sha256"] = execution_sha
            group = (context, dtype)
            common_configurations.setdefault(group, common_sha)
            require(
                common_configurations[group] == common_sha,
                "Cambia la configuración común entre controles del mismo contexto y precisión",
            )
            configuration_key = (context, dtype, arm)
            configurations.setdefault(configuration_key, configuration_sha)
            require(
                configurations[configuration_key] == configuration_sha,
                "Cambia la configuración completa entre órdenes o lotes",
            )
            shape_groups.setdefault(group, shapes)
            require(
                shapes == shape_groups[group],
                "Los nombres o formas de los pesos difieren dentro de contexto y precisión",
            )
            tensor_key = (context, dtype, arm)
            hashes.setdefault(tensor_key, row["full_tensor_sha256"])
            require(
                hashes[tensor_key] == row["full_tensor_sha256"],
                "La misma configuración cambia su huella de parámetros y buffers",
            )
            state_key = (context, dtype, batch, arm)
            snapshot = (row["initial_state_sha256"], row["model_state_sha256"])
            states.setdefault(state_key, snapshot)
            require(
                states[state_key] == snapshot,
                "El orden cambia la reconstrucción del estado",
            )
            input_groups.setdefault((dtype, batch), row["input_sha256"])
            require(
                input_groups[dtype, batch] == row["input_sha256"],
                "Las condiciones no comparten sus entradas",
            )
            rows.append(row)
        require(len(seen) == 32, "Faltan condiciones distintas")
        cpu_cache = integer(report.get("cpu_proof_tensor_peak_bytes"), "caché lógica CPU")
        require(cpu_cache <= 128 * 1024**2, "La caché lógica supera su cota")
        job_summaries.append(
            {
                "dtype": dtype,
                "batch": batch,
                "records": 32,
                "prefix_calls": 64,
                "timed_calls": 224,
                "total_process_seconds": number(
                    report.get("total_process_seconds"),
                    "tiempo de proceso",
                    positive=True,
                ),
                "cpu_proof_tensor_peak_bytes": cpu_cache,
                "environment": report.get("environment"),
            }
        )
    baselines = {
        (r["context"], r["dtype"], r["batch"], r["pass_index"]): r
        for r in rows
        if r["arm"] == LEGACY
    }
    for row in rows:
        baseline = baselines[row["context"], row["dtype"], row["batch"], row["pass_index"]]
        left, right = row["statistics"], baseline["statistics"]
        row["relative_to_legacy"] = {
            "cell_id": baseline["cell_id"],
            "pass_index": row["pass_index"],
            "median_seconds_ratio": relative_cost(left["median_seconds"], right["median_seconds"]),
            "mean_seconds_ratio": relative_cost(left["mean_seconds"], right["mean_seconds"]),
            "median_delta_seconds": left["median_seconds"] - right["median_seconds"],
        }
    return {
        "schema_version": 1,
        "kind": "pre_mac_hierarchy_matrix_summary_v1",
        "status": "passed",
        "checked": {
            "strata": 4,
            "unique_records": 128,
            "prefix_calls": 256,
            "timed_calls": 896,
            "total_prepare_calls": 1152,
            "measured_samples": 640,
            "warmups": 256,
        },
        "source_manifest": first_manifest,
        "source_manifest_sha256": identity(first_manifest),
        "hold_sha256": hold,
        "protocol": protocol,
        "jobs": job_summaries,
        "records": rows,
        "completed_process_seconds_sum": sum(j["total_process_seconds"] for j in job_summaries),
        "parameter_verification": {
            "financial_configuration": "raw_options_kind_program_and_full_hash_checked",
            "common_financial_configuration": (
                "equal_without_regional_execution_within_context_and_dtype"
            ),
            "regional_execution_identity": "complete_regional_execution_block_hash_per_record",
            "names_shapes_elements_and_bytes": "independent_stdlib_checks_within_context_and_dtype",
            "same_configuration_tensor_hashes": "checked_across_orders_and_batches_within_dtype",
            "cross_geometry_values": (
                "runtime_strict_equal_in_pinned_producer_not_reconstructed_from_archived_weights"
            ),
            "full_tensor_hash_scope": "includes_geometric_buffers_not_a_parameter_only_digest",
        },
        "scopes": {
            "latency": (
                "FinancialPredictor.prepare completo y sincronización CUDA, "
                "cinco muestras por registro"
            ),
            "torch": (
                "Máximos absolutos del asignador registrados en las cinco muestras, "
                "con modelo e inputs residentes"
            ),
            "rss": "RSS muestreado, no pico de RAM por condición",
            "lifetime_rss": (
                "Máximo de vida del proceso comunicado por otra interfaz, "
                "incluye condiciones anteriores"
            ),
            "relative_cost": (
                "Cada denominador legacy tiene el mismo contexto, precisión, lote y orden"
            ),
            "orders": "Los dos órdenes permanecen separados, sin seleccionar el más favorable",
            "process_time": (
                "Suma de los procesos completos, separada del tiempo interior de prepare "
                "y sin supervisión externa"
            ),
        },
        "scientific_evaluation": False,
        "final_test_access": False,
        "optimizer_steps": 0,
        "confidence_intervals": False,
        "tail_percentiles": False,
        "rankings": False,
    }


def aggregate_directory(root):
    jobs, manifests, artifacts = {}, {}, {}
    for dtype, batch in STRATA:
        directory = root / f"{dtype}-{batch}"
        for name, destination in (
            ("measurement.json", jobs),
            ("source-manifest.json", manifests),
        ):
            data, proof = read_json(directory / name)
            destination[dtype, batch] = data
            artifacts[f"{dtype}-{batch}/{name}"] = proof
    summary = aggregate_jobs(jobs, manifests)
    summary["input_artifacts"] = artifacts
    summary["aggregator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return summary


def write_reports(summary, output):
    require(
        summary.get("status") == "passed"
        and summary.get("checked", {}).get("strata") == 4
        and len(summary.get("records", [])) == 128,
        "Solo se publica una agregación completa y validada",
    )
    require(not output.exists(), "La salida de agregación ya existe")
    encoded = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    require(len(encoded.encode()) <= MAX_JSON_BYTES, "El resumen supera 16 MiB")
    output.mkdir(parents=True)
    columns = [
        "arm",
        "context",
        "dtype",
        "batch",
        "pass_index",
        "position",
        "min_ms",
        "median_ms",
        "mean_ms",
        "max_ms",
        "rows_per_second",
        "legacy_median_ms",
        "median_ratio_to_legacy",
        "mean_ratio_to_legacy",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "rss_sampled_max_bytes",
        "process_lifetime_peak_rss_max_reported_bytes",
        "financial_configuration_sha256",
        "common_financial_configuration_sha256",
        "regional_execution_identity_sha256",
    ]
    baseline = {
        (r["context"], r["dtype"], r["batch"], r["pass_index"]): r
        for r in summary["records"]
        if r["arm"] == LEGACY
    }
    with (output / "conditions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in summary["records"]:
            values = {key: row[key] for key in columns[:6]}
            values["context"] = row["context"] or "none"
            values.update(
                {
                    name + "_ms": 1000 * row["statistics"][name + "_seconds"]
                    for name in ("min", "median", "mean", "max")
                }
            )
            base = baseline[row["context"], row["dtype"], row["batch"], row["pass_index"]]
            values.update(
                rows_per_second=row["statistics"]["rows_per_second"],
                legacy_median_ms=1000 * base["statistics"]["median_seconds"],
                median_ratio_to_legacy=row["relative_to_legacy"]["median_seconds_ratio"],
                mean_ratio_to_legacy=row["relative_to_legacy"]["mean_seconds_ratio"],
                peak_allocated_bytes=row["memory"]["peak_allocated_max_reported_bytes"],
                peak_reserved_bytes=row["memory"]["peak_reserved_max_reported_bytes"],
                rss_sampled_max_bytes=row["memory"]["rss_sampled_max_bytes"],
                process_lifetime_peak_rss_max_reported_bytes=row["memory"][
                    "process_lifetime_peak_rss_max_reported_bytes"
                ],
                financial_configuration_sha256=row["financial_configuration_sha256"],
                common_financial_configuration_sha256=row["common_financial_configuration_sha256"],
                regional_execution_identity_sha256=row["regional_execution_identity_sha256"],
            )
            writer.writerow(values)
    note = (
        "Los cuatro estratos completaron 128 registros distintos, 256 preparaciones de prefijo, "
        "256 calentamientos y 640 muestras, 1152 preparaciones en total. "
        "Se conservan ambos órdenes.\n\n"
        "conditions.csv compara cada condición con legacy dentro del mismo contexto, precisión, "
        "lote y orden. Los cocientes son descriptivos de cinco muestras. No se selecciona un orden "
        "ni se agrupan para declarar una mejora.\n\n"
        "Los máximos de Torch, el RSS muestreado y el máximo de vida del proceso tienen ámbitos "
        "distintos. La igualdad de valores entre geometrías se apoya en las comprobaciones del "
        "productor fijado. Los artefactos no permiten reconstruir esos pesos "
        "con la biblioteca estándar. "
        "No se han medido calidad predictiva, intervalos de confianza o percentiles de cola.\n"
    )
    (output / "summary.md").write_text(note)
    (output / "summary.json").write_text(encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        summary = aggregate_directory(args.input_dir.resolve())
        write_reports(summary, args.output.resolve())
    except (ValidationError, OSError) as error:
        parser.exit(2, f"No se completó la agregación: {error}\n")
    print(json.dumps({"status": "passed", "records": 128, "output": str(args.output)}))


if __name__ == "__main__":
    main()
