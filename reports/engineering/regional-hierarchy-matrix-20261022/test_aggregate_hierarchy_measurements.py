"""Fixtures manuales del agregador, sin mediciones reales ni cargas numéricas."""

import copy
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("aggregate_hierarchy_measurements.py")
SPEC = importlib.util.spec_from_file_location("hierarchy_aggregation", SCRIPT)
AGG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGG)

ARMS = (
    "legacy_self_uniform_interleaved",
    "hierarchy_tree_local",
    "hierarchy_star_local",
    "hierarchy_permuted_tree_local",
    "hierarchy_tree_no_feedback_local",
    "hierarchy_flat_local",
    "hierarchy_tree_common_skip",
    "hierarchy_flat_common_skip",
)
STRATA = (("float32", 1), ("float32", 256), ("float64", 1), ("float64", 256))


def sha(value):
    return hashlib.sha256(repr(value).encode()).hexdigest()


def reference_configurations():
    """Opciones literales de referencia para datos de pruebas unitarias."""
    groups = [[0, 2], [1, 3, 4]]
    result = {
        ARMS[0]: {
            "selection": "fixed",
            "program": "summary_interleaved_v1",
            "context_threshold": None,
        }
    }
    controls = (
        ("tree", groups, True, "local_only"),
        ("star", groups, True, "local_only"),
        ("tree", [[0, 1], [2, 3, 4]], True, "local_only"),
        ("tree", groups, False, "local_only"),
        ("flat_control", groups, True, "local_only"),
        ("tree", groups, True, "common_skip"),
        ("flat_control", groups, True, "common_skip"),
    )
    for arm, (topology, partition, feedback, readout) in zip(ARMS[1:], controls, strict=True):
        result[arm] = {
            "topology": topology,
            "groups": copy.deepcopy(partition),
            "feedback": feedback,
            "readout": readout,
        }
    return result


def configuration_fixture(context, arm, declared):
    """Representar el contrato que se consume, sin fingir un modelo o una medición."""
    execution = copy.deepcopy(declared[arm])
    execution.update(
        schema_version=1,
        kind="pre_mac_regional_program_v1" if arm == ARMS[0] else "pre_mac_regional_hierarchy_v1",
        program="summary_interleaved_v1" if arm == ARMS[0] else "hierarchical_group_root_v1",
    )
    if arm != ARMS[0]:
        execution["rounds"] = 2
    configuration = {
        "schema_version": 1,
        "inputs": {"source_sha256": sha("unit_fixture_inputs_not_a_measurement")},
        "variant": "mac_online",
        "hidden_size": 32,
        "seed": 42,
        "max_batch": 256,
        "dropout": 0.0,
        "output": "quantile_head_v1",
        "output_head": {
            "schema_version": 1,
            "name": "quantile_head_v1",
            "levels": [0.025, 0.1, 0.5, 0.9, 0.975],
            "point_prediction": "median",
        },
        "regional_fusion": {
            "kind": "regional_fusion_v1",
            "topology": "self",
            "routing": "uniform",
            "cycles": 2,
            "local_steps": 2,
            "summary": True,
            "local_gain": 0.25,
            "summary_gain": 0.25,
        },
        "regional_execution": execution,
    }
    if context is not None:
        configuration["session_context"] = {"kind": context, "sectors": None}
    return configuration


def fixtures():
    cells = [
        {
            "cell_id": f"{context or 'none'}/{arm}",
            "variant": "mac_online",
            "context": context,
            "arm": arm,
        }
        for context in (None, "market_memory_v1")
        for arm in ARMS
    ]
    protocol = {
        "cells": cells,
        "dtypes": ["float32", "float64"],
        "batches": [1, 256],
        "condition_records": 128,
        "prefix_prepare_calls": 256,
        "timed_prepare_calls": 896,
        "total_prepare_calls": 1152,
        "warmup_per_record": 2,
        "repetitions_per_record": 5,
        "optimizer_steps": 0,
        "scientific_evaluation": False,
        "final_test_access": False,
        "parameter_comparison": "exact_values_names_shapes_within_context_and_dtype",
        "arm_configurations": reference_configurations(),
        "profile_overrides": {
            "hidden_size": 32,
            "seed": 42,
            "max_batch": 256,
            "head": "quantile_head_v1",
        },
    }
    manifest = {
        "benchmarks/regional_hierarchy_matrix_cuda.py": sha("producer"),
        "benchmarks/regional_program_matrix_cuda.py": sha("helper"),
        "src/mars_titan/models/titans/regional_hierarchy.py": sha("component"),
    }
    jobs, manifests = {}, {}
    for dtype, batch in STRATA:
        records = []
        for order in (0, 1):
            ordered = cells if order == 0 else list(reversed(cells))
            for position, cell in enumerate(ordered):
                arm, context = cell["arm"], cell["context"]
                scale = (10 if context else 1) * (2 if batch == 256 else 1)
                scale *= 2 if dtype == "float64" else 1
                scale *= 3 if order == 1 else 1
                if arm == "hierarchy_tree_local":
                    scale *= 2 if order == 0 else 0.5
                samples = [
                    {
                        "seconds": index * 0.001 * scale,
                        "allocated_bytes": 100,
                        "reserved_bytes": 200,
                        "peak_allocated_bytes": 140,
                        "peak_reserved_bytes": 240,
                        "free_global_vram_bytes": 10000,
                        "total_global_vram_bytes": 20000,
                        "rss_bytes": 1000 + index,
                        "process_lifetime_peak_rss_bytes": 900 + index,
                    }
                    for index in (1, 2, 3, 4, 5)
                ]
                shapes = {
                    "head.weight": [5, 32],
                    "buffer/regional_fusion.edges": [5, 5],
                }
                elements = 160
                if context:
                    shapes["session_context.output"] = [32, 32]
                    elements += 1024
                if arm != ARMS[0]:
                    shapes["buffer/regional_execution.edges"] = [8, 8]
                records.append(
                    {
                        "cell_id": cell["cell_id"],
                        "dtype": dtype,
                        "batch": batch,
                        "pass_index": order,
                        "position": position,
                        "status": "passed",
                        "financial_configuration": configuration_fixture(
                            context, arm, protocol["arm_configurations"]
                        ),
                        "samples": samples,
                        "warmup_seconds": [0.002 * scale, 0.003 * scale],
                        "min_ms": scale,
                        "median_ms": 3 * scale,
                        "max_ms": 5 * scale,
                        "rows_per_second": batch / (0.003 * scale),
                        "observed_regional_work": {
                            "local_node_evaluations": 20 * batch,
                            "summary_evaluations": (2 if arm == ARMS[0] else 10) * batch,
                        },
                        "selected_program_rows": {
                            "summary_interleaved_v1"
                            if arm == ARMS[0]
                            else "hierarchical_group_root_v1": batch
                        },
                        "observed_prefix_calls": {"core": 2, "mac": 2},
                        "mac_module_present": True,
                        "associative_update_enabled": True,
                        "context_present": context is not None,
                        "parameter_shapes": shapes,
                        "parameter_elements": elements,
                        "parameter_bytes": elements * (4 if dtype == "float32" else 8),
                        "parameter_sha256": sha((context, dtype, arm)),
                        "input_sha256": sha((dtype, batch, "inputs")),
                        "initial_state_sha256": sha((context, dtype, batch, arm, "initial")),
                        "model_state_sha256": sha((context, dtype, arm, "model")),
                        "initialization_seconds": 0.01,
                        "prefix_seconds": 0.02,
                    }
                )
        jobs[dtype, batch] = {
            "protocol": copy.deepcopy(protocol),
            "execution_stratum": {"dtype": dtype, "batch": batch, "expected_records": 32},
            "status": "passed",
            "fixture_kind": "manual_unit_fixture_not_measurement_data",
            "source_manifest_unchanged": True,
            "hold_unchanged": True,
            "hold_before": sha("hold"),
            "hold_sha256_final": sha("hold"),
            "records": records,
            "prefix_calls_completed": 64,
            "timed_calls_completed": 224,
            "cpu_proof_tensor_peak_bytes": 1024,
            "total_process_seconds": 10.0,
            "environment": {"gpu": "fixture", "python": "fixture"},
        }
        manifests[dtype, batch] = copy.deepcopy(manifest)
    return jobs, manifests


class HierarchyAggregationTests(unittest.TestCase):
    def test_a_star_label_with_tree_configuration_is_rejected(self):
        jobs, manifests = fixtures()
        rows = jobs["float32", 1]["records"]
        self.assertEqual(rows[2]["cell_id"], "none/hierarchy_star_local")
        rows[2]["financial_configuration"]["regional_execution"] = copy.deepcopy(
            rows[1]["financial_configuration"]["regional_execution"]
        )
        with self.assertRaisesRegex(AGG.ValidationError, "opciones regionales"):
            AGG.aggregate_jobs(jobs, manifests)

    def test_common_configuration_change_is_rejected_with_unchanged_weight_shapes(self):
        jobs, manifests = fixtures()
        rows = jobs["float32", 1]["records"]
        before = copy.deepcopy(rows[1]["parameter_shapes"])
        rows[1]["financial_configuration"]["dropout"] = 0.25
        self.assertEqual(rows[1]["parameter_shapes"], before)
        with self.assertRaisesRegex(AGG.ValidationError, "configuración común"):
            AGG.aggregate_jobs(jobs, manifests)

    def test_full_configuration_change_between_batches_is_rejected(self):
        jobs, manifests = fixtures()
        jobs["float32", 256]["records"][1]["financial_configuration"]["regional_execution"][
            "rounds"
        ] = 3
        with self.assertRaisesRegex(AGG.ValidationError, "configuración completa"):
            AGG.aggregate_jobs(jobs, manifests)

    def test_profile_and_actual_output_head_fields_are_checked(self):
        for field, value in (("hidden_size", 64), ("seed", 43), ("max_batch", 128)):
            with self.subTest(field=field):
                jobs, manifests = fixtures()
                jobs["float32", 1]["records"][1]["financial_configuration"][field] = value
                with self.assertRaisesRegex(AGG.ValidationError, "perfil"):
                    AGG.aggregate_jobs(jobs, manifests)
        jobs, manifests = fixtures()
        configuration = jobs["float32", 1]["records"][1]["financial_configuration"]
        configuration["head"] = "quantile_head_v1"
        del configuration["output_head"]
        with self.assertRaisesRegex(AGG.ValidationError, "output_head"):
            AGG.aggregate_jobs(jobs, manifests)

    def test_summary_rows_include_complete_configuration_and_regional_identity_hashes(
        self,
    ):
        jobs, manifests = fixtures()
        result = AGG.aggregate_jobs(jobs, manifests)
        first, second = result["records"][:2]
        self.assertEqual(
            first["common_financial_configuration_sha256"],
            second["common_financial_configuration_sha256"],
        )
        self.assertNotEqual(
            first["financial_configuration_sha256"],
            second["financial_configuration_sha256"],
        )
        self.assertRegex(first["financial_configuration_sha256"], r"^[0-9a-f]{64}$")
        execution = jobs["float32", 1]["records"][1]["financial_configuration"][
            "regional_execution"
        ]
        expected = hashlib.sha256(
            json.dumps(
                execution,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        self.assertEqual(second["regional_execution_identity_sha256"], expected)
        self.assertNotEqual(
            first["regional_execution_identity_sha256"],
            second["regional_execution_identity_sha256"],
        )

    def test_all_rows_and_orders_remain_separate_with_the_correct_denominator(self):
        jobs, manifests = fixtures()
        result = AGG.aggregate_jobs(jobs, manifests)
        self.assertEqual(
            result["checked"],
            {
                "strata": 4,
                "unique_records": 128,
                "prefix_calls": 256,
                "timed_calls": 896,
                "total_prepare_calls": 1152,
                "measured_samples": 640,
                "warmups": 256,
            },
        )
        rows = [row for row in result["records"] if row["arm"] == "hierarchy_tree_local"]
        self.assertEqual(len(rows), 16)
        for row in rows:
            expected = 2.0 if row["pass_index"] == 0 else 0.5
            self.assertAlmostEqual(row["relative_to_legacy"]["median_seconds_ratio"], expected)
            self.assertAlmostEqual(row["relative_to_legacy"]["mean_seconds_ratio"], expected)
        legacy = [
            row
            for row in result["records"]
            if row["arm"] == ARMS[0]
            and row["context"] is None
            and row["dtype"] == "float32"
            and row["batch"] == 1
        ]
        self.assertEqual(len(legacy), 2)
        self.assertAlmostEqual(legacy[0]["statistics"]["median_seconds"], 0.003)
        self.assertAlmostEqual(legacy[1]["statistics"]["median_seconds"], 0.009)
        self.assertEqual(result["completed_process_seconds_sum"], 40.0)
        self.assertFalse(result["rankings"])

    def test_zero_baseline_is_rejected_without_infinite_relative_cost(self):
        with self.assertRaisesRegex(AGG.ValidationError, "denominador"):
            AGG.relative_cost(1.0, 0.0)
        with self.assertRaises(AGG.ValidationError):
            AGG.relative_cost(1e308, 1e-308)

    def test_memory_interfaces_are_kept_distinct_even_when_rss_exceeds_lifetime_value(
        self,
    ):
        result = AGG.aggregate_jobs(*fixtures())
        memory = result["records"][0]["memory"]
        self.assertEqual(memory["rss_sampled_max_bytes"], 1005)
        self.assertEqual(memory["process_lifetime_peak_rss_max_reported_bytes"], 905)
        self.assertEqual(memory["peak_allocated_max_reported_bytes"], 140)
        self.assertEqual(memory["peak_reserved_max_reported_bytes"], 240)

    def test_duplicate_record_is_not_hidden_by_a_dictionary(self):
        jobs, manifests = fixtures()
        rows = jobs["float32", 1]["records"]
        rows[1] = copy.deepcopy(rows[0])
        with self.assertRaisesRegex(AGG.ValidationError, "duplicado"):
            AGG.aggregate_jobs(jobs, manifests)

    def test_a_missing_or_unfinished_stratum_prevents_a_complete_summary(self):
        for mode in ("missing", "incomplete"):
            with self.subTest(mode=mode):
                jobs, manifests = fixtures()
                if mode == "missing":
                    del jobs["float64", 256]
                else:
                    jobs["float64", 256]["status"] = "incomplete"
                with self.assertRaises(AGG.ValidationError):
                    AGG.aggregate_jobs(jobs, manifests)

    def test_forward_order_repeated_as_reverse_is_rejected(self):
        jobs, manifests = fixtures()
        rows = jobs["float32", 1]["records"]
        for position, row in enumerate(copy.deepcopy(rows[:16])):
            row.update(pass_index=1, position=position)
            rows[16 + position] = row
        with self.assertRaisesRegex(AGG.ValidationError, "orden"):
            AGG.aggregate_jobs(jobs, manifests)

    def test_changed_source_manifest_is_rejected(self):
        jobs, manifests = fixtures()
        manifests["float64", 256]["benchmarks/regional_hierarchy_matrix_cuda.py"] = sha("changed")
        with self.assertRaisesRegex(AGG.ValidationError, "manifiestos"):
            AGG.aggregate_jobs(jobs, manifests)

    def test_missing_measurement_or_warmup_is_rejected(self):
        for field in ("samples", "warmup_seconds"):
            with self.subTest(field=field):
                jobs, manifests = fixtures()
                jobs["float32", 1]["records"][0][field].pop()
                with self.assertRaises(AGG.ValidationError):
                    AGG.aggregate_jobs(jobs, manifests)

    def test_incorrect_call_count_is_rejected(self):
        for field in ("prefix_calls_completed", "timed_calls_completed"):
            with self.subTest(field=field):
                jobs, manifests = fixtures()
                jobs["float32", 1][field] -= 1
                with self.assertRaisesRegex(AGG.ValidationError, "llamadas"):
                    AGG.aggregate_jobs(jobs, manifests)

    def test_shape_change_is_rejected_even_with_consistent_element_and_byte_counts(
        self,
    ):
        jobs, manifests = fixtures()
        row = jobs["float32", 1]["records"][1]
        row["parameter_shapes"]["head.weight"] = [5, 31]
        row["parameter_elements"] -= 5
        row["parameter_bytes"] -= 20
        with self.assertRaisesRegex(AGG.ValidationError, "nombres o formas"):
            AGG.aggregate_jobs(jobs, manifests)

    def test_changed_full_tensor_hash_for_the_same_configuration_is_rejected(self):
        jobs, manifests = fixtures()
        jobs["float32", 256]["records"][1]["parameter_sha256"] = sha("different values")
        with self.assertRaisesRegex(AGG.ValidationError, "huella"):
            AGG.aggregate_jobs(jobs, manifests)

    def test_reported_median_cannot_replace_the_measured_samples(self):
        jobs, manifests = fixtures()
        jobs["float32", 1]["records"][0]["median_ms"] *= 0.5
        with self.assertRaisesRegex(AGG.ValidationError, "median_ms"):
            AGG.aggregate_jobs(jobs, manifests)

    def test_writers_preserve_128_rows_and_both_orders_without_overwriting(self):
        result = AGG.aggregate_jobs(*fixtures())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "summary"
            AGG.write_reports(result, output)
            with (output / "conditions.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 128)
            self.assertEqual({row["pass_index"] for row in rows}, {"0", "1"})
            self.assertEqual(
                rows[0]["financial_configuration_sha256"],
                result["records"][0]["financial_configuration_sha256"],
            )
            self.assertEqual(
                rows[1]["regional_execution_identity_sha256"],
                result["records"][1]["regional_execution_identity_sha256"],
            )
            before = (output / "summary.json").read_bytes()
            with self.assertRaises(AGG.ValidationError):
                AGG.write_reports(result, output)
            self.assertEqual((output / "summary.json").read_bytes(), before)

    def test_cli_rejects_incomplete_inputs_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input-dir",
                    str(root),
                    "--output",
                    str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
                timeout=10,
            )
            self.assertEqual(result.returncode, 2)
            self.assertFalse(output.exists())

    def test_json_reader_rejects_duplicate_keys_and_nonfinite_constants(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            for text in (
                '{"status":"passed","status":"incomplete"}',
                '{"seconds":NaN}',
            ):
                with self.subTest(text=text):
                    path.write_text(text)
                    with self.assertRaises(AGG.ValidationError):
                        AGG.read_json(path)


if __name__ == "__main__":
    unittest.main()
