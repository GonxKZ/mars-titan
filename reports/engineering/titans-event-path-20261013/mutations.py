"""Mutaciones dirigidas de #549. Cada una debe hacer fallar alguna prueba.

Uso: mutations.py ÁRBOL SALIDA.json [cpu|cuda]. Con cpu se ocultan las GPU, con cuda solo se
ejecutan las mutaciones que necesitan el dispositivo. Cada mutación cambia un único fragmento
del árbol, ejecuta sus pruebas con pytest del mismo intérprete, `-x` y sin orden aleatorio, y
restaura el archivo. Las pruebas del lector necesitan MARS_TITAN_EPISODIC_NATIVE.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

W, OUT, MODE = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
INPUTS = "src/mars_titan/models/titans/financial_inputs.py"
STATE = "src/mars_titan/models/titans/state.py"
FINANCIAL = "src/mars_titan/models/titans/financial.py"
CONFIG = "src/mars_titan/models/titans/config.py"
READOUT = "src/mars_titan/models/titans/episodic_readout.py"
RUN = "src/mars_titan/training/financial_run.py"
READER = "src/mars_titan/training/mars_titan_run.py"
ADAPTER = ["tests/models/titans/test_financial_adapter.py"]
CHECKS = ["tests/models/titans/test_deferred_checks.py"]
READOUT_TESTS = ["tests/models/titans/test_episodic_readout.py"]
cpu = [
    (
        "verify_without_identity",
        INPUTS,
        "value is not kept or _buffer(value) is None",
        "_buffer(value) is None",
        ADAPTER,
    ),
    ("verify_without_digest", INPUTS, "if digest != self.input_digest or any(", "if any(", ADAPTER),
    (
        "digest_without_presence",
        INPUTS,
        "        return (*(self.inputs[name] for name in MODALITIES), self.presence)",
        "        return tuple(self.inputs[name] for name in MODALITIES)",
        ADAPTER,
    ),
    (
        "inputs_not_frozen",
        INPUTS,
        "        copied[name] = _frozen(value)",
        "        copied[name] = value.copy()",
        ADAPTER,
    ),
    (
        "identity_without_suffix",
        INPUTS,
        "if not identifier.endswith(suffix) or FLOW_ID.fullmatch(flow) is None:",
        "if FLOW_ID.fullmatch(flow) is None:",
        ADAPTER,
    ),
    (
        "identity_availability_lower",
        INPUTS,
        "if min(available) < 0 or max(available) > at:",
        "if max(available) > at:",
        ADAPTER,
    ),
    (
        "identity_cut_range",
        INPUTS,
        "if len(set(moments)) != 1 or not lower <= at < FINAL_TEST_US:",
        "if len(set(moments)) != 1:",
        ADAPTER,
    ),
    (
        "indicator_l2_norm",
        STATE,
        'torch.linalg.vector_norm(value.detach(), float("inf"))',
        "torch.linalg.vector_norm(value.detach(), 2)",
        CHECKS,
    ),
    (
        "indicator_complex_norm",
        STATE,
        "if value.is_floating_point() and value.numel():",
        "if (value.is_floating_point() or value.is_complex()) and value.numel():",
        CHECKS,
    ),
    (
        "indicator_keeps_graph",
        STATE,
        "torch.linalg.vector_norm(value.detach(),",
        "torch.linalg.vector_norm(value,",
        CHECKS,
    ),
    (
        "observe_without_event_scope",
        RUN,
        "        with deferred_checks():\n            for cpu in validated:",
        "        if True:\n            for cpu in validated:",
        [
            "tests/training/test_financial_run.py::test_each_event_resolves_the_checks_of_all_its_blocks_together"
        ],
    ),
    (
        "replay_without_group_scope",
        RUN,
        "            with deferred_checks():\n                members, graphs, penalties",
        "            if True:\n                members, graphs, penalties",
        [
            "tests/training/test_financial_run.py::test_replay_resolves_the_checks_of_each_flow_group_together"
        ],
    ),
    (
        "reader_without_event_scope",
        READER,
        "        with deferred_checks():\n            for cpu in validated:",
        "        if True:\n            for cpu in validated:",
        [
            "tests/training/test_mars_titan_run.py::test_each_event_resolves_core_and_reader_checks_together"
        ],
    ),
    (
        "zero_fill_on_present_rows",
        READOUT,
        "~((read.values != 0) & ~read.presence).any()",
        "~((read.values != 0) & read.presence).any()",
        READOUT_TESTS,
    ),
    (
        "encoder_message_swapped",
        FINANCIAL,
        "            for value, message in encoder_checks:",
        "            for value, message in reversed(encoder_checks):",
        ADAPTER,
    ),
    (
        "config_id_ignores_mode",
        FINANCIAL,
        "cached = self._config_ids.get(self.training)",
        "cached = self._config_ids.get(True)",
        ADAPTER,
    ),
    (
        "seal_keeps_config_ids",
        FINANCIAL,
        "        self._config_ids = {}\n",
        "        self._config_ids = getattr(self, '_config_ids', {})\n",
        ADAPTER,
    ),
    ("fingerprint_serialized", CONFIG, '        state.pop("_fingerprint", None)\n', "", ADAPTER),
]
cuda = [
    (
        "resolve_keeps_flags",
        STATE,
        "        message = self.first_failure()\n        self.flags, self.messages = [], []\n",
        "        message = self.first_failure()\n",
        CHECKS,
    ),
    (
        "first_failure_discards",
        STATE,
        "            if not ok:\n                return message\n",
        "            if not ok:\n                self.flags = []\n                return message\n",
        CHECKS,
    ),
    (
        "resolve_ignores_bounds",
        STATE,
        "parts.append(stacked if dtype == torch.bool else torch.isfinite(stacked))",
        "parts.append(stacked if dtype == torch.bool else "
        "torch.ones_like(stacked, dtype=torch.bool))",
        CHECKS,
    ),
    (
        "nested_scope_keeps_seen",
        STATE,
        "        seen, outer.seen = outer.seen, {}\n",
        "        seen = outer.seen\n",
        CHECKS,
    ),
    (
        "encoder_checks_immediate",
        FINANCIAL,
        "                require_finite(value, message)\n",
        "                torch.isfinite(value).all().item()\n"
        "                require_finite(value, message)\n",
        ADAPTER,
    ),
    (
        "readout_without_scope",
        READOUT,
        "        with deferred_checks():\n            return self._forward(",
        "        if True:\n            return self._forward(",
        READOUT_TESTS,
    ),
]
mutants = cpu if MODE == "cpu" else cuda
environment = dict(
    os.environ,
    PYTHONPATH=str(W / "src"),
    CUBLAS_WORKSPACE_CONFIG=":4096:8",
    OMP_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
)
if MODE == "cpu":
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
results = []
for name, relative, old, new, tests in mutants:
    path = W / relative
    original = path.read_text()
    assert original.count(old) == 1, name
    path.write_text(original.replace(old, new))
    try:
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "-x",
                "-p",
                "no:randomly",
                *tests,
            ],
            capture_output=True,
            text=True,
            env=environment,
            timeout=1800,
            cwd=W,
        )
    finally:
        path.write_text(original)
    failed = [line for line in run.stdout.splitlines() if line.startswith("FAILED")]
    status = "detectada" if run.returncode else "SOBREVIVE"
    results.append(
        dict(
            name=name,
            file=relative,
            tests=tests,
            status=status,
            failed=failed[:2],
            summary=run.stdout.strip().splitlines()[-1:],
        )
    )
    print(name, status, failed[:1] or run.stdout.strip().splitlines()[-1:], flush=True)
OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
