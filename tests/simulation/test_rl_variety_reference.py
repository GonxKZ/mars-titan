"""La referencia FP64 de los objetivos de grupo y de QR-DQN coincide con el archivo versionado.

La prueba C++ `rl_variety` compara el motor nativo con
`native/tests/fixtures/rl_variety_reference.json`.
Este archivo debe salir siempre del script de referencia, nunca de una edición manual, y las
comprobaciones de aquí fijan algunas propiedades de las ecuaciones que no dependen del motor.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "native" / "tests" / "rl_variety_reference.py"
FIXTURE = ROOT / "native" / "tests" / "fixtures" / "rl_variety_reference.json"


def _generate(tmp_path):
    output = tmp_path / "reference.json"
    subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--output", str(output)], check=True, timeout=120
    )
    return output


def test_fixture_is_reproduced_by_the_reference_script(tmp_path):
    assert _generate(tmp_path).read_bytes() == FIXTURE.read_bytes()


def test_group_advantages_follow_their_source():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cases = {case["name"]: case for case in document["group"]}
    for name, case in cases.items():
        groups, returns = case["groups"], case["returns"]
        for label in set(groups):
            members = [i for i, group in enumerate(groups) if group == label]
            values = [case["advantages"][i] for i in members]
            # Toda ventaja de grupo está centrada en su grupo.
            assert abs(sum(values)) < 1e-12
            if name == "dr_grpo":
                mean = sum(returns[i] for i in members) / len(members)
                expected = [returns[i] - mean for i in members]
                assert all(abs(a - b) < 1e-15 for a, b in zip(values, expected, strict=True))
            else:
                # Normalizadas con la desviación muestral, salvo el suelo de 1e-6.
                spread = math.sqrt(sum(v * v for v in values) / (len(values) - 1))
                assert abs(spread - 1) < 1e-4


def test_aggregation_weights_match_each_identity():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for case in document["group"]:
        lengths, weights = case["lengths"], case["weights"]
        episodes, total = len(lengths), sum(lengths)
        expected = {
            "grpo": [1 / (episodes * n) for n in lengths],
            "dr_grpo": [1 / (episodes * 256)] * episodes,
            "dapo": [1 / total] * episodes,
            "gspo": [1 / episodes] * episodes,
        }[case["config"]["kind"]]
        assert weights == expected


def test_quantile_targets_do_not_bootstrap_after_a_termination():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for case in document["quantile"]:
        for row, terminated in enumerate(case["terminated"]):
            if terminated:
                assert case["targets"][row] == [case["rewards"][row]] * case["quantiles"]
