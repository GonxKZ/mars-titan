"""La etapa de políticas de la campaña solo alcanza cintas reales de la edición.

Los mundos sintéticos de experimentos anteriores (episodios, escenarios de adaptación, la
preparación del postentrenamiento emparejado y `MarketTape.from_world`) se conservan por
trazabilidad, pero ninguna ruta de `run_masked_campaign.py rl` llega a ellos. Estas pruebas
lo comprueban sobre el código y sobre la orden, sin leer datos ni aprender.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

from mars_titan.simulation import campaign_stage
from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.native_runtime import NativeLibrary
from tests.simulation.native_library import requires_native_library

ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "src"
ENTRIES = ("mars_titan.simulation.campaign_stage", "mars_titan.simulation.stage_report")
SYNTHETIC = (
    "mars_titan.episodes",
    "mars_titan.simulation.adaptation_scenarios",
    "mars_titan.posttraining.preparation",
)
GENERATORS = {
    "from_world",
    "generate_world",
    "generate_scenario",
    "prepare_adaptation_scenarios",
    "EncodedWorld",
    "write_world",
}


def _path(name):
    base = SOURCE.joinpath(*name.split("."))
    return next((p for p in (base.with_suffix(".py"), base / "__init__.py") if p.is_file()), None)


def _imports(name):
    """Módulos del proyecto que importa un módulo, también dentro de funciones."""
    path = _path(name)
    package = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
    result = set()
    for item in found:
        parts = item.split(".")
        while parts and _path(".".join(parts)) is None:
            parts.pop()
        if parts and parts[0] == "mars_titan":
            result.add(".".join(parts))
    return result


def closure(entries):
    seen, pending = set(), list(entries)
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        parts = name.split(".")
        pending.extend(".".join(parts[:i]) for i in range(1, len(parts)))
        pending.extend(_imports(name) - seen)
    return seen


def _names(name):
    tree = ast.parse(_path(name).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.alias):
            names.add(node.name.rsplit(".", 1)[-1])
    return names


def test_the_stage_code_never_reaches_a_synthetic_world():
    modules = closure(ENTRIES)
    # La cinta se define en market.py, junto a su constructor desde mundos sintéticos.
    assert "mars_titan.simulation.market" in modules
    assert sorted(m for m in modules if m.startswith(SYNTHETIC)) == []
    used = {
        name: sorted(_names(name) & GENERATORS)
        for name in modules
        if name != "mars_titan.simulation.market" and _names(name) & GENERATORS
    }
    assert used == {}


def test_the_rl_command_imports_only_its_stage(tmp_path):
    code = (
        "import json, runpy, sys\n"
        "script = runpy.run_path('scripts/run_masked_campaign.py', run_name='script')\n"
        "status = script['main'](['rl', 'check', '--stage', sys.argv[1]])\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('mars_titan'))\n"
        "print('MODULES=' + json.dumps(dict(status=status, loaded=loaded)))\n"
    )
    environment = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(SOURCE), str(ROOT)]),
        CUDA_VISIBLE_DEVICES="-1",
        # Sin binarios declarados la comprobación informa de capacidades ausentes.
        MARS_TITAN_PPO_EXECUTABLE=str(tmp_path / "missing-ppo"),
        MARS_TITAN_KLPO_EXECUTABLE=str(tmp_path / "missing-klpo"),
    )
    result = subprocess.run(
        [sys.executable, "-c", code, "configs/simulation/historical-masked-rl-stage-a.json"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    line = next(line for line in result.stdout.splitlines() if line.startswith("MODULES="))
    report = json.loads(line.removeprefix("MODULES="))
    assert report["status"] == 0
    loaded = report["loaded"]
    assert "mars_titan.simulation.campaign_stage" in loaded
    assert [m for m in loaded if m.startswith(SYNTHETIC)] == []
    # La orden no carga las demás etapas, que sí alcanzan código de mundos sintéticos. El
    # contrato de la cadena está en `training.campaign_chain`, así que no hace falta ningún
    # módulo de posentrenamiento.
    assert [m for m in loaded if m.startswith("mars_titan.posttraining")] == []


@requires_native_library
def test_the_cn_rules_probe_asks_the_library_without_building_a_tape(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("La consulta de reglas construyó una cinta")

    monkeypatch.setattr(MarketTape, "__init__", forbidden)
    probed = campaign_stage.probe_capabilities()["native_cn_a_share_rules"]
    assert probed == dict(available=True, reason=None)


@requires_native_library
def test_a_library_with_other_daily_limits_lacks_the_cn_rules(monkeypatch):
    original = NativeLibrary.price_limits

    def shifted(self, reference, band):
        limits = original(self, reference, band)
        return None if limits is None else (limits[0] + 0.01, limits[1])

    monkeypatch.setattr(NativeLibrary, "price_limits", shifted)
    probed = campaign_stage.probe_capabilities()["native_cn_a_share_rules"]
    assert probed["available"] is False and "no coinciden" in probed["reason"]
