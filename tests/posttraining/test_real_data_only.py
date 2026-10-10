"""La etapa de adaptadores de la campaña solo aprende con datos reales de la edición.

Se comprueba de tres formas. Ninguna importación del programa de la campaña, en ningún
ámbito ni función, llega a la cola, la preparación, el aumento, los mundos sintéticos o los
episodios remuestreados del postentrenamiento emparejado anterior. Las órdenes de la etapa
se ejecutan en otro proceso con esos módulos bloqueados. Y la etapa rechaza cualquier
declaración de condiciones, aumento o mundos. Los recorridos completos de la etapa en
`test_adapter_campaign_stage` y `test_chronological_stage` usan también el bloqueo.
"""

import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mars_titan.data.storage import atomic_json
from mars_titan.environments.cohort_order import Visit, real_order
from mars_titan.episodes.augmentation import training_visits
from mars_titan.posttraining import campaign_stage
from mars_titan.posttraining.augmented_inputs import AugmentedInputs
from mars_titan.posttraining.inputs import PairedInputs
from tests.posttraining.real_only import BLOCKED

ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "src"
COMMAND = ROOT / "scripts" / "run_masked_campaign.py"
CONFIGS = ROOT / "configs" / "posttraining"
# Ejecutores que la etapa importa dentro de sus funciones.
LAZY = (
    "mars_titan.posttraining.campaign_stage",
    "mars_titan.posttraining.matrix_runs",
    "mars_titan.posttraining.chronological_windows",
    "mars_titan.posttraining.readout_adapters",
    "mars_titan.posttraining.candidate_adapters",
    "mars_titan.environments.view_cohorts",
)


def _module_path(name):
    base = SOURCE.joinpath(*name.split("."))
    for path in (base / "__init__.py", base.with_suffix(".py")):
        if path.is_file():
            return path
    return None


def _imports(path, package):
    """Módulos del proyecto que importa un archivo en cualquier ámbito, también por nombre."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                parts = package.split(".")
                module = ".".join([*parts[: len(parts) - node.level + 1], *filter(None, [module])])
            found |= {module, *(f"{module}.{alias.name}" for alias in node.names)}
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Nombres que `importlib.import_module` puede recibir.
            found.add(node.value)
    result = set()
    for name in found:
        parts = name.split(".")
        result |= {
            ".".join(parts[:i])
            for i in range(2, len(parts) + 1)
            if parts[0] == "mars_titan" and _module_path(".".join(parts[:i]))
        }
    return result


def _closure(path):
    reached, pending = set(), [(path, "")]
    while pending:
        current, package = pending.pop()
        for name in _imports(current, package):
            if name not in reached:
                reached.add(name)
                module = _module_path(name)
                package = name if module.name == "__init__.py" else name.rpartition(".")[0]
                pending.append((module, package))
    return reached


def test_no_import_of_the_campaign_command_reaches_the_previous_paired_design():
    reached = _closure(COMMAND)
    # El análisis sigue también las importaciones dentro de funciones.
    assert set(LAZY) <= reached
    assert not reached & set(BLOCKED), sorted(reached & set(BLOCKED))


@pytest.mark.parametrize("variant", ["a", "b"])
def test_stage_commands_run_with_the_previous_design_blocked(variant):
    program = f"""
import importlib.abc, json, runpy, sys
BLOCKED = {BLOCKED!r}
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name in BLOCKED:
            raise ImportError(name)
sys.meta_path.insert(0, Finder())
sys.argv = ["run_masked_campaign.py", "posttraining", "check", "--stage", sys.argv[1]]
try:
    runpy.run_path({str(COMMAND)!r}, run_name="__main__")
except SystemExit as exit:
    assert exit.code in (None, 0), exit.code
loaded = sorted(name for name in BLOCKED if name in sys.modules)
print(json.dumps(dict(loaded=loaded)))
"""
    stage = CONFIGS / f"historical-masked-adapter-stage-{variant}.json"
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SOURCE), str(ROOT)]))
    result = subprocess.run(
        [sys.executable, "-c", program, str(stage)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    lines = result.stdout.strip().splitlines()
    assert json.loads(lines[-1]) == dict(loaded=[])
    assert '"status": "checked"' in result.stdout


def stage_with(tmp_path, stage_change=None, matrix_change=None):
    matrix = json.loads((CONFIGS / "adapter-matrix-v2.json").read_text())
    if matrix_change:
        matrix_change(matrix)
    atomic_json(tmp_path / "matrix.json", matrix)
    value = json.loads((CONFIGS / "historical-masked-adapter-stage-a.json").read_text())
    value.update(
        campaign=str((ROOT / "configs/baselines/historical-masked-campaign-a.json").resolve()),
        matrix=str(tmp_path / "matrix.json"),
    )
    if stage_change:
        stage_change(value)
    atomic_json(tmp_path / "stage.json", value)
    return tmp_path / "stage.json"


OTHER_DATA = {
    "synthetic_condition": dict(stage=lambda v: v.update(condition="real_synthetic")),
    "resampled_conditions": dict(stage=lambda v: v.update(conditions=["real", "real_resampled"])),
    "augmentation": dict(stage=lambda v: v.update(augmentation=dict(fraction=0.25))),
    "worlds": dict(stage=lambda v: v.update(worlds=dict(volatility=0.006))),
    "nested_value": dict(stage=lambda v: v["cohort_reading"].update(source="real_resampled")),
    "matrix_condition": dict(matrix=lambda m: m["budget"].update(condition="real_synthetic")),
    "matrix_episodes": dict(matrix=lambda m: m.update(episodes=dict(decisions=16))),
}


def test_the_repository_stage_declares_only_real_data(tmp_path):
    assert campaign_stage.check_stage(stage_with(tmp_path))["status"] == "checked"


@pytest.mark.parametrize("name", sorted(OTHER_DATA))
def test_stage_rejects_conditions_augmentation_and_worlds(tmp_path, name):
    change = OTHER_DATA[name]
    path = stage_with(tmp_path, change.get("stage"), change.get("matrix"))
    with pytest.raises(ValueError, match="solo aprende con datos reales"):
        campaign_stage.load_stage(path)


def test_stage_rejects_a_campaign_outside_the_masked_edition(tmp_path, monkeypatch):
    original = campaign_stage.load_campaign
    monkeypatch.setattr(
        campaign_stage,
        "load_campaign",
        lambda path: dict(original(path), input_policy="strict_complete_inputs"),
    )
    with pytest.raises(ValueError, match="solo aprende con datos reales"):
        campaign_stage.load_stage(stage_with(tmp_path))


def test_stage_identity_binds_the_verified_edition_of_every_market(tmp_path):
    stage = campaign_stage.load_stage(stage_with(tmp_path))
    edition = {"US": "a" * 64, "CN": "b" * 64}
    views = {
        "US": dict(edition={"US": "a" * 64}, windows={}),
        "CN": dict(edition={"CN": "b" * 64}, windows={}),
        "US+CN": dict(edition=edition, windows={}),
    }
    identity = campaign_stage._identity(stage, views)
    assert identity["training_data"] == "real_walk_forward_only"
    assert identity["editions"]["US+CN"] == edition
    for broken in ({"US": "a" * 64}, {"US": "a" * 64, "CN": "short"}, None):
        with pytest.raises(ValueError, match="edición verificada"):
            campaign_stage.real_views(stage, dict(views, **{"US+CN": dict(edition=broken)}))


def test_real_inputs_accept_no_episodes_and_no_other_condition():
    assert list(inspect.signature(PairedInputs).parameters) == ["train", "validation", "parent"]
    assert issubclass(AugmentedInputs, PairedInputs)
    data = object.__new__(PairedInputs)
    data.masked, data.train = False, SimpleNamespace(partition="train")
    for condition in ("real_resampled", "real_synthetic"):
        with pytest.raises(ValueError, match="aumento confirmado"):
            data._visits("train", condition, 0, 0)
    with pytest.raises(ValueError, match="episodios adicionales"):
        data._episode(None, 0, "real_resampled")


class _Source:
    partition, manifest_sha256 = "train", "c" * 64

    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count


@pytest.mark.parametrize(("count", "epoch", "seed"), [(1, 0, 0), (7, 3, 42), (513, 4, 44)])
def test_real_order_keeps_the_epoch_permutation_of_the_previous_design(count, epoch, seed):
    # La expresión que usaba `training_visits` antes de separar el orden real.
    rng = np.random.default_rng([seed, epoch])
    expected = [Visit("real", -1, int(i), True) for i in rng.permutation(count)]
    assert real_order(np.random.default_rng([seed, epoch]), count) == expected
    assert sorted(visit.cohort for visit in expected) == list(range(count))
    # El aumento sigue con el mismo generador después de las cohortes reales.
    windows = [SimpleNamespace(source_sha256="c" * 64, decision_start=0, stop=1)] * 3
    visits = training_visits(_Source(count), windows, epoch=epoch, seed=seed)
    order = rng.permutation(len(windows))
    assert visits[:count] == expected
    assert [visit.episode for visit in visits[count:]] == [int(i) for i in order]
