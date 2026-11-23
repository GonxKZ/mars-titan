"""Pruebas TDD para el cuaderno .ipynb de arranque (nbformat 4) y su validador."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import notebook_validator as nv
import pytest

REPO_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
PACKAGE_DIR = REPO_SCRIPTS / "colab_rtx6000"
NOTEBOOK_PATH = PACKAGE_DIR / "colab_rtx6000_bootstrap.ipynb"


def test_bootstrap_notebook_file_exists_is_valid_nbformat4_and_passes_contract():
    assert NOTEBOOK_PATH.is_file(), f"Falta el cuaderno local {NOTEBOOK_PATH}"
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert nv.validate_nbformat4_schema(nb) == []

    report = nv.validate_notebook_contract(nb, repo_scripts_dir=REPO_SCRIPTS, addon_dir=PACKAGE_DIR)
    assert report["valid"] is True, report["errors"]
    assert report["errors"] == []

    # Comprobar que invoca los scripts reales del kit repo-v11 y los nuevos complementos
    invoked = set(report["invoked_scripts"])
    expected_kit = {
        "colab_bootstrap.sh",
        "colab_site.py",
        "colab_origin.sh",
        "colab_sync.py",
        "colab_parity.py",
        "kernel_sm120_plan.py",
        "mps_probe.py",
        "concurrency_probe.py",
        "cu_budget_guard.py",
    }
    assert expected_kit <= invoked

    # Orden estricto de fases
    phases = report["phase_indices"]
    ordered = [
        phases["bootstrap"],
        phases["fp32_parity"],
        phases["mps_and_probe"],
        phases["origin_and_sync"],
        phases["cu_guard"],
        phases["safe_shutdown"],
    ]
    assert ordered == sorted(ordered) and len(set(ordered)) == len(ordered)


def test_validator_rejects_bad_nbformat_schema():
    nb = nv.build_bootstrap_notebook()
    bad_ver = copy.deepcopy(nb)
    bad_ver["nbformat"] = 3
    assert nv.validate_nbformat4_schema(bad_ver)

    bad_cell = copy.deepcopy(nb)
    bad_cell["cells"][0]["cell_type"] = "raw_invalid"
    assert nv.validate_nbformat4_schema(bad_cell)

    dirty_exec = copy.deepcopy(nb)
    for cell in dirty_exec["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = 7
            break
    assert nv.validate_nbformat4_schema(dirty_exec)


@pytest.mark.parametrize(
    "bad_snippet",
    [
        "torch.backends.cuda.matmul.allow_tf32 = True\n",
        "with torch.autocast('cuda', dtype=torch.bfloat16):\n    pass\n",
        "torch.use_deterministic_algorithms(False)\n",
        "ORIGIN = 2024\n",
        "FOLD = 'fold-019'\n",
        "rclone copy /content gdrive:bucket\n",
        "bash $K/colab_inexistente_999.sh paso\n",
    ],
)
def test_validator_rejects_fp32_violations_2024_uploads_and_missing_scripts(bad_snippet):
    nb = nv.build_bootstrap_notebook()
    tampered = copy.deepcopy(nb)
    tampered["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "tampered-cell",
            "metadata": {},
            "outputs": [],
            "source": [bad_snippet],
        }
    )
    report = nv.validate_notebook_contract(
        tampered, repo_scripts_dir=REPO_SCRIPTS, addon_dir=PACKAGE_DIR
    )
    assert report["valid"] is False
    assert report["errors"]


def test_validator_rejects_out_of_order_phases():
    nb = nv.build_bootstrap_notebook()
    swapped = copy.deepcopy(nb)
    # Mover la celda de bootstrap después del cierre seguro
    code_indices = [i for i, c in enumerate(swapped["cells"]) if c["cell_type"] == "code"]
    bootstrap_idx = next(
        i for i in code_indices if "colab_bootstrap.sh" in "".join(swapped["cells"][i]["source"])
    )
    cell = swapped["cells"].pop(bootstrap_idx)
    swapped["cells"].append(cell)
    report = nv.validate_notebook_contract(
        swapped, repo_scripts_dir=REPO_SCRIPTS, addon_dir=PACKAGE_DIR
    )
    assert report["valid"] is False
    assert any("orden" in e.lower() for e in report["errors"])


@pytest.mark.parametrize(
    "bad_extra",
    [
        "optimizer.step()\n",
        "loss.backward()\n",
        "torch.backends.cudnn.benchmark = True\n",
        "torch.set_float32_matmul_precision('high')\n",
        "git push origin main\n",
    ],
)
def test_pass3_adversarial_inline_training_cudnn_benchmark_and_missing_required_guards(bad_extra):
    nb = nv.build_bootstrap_notebook()
    tampered = copy.deepcopy(nb)
    tampered["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "adv-extra",
            "metadata": {},
            "outputs": [],
            "source": [bad_extra],
        }
    )
    report = nv.validate_notebook_contract(
        tampered, repo_scripts_dir=REPO_SCRIPTS, addon_dir=PACKAGE_DIR
    )
    assert report["valid"] is False

    # Quitar cada una de las 4 guardas obligatorias FP32/determinismo debe invalidar el cuaderno
    for needle in (
        "CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "TRITON_F32_DEFAULT=ieee",
        "allow_tf32 = False",
        "use_deterministic_algorithms(True",
    ):
        stripped = copy.deepcopy(nb)
        for cell in stripped["cells"]:
            if cell["cell_type"] == "code":
                cell["source"] = [line.replace(needle, "REMOVED_GUARD") for line in cell["source"]]
        rep = nv.validate_notebook_contract(
            stripped, repo_scripts_dir=REPO_SCRIPTS, addon_dir=PACKAGE_DIR
        )
        assert rep["valid"] is False, f"Debió fallar al eliminar guarda {needle}"
        assert any("guarda obligatoria" in e.lower() for e in rep["errors"])
