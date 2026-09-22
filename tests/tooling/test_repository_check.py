"""El verificador debe funcionar con los archivos publicados de un clon limpio."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def public_repository(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    checker = importlib.import_module("scripts.check_repository")
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    for name, value in {
        "README.md": "# Proyecto de prueba\n",
        "LICENSE": "Licencia de la prueba\n",
        "CITATION.cff": "title: Proyecto de prueba\n",
        "uv.lock": "version = 1\n",
        "docs/references/catalogs.json": "[]\n",
        ".github/planning/issues.json": json.dumps({"issues": [], "labels": [], "milestones": []}),
        "data/catalogs/macro-indicators.csv": "id,input_ids,kind,formula,verification_status\n",
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    return checker, tmp_path


def test_public_clone_does_not_require_external_documents(public_repository):
    checker, _ = public_repository
    assert checker.main() == 0


def test_missing_local_link_still_fails_validation(public_repository, capsys):
    checker, root = public_repository
    (root / "README.md").write_text("[Referencia](missing.md)\n")
    assert checker.main() == 1
    assert "Enlace local inexistente" in capsys.readouterr().out


def test_existing_local_links_and_external_urls_are_accepted(public_repository):
    checker, root = public_repository
    (root / "README.md").write_text(
        "[Cita](CITATION.cff) y [fuente](https://example.org/research).\n"
    )
    assert checker.main() == 0
