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


def write(root, files):
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)


TABLE = "docs/engineering/environment-variables.md"
# Los nombres de prueba se forman con el prefijo para que el verificador del propio repositorio
# no los tome por variables que lee este archivo.
PREFIX = "MARS_TITAN_"


def test_variable_read_by_code_must_be_in_the_single_table(public_repository, capsys):
    checker, root = public_repository
    write(root, {"src/pkg/run.py": f'os.environ.get("{PREFIX}CACHE_MIB")\n'})
    assert checker.main() == 1
    assert f"Falta {TABLE}" in capsys.readouterr().out
    write(root, {TABLE: f"| `{PREFIX}OTHER_FLAG` | Otra |\n"})
    assert checker.main() == 1
    output = capsys.readouterr().out
    assert f"Variable de entorno sin documentar en {TABLE}: {PREFIX}CACHE_MIB" in output
    assert f"Variable documentada que ningún código usa: {PREFIX}OTHER_FLAG" in output
    write(root, {TABLE: f"| `{PREFIX}CACHE_MIB` | Caché |\n"})
    assert checker.main() == 0


def test_cmake_options_c_macros_and_formatted_prefixes_are_not_runtime_variables(
    public_repository,
):
    checker, root = public_repository
    write(
        root,
        {
            "native/CMakeLists.txt": f'option({PREFIX}BUILD_EXTRA "Extra" OFF)\n',
            "native/src/run.cpp": (
                f"#ifdef {PREFIX}HAS_EXTRA\n"
                f'auto value = std::getenv("{PREFIX}RUNTIME_PATH");\n'
                "#endif\n"
            ),
            "tests/support.py": (
                'name = f"MARS_TITAN_CHECK_{suffix}"\n' + f'flag = "{PREFIX}BUILD_EXTRA"\n'
            ),
        },
    )
    assert checker.code_variables(root, checker.versioned_files(root)) == {f"{PREFIX}RUNTIME_PATH"}


def test_ignored_build_artifacts_do_not_define_variables(public_repository):
    checker, root = public_repository
    write(
        root,
        {
            ".gitignore": "native/build/\n",
            "native/build/generated.py": f'os.environ["{PREFIX}GENERATED"]\n',
        },
    )
    # rg solo respeta .gitignore dentro de un repositorio de Git.
    (root / ".git").mkdir()
    assert checker.code_variables(root, checker.versioned_files(root)) == set()


def test_documented_commands_must_exist(public_repository, capsys):
    checker, root = public_repository
    write(
        root,
        {
            "Makefile": "check: lint\nlint:\n\truff check .\n",
            "native/CMakePresets.json": json.dumps({"configurePresets": [{"name": "release"}]}),
            "scripts/run.py": "",
            "src/mars_titan/tool.py": "",
            "src/mars_titan/service/__main__.py": "",
            "CONTRIBUTING.md": (
                "`make check`, `make lint` y make deploy en prosa no se comprueba.\n\n"
                "```bash\nuv run python scripts/run.py\ncmake --preset release\n"
                "python -m mars_titan.tool\npython -m mars_titan.service\n```\n"
            ),
        },
    )
    assert checker.command_problems(root) == []
    write(
        root,
        {
            "CONTRIBUTING.md": (
                "`make test-all` y `uv run python scripts/missing.py`.\n\n"
                "```bash\ncmake --preset debug\npython -m mars_titan.absent\n```\n"
            )
        },
    )
    assert checker.main() == 1
    output = capsys.readouterr().out
    for problem in (
        "Objetivo de make inexistente en CONTRIBUTING.md: test-all",
        "Guion inexistente en CONTRIBUTING.md: scripts/missing.py",
        "Preset de CMake inexistente en CONTRIBUTING.md: debug",
        "Módulo inexistente en CONTRIBUTING.md: mars_titan.absent",
    ):
        assert problem in output


@pytest.mark.parametrize(
    ("name", "text", "term"),
    [
        ("docs/campaign.md", "Cada brazo ve las mismas filas.\n", "brazo"),
        ("docs/campaign.md", "Los Brazos comparten cortes.\n", "Brazos"),
        ("src/bound.py", '"""El oráculo conoce el futuro."""\n', "oráculo"),
        ("src/bound.py", "# Oraculos sin tilde\n", "Oraculos"),
        ("tests/test_rl.py", '"""Políticas guionizadas."""\n', "guionizadas"),
        ("site/view.mjs", "// canarios en el panel\n", "canarios"),
    ],
)
def test_calques_in_prose_and_code_fail_validation(public_repository, capsys, name, text, term):
    checker, root = public_repository
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    assert checker.main() == 1
    assert f"Término «{term}» en {name}:1" in capsys.readouterr().out


def test_declared_exceptions_and_english_identifiers_are_accepted(public_repository):
    checker, root = public_repository
    for name, text in {
        "reports/engineering/receipt.json": '{"nota": "brazo conservado como evidencia"}\n',
        "thesis/revisions/main-20261009.tex": "Versión entregada con un oráculo.\n",
        "configs/titans/chronological-training.json": '{"nota": "todos los brazos"}\n',
        "src/plan.py": "arms = {arm: oracle for arm, oracle in pairs}\n",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    assert checker.main() == 0
