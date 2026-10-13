"""Clasificación de las diferencias entre dos compilaciones nativas medidas."""

import importlib.util
import json
import shutil
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "benchmarks/native_build_variants.py"
SPECIFICATION = importlib.util.spec_from_file_location("native_build_variants", PATH)
variants = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(variants)

DIGESTS = ("a" * 64, "b" * 64)


def compare(tmp_path, objects, linked=None):
    """Comparar dos directorios con los objetos y salidas dados, cada uno con su identidad."""
    linked = linked or {}
    items = []
    for side, digest in enumerate(DIGESTS):
        folder = tmp_path / f"build-{side}"
        folder.mkdir()
        hashes = {}
        for kind, files in (("objects_sha256", objects), ("linked_sha256", linked)):
            hashes[kind] = {}
            for name, contents in files.items():
                path = folder / name
                if isinstance(contents[side], Path):
                    shutil.copyfile(contents[side], path)
                else:
                    path.write_bytes(contents[side])
                hashes[kind][name] = variants.sha256(path)
        receipt = tmp_path / f"receipt-{side}.json"
        receipt.write_text(
            json.dumps({"label": f"lado-{side}", "identity": {"sha256": digest}, **hashes})
        )
        items.append(f"{receipt}={folder}")
    output = tmp_path / "comparison.json"
    variants.command_compare(Namespace(first=items[0], second=items[1], output=output))
    return json.loads(output.read_text())


def test_objects_are_split_into_equal_identity_only_and_content(tmp_path):
    result = compare(
        tmp_path,
        {
            "same.o": (b"\x7fELF igual", b"\x7fELF igual"),
            "identity.o": tuple(b"antes " + digest.encode() + b" despues" for digest in DIGESTS),
            "code.o": (b"codigo uno", b"codigo dos"),
        },
    )["objects_sha256"]
    assert (result["total"], result["equal"]) == (3, 1)
    assert (result["identity_only"], result["content"]) == (1, 1)
    assert result["differences"]["identity.o"]["class"] == "identity_only"
    assert result["differences"]["identity.o"]["identity_occurrences"] == [1, 1]
    assert result["differences"]["code.o"]["class"] == "content"


def test_lto_bitcode_objects_are_content_without_section_table(tmp_path):
    # Con LTO los objetos son bitcode de LLVM y readelf no puede leerlos.
    result = compare(tmp_path, {"lto.o": (b"BC\xc0\xde uno", b"BC\xc0\xde dos")})
    difference = result["objects_sha256"]["differences"]["lto.o"]
    assert difference == {
        "class": "content",
        "identity_occurrences": [0, 0],
        "differing_sections": None,
    }


@pytest.fixture(scope="module")
def executables(tmp_path_factory):
    """Ejecutables mínimos que solo cambian en la RUNPATH o en el código."""
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None or shutil.which("readelf") is None:
        pytest.skip("Hace falta un compilador de C y readelf")
    folder = tmp_path_factory.mktemp("elf")
    built = {}
    # El nombre del fuente queda en la tabla de símbolos, así que cada valor usa el mismo.
    for name, value, runpath in (
        ("uno", 1, "/opt/ruta-a"),
        ("otro", 1, "/opt/ruta-b"),
        ("dos", 2, "/opt/ruta-a"),
    ):
        source = folder / f"valor-{value}" / "main.c"
        source.parent.mkdir(exist_ok=True)
        source.write_text(f"int main(void) {{ return {value}; }}\n")
        output = folder / name
        subprocess.run(
            [compiler, "-O2", f"-Wl,-rpath,{runpath}", "-o", str(output), "main.c"],
            cwd=source.parent,
            check=True,
            capture_output=True,
        )
        built[name] = output
    return built


def test_linked_outputs_that_only_change_link_metadata_are_separated(tmp_path, executables):
    result = compare(
        tmp_path,
        {},
        {
            "runpath": (executables["uno"], executables["otro"]),
            "code": (executables["uno"], executables["dos"]),
        },
    )["linked_sha256"]
    runpath, code = result["differences"]["runpath"], result["differences"]["code"]
    assert runpath["class"] == "identity_and_link_metadata"
    assert ".dynstr" in runpath["differing_sections"]
    assert set(runpath["differing_sections"]) <= variants.LINK_METADATA
    assert code["class"] == "content"
    assert ".text" in code["differing_sections"]


def test_comparison_refuses_builds_with_different_outputs(tmp_path):
    folder = tmp_path / "build"
    folder.mkdir()
    receipts = []
    for side, names in enumerate((["a.o"], ["a.o", "b.o"])):
        receipt = tmp_path / f"receipt-{side}.json"
        receipt.write_text(
            json.dumps(
                {
                    "label": str(side),
                    "identity": {"sha256": DIGESTS[side]},
                    "objects_sha256": dict.fromkeys(names, "0"),
                    "linked_sha256": {},
                }
            )
        )
        receipts.append(f"{receipt}={folder}")
    with pytest.raises(SystemExit, match="mismas salidas"):
        variants.command_compare(
            Namespace(first=receipts[0], second=receipts[1], output=tmp_path / "salida.json")
        )
