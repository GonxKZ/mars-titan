"""Las medidas de coste escriben su recibo con atomic_json en un destino nuevo (#802).

`run` se sustituye por una función que devuelve un recibo pequeño, así que no se mide nada.
"""

import importlib
import json

import pytest

# Medidas cuya orden separa el protocolo, sin --run, de la medida.
PROTOCOL_MEASURES = (
    "cross_asset_attention_cost",
    "holographic_network_cost",
    "online_arbiters_cost",
    "regional_variants_cost",
    "nested_equal_depth_cost",
)


def measure(name, monkeypatch, result):
    """El módulo de la medida con `run` sustituido y la lista de sus llamadas."""
    module = importlib.import_module(f"benchmarks.{name}")
    calls = []

    def run(*arguments):
        calls.append(arguments)
        return result

    monkeypatch.setattr(module, "run", run)
    return module, calls


@pytest.mark.parametrize("name", PROTOCOL_MEASURES)
def test_the_protocol_does_not_measure(name, monkeypatch, capsys):
    module, calls = measure(name, monkeypatch, {})
    monkeypatch.setattr("sys.argv", [name])
    module.main()
    assert json.loads(capsys.readouterr().out) and not calls


@pytest.mark.parametrize("name", PROTOCOL_MEASURES)
def test_an_existing_destination_is_refused_before_measuring(name, monkeypatch, tmp_path):
    module, calls = measure(name, monkeypatch, {})
    existing = tmp_path / "cost.json"
    existing.write_text("{}", encoding="utf-8")
    for argv in ([name, "--run"], [name, "--run", "--output", str(existing)]):
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(SystemExit, match="destino nuevo"):
            module.main()
    assert not calls and existing.read_text(encoding="utf-8") == "{}"


@pytest.mark.parametrize("name", PROTOCOL_MEASURES)
def test_the_receipt_is_written_whole_in_a_new_folder(name, monkeypatch, tmp_path):
    receipt = dict(summary=[1.5], label="coste")
    module, calls = measure(name, monkeypatch, receipt)
    output = tmp_path / "new" / "folder" / "cost.json"
    monkeypatch.setattr("sys.argv", [name, "--run", "--output", str(output)])
    module.main()
    assert len(calls) == 1
    assert json.loads(output.read_text(encoding="utf-8")) == receipt
    # El temporal de la escritura no queda junto al recibo.
    assert [path.name for path in output.parent.iterdir()] == ["cost.json"]


@pytest.mark.parametrize("name", PROTOCOL_MEASURES)
def test_a_receipt_with_nan_is_refused_without_leaving_a_file(name, monkeypatch, tmp_path):
    module, _ = measure(name, monkeypatch, dict(value=float("nan")))
    output = tmp_path / "cost.json"
    monkeypatch.setattr("sys.argv", [name, "--run", "--output", str(output)])
    with pytest.raises(ValueError):
        module.main()
    assert not list(tmp_path.iterdir())
