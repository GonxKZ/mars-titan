"""Las medidas de coste escriben su recibo con atomic_json en un destino nuevo (#802).

`run` se sustituye por una función que devuelve un recibo pequeño, así que no se mide nada.
"""

import importlib
import json

import pytest

from mars_titan.calibration.regime_weighted_conformal import WEIGHTINGS

# Medidas cuya orden separa el protocolo, sin --run, de la medida.
PROTOCOL_MEASURES = (
    "axial_time_region_cost",
    "cross_asset_attention_cost",
    "fractal_shared_operator_cost",
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


# Medidas que escribían el recibo en línea. Cada una lleva los argumentos obligatorios además de
# --output y un recibo con las claves que lee su resumen impreso.
INLINE_MEASURES = {
    "asof_ledger_audit_cost": ([], dict(cases=dict(c=dict(audit=dict(median_s=1.5))))),
    "calibrated_quantile_rule_cost": ([], dict(cases=dict(c=dict(compare=dict(median_s=1.5))))),
    "output_stage_cost": ([], dict(cases=dict(c=dict(emit_with_four_controls=1.5)))),
    "predictive_ability_cost": ([], dict(median_elapsed_s=1.5, days=3)),
    "regime_weighted_conformal_cost": (
        [],
        dict(
            cases=dict(c={weighting: dict(replay=dict(median_s=1.5)) for weighting in WEIGHTINGS})
        ),
    ),
    "pit_region_gate_cost": ([], dict(cases=dict(c=dict(fit=1.5)))),
    "output_self_edit_cost": ([], dict(cases=dict(c=dict(full_cycle=1.5)))),
    "online_conformal_cost": ([], dict(cases=dict(c=dict(online_replay=1.5)))),
    "regime_routing_cost": (
        ["--core-rows-per-second", "100"],
        dict(cases=dict(c=dict(event_p50_ms=1.5, share_of_core_event=0.25))),
    ),
    "associative_memory_cost": ([], dict(cases=dict(c=dict(write=dict(p50_ms=1.5, p95_ms=2.5))))),
}


@pytest.mark.parametrize("name", INLINE_MEASURES)
def test_an_inline_measure_refuses_an_existing_destination(name, monkeypatch, tmp_path):
    extra, receipt = INLINE_MEASURES[name]
    module, calls = measure(name, monkeypatch, receipt)
    existing = tmp_path / "cost.json"
    existing.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("sys.argv", [name, *extra, "--output", str(existing)])
    with pytest.raises(SystemExit, match="destino nuevo"):
        module.main()
    assert not calls and existing.read_text(encoding="utf-8") == "{}"


@pytest.mark.parametrize("name", INLINE_MEASURES)
def test_an_inline_measure_writes_its_receipt_in_a_new_folder(name, monkeypatch, tmp_path, capsys):
    extra, receipt = INLINE_MEASURES[name]
    module, calls = measure(name, monkeypatch, receipt)
    output = tmp_path / "new" / "folder" / "cost.json"
    monkeypatch.setattr("sys.argv", [name, *extra, "--output", str(output)])
    module.main()
    # La medida recibe los argumentos ya leídos, con el destino de la orden.
    assert len(calls) == 1 and calls[0][0].output == output
    assert json.loads(output.read_text(encoding="utf-8")) == receipt
    assert [path.name for path in output.parent.iterdir()] == ["cost.json"]
    assert "1.5" in capsys.readouterr().out


@pytest.mark.parametrize("name", INLINE_MEASURES)
def test_an_inline_receipt_with_nan_leaves_no_file(name, monkeypatch, tmp_path):
    extra, receipt = INLINE_MEASURES[name]
    module, _ = measure(name, monkeypatch, dict(receipt, value=float("nan")))
    output = tmp_path / "cost.json"
    monkeypatch.setattr("sys.argv", [name, *extra, "--output", str(output)])
    with pytest.raises(ValueError):
        module.main()
    assert not list(tmp_path.iterdir())
